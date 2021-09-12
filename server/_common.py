import logging
from gabriel_server import cognitive_engine
from gabriel_protocol import gabriel_pb2
import numpy as np
from collections import namedtuple
import sterling_pb2
import cv2
from functools import lru_cache
import threading
import tensorflow as tf
import torch
import torch.nn as nn
from torch.autograd import Function
from PIL import Image
import imagehash
import io

INPUT_QUEUE_MAXSIZE = 60
PORT = 9099
NUM_TOKENS = 1
NUM_SAME_HASH = 2
NUM_COMPLETED_STEP = 0
THRESHOLD = 0.4
DETECTOR_ONES_SIZE = (1, 480, 640, 3)
MAX_CACHE_SIZE = 3
MODEL_ON_STARTUP_NUM = 2
MODELS_NUM = 4

Model = namedtuple(
    "Model", ["obj_det_path", "classifer_path", "classes", "class_num"]
)
MODELS_INFO = [
    Model(
        obj_det_path="/object_detectors/cylinder/saved_model/",
        classifer_path="/classifiers/cylinder/model_best.pth.tar",
        classes=[
            "1screw",
            "2screws",
            "3screws",
            "nocylinder",
            "nopad",
            "nopiston",
            "noring",
            "noscrews",
        ],
        class_num=8,
    ),
    Model(
        obj_det_path="/object_detectors/rods/saved_model/",
        classifer_path="/classifiers/rods/model_best.pth.tar",
        classes=["2rods", "1strodoff", "1rod", "2ndrodon", "0rod"],
        class_num=5,
    ),
    Model(
        obj_det_path="/object_detectors/wheels/saved_model/",
        classifer_path="/classifiers/wheels/model_best.pth.tar",
        classes=["2wheels", "1wheel", "0wheel", "noshaft"],
        class_num=4,
    ),
    Model(
        obj_det_path="/object_detectors/screws/saved_model/",
        classifer_path="/classifiers/screws/model_best.pth.tar",
        classes=["finished", "2screws", "1screw", "0screw", ],
        class_num=4,
    ),
]

IMAGE_FORMAT = "/stirling/server/images/{}.jpg"
DONE_SPEECH = "You are done."
NOT_COMPLETE = "This step was not done."
NO_DETECTION = "Refocus on the engine."

CYLINDER_OBJ_DET = 0
CYLINDER_CLASSIFIER = 0
RODS_OBJ_DET = 1
RODS_CLASSIFIER = 1
WHEELS_OBJ_DET = 2
WHEELS_CLASSIFIER = 2
SCREWS_OBJ_DET = 3
SCREWS_CLASSIFIER = 3

ONESCREW = 0
TWOSCREWS = 1
THREESCREWS = 2
NOCYLINDER = 3
NOPAD = 4
NOPISTON = 5
NORING = 6
NOSCREWS = 7

TWORODS = 0
FIRSTRODOFF = 1
ONEROD = 2
SECONDRODON = 3
NOROD = 4

TWOWHEELS = 0
ONEWHEEL = 1
NOWHEEL = 2
NOSHAFT = 3

TWOSCREWS_BASE = 1
ONESCREW_BASE = 2
NOSCREW_BASE = 3
FINISHED = 0

logging.getLogger('gabriel_server.websocket_server').setLevel(logging.INFO)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class State:
    def __init__(
        self,
        speech,
        image_name,
        next_step,
        correct_class,
        obj_det,
        classifier,
        is_finishing_step=False,
    ):
        self._speech = speech
        self._image = open(IMAGE_FORMAT.format(image_name), "rb").read()
        self._next_step = next_step
        self._correct_class = correct_class
        self._obj_det = obj_det
        self._classifier = classifier
        self._is_finishing_step = is_finishing_step
        self._next_model_to_be_loaded = (
            (obj_det + MODEL_ON_STARTUP_NUM)
            if (obj_det + MODEL_ON_STARTUP_NUM) < MODELS_NUM
            else None
        )

    def get_speech(self):
        return self._speech

    def get_image(self):
        return self._image

    def get_next_step(self):
        return self._next_step

    def get_correct_class(self):
        return self._correct_class

    def get_obj_det(self):
        return self._obj_det

    def get_classifier(self):
        return self._classifier

    def is_finishing_step(self):
        return self._is_finishing_step

    def get_next_model_to_be_loaded(self):
        return self._next_model_to_be_loaded


STATES = {
    sterling_pb2.Step.FOURSCREWS: State(
        speech="Position the camera as shown",
        image_name="3screws",
        next_step=sterling_pb2.Step.THREESCREWS,
        correct_class=THREESCREWS,
        obj_det=CYLINDER_OBJ_DET,
        classifier=CYLINDER_CLASSIFIER,
    ),
    sterling_pb2.Step.THREESCREWS: State(
        speech="Remove the screw in the circled position.",
        image_name="2screws",
        next_step=sterling_pb2.Step.TWOSCREWS,
        correct_class=TWOSCREWS,
        obj_det=CYLINDER_OBJ_DET,
        classifier=CYLINDER_CLASSIFIER,
    ),
    sterling_pb2.Step.TWOSCREWS: State(
        speech="Remove the screw in the circled position.",
        image_name="1screw",
        next_step=sterling_pb2.Step.TWOSCREWSVISIBLE,
        correct_class=ONESCREW,
        obj_det=CYLINDER_OBJ_DET,
        classifier=CYLINDER_CLASSIFIER,
    ),
    sterling_pb2.Step.TWOSCREWSVISIBLE: State(
        speech="Flip the engine so that both remaining screws are visible.",
        image_name="2screwsflipped",
        next_step=sterling_pb2.Step.ONESCREW,
        correct_class=TWOSCREWS,
        obj_det=CYLINDER_OBJ_DET,
        classifier=CYLINDER_CLASSIFIER,
    ),
    sterling_pb2.Step.ONESCREW: State(
        speech="Remove the screw in the circled position.",
        image_name="1screwflipped",
        next_step=sterling_pb2.Step.NOSCREWS,
        correct_class=ONESCREW,
        obj_det=CYLINDER_OBJ_DET,
        classifier=CYLINDER_CLASSIFIER,
    ),
    sterling_pb2.Step.NOSCREWS: State(
        speech="Remove the last black screw.",
        image_name="noscrews",
        next_step=sterling_pb2.Step.NOPAD,
        correct_class=NOSCREWS,
        obj_det=CYLINDER_OBJ_DET,
        classifier=CYLINDER_CLASSIFIER,
    ),
    sterling_pb2.Step.NOPAD: State(
        speech="Remove the pad that was under the screws.",
        image_name="nopad",
        next_step=sterling_pb2.Step.NORING,
        correct_class=NOPAD,
        obj_det=CYLINDER_OBJ_DET,
        classifier=CYLINDER_CLASSIFIER,
    ),
    sterling_pb2.Step.NORING: State(
        speech="Remove the silicone ring.",
        image_name="noring",
        next_step=sterling_pb2.Step.NOCYLINDER,
        correct_class=NORING,
        obj_det=CYLINDER_OBJ_DET,
        classifier=CYLINDER_CLASSIFIER,
    ),
    sterling_pb2.Step.NOCYLINDER: State(
        speech="Remove the cylinder.",
        image_name="nocylinder",
        next_step=sterling_pb2.Step.NOPISTON,
        correct_class=NOCYLINDER,
        obj_det=CYLINDER_OBJ_DET,
        classifier=CYLINDER_CLASSIFIER,
    ),
    sterling_pb2.Step.NOPISTON: State(
        speech="Remove the piston.",
        image_name="nopiston",
        next_step=sterling_pb2.Step.TWORODS,
        correct_class=NOPISTON,
        obj_det=CYLINDER_OBJ_DET,
        classifier=CYLINDER_CLASSIFIER,
        is_finishing_step=True,
    ),
    sterling_pb2.Step.TWORODS: State(
        speech="position the camera as shown",
        image_name="1_2rods",
        next_step=sterling_pb2.Step.FIRSTRODOFF,
        correct_class=TWORODS,
        obj_det=RODS_OBJ_DET,
        classifier=RODS_CLASSIFIER,
    ),
    sterling_pb2.Step.FIRSTRODOFF: State(
        speech="Remove the front rod from the peg.",
        image_name="2_1strodoff",
        next_step=sterling_pb2.Step.ONEROD,
        correct_class=FIRSTRODOFF,
        obj_det=RODS_OBJ_DET,
        classifier=RODS_CLASSIFIER,
    ),
    sterling_pb2.Step.ONEROD: State(
        speech="Slide the front rod assembly to the right and then remove it from the engine.",
        image_name="3_1rod",
        next_step=sterling_pb2.Step.SECONDRODON,
        correct_class=ONEROD,
        obj_det=RODS_OBJ_DET,
        classifier=RODS_CLASSIFIER,
    ),
    sterling_pb2.Step.SECONDRODON: State(
        speech="Flip the engine so the remaining rod is facing you.",
        image_name="4_2ndrodon",
        next_step=sterling_pb2.Step.NOROD,
        correct_class=SECONDRODON,
        obj_det=RODS_OBJ_DET,
        classifier=RODS_CLASSIFIER,
    ),
    sterling_pb2.Step.NOROD: State(
        speech="Remove the front rod assembly from the engine",
        image_name="5_0rod",
        next_step=sterling_pb2.Step.TWOWHEELS,
        correct_class=NOROD,
        obj_det=RODS_OBJ_DET,
        classifier=RODS_CLASSIFIER,
        is_finishing_step=True,
    ),
    sterling_pb2.Step.TWOWHEELS: State(
        speech="Flip the engine so the small wheel is facing you.",
        image_name="6_2wheels",
        next_step=sterling_pb2.Step.ONEWHEEL,
        correct_class=TWOWHEELS,
        obj_det=WHEELS_OBJ_DET,
        classifier=WHEELS_CLASSIFIER,
    ),
    sterling_pb2.Step.ONEWHEEL: State(
        speech="Remove the small wheel in the circled position.",
        image_name="7_1wheel",
        next_step=sterling_pb2.Step.NOWHEEL,
        correct_class=ONEWHEEL,
        obj_det=WHEELS_OBJ_DET,
        classifier=WHEELS_CLASSIFIER,
    ),
    sterling_pb2.Step.NOWHEEL: State(
        speech="Remove the big wheel in the circled position.",
        image_name="8_0wheel",
        next_step=sterling_pb2.Step.NOSHAFT,
        correct_class=NOWHEEL,
        obj_det=WHEELS_OBJ_DET,
        classifier=WHEELS_CLASSIFIER,
    ),
    sterling_pb2.Step.NOSHAFT: State(
        speech="Remove the rod.",
        image_name="noshaft",
        next_step=sterling_pb2.Step.TWOSCREWS_BASE,
        correct_class=NOSHAFT,
        obj_det=WHEELS_OBJ_DET,
        classifier=WHEELS_CLASSIFIER,
    ),
    sterling_pb2.Step.TWOSCREWS_BASE: State(
        speech="Flip the engine and make sure the cylinder base is visible to the camera.",
        image_name="9_2screws",
        next_step=sterling_pb2.Step.ONESCREW_BASE,
        correct_class=TWOSCREWS_BASE,
        obj_det=SCREWS_OBJ_DET,
        classifier=SCREWS_CLASSIFIER,
    ),
    sterling_pb2.Step.ONESCREW_BASE: State(
        speech="Remove the screw in the circled position.",
        image_name="10_1screw",
        next_step=sterling_pb2.Step.NOSCREW_BASE,
        correct_class=ONESCREW_BASE,
        obj_det=SCREWS_OBJ_DET,
        classifier=SCREWS_CLASSIFIER,
    ),
    sterling_pb2.Step.NOSCREW_BASE: State(
        speech="remove the screw in the circled position.",
        image_name="11_0screw",
        next_step=sterling_pb2.Step.FINISHED,
        correct_class=NOSCREW_BASE,
        obj_det=SCREWS_OBJ_DET,
        classifier=SCREWS_CLASSIFIER,
    ),
    sterling_pb2.Step.FINISHED: State(
        speech="remove the cylinder base.",
        image_name="12_finished",
        next_step=sterling_pb2.Step.DONE,
        correct_class=FINISHED,
        obj_det=SCREWS_OBJ_DET,
        classifier=SCREWS_CLASSIFIER,
    ),
}

prev_step_dict = {
    sterling_pb2.Step.START: sterling_pb2.Step.START,
    sterling_pb2.Step.FOURSCREWS: sterling_pb2.Step.START,
}
for state_key in STATES:
    state = STATES[state_key]
    next_state_key = state.get_next_step()
    prev_step_dict[next_state_key] = state_key


@lru_cache(maxsize=MAX_CACHE_SIZE)
def fetch_model(model_num):
    logger.debug("loading model %d", model_num)
    model_info = MODELS_INFO[model_num]
    object_detector = tf.saved_model.load(model_info.obj_det_path)
    ones = tf.ones(DETECTOR_ONES_SIZE, dtype=tf.uint8)
    object_detector(ones)

    representation = {
        "function": MPNCOV,
        "iterNum": 5,
        "is_sqrt": True,
        "is_vec": True,
        "input_dim": 2048,
        "dimension_reduction": None,
    }
    freezed_layer = 0
    classifier = Newmodel(representation, model_info.class_num, freezed_layer)
    classifier.features = torch.nn.DataParallel(classifier.features)
    classifier.cuda()
    trained_model = torch.load(model_info.classifer_path)
    classifier.load_state_dict(trained_model["state_dict"])
    classifier.eval()
    return (object_detector, classifier)


def pack_to_client_extras(frames_completed_step, step, last_hash,
                          frames_same_hash):
    to_client_extras = sterling_pb2.ToClientExtras()
    to_client_extras.viewfinder_change = sterling_pb2.ViewfinderChange.DoNothing
    to_client_extras.frames_completed_step = frames_completed_step
    to_client_extras.step = step
    to_client_extras.last_hash = last_hash
    to_client_extras.frames_same_hash = frames_same_hash
    # str(last_hash) gives a type str with values 0-f
    return to_client_extras


def to_client_extras_from_to_server_extra(to_server_extras):
    return pack_to_client_extras(
        to_server_extras.frames_completed_step,
        to_server_extras.step,
        to_server_extras.last_hash,
        to_server_extras.frames_same_hash
    )


def get_last_hash(to_server_extras):
    if to_server_extras.last_hash == "":
        return None
    return imagehash.hex_to_hash(to_server_extras.last_hash)


def get_img_data(input_frame):
    np_data = np.frombuffer(input_frame.payloads[0], dtype=np.uint8)
    img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Image.fromarray(img) will give different values
    pil_img = Image.open(io.BytesIO(input_frame.payloads[0]))
    # phash requires a PIL instance
    img_hash = imagehash.phash(pil_img)
    return img, pil_img, img_hash


def _go_to_prev_step(to_server_extras):
    status = gabriel_pb2.ResultWrapper.Status.SUCCESS
    result_wrapper = cognitive_engine.create_result_wrapper(status)
    to_client_extras = to_client_extras_from_to_server_extra(to_server_extras)
    to_client_extras.step = prev_step_dict[to_server_extras.step]
    if to_client_extras.step != sterling_pb2.Step.START:
        prev_state = STATES[to_client_extras.step]
        to_client_extras.image = prev_state.get_image()
        to_client_extras.speech = prev_state.get_speech()
    to_client_extras.frames_completed_step = 0
    to_client_extras.frames_same_hash = 0
    result_wrapper.extras.Pack(to_client_extras)
    return result_wrapper


def _pack_start_or_done(to_server_extras):
    status = gabriel_pb2.ResultWrapper.Status.SUCCESS
    result_wrapper = cognitive_engine.create_result_wrapper(status)
    to_client_extras = to_client_extras_from_to_server_extra(to_server_extras)
    if to_server_extras.step == sterling_pb2.Step.START:
        to_client_extras.step = sterling_pb2.Step.FOURSCREWS
        next_state = STATES[to_client_extras.step]
        to_client_extras.image = next_state.get_image()
        to_client_extras.speech = next_state.get_speech()

    elif to_server_extras.step == sterling_pb2.Step.DONE:
        to_client_extras.step = sterling_pb2.Step.DONE

    result_wrapper.extras.Pack(to_client_extras)
    return result_wrapper


def check_early_return(to_server_extras, input_frame):

    if (
        to_server_extras.go_back
        and to_server_extras.step != sterling_pb2.Step.START
    ):
        logger.debug("back!")
        return _go_to_prev_step(to_server_extras)

    if (
        to_server_extras.step == sterling_pb2.Step.START
        or to_server_extras.step == sterling_pb2.Step.DONE
    ):
        return _pack_start_or_done(to_server_extras)

    if input_frame.payload_type != gabriel_pb2.PayloadType.IMAGE:
        status = gabriel_pb2.ResultWrapper.Statuss.SUCCESS
        result_wrapper = cognitive_engine.create_result_wrapper(status)
        status = gabriel_pb2.ResultWrapper.Status.WRONG_INPUT_FORMAT
        to_client_extras = to_client_extras_from_to_server_extra(
            to_server_extras
        )
        result_wrapper.extras.Pack(to_client_extras)
        return result_wrapper
    return None


def get_measurement(box, img):
    # from https://github.com/tensorflow/models/blob/39f98e30e7fb51c8b7ad58b0fc65ee5312829deb/research/object_detection/utils/visualization_utils.py#L1232
    ymin, xmin, ymax, xmax = box
    im_height, im_width = img.shape[:2]
    # from https://github.com/tensorflow/models/blob/39f98e30e7fb51c8b7ad58b0fc65ee5312829deb/official/vision/detection/utils/object_detection/visualization_utils.py#L192
    left = xmin * im_width
    right = xmax * im_width
    top = ymin * im_height
    bottom = ymax * im_height
    return (left, right, top, bottom)


def get_cropped_result(box, img):
    left, right, top, bottom = get_measurement(box, img)
    cropped = img[int(top): int(bottom), int(left): int(right)]

    # opencv needs cropped in BGR to encode properly
    cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
    _, jpeg_img = cv2.imencode(".jpg", cropped)
    result = gabriel_pb2.ResultWrapper.Result()
    result.payload_type = gabriel_pb2.PayloadType.IMAGE
    result.payload = jpeg_img.tobytes()
    return result


def handle_same_hash_frame(to_server_extras):
    status = gabriel_pb2.ResultWrapper.Status.SUCCESS
    result_wrapper = cognitive_engine.create_result_wrapper(status)
    to_client_extras = to_client_extras_from_to_server_extra(to_server_extras)
    to_client_extras.frames_same_hash = to_server_extras.frames_same_hash + 1
    result_wrapper.extras.Pack(to_client_extras)
    return result_wrapper


def get_box(scores, boxes):
    for score, box in zip(scores, boxes):
        if score >= THRESHOLD:
            return box
    return None


def handle_step_completed(to_server_extras, state, last_hash, frames_same_hash):

    if state.is_finishing_step():
        next_model_num = state.get_next_model_to_be_loaded()
        thread = threading.Thread(target=fetch_model, args=(next_model_num,))
        thread.start()

    to_client_extras = pack_to_client_extras(
        frames_completed_step=0,
        step=state.get_next_step(),
        last_hash=str(last_hash),
        frames_same_hash = frames_same_hash
    )
    if to_server_extras.step == sterling_pb2.Step.FINISHED:
        to_client_extras.speech = DONE_SPEECH
    else:
        next_state = STATES[to_client_extras.step]
        logger.debug("sending %s", next_state.get_speech())
        to_client_extras.image = next_state.get_image()
        to_client_extras.speech = next_state.get_speech()
    return to_client_extras


# The following code is Copyright (c) 2018 Peihua Li and Jiangtao Xie
# It came from: https://github.com/jiangtaoxie/fast-MPN-COV
# If you use this code, please cite the following paper:
# Peihua Li, Jiangtao Xie, Qilong Wang and Zilin Gao. Towards Faster Training of Global Covariance Pooling Networks by Iterative Matrix Square Root Normalization. IEEE Int. Conf. on Computer Vision and Pattern Recognition (CVPR), pp. 947-955, 2018.


class Basemodel(nn.Module):
    """Load backbone model and reconstruct it into three part:
    1) feature extractor
    2) global image representaion
    3) classifier
    """

    def __init__(self):
        super(Basemodel, self).__init__()
        basemodel = MPNCOVResNet(Bottleneck, [3, 4, 6, 3])
        basemodel = self._reconstruct_mpncovresnet(basemodel)

        self.features = basemodel.features
        self.representation = basemodel.representation
        self.classifier = basemodel.classifier
        self.representation_dim = basemodel.representation_dim

    def _reconstruct_mpncovresnet(self, basemodel):
        model = nn.Module()

        model.features = nn.Sequential(*list(basemodel.children())[:-1])
        model.representation_dim = basemodel.layer_reduce.weight.size(0)

        model.representation = None
        model.classifier = basemodel.fc
        return model

    def forward(self, x):
        x = self.features(x)
        x = self.representation(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


class Newmodel(Basemodel):
    def __init__(self, representation, num_classes, freezed_layer):
        super(Newmodel, self).__init__()

        representation_method = representation["function"]
        representation.pop("function")
        representation_args = representation
        representation_args["input_dim"] = self.representation_dim
        self.representation = representation_method(**representation_args)
        fc_input_dim = self.representation.output_dim

        self.classifier = nn.Linear(fc_input_dim, num_classes)

        index_before_freezed_layer = 0
        if freezed_layer:
            for m in self.features.children():
                if index_before_freezed_layer < freezed_layer:
                    m = self._freeze(m)
                index_before_freezed_layer += 1

    def _freeze(self, modules):
        for param in modules.parameters():
            param.requires_grad = False
        return modules


class MPNCOV(nn.Module):
    """Matrix power normalized Covariance pooling (MPNCOV)
       implementation of fast MPN-COV (i.e.,iSQRT-COV)
       https://arxiv.org/abs/1712.01034
    Args:
        iterNum: #iteration of Newton-schulz method
        is_sqrt: whether perform matrix square root or not
        is_vec: whether the output is a vector or not
        input_dim: the #channel of input feature
        dimension_reduction: if None, it will not use 1x1 conv to
                              reduce the #channel of feature.
                             if 256 or others, the #channel of feature
                              will be reduced to 256 or others.
    """

    def __init__(
        self,
        iterNum=3,
        is_sqrt=True,
        is_vec=True,
        input_dim=2048,
        dimension_reduction=None,
    ):

        super(MPNCOV, self).__init__()
        self.iterNum = iterNum
        self.is_sqrt = is_sqrt
        self.is_vec = is_vec
        self.dr = dimension_reduction
        if self.dr is not None:
            self.conv_dr_block = nn.Sequential(
                nn.Conv2d(
                    input_dim, self.dr, kernel_size=1, stride=1, bias=False
                ),
                nn.BatchNorm2d(self.dr),
                nn.ReLU(inplace=True),
            )
        output_dim = self.dr if self.dr else input_dim
        if self.is_vec:
            self.output_dim = int(output_dim * (output_dim + 1) / 2)
        else:
            self.output_dim = int(output_dim * output_dim)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _cov_pool(self, x):
        return Covpool.apply(x)

    def _sqrtm(self, x):
        return Sqrtm.apply(x, self.iterNum)

    def _triuvec(self, x):
        return Triuvec.apply(x)

    def forward(self, x):
        if self.dr is not None:
            x = self.conv_dr_block(x)
        x = self._cov_pool(x)
        if self.is_sqrt:
            x = self._sqrtm(x)
        if self.is_vec:
            x = self._triuvec(x)
        return x


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MPNCOVResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(MPNCOVResNet, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.layer_reduce = nn.Conv2d(
            512 * block.expansion,
            256,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.layer_reduce_bn = nn.BatchNorm2d(256)
        self.layer_reduce_relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(int(256 * (256 + 1) / 2), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 1x1 Conv. for dimension reduction
        x = self.layer_reduce(x)
        x = self.layer_reduce_bn(x)
        x = self.layer_reduce_relu(x)

        x = MPNCOV.CovpoolLayer(x)
        x = MPNCOV.SqrtmLayer(x, 5)
        x = MPNCOV.TriuvecLayer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class Covpool(Function):
    @staticmethod
    def forward(ctx, input):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        h = x.data.shape[2]
        w = x.data.shape[3]
        M = h * w
        x = x.reshape(batchSize, dim, M)
        I_hat = (-1.0 / M / M) * torch.ones(M, M, device=x.device) + (
            1.0 / M
        ) * torch.eye(M, M, device=x.device)
        I_hat = I_hat.view(1, M, M).repeat(batchSize, 1, 1).type(x.dtype)
        y = x.bmm(I_hat).bmm(x.transpose(1, 2))
        ctx.save_for_backward(input, I_hat)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        input, I_hat = ctx.saved_tensors
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        h = x.data.shape[2]
        w = x.data.shape[3]
        M = h * w
        x = x.reshape(batchSize, dim, M)
        grad_input = grad_output + grad_output.transpose(1, 2)
        grad_input = grad_input.bmm(x).bmm(I_hat)
        grad_input = grad_input.reshape(batchSize, dim, h, w)
        return grad_input


class Sqrtm(Function):
    @staticmethod
    def forward(ctx, input, iterN):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        I3 = 3.0 * torch.eye(dim, dim, device=x.device).view(
            1, dim, dim
        ).repeat(batchSize, 1, 1).type(dtype)
        normA = (1.0 / 3.0) * x.mul(I3).sum(dim=1).sum(dim=1)
        A = x.div(normA.view(batchSize, 1, 1).expand_as(x))
        Y = torch.zeros(
            batchSize, iterN, dim, dim, requires_grad=False, device=x.device
        ).type(dtype)
        Z = (
            torch.eye(dim, dim, device=x.device)
            .view(1, dim, dim)
            .repeat(batchSize, iterN, 1, 1)
            .type(dtype)
        )
        if iterN < 2:
            ZY = 0.5 * (I3 - A)
            YZY = A.bmm(ZY)
        else:
            ZY = 0.5 * (I3 - A)
            Y[:, 0, :, :] = A.bmm(ZY)
            Z[:, 0, :, :] = ZY
            for i in range(1, iterN - 1):
                ZY = 0.5 * (I3 - Z[:, i - 1, :, :].bmm(Y[:, i - 1, :, :]))
                Y[:, i, :, :] = Y[:, i - 1, :, :].bmm(ZY)
                Z[:, i, :, :] = ZY.bmm(Z[:, i - 1, :, :])
            YZY = 0.5 * Y[:, iterN - 2, :, :].bmm(
                I3 - Z[:, iterN - 2, :, :].bmm(Y[:, iterN - 2, :, :])
            )
        y = YZY * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
        ctx.save_for_backward(input, A, YZY, normA, Y, Z)
        ctx.iterN = iterN
        return y

    @staticmethod
    def backward(ctx, grad_output):
        input, A, ZY, normA, Y, Z = ctx.saved_tensors
        iterN = ctx.iterN
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        der_postCom = grad_output * torch.sqrt(normA).view(
            batchSize, 1, 1
        ).expand_as(x)
        der_postComAux = (
            (grad_output * ZY).sum(dim=1).sum(dim=1).div(2 * torch.sqrt(normA))
        )
        I3 = 3.0 * torch.eye(dim, dim, device=x.device).view(
            1, dim, dim
        ).repeat(batchSize, 1, 1).type(dtype)
        if iterN < 2:
            der_NSiter = 0.5 * (der_postCom.bmm(I3 - A) - A.bmm(der_postCom))
        else:
            dldY = 0.5 * (
                der_postCom.bmm(
                    I3 - Y[:, iterN - 2, :, :].bmm(Z[:, iterN - 2, :, :])
                )
                - Z[:, iterN - 2, :, :]
                .bmm(Y[:, iterN - 2, :, :])
                .bmm(der_postCom)
            )
            dldZ = -0.5 * Y[:, iterN - 2, :, :].bmm(der_postCom).bmm(
                Y[:, iterN - 2, :, :]
            )
            for i in range(iterN - 3, -1, -1):
                YZ = I3 - Y[:, i, :, :].bmm(Z[:, i, :, :])
                ZY = Z[:, i, :, :].bmm(Y[:, i, :, :])
                dldY_ = 0.5 * (
                    dldY.bmm(YZ)
                    - Z[:, i, :, :].bmm(dldZ).bmm(Z[:, i, :, :])
                    - ZY.bmm(dldY)
                )
                dldZ_ = 0.5 * (
                    YZ.bmm(dldZ)
                    - Y[:, i, :, :].bmm(dldY).bmm(Y[:, i, :, :])
                    - dldZ.bmm(ZY)
                )
                dldY = dldY_
                dldZ = dldZ_
            der_NSiter = 0.5 * (dldY.bmm(I3 - A) - dldZ - A.bmm(dldY))
        der_NSiter = der_NSiter.transpose(1, 2)
        grad_input = der_NSiter.div(normA.view(batchSize, 1, 1).expand_as(x))
        grad_aux = der_NSiter.mul(x).sum(dim=1).sum(dim=1)
        for i in range(batchSize):
            update = (
                der_postComAux[i] - (grad_aux[i] / (normA[i] * normA[i]))
            ) * torch.ones(dim, device=x.device).diag().type(dtype)
            grad_input[i, :, :] += update
        return grad_input, None


class Triuvec(Function):
    @staticmethod
    def forward(ctx, input):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        x = x.reshape(batchSize, dim * dim)
        I = torch.ones(dim, dim).triu().reshape(dim * dim)
        index = I.nonzero()
        y = torch.zeros(
            batchSize, int(dim * (dim + 1) / 2), device=x.device
        ).type(dtype)
        y = x[:, index]
        ctx.save_for_backward(input, index)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        input, index = ctx.saved_tensors
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        grad_input = torch.zeros(
            batchSize, dim * dim, device=x.device, requires_grad=False
        ).type(dtype)
        grad_input[:, index] = grad_output
        grad_input = grad_input.reshape(batchSize, dim, dim)
        return grad_input


def CovpoolLayer(var):
    return Covpool.apply(var)


def SqrtmLayer(var, iterN):
    return Sqrtm.apply(var, iterN)


def TriuvecLayer(var):
    return Triuvec.apply(var)
