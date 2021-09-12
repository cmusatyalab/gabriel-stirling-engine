import logging
from gabriel_server import cognitive_engine
from gabriel_server import local_engine
from gabriel_protocol import gabriel_pb2
import numpy as np
from torchvision import transforms
import tensorflow as tf
import sterling_pb2
import _common

SOURCE = "stirling_phone"
DIFF_THRESHOLD = 10

class InferenceEngine(cognitive_engine.Engine):
    def __init__(self):
        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self._transform = transforms.Compose(
            [
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                normalize,
            ]
        )
        for i in range(_common.MODEL_ON_STARTUP_NUM):
            _common.fetch_model(i)

    def _get_prediction(
        self, state, box, img, pil_img, classifier_model, model_num
    ):
        left, right, top, bottom = _common.get_measurement(box, img)
        cropped_pil = pil_img.crop((left, top, right, bottom))
        transformed = self._transform(cropped_pil).cuda()

        output = classifier_model(transformed[None, ...])
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        model_info = _common.MODELS_INFO[model_num]
        _common.logger.debug(
            "Predicted: %s, Current: %s",
            model_info.classes[pred],
            model_info.classes[state.get_correct_class()],
        )
        return pred

    def handle_detection(
        self,
        to_server_extras,
        frames_completed_step,
        img,
        pil_img,
        last_hash,
        frames_same_hash
    ):
        state = _common.STATES[to_server_extras.step]
        model_num = state.get_obj_det()
        obj_det_model, classifier_model = _common.fetch_model(model_num)

        detections = obj_det_model(np.expand_dims(img, 0))
        scores = detections["detection_scores"][0].numpy()
        boxes = detections["detection_boxes"][0].numpy()
        box = _common.get_box(scores, boxes)
        to_client_extras = self._get_to_client_extras_from_detection(
            state,
            model_num,
            to_server_extras,
            frames_completed_step,
            img,
            pil_img,
            last_hash,
            classifier_model,
            box,
            frames_same_hash
        )
        return (box, to_client_extras)

    def _get_to_client_extras_from_detection(
        self,
        state,
        model_num,
        to_server_extras,
        frames_completed_step,
        img,
        pil_img,
        last_hash,
        classifier_model,
        box,
        frames_same_hash
    ):
        if (box is None) or (frames_same_hash == 0):
            return _common.pack_to_client_extras(
                frames_completed_step=frames_completed_step,
                step=to_server_extras.step,
                last_hash=str(last_hash),
                frames_same_hash=frames_same_hash
            )
        else:
            _common.logger.debug("found object")
            pred = self._get_prediction(
                state, box, img, pil_img, classifier_model, model_num
            )
            if pred == state.get_correct_class():
                if (
                    to_server_extras.frames_completed_step
                    < _common.NUM_COMPLETED_STEP
                ):
                    to_client_extras = _common.pack_to_client_extras(
                        frames_completed_step=(
                            to_server_extras.frames_completed_step + 1
                        ),
                        step=to_server_extras.step,
                        last_hash=str(last_hash),
                        frames_same_hash=frames_same_hash
                    )
                else:
                    to_client_extras = _common.handle_step_completed(
                        to_server_extras, state, last_hash, frames_same_hash
                    )
            else:
                to_client_extras = _common.pack_to_client_extras(
                    frames_completed_step=0,
                    step=to_server_extras.step,
                    last_hash=str(last_hash),
                    frames_same_hash=frames_same_hash
                )
            return to_client_extras

    def handle(self, input_frame):
        to_server_extras = cognitive_engine.unpack_extras(
            sterling_pb2.ToServerExtras, input_frame
        )
        res = _common.check_early_return(to_server_extras, input_frame)
        if res is not None:
            return res

        frames_completed_step = to_server_extras.frames_completed_step
        frames_same_hash = to_server_extras.frames_same_hash
        last_hash = _common.get_last_hash(to_server_extras)
        img, pil_img, img_hash = _common.get_img_data(input_frame)

        if(
            last_hash is None
            or (img_hash - last_hash) >= DIFF_THRESHOLD
        ):
            _common.logger.debug("new hash")
            frames_completed_step = 0
            last_hash = img_hash
            frames_same_hash = 0
            # Do not return the result wrapper so we can show the
            # user the crop.
        elif frames_same_hash != _common.NUM_SAME_HASH:
            return _common.handle_same_hash_frame(to_server_extras)
        else:
            frames_same_hash += 1
    
        box, to_client_extras = self.handle_detection(
            to_server_extras,
            frames_completed_step,
            img,
            pil_img,
            last_hash,
            frames_same_hash
        )
        status = gabriel_pb2.ResultWrapper.Status.SUCCESS
        result_wrapper = cognitive_engine.create_result_wrapper(status)
        if box is not None:
            cropped_result = _common.get_cropped_result(box, img)
            result_wrapper.results.append(cropped_result)
        result_wrapper.extras.Pack(to_client_extras)
        return result_wrapper


def main():
    
    def engine_factory():
        return InferenceEngine()

    local_engine.run(
        engine_factory,
        SOURCE,
        _common.INPUT_QUEUE_MAXSIZE,
        _common.PORT,
        _common.NUM_TOKENS,
    )


if __name__ == "__main__":
    main()
