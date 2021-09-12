import logging
from gabriel_server import cognitive_engine
from gabriel_server import local_engine
from gabriel_protocol import gabriel_pb2
import sterling_pb2
import server_phone
import _common

SOURCE = "stirling_glass"
MAX_FRAMES_NOT_DETECTED = 10
MIN_FRAMES_DETECTED = 1
DIFF_THRESHOLD = 10


def _check_for_refocus(to_server_extras, frames_completed_step):
    status = gabriel_pb2.ResultWrapper.Status.SUCCESS
    result_wrapper = cognitive_engine.create_result_wrapper(status)
    to_client_extras = _common.to_client_extras_from_to_server_extra(
        to_server_extras
    )
    to_client_extras.frames_completed_step = frames_completed_step
    if (
        to_server_extras.viewfinder_status
        == sterling_pb2.ViewfinderStatus.IsOff
    ):
        to_client_extras.viewfinder_change = (
            sterling_pb2.ViewfinderChange.TurnOn
        )
        to_client_extras.detected_frames = 0
        to_client_extras.undetected_frames = 0
        _common.logger.debug("can't detect object! ")
        to_client_extras.speech = _common.NO_DETECTION
    result_wrapper.extras.Pack(to_client_extras)
    return result_wrapper


def _handle_same_hash_frame_glass(
    to_server_extras, undetected_frames, detected_frames, frames_completed_step
):
    to_client_extras = _common.to_client_extras_from_to_server_extra(
        to_server_extras
    )
    status = gabriel_pb2.ResultWrapper.Status.SUCCESS
    result_wrapper = cognitive_engine.create_result_wrapper(status)
    if to_server_extras.last_frame_undetected:
        undetected_frames += 1
        if undetected_frames == MAX_FRAMES_NOT_DETECTED:
            return _check_for_refocus(to_server_extras, frames_completed_step)
        to_client_extras.detected_frames = 0
        to_client_extras.last_frame_undetected = True
        to_client_extras.undetected_frames = undetected_frames
    else:
        to_client_extras.detected_frames = detected_frames
        to_client_extras.last_frame_undetected = False
        to_client_extras.undetected_frames = 0
    to_client_extras.frames_same_hash = to_server_extras.frames_same_hash + 1
    result_wrapper.extras.Pack(to_client_extras)
    return result_wrapper


class GlassInferenceEngine(cognitive_engine.Engine):
    def __init__(self):
        self._base_engine = server_phone.InferenceEngine()

    def handle_step_completed_glass(
        self,
        to_server_extras,
        state,
        detected_frames,
        last_hash,
        frames_same_hash,
    ):
        to_client_extras = _common.handle_step_completed(
            to_server_extras, state, last_hash, frames_same_hash
        )
        to_client_extras.detected_frames = detected_frames + 1
        to_client_extras.undetected_frames = 0
        return to_client_extras

    def handle_detection_glass(
        self,
        to_server_extras,
        frames_completed_step,
        img,
        pil_img,
        last_hash,
        detected_frames,
        undetected_frames,
        frames_same_hash,
    ):
        box, to_client_extras = self._base_engine.handle_detection(
            to_server_extras,
            frames_completed_step,
            img,
            pil_img,
            last_hash,
            frames_same_hash,
        )
        status = gabriel_pb2.ResultWrapper.Status.SUCCESS
        result_wrapper = cognitive_engine.create_result_wrapper(status)
        if box is None:
            undetected_frames += 1
            if undetected_frames == MAX_FRAMES_NOT_DETECTED:
                return _check_for_refocus(
                    to_server_extras, frames_completed_step
                )
            to_client_extras.detected_frames = 0
            to_client_extras.undetected_frames = undetected_frames
            to_client_extras.last_frame_undetected = True
        else:
            cropped_result = _common.get_cropped_result(box, img)
            result_wrapper.results.append(cropped_result)
            to_client_extras.detected_frames = detected_frames + 1
            to_client_extras.undetected_frames = 0
            to_client_extras.last_frame_undetected = False
            if to_client_extras.detected_frames >= MIN_FRAMES_DETECTED:
                _common.logger.debug("Detected")
                if (
                    to_server_extras.viewfinder_status
                    == sterling_pb2.ViewfinderStatus.IsOn
                ):
                    to_client_extras.viewfinder_change = (
                        sterling_pb2.ViewfinderChange.TurnOff
                    )
                    to_client_extras.detected_frames = 0
        result_wrapper.extras.Pack(to_client_extras)
        return result_wrapper

    def handle(self, input_frame):
        to_server_extras = cognitive_engine.unpack_extras(
            sterling_pb2.ToServerExtras, input_frame
        )
        res = _common.check_early_return(to_server_extras, input_frame)
        if res is not None:
            return res

        frames_completed_step = to_server_extras.frames_completed_step
        frames_same_hash = to_server_extras.frames_same_hash
        detected_frames = to_server_extras.detected_frames
        undetected_frames = to_server_extras.undetected_frames
        img, pil_img, img_hash = _common.get_img_data(input_frame)
        last_hash = _common.get_last_hash(to_server_extras)

        if last_hash is None or (img_hash - last_hash) >= DIFF_THRESHOLD:
            _common.logger.debug("new hash")
            undetected_frames = 0
            frames_completed_step = 0
            last_hash = img_hash
            frames_same_hash = 0
            # Do not return the result wrapper so we can show the
            # user the crop.

        elif frames_same_hash != _common.NUM_SAME_HASH:
            return _handle_same_hash_frame_glass(
                to_server_extras,
                undetected_frames,
                detected_frames,
                frames_completed_step,
            )
        else:
            frames_same_hash += 1

        return self.handle_detection_glass(
            to_server_extras,
            frames_completed_step,
            img,
            pil_img,
            last_hash,
            detected_frames,
            undetected_frames,
            frames_same_hash,
        )


def main():
    def engine_factory():
        return GlassInferenceEngine()

    local_engine.run(
        engine_factory,
        SOURCE,
        _common.INPUT_QUEUE_MAXSIZE,
        _common.PORT,
        _common.NUM_TOKENS,
    )


if __name__ == "__main__":
    main()
