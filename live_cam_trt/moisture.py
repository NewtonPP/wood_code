# live_cam_trt/moisture.py
"""
Compatibility shim. The crop preprocessing / aggregation now live in
``woodchip_core.moisture``. The device keeps its original private signatures
(reading module-global ``config`` and the Jetson class-labels file) so
``loop.py`` is unchanged.
"""

import os

from woodchip_core.moisture import (
    prep_crop as _prep_crop,
    topk_crops as _topk_crops,
    moisture_aggregate as _moisture_aggregate_core,
)

from . import config


def _read_moist_classes(path: str):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return [l.strip() for l in f if l.strip()]


MOISTURE_CLASSES = _read_moist_classes(config.MOIST_CLASSES_PATH)


def _prep_moist_crop_bgr(crop_bgr, size):
    return _prep_crop(crop_bgr, size)


def _boxes_to_topk_crops(frame_bgr, boxes_xyxy, scores, topk):
    return _topk_crops(
        frame_bgr, boxes_xyxy, scores, topk,
        min_box_px=config.MOISTURE_MIN_BOX_PX,
        input_size=config.MOISTURE_INPUT_SIZE,
    )


def _moisture_aggregate(probs):
    return _moisture_aggregate_core(probs, MOISTURE_CLASSES)
