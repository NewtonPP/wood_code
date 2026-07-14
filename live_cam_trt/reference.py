# live_cam_trt/reference.py
"""
Compatibility shim. Disk *detection* now lives in
``woodchip_core.reference.detect_reference_object`` (pure). The device's
single-camera online scale update (mutating module-global ``config``) stays
here so ``loop.py`` is unchanged.
"""

from woodchip_core.reference import detect_reference_object  # noqa: F401

from . import config


def update_scale_from_reference(frame_bgr):
    ref = detect_reference_object(frame_bgr)
    if ref is None or config.REF_DIAM_MM <= 0:
        return None

    cx, cy, diameter_px = ref
    ppm_new = diameter_px / float(config.REF_DIAM_MM)

    config.SCALE_BUF.append(ppm_new)
    if len(config.SCALE_BUF) > config.SCALE_ROLLING_MAX:
        del config.SCALE_BUF[: len(config.SCALE_BUF) - config.SCALE_ROLLING_MAX]

    if config.PIXELS_PER_MM is None:
        config.PIXELS_PER_MM = ppm_new
    else:
        config.PIXELS_PER_MM = 0.9 * config.PIXELS_PER_MM + 0.1 * ppm_new

    config.UNITS = "mm"
    return cx, cy, diameter_px
