# live_cam_trt/overlay.py
"""
Compatibility shim — stats + on-frame rendering now live in
``woodchip_core.overlay``. Re-exported so ``loop.py`` / ``mock.py`` are unchanged.
"""

from woodchip_core.overlay import (  # noqa: F401
    classify_batch,
    compute_diameter_stats,
    draw_stats_panel,
    draw_color_legend,
)
