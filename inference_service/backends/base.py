# inference_service/backends/base.py
"""
Model backend interface.

A backend is responsible *only* for the model math; all post-processing is done
by ``woodchip_core.FrameProcessor``. Implementations must be safe to call from
multiple per-device processors sharing the one backend instance.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np


class ModelBackend(ABC):
    #: Moisture class labels in model-output order (e.g. ["Dry", "Normal", "Wet"]).
    moisture_classes: Optional[List[str]] = None

    #: For mock/demo backends, an optional forced px/mm so the UI shows mm even
    #: without a physical reference disk in frame. ``None`` for real models.
    default_pixels_per_mm: Optional[float] = None

    @abstractmethod
    def infer_detr(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (boxes_xyxy float32 [N,4], scores float32 [N]) in frame pixels."""
        raise NotImplementedError

    @abstractmethod
    def infer_moisture(self, crops_bhwc: np.ndarray) -> Optional[np.ndarray]:
        """Return per-crop class probabilities float32 [N, num_classes], or None."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        return type(self).__name__
