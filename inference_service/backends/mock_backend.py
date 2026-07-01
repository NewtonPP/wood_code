# inference_service/backends/mock_backend.py
"""
Synthetic backend: fabricates plausible DETR detections and moisture
probabilities so the whole capture -> cloud -> overlay loop works end-to-end
before real model files exist. Mirrors the logic in ``live_cam_trt/mock.py`` but
returns *raw* model outputs (boxes/scores, probs) for the shared
``FrameProcessor`` to post-process — so the mock and real paths are identical
downstream.
"""

from typing import List, Optional, Tuple

import numpy as np

from .base import ModelBackend

# A fixed scale so diameters render in mm and the histogram uses a stable range,
# matching the previous on-device mock demo.
_MOCK_PX_PER_MM = 4.0


class MockBackend(ModelBackend):
    moisture_classes: List[str] = ["Dry", "Normal", "Wet"]
    default_pixels_per_mm: Optional[float] = _MOCK_PX_PER_MM

    def __init__(self):
        self._rng = np.random.default_rng()

    def infer_detr(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = frame_bgr.shape[:2]
        rng = self._rng
        n = int(rng.integers(4, 10))
        # diameters (mm) -> square-ish boxes (px) via the mock scale
        d_mm = np.clip(rng.normal(45.0, 12.0, size=n), 5.0, 120.0)
        boxes = []
        scores = []
        for d in d_mm:
            side = max(4, int(d * _MOCK_PX_PER_MM / 1.4142))
            side = min(side, max(4, min(w, h) - 20))
            x1 = int(rng.integers(5, max(6, w - side - 5)))
            y1 = int(rng.integers(5, max(6, h - side - 5)))
            boxes.append([x1, y1, x1 + side, y1 + side])
            scores.append(float(rng.uniform(0.55, 0.98)))
        return np.asarray(boxes, np.float32), np.asarray(scores, np.float32)

    def infer_moisture(self, crops_bhwc: np.ndarray) -> Optional[np.ndarray]:
        n = int(crops_bhwc.shape[0])
        k = len(self.moisture_classes)
        logits = self._rng.normal(0.0, 1.0, size=(n, k))
        ex = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = ex / np.sum(ex, axis=1, keepdims=True)
        return probs.astype(np.float32)
