# live_cam_trt/engines.py
"""
Stable TensorRT engine wrappers.

Stability rules preserved from the original implementation:
- single global TRT_LOGGER
- re-allocate bindings when the FULL input shape changes
- resolve outputs by binding name when possible
"""

import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

from .geometry import safe_softmax_2d


# =================== TRT LOGGER (single global) ===================
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class GenericTRTWrapper:
    """
    Stable TRT wrapper:
    - single global TRT_LOGGER
    - re-alloc bindings when FULL input shape changes
    - uses binding names to resolve outputs if needed
    """
    def __init__(self, engine_path: str):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine not found: {engine_path}")

        self.logger = TRT_LOGGER
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.num_bindings = self.engine.num_bindings
        self.binding_names = [self.engine.get_binding_name(i) for i in range(self.num_bindings)]

        self.input_idx = [i for i in range(self.num_bindings) if self.engine.binding_is_input(i)][0]
        self.output_indices = [i for i in range(self.num_bindings) if not self.engine.binding_is_input(i)]

        self.bindings = [None] * self.num_bindings
        self.host_mem = {}
        self.device_mem = {}
        self.last_input_shape = None

    def _alloc(self, idx: int, shape: tuple):
        dtype = trt.nptype(self.engine.get_binding_dtype(idx))
        n_elts = int(np.prod(shape))
        host = cuda.pagelocked_empty(n_elts, dtype)
        dev = cuda.mem_alloc(host.nbytes)
        self.host_mem[idx] = host
        self.device_mem[idx] = dev
        self.bindings[idx] = int(dev)

    def _ensure_buffers(self, input_shape: tuple):
        input_shape = tuple(input_shape)
        if self.last_input_shape == input_shape:
            return

        self.context.set_binding_shape(self.input_idx, input_shape)

        # allocate ALL bindings based on shapes AFTER set_binding_shape
        for i in range(self.num_bindings):
            shape = tuple(self.context.get_binding_shape(i))
            self._alloc(i, shape)

        self.last_input_shape = input_shape

    def infer_all(self, x: np.ndarray):
        x = np.ascontiguousarray(x, dtype=np.float32)
        self._ensure_buffers(x.shape)

        np.copyto(self.host_mem[self.input_idx], x.ravel())
        cuda.memcpy_htod_async(self.device_mem[self.input_idx], self.host_mem[self.input_idx], self.stream)

        self.context.execute_async_v2(self.bindings, self.stream.handle)

        for oi in self.output_indices:
            cuda.memcpy_dtoh_async(self.host_mem[oi], self.device_mem[oi], self.stream)

        self.stream.synchronize()

        outs = {}
        for oi in self.output_indices:
            shape = tuple(self.context.get_binding_shape(oi))
            name = self.binding_names[oi] or f"out_{oi}"
            outs[name] = np.array(self.host_mem[oi], copy=True).reshape(shape)
        return outs


class DETRTRT(GenericTRTWrapper):
    """Return (logits, pred_boxes) exactly like your old code expected."""
    def infer(self, x: np.ndarray):
        outs = self.infer_all(x)

        logits = None
        pred_boxes = None

        for k, v in outs.items():
            lk = (k or "").lower()
            if ("logit" in lk) and (logits is None):
                logits = v
            if (("pred_boxes" in lk) or ("box" in lk)) and (pred_boxes is None):
                pred_boxes = v

        # Fallback if names aren't helpful
        if logits is None or pred_boxes is None:
            vals = list(outs.values())
            if len(vals) == 2:
                a, b = vals
                if a.shape[-1] == 4:
                    pred_boxes, logits = a, b
                elif b.shape[-1] == 4:
                    pred_boxes, logits = b, a
                else:
                    logits, pred_boxes = a, b
            else:
                raise RuntimeError("Unable to resolve DETR TRT outputs (logits/pred_boxes).")

        return logits, pred_boxes


class MoistureTRT(GenericTRTWrapper):
    """Return probabilities (N,C)."""
    def infer_probs_bhwc(self, crops_bhwc: np.ndarray):
        if crops_bhwc is None or len(crops_bhwc) == 0:
            return None

        # engine may expect BCHW or BHWC
        input_shape = tuple(self.engine.get_binding_shape(self.input_idx))
        if len(input_shape) == 4 and input_shape[-1] == 3:
            x = crops_bhwc
        else:
            x = np.transpose(crops_bhwc, (0, 3, 1, 2))

        outs = self.infer_all(x)
        out = list(outs.values())[0]
        if out.ndim > 2:
            out = out.reshape(out.shape[0], -1)

        # softmax if not already probs
        row_sum = np.sum(out, axis=1)
        if np.any(row_sum < 0.8) or np.any(row_sum > 1.2) or np.any(out < 0.0) or np.any(out > 1.0):
            out = safe_softmax_2d(out, axis=1)

        return out
