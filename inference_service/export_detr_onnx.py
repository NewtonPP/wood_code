# inference_service/export_detr_onnx.py
"""
Export the trained DETR Lightning checkpoint to ONNX for the onnx backend.

The training script (Deep Learning Models/2DETR) wraps HuggingFace
``DetrForObjectDetection`` ("facebook/detr-resnet-101", num_labels=1) in a
PyTorch Lightning module whose attribute is ``self.model``, so the checkpoint's
state_dict keys are the HF keys with a ``model.`` prefix.

The exported graph matches the contract in ``inference_service/model.py`` +
``woodchip_core.geometry``:
  input:   pixel_values  float32 (1, 3, FIXED_H, FIXED_W)   # 800 x 1333
  outputs: logits        (1, num_queries, num_labels + 1)   # raw, softmaxed later
           pred_boxes    (1, num_queries, 4)                # normalized cxcywh

Usage (needs torch/transformers/onnx, NOT part of the serving image):
  python inference_service/export_detr_onnx.py \
      --ckpt /path/to/best-detr-repaired.ckpt \
      --out  inference_service/models/detr_resnet101.onnx
"""

import argparse

import torch
from transformers import DetrConfig, DetrForObjectDetection

FIXED_H, FIXED_W = 800, 1333  # woodchip_core.config.FIXED_H / FIXED_W


class _Wrapper(torch.nn.Module):
    """Return a plain (logits, pred_boxes) tuple so the ONNX graph has two
    named outputs instead of the HF output dataclass."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values)
        return out.logits, out.pred_boxes


def load_model(ckpt_path: str) -> DetrForObjectDetection:
    # NOTE: requires transformers<5 — v5 renamed the DETR backbone keys and no
    # longer matches this 2025 (v4-era) checkpoint.
    cfg = DetrConfig.from_pretrained(
        "facebook/detr-resnet-101",
        num_labels=1,
        id2label={0: "woodchip"},
        label2id={"woodchip": 0},
    )
    cfg.use_pretrained_backbone = False  # weights come from the checkpoint
    model = DetrForObjectDetection(cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    # Lightning module attribute is `self.model` -> strip that prefix.
    sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Tolerate only non-weight bookkeeping; anything else is a wrong checkpoint.
    problems = [k for k in missing + unexpected if "position_embedding" not in k]
    if problems:
        raise SystemExit(f"state_dict mismatch: {problems[:10]}")
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # eval() the wrapper too: onnx.export restores the top module's original
    # mode after tracing, which would flip the whole tree (dropout!) back on.
    model = _Wrapper(load_model(args.ckpt)).eval()
    dummy = torch.zeros(1, 3, FIXED_H, FIXED_W, dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy,),
        args.out,
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes"],
        opset_version=17,
        dynamo=False,
    )
    print(f"exported: {args.out}")

    # Parity check: ONNX vs eager torch on a random frame-like input.
    import numpy as np
    import onnxruntime as ort

    x = torch.rand(1, 3, FIXED_H, FIXED_W)
    with torch.no_grad():
        t_logits, t_boxes = model(x)
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    o_logits, o_boxes = sess.run(None, {"pixel_values": x.numpy()})
    dl = float(np.abs(t_logits.numpy() - o_logits).max())
    db = float(np.abs(t_boxes.numpy() - o_boxes).max())
    print(f"parity: max|dlogits|={dl:.2e}  max|dboxes|={db:.2e}")
    if dl > 1e-3 or db > 1e-3:
        raise SystemExit("ONNX output diverges from torch — do not deploy this file")


if __name__ == "__main__":
    main()
