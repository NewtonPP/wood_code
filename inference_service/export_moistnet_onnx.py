# inference_service/export_moistnet_onnx.py
"""
Export the trained MoistNetLite Keras weights to ONNX for the moisture model.

The training script (Deep Learning Models/3moistnetlite.py) saved weights-only
HDF5 (Keras 2.10). The graph is rebuilt here in inference form — the
``random_translation`` augmentation layer is dropped (it has no weights and is
identity at inference) — and the weights are loaded ``by_name``.

Hyperparameters are fixed to what the checkpoint actually contains (verified
against the h5 tensor shapes): num_layers=3, num_filters=32,
dense_layer_size=256, 3-class softmax head named ``classification_head_2``.

The exported graph matches the contract in ``inference_service/model.py``:
  input:  float32 (N, 224, 224, 3) NHWC crops as produced by
          ``woodchip_core.moisture.prep_crop`` — scaled to 0-1 then
          ImageNet mean/std normalized.
  output: float32 (N, 3) class probabilities (softmax already applied).

Because the runtime always feeds ImageNet-normalized crops, the graph starts
with an affine layer that converts them back to whatever the model saw during
training (``--train-preproc``):
  raw       pixels 0-255 (default: the training script's Normalization layer
            is commented out, so the generators fed raw pixels)
  unit      pixels 0-1 (rescale=1/255 generators)
  imagenet  same as runtime — no compensation

Usage (needs tensorflow 2.x [Keras 2] + tf2onnx, NOT part of the serving image):
  python inference_service/export_moistnet_onnx.py \
      --weights moistnetlite_best_weights.h5 \
      --out inference_service/models/moistnetlite.onnx
"""

import argparse

import h5py
import numpy as np
import tensorflow as tf

INPUT_SIZE = 224  # woodchip_core.config.MOISTURE_INPUT_SIZE
# woodchip_core.config.MEAN / STD (ImageNet), applied by prep_crop at runtime.
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# Layers that carry weights in the checkpoint: name -> (kernel_shape, bias_shape)
EXPECTED_WEIGHTS = {
    "conv2d_0_0": ((3, 3, 3, 32), (32,)),
    "conv2d_1_0": ((3, 3, 32, 32), (32,)),
    "conv2d_2_0": ((3, 3, 32, 512), (512,)),
    "dense": ((512, 256), (256,)),
    "classification_head_2": ((256, 3), (3,)),
}


def build_inference_model(train_preproc: str) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3), name="input_1")

    # Undo the runtime ImageNet normalization to recover the training input.
    if train_preproc == "raw":
        x = inputs * (STD * 255.0) + (MEAN * 255.0)
    elif train_preproc == "unit":
        x = inputs * STD + MEAN
    else:  # imagenet: runtime input == training input
        x = inputs

    for i, filters in enumerate((32, 32, 512)):
        x = tf.keras.layers.Conv2D(filters, (3, 3), activation="relu", name=f"conv2d_{i}_0")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), name=f"max_pooling2d_{i}")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense")(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax", name="classification_head_2")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="moistnetlite")


def load_and_verify_weights(model: tf.keras.Model, weights_path: str) -> None:
    """load_weights(by_name=True) silently skips mismatched layers, so compare
    every tensor against the h5 file afterwards."""
    model.load_weights(weights_path, by_name=True)
    with h5py.File(weights_path, "r") as f:
        for name, (kshape, bshape) in EXPECTED_WEIGHTS.items():
            kernel, bias = model.get_layer(name).get_weights()
            h5_kernel = np.asarray(f[name][name]["kernel:0"])
            h5_bias = np.asarray(f[name][name]["bias:0"])
            if kernel.shape != kshape or not np.array_equal(kernel, h5_kernel):
                raise SystemExit(f"layer {name!r}: kernel not loaded from checkpoint")
            if bias.shape != bshape or not np.array_equal(bias, h5_bias):
                raise SystemExit(f"layer {name!r}: bias not loaded from checkpoint")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--train-preproc",
        choices=("raw", "unit", "imagenet"),
        default="raw",
        help="pixel format the model was trained on (see module docstring)",
    )
    args = ap.parse_args()

    model = build_inference_model(args.train_preproc)
    load_and_verify_weights(model, args.weights)

    import tf2onnx

    spec = (tf.TensorSpec((None, INPUT_SIZE, INPUT_SIZE, 3), tf.float32, name="input_1"),)
    tf2onnx.convert.from_keras(model, input_signature=spec, opset=17, output_path=args.out)
    print(f"exported: {args.out}  (train_preproc={args.train_preproc})")

    # Parity check: ONNX vs Keras on random runtime-like (normalized) crops.
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    raw = rng.uniform(0.0, 255.0, size=(4, INPUT_SIZE, INPUT_SIZE, 3)).astype(np.float32)
    x = (raw / 255.0 - MEAN) / STD  # what prep_crop feeds at runtime
    k_probs = model.predict(x, verbose=0)
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    o_probs = sess.run(None, {inp: x})[0]
    dp = float(np.abs(k_probs - o_probs).max())
    print(f"parity: max|dprobs|={dp:.2e}")
    print(f"sample probs (dry/medium/wet): {np.round(o_probs, 4).tolist()}")
    if dp > 1e-4:
        raise SystemExit("ONNX output diverges from Keras — do not deploy this file")
    if not np.allclose(o_probs.sum(axis=1), 1.0, atol=1e-3):
        raise SystemExit("outputs are not probabilities — check the softmax head")


if __name__ == "__main__":
    main()
