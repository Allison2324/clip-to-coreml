import time
import os
import math

import numpy as np
import torch
import coremltools as ct
import open_clip


MODEL_NAME = "ViT-B-16"
PRETRAINED = "laion2b_s34b_b88k"

BATCH_SIZE = 32
IMAGE_SIZE = 224

ONNX_PATH = f"clip_image_encoder_{MODEL_NAME}_B{BATCH_SIZE}.onnx"
OUT_NAME = f"ClipImageEncoder_{MODEL_NAME}_B{BATCH_SIZE}"
MLPACKAGE_PATH = f"{OUT_NAME}.mlpackage"


class ImageEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        feats = self.model.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return feats


def fmt_time(sec: float) -> str:
    sec = float(sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def main():
    t0 = time.time()

    print("Step 0/4: environment info")
    print(f"torch:        {torch.__version__}")
    print(f"coremltools:  {ct.__version__}")
    try:
        print(f"open_clip:    {open_clip.__version__}")
    except Exception:
        print("open_clip:    (no __version__ attr)")
    print(f"model name:   {MODEL_NAME}")
    print(f"pretrained:   {PRETRAINED}")
    print(f"batch size:   {BATCH_SIZE}")
    print(f"image size:   {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"onnx path:    {ONNX_PATH}")
    print(f"output name:  {OUT_NAME}.mlpackage")
    print("------------------------------------------------------------", flush=True)

    # ---------------------------------------
    # Step 1: загрузка CLIP
    # ---------------------------------------
    t1 = time.time()
    print("Step 1/4: loading CLIP model...")

    device = "cpu"
    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
        device=device
    )
    model.eval()
    encoder = ImageEncoder(model).eval()

    t2 = time.time()
    print(f"Step 1/4 done in {fmt_time(t2 - t1)}", flush=True)

    # ---------------------------------------
    # Step 2: dummy-прогон и экспорт в ONNX
    # ---------------------------------------
    print("Step 2/4: building encoder and running dummy forward + ONNX export...")

    dummy = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    with torch.no_grad():
        out = encoder(dummy)

    out_shape = tuple(out.shape)
    emb_dim = out_shape[-1]
    total_params = sum(p.numel() for p in encoder.parameters())

    print(f"Encoder dummy input shape: {tuple(dummy.shape)}")
    print(f"Encoder output shape:      {out_shape}")
    print(f"Embedding dim:             {emb_dim}")
    print(f"Total parameters:          {total_params:,}")

    t2a = time.time()
    print("Exporting to ONNX...")
    torch.onnx.export(
        encoder,
        dummy,
        ONNX_PATH,
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={
            "image": {0: "batch"},
            "embedding": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    t3 = time.time()
    print(f"ONNX exported to:          {ONNX_PATH}")
    print(f"Step 2/4 done in {fmt_time(t3 - t2)}", flush=True)

    # ---------------------------------------
    # Step 3: конвертация ONNX -> CoreML
    # ---------------------------------------
    print("Step 3/4: converting ONNX to CoreML (this step can take a long time)...", flush=True)

    mlmodel = ct.convert(
        ONNX_PATH,
        source="onnx",
        minimum_deployment_target=ct.target.iOS15,
        compute_units=ct.ComputeUnit.ALL,
    )

    t4 = time.time()
    print(f"Step 3/4 done in {fmt_time(t4 - t3)}", flush=True)

    # ---------------------------------------
    # Step 4: сохранение mlpackage
    # ---------------------------------------
    print("Step 4/4: saving CoreML model...")
    mlmodel.save(MLPACKAGE_PATH)
    t5 = time.time()
    print(f"CoreML model saved to:     {MLPACKAGE_PATH}")
    print(f"Step 4/4 done in {fmt_time(t5 - t4)}", flush=True)

    print("------------------------------------------------------------")
    print(f"Total time: {fmt_time(t5 - t0)}")


if __name__ == "__main__":
    main()
