import time
import math
import os

import numpy as np
import torch
import coremltools as ct
import open_clip
import onnx
import coremltools.converters.onnx as onnx_converter


MODEL_NAME = "ViT-B-16"
PRETRAINED = "laion2b_s34b_b88k"
BATCH_SIZE = 32
IMAGE_SIZE = 224

ONNX_PATH = f"clip_image_encoder_{MODEL_NAME}_B{BATCH_SIZE}.onnx"
OUTPUT_NAME = f"ClipImageEncoder_{MODEL_NAME}_B{BATCH_SIZE}.mlpackage"


def fmt_hms(sec: float) -> str:
    sec = float(sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


class ImageEncoder(torch.nn.Module):
    def __init__(self, clip_model: torch.nn.Module):
        super().__init__()
        self.visual = clip_model.visual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.visual(x)


def main():
    t0 = time.time()
    torch_version = torch.__version__
    ct_version = ct.__version__
    oc_version = getattr(open_clip, "__version__", "unknown")

    print("Step 0/4: environment info")
    print(f"torch:        {torch_version}")
    print(f"coremltools:  {ct_version}")
    print(f"open_clip:    {oc_version}")
    print(f"model name:   {MODEL_NAME}")
    print(f"pretrained:   {PRETRAINED}")
    print(f"batch size:   {BATCH_SIZE}")
    print(f"image size:   {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"onnx path:    {ONNX_PATH}")
    print(f"output name:  {OUTPUT_NAME}")
    print("-" * 60)

    # ---------------------------------------------------------
    # Step 1: загрузка CLIP
    # ---------------------------------------------------------
    t1 = time.time()
    print("Step 1/4: loading CLIP model...")

    device = "cpu"
    clip_model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
        device=device,
    )
    clip_model.eval()

    encoder = ImageEncoder(clip_model).to(device)
    encoder.eval()

    print(f"Step 1/4 done in {fmt_hms(time.time() - t1)}")

    # ---------------------------------------------------------
    # Step 2: dummy forward + экспорт в ONNX
    # ---------------------------------------------------------
    t2 = time.time()
    print("Step 2/4: building encoder and running dummy forward + ONNX export...")

    dummy = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    with torch.no_grad():
        out = encoder(dummy)

    print(f"Encoder dummy input shape: {tuple(dummy.shape)}")
    print(f"Encoder output shape:      {tuple(out.shape)}")

    embed_dim = out.shape[-1]
    n_params = sum(int(p.numel()) for p in encoder.parameters())
    print(f"Embedding dim:             {embed_dim}")
    print(f"Total parameters:          {n_params:,}")

    print("Exporting to ONNX...")
    torch.onnx.export(
        encoder,
        dummy,
        ONNX_PATH,
        input_names=["image"],
        output_names=["embedding"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=None,  # фикcированный батч BATCH_SIZE
    )
    print(f"ONNX exported to:          {ONNX_PATH}")

    # Быстрая проверка ONNX-модели
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("ONNX model check:          OK")

    print(f"Step 2/4 done in {fmt_hms(time.time() - t2)}")

    # ---------------------------------------------------------
    # Step 3: конвертация ONNX -> CoreML
    # ---------------------------------------------------------
    t3 = time.time()
    print("Step 3/4: converting ONNX to CoreML (this step can take a long time)...")

    mlmodel = onnx_converter.convert(
        model=ONNX_PATH,
        minimum_ios_deployment_target="15",
        image_input_names=["image"],
    )

    print(f"Step 3/4 done in {fmt_hms(time.time() - t3)}")

    # ---------------------------------------------------------
    # Step 4: сохранение CoreML + краткая инфа по spec
    # ---------------------------------------------------------
    t4 = time.time()
    print("Step 4/4: saving CoreML model...")

    mlmodel.short_description = (
        f"CLIP {MODEL_NAME} image encoder (OpenCLIP {PRETRAINED}), "
        f"batch={BATCH_SIZE}, embed_dim={embed_dim}"
    )

    mlmodel.save(OUTPUT_NAME)

    spec = mlmodel.get_spec()
    if spec.description.input:
        inp = spec.description.input[0]
        print("CoreML input:")
        print(f"  name: {inp.name}")
        if inp.type.WhichOneof("Type") == "imageType":
            it = inp.type.imageType
            print(f"  type: image {it.width}x{it.height}, color={it.colorSpace}")
        elif inp.type.WhichOneof("Type") == "multiArrayType":
            mt = inp.type.multiArrayType
            print(f"  type: multiArray, shape={list(mt.shape)}")
    if spec.description.output:
        outd = spec.description.output[0]
        print("CoreML output:")
        print(f"  name: {outd.name}")
        if outd.type.WhichOneof("Type") == "multiArrayType":
            mt = outd.type.multiArrayType
            print(f"  type: multiArray, shape={list(mt.shape)}")

    print(f"Step 4/4 done in {fmt_hms(time.time() - t4)}")
    print("-" * 60)
    print(f"Total time: {fmt_hms(time.time() - t0)}")


if __name__ == "__main__":
    main()
