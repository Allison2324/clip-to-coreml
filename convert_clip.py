import time
from pathlib import Path

import numpy as np
import torch
import coremltools as ct
from coremltools.converters import onnx as onnx_converter
import open_clip


MODEL_NAME = "ViT-B-16"
PRETRAINED = "laion2b_s34b_b88k"

BATCH_SIZE = 32
IMAGE_SIZE = 224

ONNX_PATH = Path("clip_image_encoder_ViT-B-16_B32.onnx")
OUTPUT_COREML = Path("ClipImageEncoder_ViT-B-16_B32.mlpackage")


def format_time(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    else:
        return f"{m:02d}:{s:02d}"


class ImageEncoder(torch.nn.Module):
    def __init__(self, clip_model: torch.nn.Module):
        super().__init__()
        self.clip_model = clip_model
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        feats = self.clip_model.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return feats


def main():
    device = "cpu"

    print("Step 0/4: environment info", flush=True)
    print(f"torch:        {torch.__version__}", flush=True)
    print(f"coremltools:  {ct.__version__}", flush=True)
    oc_ver = getattr(open_clip, "__version__", "unknown")
    print(f"open_clip:    {oc_ver}", flush=True)
    print(f"model name:   {MODEL_NAME}", flush=True)
    print(f"pretrained:   {PRETRAINED}", flush=True)
    print(f"batch size:   {BATCH_SIZE}", flush=True)
    print(f"image size:   {IMAGE_SIZE}x{IMAGE_SIZE}", flush=True)
    print(f"onnx path:    {ONNX_PATH}", flush=True)
    print(f"output name:  {OUTPUT_COREML}", flush=True)
    print("-" * 60, flush=True)

    print("Step 1/4: loading CLIP model...", flush=True)
    t1 = time.time()
    clip_model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
        device=device,
    )
    clip_model.eval()
    print(f"Step 1/4 done in {format_time(time.time() - t1)}", flush=True)

    print("Step 2/4: building encoder and running dummy forward + ONNX export...", flush=True)
    t2 = time.time()
    encoder = ImageEncoder(clip_model).to(device)
    encoder.eval()

    dummy = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32, device=device)
    with torch.no_grad():
        out = encoder(dummy)

    out_shape = tuple(out.shape)
    emb_dim = out_shape[-1]
    n_params = sum(p.numel() for p in encoder.parameters())

    print(f"Encoder dummy input shape: {tuple(dummy.shape)}", flush=True)
    print(f"Encoder output shape:      {out_shape}", flush=True)
    print(f"Embedding dim:             {emb_dim}", flush=True)
    print(f"Total parameters:          {n_params:,}", flush=True)

    print("Exporting to ONNX...", flush=True)
    torch.onnx.export(
        encoder,
        dummy,
        str(ONNX_PATH),
        input_names=["image"],
        output_names=["embedding"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    print(f"ONNX exported to:          {ONNX_PATH}", flush=True)
    print(f"Step 2/4 done in {format_time(time.time() - t2)}", flush=True)

    print("Step 3/4: converting ONNX to CoreML (this step can take a long time)...", flush=True)
    t3 = time.time()
    mlmodel = onnx_converter.convert(
        model=str(ONNX_PATH)
    )
    print(f"Step 3/4 done in {format_time(time.time() - t3)}", flush=True)

    print("Step 4/4: saving CoreML model...", flush=True)
    t4 = time.time()
    mlmodel.save(str(OUTPUT_COREML))
    print(f"Saved CoreML model to:     {OUTPUT_COREML}", flush=True)
    print(f"Step 4/4 done in {format_time(time.time() - t4)}", flush=True)
    print("All done.", flush=True)


if __name__ == "__main__":
    main()
