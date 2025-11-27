import time
import os
import torch
import numpy as np
import coremltools as ct
import open_clip
import onnx
from onnx_coreml import convert as onnx_to_coreml


CLIP_MODEL_NAME = "ViT-B-16"
CLIP_PRETRAINED = "laion2b_s34b_b88k"
BATCH_SIZE = 32
INPUT_RES = 224

ONNX_PATH = "clip_image_encoder_ViT-B-16_B32.onnx"
OUTPUT_NAME = "ClipImageEncoder_ViT-B-16_B32.mlpackage"


def fmt_hms(sec: float) -> str:
    sec = int(max(0, sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class ImageEncoder(torch.nn.Module):
    def __init__(self, clip_model: torch.nn.Module):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.clip_model.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return feats


def main():
    t_total0 = time.time()

    print("Step 0/4: environment info")
    print("torch:       ", torch.__version__)
    print("coremltools: ", ct.__version__)
    print("open_clip:   ", getattr(open_clip, "__version__", "unknown"))
    print("model name:  ", CLIP_MODEL_NAME)
    print("pretrained:  ", CLIP_PRETRAINED)
    print("batch size:  ", BATCH_SIZE)
    print("image size:  ", f"{INPUT_RES}x{INPUT_RES}")
    print("onnx path:   ", ONNX_PATH)
    print("output name: ", OUTPUT_NAME)
    print("-" * 60, flush=True)

    t0 = time.time()
    print("Step 1/4: loading CLIP model...", flush=True)
    device = "cpu"
    clip_model, _, _ = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME,
        pretrained=CLIP_PRETRAINED,
        device=device,
    )
    clip_model.eval()
    clip_model.to(device)
    t1 = time.time()
    print(f"Step 1/4 done in {fmt_hms(t1 - t0)}", flush=True)

    t0 = time.time()
    print("Step 2/4: building encoder and running dummy forward + ONNX export...", flush=True)

    encoder = ImageEncoder(clip_model).to(device)
    encoder.eval()

    dummy = torch.randn(BATCH_SIZE, 3, INPUT_RES, INPUT_RES, dtype=torch.float32, device=device)
    with torch.no_grad():
        out = encoder(dummy)

    print("Encoder dummy input shape:", tuple(dummy.shape))
    print("Encoder output shape:     ", tuple(out.shape))
    emb_dim = out.shape[-1]
    total_params = sum(p.numel() for p in encoder.parameters())
    print("Embedding dim:            ", emb_dim)
    print("Total parameters:         ", total_params)

    print("Exporting to ONNX...", flush=True)
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
    print("ONNX exported to:         ", ONNX_PATH, flush=True)

    t1 = time.time()
    print(f"Step 2/4 done in {fmt_hms(t1 - t0)}", flush=True)

    t0 = time.time()
    print("Step 3/4: converting ONNX to CoreML (this step can take a long time)...", flush=True)

    onnx_model = onnx.load(ONNX_PATH)

    mlmodel = onnx_to_coreml(
        onnx_model,
        minimum_ios_deployment_target="16",
        image_input_names=["image"],
    )

    t1 = time.time()
    print(f"Step 3/4 done in {fmt_hms(t1 - t0)}", flush=True)

    t0 = time.time()
    print("Step 4/4: saving CoreML model...", flush=True)
    mlmodel.save(OUTPUT_NAME)
    t1 = time.time()
    print(f"Step 4/4 done in {fmt_hms(t1 - t0)}", flush=True)

    t_total1 = time.time()
    print("-" * 60)
    print("All done.")
    print("Total time:", fmt_hms(t_total1 - t_total0))


if __name__ == "__main__":
    main()
