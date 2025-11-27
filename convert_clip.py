import time
import numpy as np
import torch
import coremltools as ct
import onnx
import open_clip


MODEL_NAME = "ViT-B-16"
PRETRAINED = "laion2b_s34b_b88k"
BATCH_SIZE = 32
INPUT_RES = 224

ONNX_PATH = f"clip_image_encoder_{MODEL_NAME}_B{BATCH_SIZE}.onnx"
OUT_NAME = f"ClipImageEncoder_{MODEL_NAME}_B{BATCH_SIZE}.mlpackage"


def fmt_hms(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def log(msg: str):
    print(msg, flush=True)


class ImageEncoder(torch.nn.Module):
    def __init__(self, clip_model: torch.nn.Module):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        return self.clip_model.encode_image(x)


def main():
    log("Step 0/4: environment info")
    oc_ver = getattr(open_clip, "__version__", "unknown")
    log(f"torch:        {torch.__version__}")
    log(f"coremltools:  {ct.__version__}")
    log(f"open_clip:    {oc_ver}")
    log(f"model name:   {MODEL_NAME}")
    log(f"pretrained:   {PRETRAINED}")
    log(f"batch size:   {BATCH_SIZE}")
    log(f"image size:   {INPUT_RES}x{INPUT_RES}")
    log(f"onnx path:    {ONNX_PATH}")
    log(f"output name:  {OUT_NAME}")
    log("-" * 60)

    t0 = time.time()
    log("Step 1/4: loading CLIP model...")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
        device="cpu",
    )
    clip_model.eval()
    clip_model = clip_model.to(dtype=torch.float32)
    log(f"Step 1/4 done in {fmt_hms(time.time() - t0)}")

    t1 = time.time()
    log("Step 2/4: building encoder and running dummy forward + ONNX export...")
    encoder = ImageEncoder(clip_model)
    encoder.eval()

    dummy = torch.randn(
        BATCH_SIZE,
        3,
        INPUT_RES,
        INPUT_RES,
        dtype=torch.float32,
    )

    with torch.no_grad():
        out = encoder(dummy)

    log(f"Encoder dummy input shape: {tuple(dummy.shape)}")
    log(f"Encoder output shape:      {tuple(out.shape)}")
    emb_dim = int(out.shape[-1])
    total_params = sum(p.numel() for p in encoder.parameters())
    log(f"Embedding dim:             {emb_dim}")
    log(f"Total parameters:          {total_params:,}")

    log("Exporting to ONNX...")
    torch.onnx.export(
        encoder,
        dummy,
        ONNX_PATH,
        input_names=["image"],
        output_names=["embedding"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    log(f"ONNX exported to:          {ONNX_PATH}")
    log(f"Step 2/4 done in {fmt_hms(time.time() - t1)}")

    t2 = time.time()
    log("Step 3/4: converting ONNX to CoreML (this step can take a long time)...")
    onnx_model = onnx.load(ONNX_PATH)

    mlmodel = ct.convert(
        onnx_model,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS15,
        compute_units=ct.ComputeUnit.ALL,
        # inputs можно не указывать: формы берутся из ONNX-графа
    )
    log(f"Step 3/4 done in {fmt_hms(time.time() - t2)}")

    t3 = time.time()
    log("Step 4/4: saving CoreML model...")
    mlmodel.save(OUT_NAME)
    log(f"Saved CoreML model to:     {OUT_NAME}")
    log(f"Step 4/4 done in {fmt_hms(time.time() - t3)}")
    log("All done.")


if __name__ == "__main__":
    main()
