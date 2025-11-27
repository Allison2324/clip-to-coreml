import time
import numpy as np
import torch
import coremltools as ct
import open_clip


MODEL_NAME = "ViT-B-16"
PRETRAINED = "laion2b_s34b_b88k"
IMAGE_SIZE = 224
BATCH_SIZE = 32
TS_PATH = f"clip_image_encoder_{MODEL_NAME}_B{BATCH_SIZE}.pt"
OUTPUT_NAME = f"ClipImageEncoder_{MODEL_NAME}_B{BATCH_SIZE}.mlpackage"


def main():
    t0 = time.time()

    print("Step 0/4: environment info")
    print(f"torch:        {torch.__version__}")
    print(f"coremltools:  {ct.__version__}")
    print(f"open_clip:    {getattr(open_clip, '__version__', 'open_clip')}")
    print(f"model name:   {MODEL_NAME}")
    print(f"pretrained:   {PRETRAINED}")
    print(f"batch size:   {BATCH_SIZE}")
    print(f"image size:   {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"ts path:      {TS_PATH}")
    print(f"output name:  {OUTPUT_NAME}")
    print("-" * 60, flush=True)

    # ----------------- Step 1: load CLIP -----------------
    t1 = time.time()
    print("Step 1/4: loading CLIP model...", flush=True)
    device = "cpu"
    clip_model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
        device=device,
    )
    clip_model.eval()
    clip_model.to(device)
    print("Step 1/4 done in %.1f s" % (time.time() - t1), flush=True)

    # ----------------- Step 2: wrap encoder + TorchScript -----------------
    t2 = time.time()
    print("Step 2/4: building image encoder wrapper and tracing to TorchScript...", flush=True)

    class ImageEncoder(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feats = self.base_model.encode_image(x)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            return feats

    encoder = ImageEncoder(clip_model)
    encoder.eval()
    encoder.to(device)
    encoder.float()

    dummy = torch.randn(
        BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE,
        dtype=torch.float32, device=device
    )

    with torch.no_grad():
        out = encoder(dummy)

    out_shape = tuple(out.shape)
    emb_dim = out_shape[-1]
    n_params = sum(p.numel() for p in encoder.parameters())

    print(f"Encoder dummy input shape: {tuple(dummy.shape)}", flush=True)
    print(f"Encoder output shape:      {out_shape}", flush=True)
    print(f"Embedding dim:             {emb_dim}", flush=True)
    print(f"Total parameters:          {n_params:,}", flush=True)

    print("Tracing with torch.jit.trace...", flush=True)
    with torch.no_grad():
        ts_model = torch.jit.trace(encoder, dummy, strict=False)
    ts_model.eval()
    ts_model = torch.jit.freeze(ts_model)
    torch.jit.save(ts_model, TS_PATH)
    print(f"TorchScript saved to:      {TS_PATH}", flush=True)
    print("Step 2/4 done in %.1f s" % (time.time() - t2), flush=True)

    # ----------------- Step 3: convert to CoreML -----------------
    t3 = time.time()
    print("Step 3/4: converting TorchScript to CoreML (this step can take a long time)...", flush=True)

    mlmodel = ct.convert(
        ts_model,
        source="pytorch",
        inputs=[
            ct.TensorType(
                name="image",
                shape=(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE),
                dtype=np.float32,
            )
        ],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS15,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )

    print("Step 3/4 done in %.1f s" % (time.time() - t3), flush=True)

    # ----------------- Step 4: save model -----------------
    t4 = time.time()
    print("Step 4/4: saving CoreML package...", flush=True)
    mlmodel.save(OUTPUT_NAME)
    print("Step 4/4 done in %.1f s" % (time.time() - t4), flush=True)

    print("-" * 60)
    print("Total time: %.1f s" % (time.time() - t0))
    print("Saved to:", OUTPUT_NAME)


if __name__ == "__main__":
    main()
