import time
import math
import types

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
import open_clip


MODEL_NAME = "ViT-B-16"
PRETRAINED = "laion2b_s34b_b88k"
BATCH_SIZE = 32
IMAGE_SIZE = 224
OUT_PATH = "ClipImageEncoder_ViT-B-16_B32.mlpackage"
TS_PATH = "clip_image_encoder_ViT-B-16_B32.pt"


def log(msg: str):
    print(msg, flush=True)


class ImageEncoder(nn.Module):
    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.model = clip_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(x)


class MyMultiheadAttention(nn.Module):
    def __init__(self, mha: nn.MultiheadAttention):
        super().__init__()
        self.embed_dim = mha.embed_dim
        self.num_heads = mha.num_heads
        self.dropout = mha.dropout
        self.batch_first = getattr(mha, "batch_first", False)

        self.in_proj_weight = nn.Parameter(mha.in_proj_weight.detach().clone())
        if mha.in_proj_bias is not None:
            self.in_proj_bias = nn.Parameter(mha.in_proj_bias.detach().clone())
        else:
            self.in_proj_bias = nn.Parameter(torch.zeros(3 * self.embed_dim))

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        with torch.no_grad():
            self.out_proj.weight.copy_(mha.out_proj.weight.detach())
            if mha.out_proj.bias is not None:
                self.out_proj.bias.copy_(mha.out_proj.bias.detach())
            else:
                self.out_proj.bias.zero_()

        self.attn_drop = nn.Dropout(self.dropout if self.dropout is not None else 0.0)
        self.proj_drop = nn.Dropout(self.dropout if self.dropout is not None else 0.0)

    def forward(self, x, key_padding_mask=None, need_weights=False, attn_mask=None):
        if not self.batch_first:
            x = x.transpose(0, 1)  # (N, S, E)
        N, S, E = x.shape

        qkv = torch.nn.functional.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        head_dim = E // self.num_heads
        scale = head_dim ** -0.5

        q = q.view(N, S, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(N, S, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(N, S, self.num_heads, head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attn_mask is not None:
            attn = attn + attn_mask

        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        y = torch.matmul(attn, v)
        y = y.transpose(1, 2).contiguous().view(N, S, E)

        y = self.out_proj(y)
        y = self.proj_drop(y)

        if not self.batch_first:
            y = y.transpose(0, 1)

        if need_weights:
            w = attn.mean(dim=1)
            return y, w
        return y


def patch_attention_modules(clip_model: nn.Module) -> int:
    patched = 0
    for module in clip_model.modules():
        if module.__class__.__name__ == "Attention" and hasattr(module, "qkv") and hasattr(module, "num_heads"):
            def forward(self, x):
                B, N, C = x.shape
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                head_dim = C // self.num_heads
                scale = head_dim ** -0.5
                attn = (q @ k.transpose(-2, -1)) * scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = attn @ v
                x = x.transpose(1, 2).reshape(B, N, C)
                x = self.proj(x)
                x = self.proj_drop(x)
                return x

            module.forward = types.MethodType(forward, module)
            patched += 1
    return patched


def replace_multihead_attention(clip_model: nn.Module) -> int:
    replaced = 0
    for module in clip_model.modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.MultiheadAttention) or child.__class__.__name__.lower().endswith("multiheadattention"):
                new_mha = MyMultiheadAttention(child)
                setattr(module, child_name, new_mha)
                replaced += 1
    return replaced


def main():
    t0 = time.time()
    log("Step 0/4: environment info")
    log(f"torch:        {torch.__version__}")
    log(f"coremltools:  {ct.__version__}")
    log(f"open_clip:    {getattr(open_clip, '__version__', 'unknown')}")
    log(f"model name:   {MODEL_NAME}")
    log(f"pretrained:   {PRETRAINED}")
    log(f"batch size:   {BATCH_SIZE}")
    log(f"image size:   {IMAGE_SIZE}x{IMAGE_SIZE}")
    log(f"ts path:      {TS_PATH}")
    log(f"output path:  {OUT_PATH}")
    log("-" * 60)

    device = "cpu"
    torch.set_grad_enabled(False)

    log("Step 1/4: loading CLIP model...")
    t1 = time.time()
    clip_model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
        device=device
    )
    clip_model.eval()
    clip_model.to(device)

    total_params = sum(p.numel() for p in clip_model.parameters())
    log(f"Total parameters (CLIP): {total_params:,}")

    patched_att = patch_attention_modules(clip_model)
    log(f"Patched Attention modules: {patched_att}")

    replaced_mha = replace_multihead_attention(clip_model)
    log(f"Replaced MultiheadAttention modules: {replaced_mha}")

    log(f"Step 1/4 done in {time.time() - t1:.1f} s")

    log("Step 2/4: building image encoder wrapper and tracing TorchScript...")
    t2 = time.time()
    encoder = ImageEncoder(clip_model)
    encoder.eval()
    encoder.to(device)

    dummy = torch.randn(
        BATCH_SIZE,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():
        out = encoder(dummy)
    out_shape = tuple(out.shape)
    embed_dim = out_shape[-1]

    log(f"Encoder dummy input shape: {tuple(dummy.shape)}")
    log(f"Encoder output shape:      {out_shape}")
    log(f"Embedding dim:             {embed_dim}")
    enc_params = sum(p.numel() for p in encoder.parameters())
    log(f"Total parameters (encoder): {enc_params:,}")

    log("Tracing with torch.jit.trace...")
    ts_encoder = torch.jit.trace(encoder, dummy)
    ts_encoder.eval()
    ts_encoder.save(TS_PATH)
    log(f"TorchScript saved to:      {TS_PATH}")
    log(f"Step 2/4 done in {time.time() - t2:.1f} s")

    log("Step 3/4: converting TorchScript to CoreML (this step can take a long time)...")
    t3 = time.time()

    image_input = ct.TensorType(
        name="image",
        shape=dummy.shape,
        dtype=np.float32
    )

    mlmodel = ct.convert(
        ts_encoder,
        inputs=[image_input],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS16,
        source="pytorch"
    )

    log(f"Step 3/4 done in {time.time() - t3:.1f} s")

    log("Step 4/4: saving CoreML model...")
    t4 = time.time()
    mlmodel.save(OUT_PATH)
    log(f"Model saved to: {OUT_PATH}")
    log(f"Step 4/4 done in {time.time() - t4:.1f} s")

    log(f"Total time: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
