import time
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
TS_PATH = "clip_image_encoder_ViT-B-16_B32.pt"
OUT_PATH = "ClipImageEncoder_ViT-B-16_B32.mlpackage"


def log(msg: str):
    print(msg, flush=True)


class ImageEncoder(nn.Module):
    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.model = clip_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224), float32, в диапазоне [0, 1] с нормализацией внутри CoreML уже не делаем
        return self.model.encode_image(x)


def patch_residual_attention_blocks(clip_model: nn.Module) -> int:
    """
    Переопределяем метод attention в ResidualAttentionBlock так, чтобы
    он НЕ использовал nn.MultiheadAttention.forward (и, соответственно,
    не вызывал _native_multi_head_attention), а делал attention руками.
    """
    patched = 0

    for m in clip_model.modules():
        if (
            m.__class__.__name__ == "ResidualAttentionBlock"
            and hasattr(m, "attn")
            and isinstance(m.attn, nn.MultiheadAttention)
        ):
            mha = m.attn

            def attention(self, q_x, k_x=None, v_x=None, attn_mask=None):
                mha_local = self.attn
                batch_first = getattr(mha_local, "batch_first", False)
                embed_dim = mha_local.embed_dim
                num_heads = mha_local.num_heads
                head_dim = embed_dim // num_heads
                scale = head_dim ** -0.5

                if k_x is None:
                    k_x = q_x
                if v_x is None:
                    v_x = k_x

                # Приводим к batch_first=(B, T, C)
                if batch_first:
                    q = q_x
                    k = k_x
                    v = v_x
                else:
                    # (T, B, C) -> (B, T, C)
                    q = q_x.transpose(0, 1)
                    k = k_x.transpose(0, 1)
                    v = v_x.transpose(0, 1)

                W = mha_local.in_proj_weight    # (3*E, E)
                b = mha_local.in_proj_bias      # (3*E,) или None
                if W is None:
                    raise ValueError("Expected in_proj_weight in MultiheadAttention")

                w_q, w_k, w_v = W.chunk(3, dim=0)
                if b is not None:
                    b_q, b_k, b_v = b.chunk(3, dim=0)
                else:
                    b_q = b_k = b_v = None

                q = torch.nn.functional.linear(q, w_q, b_q)
                k = torch.nn.functional.linear(k, w_k, b_k)
                v = torch.nn.functional.linear(v, w_v, b_v)

                B, T_q, _ = q.shape
                _, T_k, _ = k.shape

                q = q.view(B, T_q, num_heads, head_dim).transpose(1, 2)  # (B, H, T_q, d)
                k = k.view(B, T_k, num_heads, head_dim).transpose(1, 2)  # (B, H, T_k, d)
                v = v.view(B, T_k, num_heads, head_dim).transpose(1, 2)  # (B, H, T_k, d)

                attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T_q, T_k)

                if attn_mask is not None:
                    # Ожидаем broadcast совместимый attn_mask.
                    attn = attn + attn_mask

                attn = torch.softmax(attn, dim=-1)
                attn = torch.nn.functional.dropout(
                    attn, p=mha_local.dropout, training=self.training
                )

                y = torch.matmul(attn, v)  # (B, H, T_q, d)
                y = y.transpose(1, 2).contiguous().view(B, T_q, embed_dim)  # (B, T_q, E)

                y = mha_local.out_proj(y)  # (B, T_q, E)

                if not batch_first:
                    y = y.transpose(0, 1)  # обратно в (T, B, E)

                return y

            m.attention = types.MethodType(attention, m)
            patched += 1

    return patched


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

    # 1. Загружаем CLIP
    log("Step 1/4: loading CLIP model...")
    t1 = time.time()
    clip_model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
        device=device,
    )
    clip_model.eval()
    clip_model.to(device)

    total_params = sum(p.numel() for p in clip_model.parameters())
    log(f"Total parameters (CLIP): {total_params:,}")

    patched_ra = patch_residual_attention_blocks(clip_model)
    log(f"Patched ResidualAttentionBlock.attention: {patched_ra}")

    log(f"Step 1/4 done in {time.time() - t1:.1f} s")

    # 2. Оборачиваем в ImageEncoder и трассим в TorchScript
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
        device=device,
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

    # 3. Конвертация TorchScript -> CoreML
    log("Step 3/4: converting TorchScript to CoreML (this step can take a long time)...")
    t3 = time.time()

    image_input = ct.TensorType(
        name="image",
        shape=dummy.shape,  # (32, 3, 224, 224)
        dtype=np.float32,
    )

    mlmodel = ct.convert(
        ts_encoder,
        source="pytorch",
        inputs=[image_input],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS16,
        compute_units=ct.ComputeUnit.ALL,
    )

    log(f"Step 3/4 done in {time.time() - t3:.1f} s")

    # 4. Сохранение mlpackage
    log("Step 4/4: saving CoreML model...")
    t4 = time.time()
    mlmodel.save(OUT_PATH)
    log(f"Model saved to: {OUT_PATH}")
    log(f"Step 4/4 done in {time.time() - t4:.1f} s")

    log(f"Total time: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
