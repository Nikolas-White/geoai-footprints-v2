"""Model construction.

A single U-Net with a two-channel output head. Two channels from one decoder
rather than two separate decoders: the interior and boundary tasks share
almost all of their useful features, and a shared decoder is both cheaper and
empirically at least as good on this task.
"""
from __future__ import annotations

from typing import Any, Dict

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def build_model(cfg) -> nn.Module:
    tcfg = cfg["train"]
    # Extra derived input channels (e.g. ExG) widen the stem. smp handles the
    # pretrained-weight surgery for us, replicating the RGB filters.
    in_ch = 3 + len(cfg.get("preprocess", {}).get("extra_channels") or [])
    kwargs: Dict[str, Any] = dict(
        encoder_name=tcfg["encoder"],
        encoder_weights=tcfg["encoder_weights"],
        in_channels=in_ch,
        classes=2,  # 0 = interior, 1 = boundary
    )
    arch = str(tcfg.get("arch", "unet")).lower()

    attn = tcfg.get("decoder_attention", None)
    # Transformer encoders do not accept the decoder attention argument.
    if attn and not str(tcfg["encoder"]).startswith(("mit_", "tu-")):
        kwargs["decoder_attention_type"] = attn

    if arch == "unet":
        return smp.Unet(**kwargs)
    if arch == "unetplusplus":
        return smp.UnetPlusPlus(**kwargs)
    if arch == "segformer":
        kwargs.pop("decoder_attention_type", None)
        return smp.Segformer(**kwargs)
    raise ValueError(f"Unsupported arch: {arch}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(path: str, cfg, device: torch.device) -> nn.Module:
    """Rebuild a model and restore weights saved by ``src.train.train``."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model
