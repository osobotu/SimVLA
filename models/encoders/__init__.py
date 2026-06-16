"""
Pluggable perception encoder registry for SimVLA.

Usage
-----
    from models.encoders import build_encoder
    encoder = build_encoder("smolvlm", cfg, vlm=loaded_vlm_model)
    encoder = build_encoder("dinov2",  cfg, vlm=loaded_vlm_model)
    encoder = build_encoder("ijepa",   cfg, vlm=loaded_vlm_model)
    encoder = build_encoder("vjepa2",  cfg, vlm=loaded_vlm_model)

All encoders satisfy:
    forward(pixel_values, image_mask, input_ids) -> {"vlm_features": [B, T, D]}
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from .base import PerceptionEncoder

if TYPE_CHECKING:
    import torch.nn as nn

ENCODER_NAMES = ("smolvlm", "dinov2", "ijepa", "vjepa2")


def build_encoder(name: str, cfg, vlm) -> PerceptionEncoder:
    """
    Instantiate and return a PerceptionEncoder.

    Parameters
    ----------
    name : str
        One of "smolvlm" | "dinov2" | "ijepa" | "vjepa2".
    cfg  : SmolVLMVLAConfig
        Model config. Reads: encoder_ckpt, encoder_frozen, encoder_lora,
        and (for vjepa2) encoder_t_pad, encoder_video_size.
    vlm  : nn.Module
        Loaded SmolVLM model (AutoModelForImageTextToText).
        Always required — even for non-smolvlm encoders, the SmolLM
        text_model is shared for language fusion.

    Returns
    -------
    PerceptionEncoder
    """
    name = name.lower()
    if name not in ENCODER_NAMES:
        raise ValueError(
            f"Unknown encoder '{name}'. Available: {ENCODER_NAMES}"
        )

    ckpt = getattr(cfg, "encoder_ckpt", "") or ""
    frozen = getattr(cfg, "encoder_frozen", True)
    lora = getattr(cfg, "encoder_lora", False)
    lm_hidden_size: int = vlm.config.text_config.hidden_size
    text_model = vlm.model.text_model

    if name == "smolvlm":
        from .smolvlm_encoder import SmolVLMEncoder
        return SmolVLMEncoder(vlm)

    if name == "dinov2":
        from .dinov2_encoder import DINOv2Encoder
        return DINOv2Encoder(
            text_model=text_model,
            ckpt=ckpt or "facebook/dinov2-base",
            lm_hidden_size=lm_hidden_size,
            frozen=frozen,
            lora=lora,
        )

    if name == "ijepa":
        from .ijepa_encoder import IJEPAEncoder
        arch = getattr(cfg, "encoder_arch", "vith14")
        return IJEPAEncoder(
            text_model=text_model,
            ckpt=ckpt or "facebook/ijepa_vith14_1k",
            arch=arch,
            lm_hidden_size=lm_hidden_size,
            frozen=frozen,
            lora=lora,
        )

    if name == "vjepa2":
        from .vjepa2_encoder import VJEPA2Encoder
        t_pad = getattr(cfg, "encoder_t_pad", VJEPA2Encoder.T_PAD_DEFAULT)
        video_size = getattr(cfg, "encoder_video_size", 224)
        return VJEPA2Encoder(
            text_model=text_model,
            ckpt=ckpt or "facebook/vjepa2-vitl-16",
            lm_hidden_size=lm_hidden_size,
            frozen=frozen,
            lora=lora,
            t_pad=t_pad,
            video_size=video_size,
        )


__all__ = ["PerceptionEncoder", "build_encoder", "ENCODER_NAMES"]
