from __future__ import annotations
from typing import Optional, Dict
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import PerceptionEncoder, build_llm_fused_output

logger = logging.getLogger(__name__)


class VJEPA2Encoder(PerceptionEncoder):
    """
    E3 (headline) encoder: V-JEPA 2 — video predictive SSL.

    Accepts REAL consecutive-frame clips from the dataset (not repeated padding).
    The LIBERO data handler yields `image_input: [V, T, C, H, W]` when
    `num_frames > 1`; all other encoders receive `[V, C, H, W]` (unchanged).

    Token budget: T frames × P patches per frame.
      Default: T=4, resize=224, patch=16  →  4 × (14×14) = 784 tokens/view.
      2 valid LIBERO views                →  1568 image tokens + text.
    For larger T or 384px, reduce via t_pad or set video_size=224 (default).

    Checkpoint loading priority:
      1. HF AutoModel  (if model is on the Hub with trust_remote_code)
      2. timm ViT-L/16 (local .pt or hub fallback with key remapping)
      3. Random init with warning (smoke runs without weights)

    Architecture defaults: ViT-L/16 (D_vis=1024).

    Args:
        text_model:     shared SmolLM text model (non-registered reference)
        ckpt:           HF repo id or local .pt path
        lm_hidden_size: must match SmolVLM text_config.hidden_size
        frozen:         freeze backbone weights
        lora:           inject LoRA adapters (overrides frozen)
        t_pad:          frames to repeat when dataset sends single frames
                        (fallback only — real clips from dataset are preferred)
        video_size:     resize each frame to this square resolution before
                        encoding. Default=224 (V-JEPA 2 training size).
                        Set to 0 to skip resize (use raw 384px, more VRAM).
    """

    T_PAD_DEFAULT: int = 4  # fallback for single-frame datasets

    def __init__(
        self,
        text_model: nn.Module,
        ckpt: str = "facebook/vjepa2-vitl-16",
        lm_hidden_size: int = 576,
        frozen: bool = True,
        lora: bool = False,
        t_pad: int = T_PAD_DEFAULT,
        video_size: int = 224,
    ) -> None:
        super().__init__()

        self.t_pad = t_pad
        self.video_size = video_size

        self.backbone, self.vision_dim, self._is_native_video = self._load_backbone(ckpt)
        self.connector = nn.Linear(self.vision_dim, lm_hidden_size)
        self._text_model_list: list[nn.Module] = [text_model]
        self.output_dim = lm_hidden_size

        if frozen and not lora:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        if lora:
            try:
                from peft import get_peft_model, LoraConfig
                lora_cfg = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    target_modules=["qkv", "proj", "query", "key", "value"],
                    lora_dropout=0.05,
                    bias="none",
                )
                self.backbone = get_peft_model(self.backbone, lora_cfg)
            except ImportError:
                raise ImportError("peft required for lora. Install: pip install peft")

    @property
    def _text_model(self) -> nn.Module:
        return self._text_model_list[0]

    # ------------------------------------------------------------------
    # Backbone loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_backbone(ckpt: str):
        """Returns (model, vision_dim, is_native_video_model)."""

        # 1. HuggingFace AutoModel (official HF release)
        try:
            from transformers import AutoModel, AutoConfig
            cfg = AutoConfig.from_pretrained(ckpt, trust_remote_code=True)
            model = AutoModel.from_pretrained(ckpt, trust_remote_code=True)
            dim = cfg.hidden_size
            logger.info("V-JEPA 2: loaded HF model from '%s' (D=%d)", ckpt, dim)
            return model, dim, True
        except Exception:
            pass

        # 2. timm ViT-L/16
        try:
            import timm
            model = timm.create_model(
                "vit_large_patch16_224",
                pretrained=False,
                num_classes=0,
                global_pool="",
            )
            dim = model.embed_dim  # 1024

            if ckpt.endswith((".pt", ".pth")):
                VJEPA2Encoder._load_local_pt(model, ckpt)
            else:
                VJEPA2Encoder._load_from_hub(model, ckpt)

            logger.info("V-JEPA 2: using timm ViT-L/16 (D=%d)", dim)
            return model, dim, False
        except ImportError:
            raise ImportError("timm required for VJEPA2Encoder. Install: pip install timm")

    @staticmethod
    def _load_local_pt(model: nn.Module, path: str) -> None:
        state = torch.load(path, map_location="cpu")
        for key in ("encoder", "model", "state_dict", "vision_encoder"):
            if key in state:
                state = state[key]
                break
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.warning("V-JEPA 2: %d missing keys in backbone", len(missing))
        if unexpected:
            logger.warning("V-JEPA 2: %d unexpected keys in backbone", len(unexpected))

    @staticmethod
    def _load_from_hub(model: nn.Module, repo_id: str) -> None:
        try:
            from huggingface_hub import hf_hub_download
            pt_path = hf_hub_download(repo_id=repo_id, filename="vjepa2_encoder.pt")
            VJEPA2Encoder._load_local_pt(model, pt_path)
        except Exception as e:
            logger.warning(
                "V-JEPA 2: could not load weights from '%s' (%s). "
                "Using random init — suitable for smoke runs only.",
                repo_id, e,
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        image_mask: torch.BoolTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pixel_values: [B, V, T, C, H, W]  real clip from dataset  (preferred)
                       or [B, V, C, H, W]      single frame            (fallback → t_pad repeat)
            image_mask:  [B, V]
            input_ids:   [B, L]
        """
        if pixel_values.dim() == 5:
            # Single-frame input: repeat along a new time axis
            # Shape: [B, V, C, H, W] → [B, V, t_pad, C, H, W]
            pixel_values = pixel_values.unsqueeze(2).expand(
                -1, -1, self.t_pad, -1, -1, -1
            ).contiguous()

        # pixel_values is now [B, V, T, C, H, W]
        B, V, T, C, H, W = pixel_values.shape
        flat_clips = pixel_values.flatten(0, 1)            # [B*V, T, C, H, W]
        flat_mask = image_mask.view(-1).bool()
        valid_clips = flat_clips[flat_mask]                # [num_valid, T, C, H, W]

        if valid_clips.shape[0] == 0:
            raise ValueError("At least one image view must be valid.")

        # Optional resize to video_size (e.g. 224) to control token budget
        if self.video_size > 0 and (H != self.video_size or W != self.video_size):
            N_v = valid_clips.shape[0]
            valid_clips = F.interpolate(
                valid_clips.flatten(0, 1),          # [N*T, C, H, W]
                size=(self.video_size, self.video_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).reshape(N_v, T, C, self.video_size, self.video_size)

        features = self._encode_clips(valid_clips)         # [num_valid, T*P, D_vis]
        image_features = self.connector(features)          # [num_valid, T*P, D_lm]
        return build_llm_fused_output(image_features, image_mask, input_ids, self._text_model)

    def _encode_clips(self, clips: torch.Tensor) -> torch.Tensor:
        """
        clips: [N, T, C, H, W]
        Returns: [N, T*P, D_vis]
        """
        N, T, C, H, W = clips.shape

        if self._is_native_video:
            # Native video model: expects [N, C, T, H, W]
            video = clips.permute(0, 2, 1, 3, 4)  # [N, C, T, H, W]
            out = self.backbone(video)
            if hasattr(out, "last_hidden_state"):
                return out.last_hidden_state        # [N, T*P, D]
            return out

        # timm image ViT: encode each frame independently, concat tokens.
        # [N, T, C, H, W] → T × forward_features([N, C, H, W]) → [N, T*P, D]
        per_frame: list[torch.Tensor] = []
        for t in range(T):
            feat = self.backbone.forward_features(clips[:, t])  # [N, 1+P, D]
            if feat.dim() == 3 and feat.shape[1] > 1:
                feat = feat[:, 1:, :]               # drop CLS
            per_frame.append(feat)
        return torch.cat(per_frame, dim=1)          # [N, T*P, D]
