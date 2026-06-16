from __future__ import annotations
from typing import Optional, Dict
import logging
import torch
import torch.nn as nn
from .base import PerceptionEncoder, build_llm_fused_output

logger = logging.getLogger(__name__)


# Canonical I-JEPA architecture configs (patch_size for reference only)
_IJEPA_ARCH = {
    "vith14": dict(model_name="vit_huge_patch14_224", embed_dim=1280),
    "vitl16": dict(model_name="vit_large_patch16_224", embed_dim=1024),
    "vitb16": dict(model_name="vit_base_patch16_224", embed_dim=768),
}


class IJEPAEncoder(PerceptionEncoder):
    """
    I-JEPA (Image JEPA) — image-level predictive SSL.

    Trains by predicting target patch representations from context patches
    in latent space (no pixel reconstruction), yielding abstract semantic
    features that may transfer better to downstream robotic tasks.

    Checkpoint loading (in order of precedence):
      1. Local .pt / .pth file: loads {"encoder": ...} or {"model": ...} key.
      2. HuggingFace Hub: tries hf_hub_download for "ijepa_encoder.pt".
      3. Random init with a warning (useful for smoke runs without weights).

    Architecture: timm ViT (vith14 default — D_vis=1280, patch=14).
    Preprocessing: ImageNet normalisation — identical to dataset pipeline,
    no re-normalisation inside the wrapper.

    Args:
        text_model:     shared SmolLM text model (non-registered reference)
        ckpt:           HF repo id (e.g. "facebook/ijepa_vith14_1k") or
                        local path to a .pt checkpoint
        arch:           one of "vith14" | "vitl16" | "vitb16"
        lm_hidden_size: D_lm to project into (must match SmolVLM text_config.hidden_size)
        frozen:         freeze backbone weights
        lora:           inject LoRA adapters via peft (overrides frozen)
    """

    def __init__(
        self,
        text_model: nn.Module,
        ckpt: str = "facebook/ijepa_vith14_1k",
        arch: str = "vith14",
        lm_hidden_size: int = 576,
        frozen: bool = True,
        lora: bool = False,
    ) -> None:
        super().__init__()

        arch_cfg = _IJEPA_ARCH.get(arch)
        if arch_cfg is None:
            raise ValueError(f"Unknown I-JEPA arch '{arch}'. Choose from: {list(_IJEPA_ARCH)}")

        self.backbone = self._load_timm_backbone(ckpt, arch_cfg)
        vision_dim: int = arch_cfg["embed_dim"]

        self.connector = nn.Linear(vision_dim, lm_hidden_size)
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
                    # timm ViT uses "qkv" fused projection + "proj" output
                    target_modules=["qkv", "proj"],
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
    # Backbone loading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_timm_backbone(ckpt: str, arch_cfg: dict) -> nn.Module:
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required for IJEPAEncoder. Install: pip install timm")

        model = timm.create_model(
            arch_cfg["model_name"],
            pretrained=False,
            num_classes=0,     # remove classification head
            global_pool="",    # return all patch tokens, not pooled
        )

        # Try to load weights
        if ckpt.endswith((".pt", ".pth")):
            IJEPAEncoder._load_local_pt(model, ckpt)
        else:
            IJEPAEncoder._load_from_hub(model, ckpt)

        return model

    @staticmethod
    def _load_local_pt(model: nn.Module, path: str) -> None:
        state = torch.load(path, map_location="cpu")
        for key in ("encoder", "model", "state_dict", "target_encoder"):
            if key in state:
                state = state[key]
                break
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.warning("I-JEPA: %d missing keys in backbone", len(missing))
        if unexpected:
            logger.warning("I-JEPA: %d unexpected keys in backbone", len(unexpected))

    @staticmethod
    def _load_from_hub(model: nn.Module, repo_id: str) -> None:
        try:
            from huggingface_hub import hf_hub_download
            pt_path = hf_hub_download(repo_id=repo_id, filename="ijepa_encoder.pt")
            IJEPAEncoder._load_local_pt(model, pt_path)
        except Exception as e:
            logger.warning(
                "I-JEPA: could not load weights from HF hub '%s' (%s). "
                "Using random initialisation — suitable for smoke runs only.",
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
        if pixel_values.dim() == 6:
            pixel_values = (
                pixel_values.squeeze(2)
                if pixel_values.size(2) == 1
                else pixel_values[:, :, 0]
            )

        B, V, C, H, W = pixel_values.shape
        flat_images = pixel_values.flatten(0, 1)
        flat_mask = image_mask.view(-1).bool()
        valid_images = flat_images[flat_mask]

        if valid_images.shape[0] == 0:
            raise ValueError("At least one image view must be valid.")

        # timm forward_features returns [N, num_tokens, D]
        # For ViTs with global_pool="", first token is CLS; 1: are patch tokens.
        features = self.backbone.forward_features(valid_images)  # [N, 1+P, D]
        if features.dim() == 3 and features.shape[1] > 1:
            features = features[:, 1:, :]  # drop CLS

        image_features = self.connector(features)  # [num_valid, P, D_lm]
        return build_llm_fused_output(image_features, image_mask, input_ids, self._text_model)
