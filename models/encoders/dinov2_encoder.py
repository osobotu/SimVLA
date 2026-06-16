from __future__ import annotations
from typing import Optional, Dict
import torch
import torch.nn as nn
from .base import PerceptionEncoder, build_llm_fused_output


class DINOv2Encoder(PerceptionEncoder):
    """
    Control encoder: DINOv2 (self-supervised ViT, non-predictive).

    Purpose: isolate "SSL representation helps" from "predictive objective helps"
    by comparing against I-JEPA / V-JEPA 2 on the same downstream pipeline.

    Vision backbone: facebook/dinov2-base  (D_vis=768, patch=14)
                  or facebook/dinov2-large (D_vis=1024, patch=14)
    Language fusion: SmolLM text_model — same as baseline.
    Connector: fresh nn.Linear(vision_dim -> D_lm), trainable from step 0.

    Preprocessing note: DINOv2 uses ImageNet normalisation (mean/std identical
    to the dataset pipeline), so the tensors from the dataloader are fed directly
    to the backbone with no re-normalisation. The backbone accepts any spatial
    resolution ≥ patch_size; position embeddings are bicubic-interpolated
    automatically by the HF implementation.

    Freezing: when frozen=True, backbone.requires_grad=False. Only the
    connector (and the shared SmolLM) accumulate gradients.
    LoRA: when lora=True, peft LoRA adapters are injected into
    query/key/value projections; only adapter weights are trainable.
    """

    def __init__(
        self,
        text_model: nn.Module,
        ckpt: str = "facebook/dinov2-base",
        lm_hidden_size: int = 576,
        frozen: bool = True,
        lora: bool = False,
    ) -> None:
        super().__init__()

        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(ckpt)
        vision_dim: int = self.backbone.config.hidden_size

        self.connector = nn.Linear(vision_dim, lm_hidden_size)
        # Non-registered reference to shared text model (owned by SmolVLMVLA.vlm)
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
                    # DINOv2 HF uses "query", "key", "value" in attention layers
                    target_modules=["query", "key", "value"],
                    lora_dropout=0.05,
                    bias="none",
                )
                self.backbone = get_peft_model(self.backbone, lora_cfg)
            except ImportError:
                raise ImportError(
                    "peft is required for encoder_lora=True. "
                    "Install with: pip install peft"
                )

    @property
    def _text_model(self) -> nn.Module:
        return self._text_model_list[0]

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
        valid_images = flat_images[flat_mask]  # [num_valid, C, H, W]

        if valid_images.shape[0] == 0:
            raise ValueError("At least one image view must be valid.")

        # DINOv2 forward — data already ImageNet-normalised by the dataloader
        out = self.backbone(pixel_values=valid_images, return_dict=True)
        # last_hidden_state: [num_valid, 1+num_patches, D_vis]  (CLS at index 0)
        image_features = out.last_hidden_state[:, 1:, :]  # drop CLS token

        image_features = self.connector(image_features)  # [num_valid, P, D_lm]
        return build_llm_fused_output(image_features, image_mask, input_ids, self._text_model)
