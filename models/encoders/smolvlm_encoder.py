from __future__ import annotations
from typing import Optional, Dict
import torch
import torch.nn as nn
from .base import PerceptionEncoder


class SmolVLMEncoder(PerceptionEncoder):
    """
    Baseline encoder — wrapper around the existing SmolVLM pipeline.

    Wraps SigLIP vision_model + connector + SmolLM text_model.
    """

    def __init__(self, vlm: nn.Module) -> None:
        super().__init__()
        # Non-registered reference: store in a plain list so nn.Module.__setattr__
        # does not register it as a child module (avoids double param counting).
        self._vlm_list: list[nn.Module] = [vlm]
        self.output_dim: int = vlm.config.text_config.hidden_size

    @property
    def _vlm(self) -> nn.Module:
        return self._vlm_list[0]

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        image_mask: torch.BoolTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Normalise 6-D legacy inputs
        if pixel_values.dim() == 6:
            pixel_values = (
                pixel_values.squeeze(2)
                if pixel_values.size(2) == 1
                else pixel_values[:, :, 0]
            )

        B, V, C, H, W = pixel_values.shape
        device = pixel_values.device
        dtype = pixel_values.dtype

        # ---- Step 1: SigLIP vision encoder ----
        flat_images = pixel_values.flatten(0, 1)
        flat_mask = image_mask.view(-1).bool()
        valid_images = flat_images[flat_mask]  # [num_valid, C, H, W]

        if valid_images.shape[0] == 0:
            raise ValueError("At least one image view must be valid.")

        vision_outputs = self._vlm.model.vision_model(
            pixel_values=valid_images,
            output_hidden_states=True,
            return_dict=True,
        )
        image_features = vision_outputs.last_hidden_state  # [num_valid, P, D_vis]

        # ---- Step 2: connector (vision dim → LM dim) ----
        if hasattr(self._vlm.model, "connector"):
            image_features = self._vlm.model.connector(image_features)
        elif hasattr(self._vlm.model, "multi_modal_projector"):
            image_features = self._vlm.model.multi_modal_projector(image_features)

        # ---- Step 3: rebuild batch structure [B, V, P, D] ----
        hidden_size = image_features.shape[-1]
        num_patches = image_features.shape[1]
        full_features = image_features.new_zeros(B * V, num_patches, hidden_size)
        full_features[flat_mask] = image_features
        full_features = full_features.view(B, V, num_patches, hidden_size)

        valid_per_sample = image_mask.sum(dim=1).int()
        text_embeds = self._vlm.model.text_model.get_input_embeddings()(input_ids)

        batch_inputs_embeds = []
        max_seq_len = 0
        for b in range(B):
            num_valid = int(valid_per_sample[b].item())
            sample_feats = full_features[b, :num_valid].reshape(-1, hidden_size)
            combined = torch.cat([sample_feats, text_embeds[b]], dim=0)
            batch_inputs_embeds.append(combined)
            max_seq_len = max(max_seq_len, combined.shape[0])

        # ---- Step 4: pad + LM forward ----
        padded = torch.zeros(B, max_seq_len, hidden_size, device=device, dtype=dtype)
        attn_mask = torch.zeros(B, max_seq_len, device=device, dtype=torch.long)
        for b, embeds in enumerate(batch_inputs_embeds):
            padded[b, :embeds.shape[0]] = embeds
            attn_mask[b, :embeds.shape[0]] = 1

        lm_out = self._vlm.model.text_model(
            inputs_embeds=padded,
            attention_mask=attn_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return {"vlm_features": lm_out.last_hidden_state}
