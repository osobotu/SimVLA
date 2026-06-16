from __future__ import annotations
from typing import Optional, Dict
import torch
import torch.nn as nn


class PerceptionEncoder(nn.Module):
    """
    Abstract base class for all perception encoders.

    Interface contract with action head:
        Input:  pixel_values [B, V, C, H, W]
                image_mask   [B, V]            (bool, True = valid view)
                input_ids    [B, L]            (optional tokenized instruction)
        Output: {"vlm_features": [B, T_enc, D_out]}
    """
    output_dim: int = 0

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        image_mask: torch.BoolTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


def build_llm_fused_output(
    image_features_lm: torch.Tensor,
    image_mask: torch.BoolTensor,
    input_ids: torch.LongTensor,
    text_model: nn.Module,
) -> Dict[str, torch.Tensor]:
    """
    Shared LLM-fusion step encoders with language model.

    Reconstructs the per-sample multi-view token sequence, concatenates text
    embeddings, pads to a uniform length, and runs one forward pass through
    the SmolLM text model.

    Args:
        image_features_lm: [num_valid, num_patches, D_lm]  (connector output)
        image_mask:         [B, V]
        input_ids:          [B, L]
        text_model:         SmolLM / Idefics3 text_model module

    Returns:
        {"vlm_features": [B, T_seq, D_lm]}
    """
    B, V = image_mask.shape
    device = image_features_lm.device
    dtype = image_features_lm.dtype
    flat_mask = image_mask.view(-1).bool()

    hidden_size = image_features_lm.shape[-1]
    num_patches = image_features_lm.shape[1]

    # Scatter valid features back into [B, V, P, D]
    full_features = image_features_lm.new_zeros(B * V, num_patches, hidden_size)
    full_features[flat_mask] = image_features_lm
    full_features = full_features.view(B, V, num_patches, hidden_size)

    valid_per_sample = image_mask.sum(dim=1).int()
    text_embeds = text_model.get_input_embeddings()(input_ids)  # [B, L, D]

    batch_inputs_embeds = []
    for b in range(B):
        num_valid = int(valid_per_sample[b].item())
        sample_feats = full_features[b, :num_valid].reshape(-1, hidden_size)
        combined = torch.cat([sample_feats, text_embeds[b]], dim=0)
        batch_inputs_embeds.append(combined)

    max_seq_len = max(e.shape[0] for e in batch_inputs_embeds)
    padded = torch.zeros(B, max_seq_len, hidden_size, device=device, dtype=dtype)
    attn_mask = torch.zeros(B, max_seq_len, device=device, dtype=torch.long)
    for b, embeds in enumerate(batch_inputs_embeds):
        padded[b, :embeds.shape[0]] = embeds
        attn_mask[b, :embeds.shape[0]] = 1

    lm_out = text_model(
        inputs_embeds=padded,
        attention_mask=attn_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    return {"vlm_features": lm_out.last_hidden_state}
