"""
SmolVLM-VLA Model

HuggingFace-compatible Vision-Language-Action policy using SmolVLM-500M-Instruct
as the visual-language backbone, with a pluggable perception encoder.

Encoder abstraction
-------------------
All vision encoding is delegated to `self.encoder` (a PerceptionEncoder subclass).
The interface contract is:

    encoder(pixel_values, image_mask, input_ids)
        -> {"vlm_features": [B, T_enc, D]}

For encoder_name="smolvlm" (baseline B0): SmolVLMEncoder wraps the original
forward_vlm_efficient logic unchanged. The SmolVLM weights live in `self.vlm`
(parameter owner). SmolVLMEncoder holds a non-registered reference so the
state-dict layout ("vlm.*") is preserved and old checkpoints remain loadable.

For encoder_name in {dinov2, ijepa, vjepa2}: the new backbone lives in
`self.encoder.backbone` + `self.encoder.connector`. The SmolLM text_model
(in `self.vlm`) is shared for language fusion. SigLIP vision_model inside
self.vlm is frozen (unused) for non-smolvlm encoders.

Temporal clips (V-JEPA 2)
--------------------------
When the dataset is configured with num_frames > 1, pixel_values has shape
[B, V, T, C, H, W]. SmolVLMEncoder / DINOv2Encoder / IJEPAEncoder discard
all but the last frame (current timestep). VJEPA2Encoder processes all T frames.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Dict

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn
import json_numpy
import cv2

from transformers import PreTrainedModel, AutoProcessor, AutoModelForImageTextToText
from .transformer_smolvlm import SmolVLMActionTransformer
from .action_hub import build_action_space
from .configuration_smolvlm_vla import SmolVLMVLAConfig
from .encoders import build_encoder


class SmolVLMVLA(PreTrainedModel):
    """
    SmolVLM-VLA: HuggingFace-compatible Vision-Language-Action policy.

    Components:
      • self.vlm  — SmolVLM-500M-Instruct (always loaded; owns SmolLM weights)
      • self.encoder — pluggable PerceptionEncoder (smolvlm | dinov2 | ijepa | vjepa2)
      • self.transformer — SmolVLMActionTransformer (flow matching action head)
      • self.action_space — action pre/post-processing + loss
    """
    config_class = SmolVLMVLAConfig
    base_model_prefix = "smolvlm_vla"
    supports_gradient_checkpointing = True

    def __init__(self, config: SmolVLMVLAConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.num_actions: int = config.num_actions
        self.use_proprio: bool = config.use_proprio
        self.action_mode: str = config.action_mode.lower()
        self.image_size: int = config.image_size
        self.num_views: int = config.num_views

        # Action space
        self.action_space = build_action_space(config.action_mode.lower())
        dim_action = self.action_space.dim_action
        dim_proprio = getattr(self.action_space, "dim_proprio", dim_action)

        # ── SmolVLM (always loaded — owns LM weights regardless of encoder) ──
        logging.info(f"Loading SmolVLM from: {config.smolvlm_model_path}")
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            config.smolvlm_model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        self.vlm_processor = AutoProcessor.from_pretrained(
            config.smolvlm_model_path,
            trust_remote_code=True,
        )

        # For non-smolvlm encoders: freeze SigLIP (unused in forward pass)
        encoder_name = getattr(config, "encoder_name", "smolvlm")
        if encoder_name != "smolvlm":
            logging.info(
                f"encoder_name='{encoder_name}': freezing SigLIP "
                "(self.vlm.model.vision_model + connector — unused by this encoder)"
            )
            for p in self.vlm.model.vision_model.parameters():
                p.requires_grad_(False)
            if hasattr(self.vlm.model, "connector"):
                for p in self.vlm.model.connector.parameters():
                    p.requires_grad_(False)

        # ── Pluggable perception encoder ──
        logging.info(f"Building encoder: '{encoder_name}'")
        self.encoder = build_encoder(encoder_name, config, vlm=self.vlm)
        vlm_hidden_size: int = self.encoder.output_dim
        logging.info(f"Encoder output_dim: {vlm_hidden_size}")

        # ── DiT/AdaLN mode ──
        self.use_adaln = getattr(config, "use_adaln", False)

        # ── Flow matching action head ──
        self.transformer = SmolVLMActionTransformer(
            hidden_size=config.hidden_size,
            vlm_hidden_size=vlm_hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            dim_action=dim_action,
            dim_propio=dim_proprio,
            dim_time=config.dim_time,
            max_len_seq=config.max_len_seq,
            use_adaln=self.use_adaln,
        )

        if self.use_adaln:
            logging.info("DiT/AdaLN mode enabled")
        else:
            logging.info("Concat mode enabled")

        self.app: FastAPI | None = None

    # ========================= parameter grouping helper ====================

    def vlm_parameters(self):
        """
        Parameters belonging to the language-model portion of self.vlm.

        For encoder_name="smolvlm": returns all vlm params (SigLIP + LLM).
        For other encoders: returns only text_model params (LLM);
            SigLIP params are frozen and excluded to avoid wasted optimizer state.
        """
        encoder_name = getattr(self.config, "encoder_name", "smolvlm")
        if encoder_name == "smolvlm":
            return list(self.vlm.parameters())
        # Only the text model trains; SigLIP is frozen (requires_grad=False)
        return [p for p in self.vlm.model.text_model.parameters()]

    # ========================= perception forward ===========================

    def forward_vlm_efficient(
        self,
        pixel_values: torch.FloatTensor,    # [B, V, C, H, W] or [B, V, T, C, H, W]
        image_mask: torch.Tensor,           # [B, V]
        input_ids: torch.LongTensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Delegate to the pluggable encoder.
        Returns {"vlm_features": [B, T_enc, D]}.
        """
        return self.encoder(pixel_values, image_mask, input_ids)

    # ====================== legacy (non-efficient) path =====================

    def forward_vlm(
        self,
        pixel_values: torch.FloatTensor,
        image_mask: torch.Tensor,
        language_instruction: list[str] | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Slow VLM forward (PIL round-trip). Kept for compatibility; not used
        in training. Non-smolvlm encoders are not supported here.
        """
        if pixel_values.dim() == 6:
            if pixel_values.size(2) == 1:
                pixel_values = pixel_values.squeeze(2)
            else:
                pixel_values = pixel_values[:, :, 0]

        B, V, C, H, W = pixel_values.shape
        device = pixel_values.device

        batch_features = []

        for b in range(B):
            valid_mask = image_mask[b].bool()
            valid_images = pixel_values[b][valid_mask]

            if valid_images.shape[0] == 0:
                raise ValueError("At least one image view must be valid per batch.")

            pil_images = []
            for img_tensor in valid_images:
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img_np))

            content = [{"type": "image", "image": img} for img in pil_images]
            if language_instruction is not None and b < len(language_instruction):
                content.append({"type": "text", "text": language_instruction[b]})
            else:
                content.append({"type": "text", "text": "Describe the robot's observation."})

            messages = [{"role": "user", "content": content}]

            inputs = self.vlm_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = self.vlm(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )

            hidden_states = outputs.hidden_states[-1]
            batch_features.append(hidden_states.squeeze(0))

        max_len = max(f.shape[0] for f in batch_features)
        hidden_size = batch_features[0].shape[-1]

        padded_features = torch.zeros(B, max_len, hidden_size, device=device, dtype=batch_features[0].dtype)
        for b, feat in enumerate(batch_features):
            padded_features[b, :feat.shape[0]] = feat

        return {"vlm_features": padded_features}

    # ================================= training =================================

    def forward(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,     # [B, V, C, H, W] or [B, V, T, C, H, W]
        image_mask: torch.Tensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Flow Matching training forward.

        1) t ~ Beta(1.5, 1) * 0.999 + 0.001
        2) x_t = t * noise + (1-t) * action_norm
        3) u_t = noise - action_norm
        4) v_t = transformer(vlm_features, x_t, t, proprio_norm)
        5) loss = MSE(v_t, u_t)
        """
        enc = self.forward_vlm_efficient(image_input, image_mask, input_ids)

        B = input_ids.shape[0]
        device = input_ids.device

        beta_dist = torch.distributions.Beta(
            torch.tensor(1.5, device=device),
            torch.tensor(1.0, device=device)
        )
        t = beta_dist.sample((B,)) * 0.999 + 0.001

        if hasattr(self.action_space, "normalize_action"):
            action_norm = self.action_space.normalize_action(action)
        elif hasattr(self.action_space, "normalize"):
            action_norm = self.action_space.normalize(action)
        else:
            action_norm = action

        if hasattr(self.action_space, "normalize_state"):
            proprio_norm = self.action_space.normalize_state(proprio)
        elif hasattr(self.action_space, "normalize"):
            proprio_norm = self.action_space.normalize(proprio)
        else:
            proprio_norm = proprio

        noise = torch.randn_like(action_norm)
        t_expanded = t.view(-1, 1, 1)
        x_t = t_expanded * noise + (1 - t_expanded) * action_norm
        u_t = noise - action_norm

        v_t = self.transformer(
            vlm_features=enc["vlm_features"],
            action_with_noise=x_t,
            t=t,
            proprio=proprio_norm,
        )

        velocity_loss = torch.mean(torch.square(v_t - u_t))
        return {"velocity_loss": velocity_loss}

    # ================================= inference =================================

    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        proprio: torch.Tensor,
        steps: int = 10,
    ) -> torch.Tensor:
        """Flow Matching inference (Euler integration)."""
        self.eval()
        enc = self.forward_vlm_efficient(image_input, image_mask, input_ids)

        B = input_ids.shape[0]
        D = self.action_space.dim_action
        device = proprio.device
        dtype = proprio.dtype

        if hasattr(self.action_space, "normalize_state"):
            proprio_norm = self.action_space.normalize_state(proprio)
        elif hasattr(self.action_space, "normalize"):
            proprio_norm = self.action_space.normalize(proprio)
        else:
            proprio_norm = proprio

        steps = max(1, int(steps))
        dt = -1.0 / steps

        x_t = torch.randn(B, self.num_actions, D, device=device, dtype=dtype)
        t = 1.0

        while t > -dt / 2:
            t_tensor = torch.full((B,), t, device=device, dtype=dtype)
            v_t = self.transformer(
                vlm_features=enc["vlm_features"],
                action_with_noise=x_t,
                proprio=proprio_norm,
                t=t_tensor,
            )
            x_t = x_t + dt * v_t
            t = t + dt

        return self.action_space.postprocess(x_t)

    # =============================== FastAPI service =============================

    def _build_app(self, processor):
        """Build FastAPI app for SmolVLM-VLA inference."""
        if self.app is not None:
            return

        app = FastAPI()

        @app.post("/act")
        def act(payload: Dict[str, Any]):
            try:
                self.eval()
                images = []
                for key in ("image0", "image1", "image2"):
                    if key not in payload:
                        continue
                    v = json_numpy.loads(payload[key])
                    if isinstance(v, np.ndarray):
                        if v.ndim == 1:
                            v = cv2.imdecode(v, cv2.IMREAD_COLOR)
                        images.append(Image.fromarray(v))
                    elif isinstance(v, (list, tuple)):
                        images.append(Image.fromarray(np.array(v)))
                    elif isinstance(v, str):
                        images.append(Image.open(v))

                if not images:
                    return JSONResponse({"error": "No valid images found."}, status_code=400)

                inputs = processor(images, payload["language_instruction"])
                if not {"input_ids", "image_input", "image_mask"}.issubset(inputs):
                    return JSONResponse({"error": "Processor returned incomplete inputs."}, status_code=400)

                proprio = torch.as_tensor(np.asarray(json_numpy.loads(payload["proprio"])))

                device = next(self.parameters()).device
                dtype = next(self.parameters()).dtype

                def to_model(t: torch.Tensor) -> torch.Tensor:
                    if not isinstance(t, torch.Tensor):
                        t = torch.as_tensor(t)
                    return t.to(device=device, dtype=dtype) if t.is_floating_point() else t.to(device=device)

                inputs = {k: to_model(v) for k, v in inputs.items()}
                inputs["proprio"] = to_model(proprio.unsqueeze(0))

                steps = int(payload.get("steps", 10))
                action = self.generate_actions(**inputs, steps=steps).squeeze(0).float().cpu().numpy()
                return JSONResponse({"action": action.tolist()})

            except Exception:
                logging.error(traceback.format_exc())
                return JSONResponse({"error": "Request failed"}, status_code=400)

        self.app = app

    def run(self, processor, host: str = "0.0.0.0", port: int = 8000):
        """Launch the FastAPI service."""
        self._build_app(processor)
        assert self.app is not None
        uvicorn.run(self.app, host=host, port=port)
