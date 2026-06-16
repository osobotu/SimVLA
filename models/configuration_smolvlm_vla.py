"""
SmolVLM-VLA Configuration

Configuration class for SmolVLM-500M-Instruct based VLA model.
Uses SmolVLM as the vision-language backbone instead of Florence2.
"""

from transformers.configuration_utils import PretrainedConfig


class SmolVLMVLAConfig(PretrainedConfig):
    """
    Configuration class for the **SmolVLM-VLA (SmolVLM Vision-Language-Action)** model.

    This configuration defines all submodules of SmolVLM-VLA:
      - The visual-language backbone (SmolVLM-500M-Instruct)
      - The pluggable perception encoder (smolvlm | dinov2 | ijepa | vjepa2)
      - The temporal/action transformer
      - The action/proprio setup

    Encoder fields
    --------------
    encoder_name : str
        Which vision encoder to use. One of:
          "smolvlm" (B0 baseline) | "dinov2" (E1) | "ijepa" (E2) | "vjepa2" (E3).
    encoder_ckpt : str
        HuggingFace repo id or local path to encoder weights.
        Falls back to canonical defaults when empty.
    encoder_frozen : bool
        Freeze the vision backbone (connector + LLM still train). Default True.
    encoder_lora : bool
        Inject LoRA adapters into the frozen backbone (overrides encoder_frozen).
    encoder_arch : str
        Sub-architecture for I-JEPA ("vith14" | "vitl16" | "vitb16").
    encoder_t_pad : int
        Fallback frame-repeat count for V-JEPA 2 when dataset sends single
        frames. Ignored when the dataset yields real clips (num_frames > 1).
    encoder_video_size : int
        Resize each frame to this square before V-JEPA 2 encoding (224 default).
        Set to 0 to skip resize (uses raw dataset resolution, more VRAM).
    fusion : str
        Language fusion strategy. "llm" (keep SmolLM, all encoders) or
        "xattn" (cross-attention, stretch goal, not yet implemented).
    """

    model_type = "smolvlm_vla"

    def __init__(
        self,
        # === SmolVLM backbone (always loaded as LM backbone) ===
        smolvlm_model_path: str = "HuggingFaceTB/SmolVLM-500M-Instruct",

        # === Pluggable encoder ===
        encoder_name: str = "smolvlm",
        encoder_ckpt: str = "",
        encoder_frozen: bool = True,
        encoder_lora: bool = False,
        encoder_arch: str = "vith14",       # I-JEPA sub-arch
        encoder_t_pad: int = 4,             # V-JEPA 2 fallback temporal pad
        encoder_video_size: int = 224,      # V-JEPA 2 internal resize
        fusion: str = "llm",

        # === Transformer head ===
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dim_time: int = 32,
        max_len_seq: int = 512,

        # === Action & proprioception ===
        num_actions: int = 30,
        action_mode: str = "galaxea_joint",
        use_proprio: bool = True,

        # === DiT/AdaLN Mode ===
        use_adaln: bool = False,

        # === Image settings ===
        image_size: int = 384,
        num_views: int = 3,

        **kwargs,
    ):
        # SmolVLM backbone path
        self.smolvlm_model_path = smolvlm_model_path

        # Encoder config
        self.encoder_name = encoder_name
        self.encoder_ckpt = encoder_ckpt
        self.encoder_frozen = encoder_frozen
        self.encoder_lora = encoder_lora
        self.encoder_arch = encoder_arch
        self.encoder_t_pad = encoder_t_pad
        self.encoder_video_size = encoder_video_size
        self.fusion = fusion

        # Transformer hyperparameters
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dim_time = dim_time
        self.max_len_seq = max_len_seq

        # Action/proprioception settings
        self.num_actions = num_actions
        self.action_mode = action_mode
        self.use_proprio = use_proprio

        # DiT/AdaLN settings
        self.use_adaln = use_adaln

        # Image settings
        self.image_size = image_size
        self.num_views = num_views

        super().__init__(**kwargs)

    def to_dict(self):
        output = super().to_dict()
        return output
