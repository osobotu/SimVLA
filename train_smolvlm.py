"""
SmolVLM-VLA Training Script

Training script for SmolVLM-VLA with pluggable perception encoders.

Usage (baseline B0):
    python train_smolvlm.py \\
        --output_dir ./runs/smolvlm_vla \\
        --train_metas_path ./train_metas.json \\
        --encoder_name smolvlm \\
        --batch_size 32 \\
        --action_mode libero_joint \\
        --num_actions 10 \\
        --data_frac 1.0

Usage (V-JEPA 2 with temporal clips):
    python train_smolvlm.py \\
        --encoder_name vjepa2 \\
        --encoder_ckpt facebook/vjepa2-vitl-16 \\
        --encoder_frozen \\
        --num_frames 4 \\
        --output_dir ./runs/vjepa2_f1.0_s0
"""

import os
import math
import time
import json
import random
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.optim import AdamW

from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import create_smolvlm_dataloader
from models.modeling_smolvlm_vla import SmolVLMVLA
from models.processing_smolvlm_vla import SmolVLMVLAProcessor

import logging
import sys

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


# ============================================================
# Logger
# ============================================================
def get_logger(name="train_smolvlm", output_dir=None, accelerator=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger
    is_main = accelerator is None or accelerator.is_main_process
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    if is_main:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(level)
        logger.addHandler(ch)
    if output_dir and is_main:
        os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(output_dir, "train_smolvlm.log"), mode="a")
        fh.setFormatter(formatter)
        fh.setLevel(level)
        logger.addHandler(fh)
    return logger


# ============================================================
# Argument Parser
# ============================================================
def get_args_parser():
    parser = argparse.ArgumentParser("SmolVLM-VLA Training", add_help=False)

    # I/O
    parser.add_argument("--models", type=str, default=None,
                        help="Path to pretrained SmolVLM-VLA checkpoint (optional)")
    parser.add_argument("--output_dir", type=str, default="runnings_smolvlm",
                        help="Directory to save checkpoints")

    # SmolVLM backbone
    parser.add_argument("--smolvlm_model_path", type=str,
                        default="HuggingFaceTB/SmolVLM-500M-Instruct",
                        help="Path or HF repo for SmolVLM backbone")

    # ── Pluggable encoder ──
    parser.add_argument("--encoder_name", type=str, default="smolvlm",
                        choices=["smolvlm", "dinov2", "ijepa", "vjepa2"],
                        help="Which vision encoder to use (B0/E1/E2/E3)")
    parser.add_argument("--encoder_ckpt", type=str, default="",
                        help="HF repo id or local path to encoder weights")
    parser.add_argument("--encoder_frozen", action="store_true", default=True,
                        help="Freeze encoder backbone (default: True)")
    parser.add_argument("--encoder_no_frozen", dest="encoder_frozen",
                        action="store_false",
                        help="Unfreeze encoder backbone from step 0")
    parser.add_argument("--encoder_lora", action="store_true", default=False,
                        help="Inject LoRA adapters into encoder backbone")
    parser.add_argument("--encoder_arch", type=str, default="vith14",
                        help="I-JEPA sub-architecture (vith14|vitl16|vitb16)")
    parser.add_argument("--encoder_t_pad", type=int, default=4,
                        help="V-JEPA 2 fallback frame-repeat (single-frame datasets)")
    parser.add_argument("--encoder_video_size", type=int, default=224,
                        help="V-JEPA 2 internal frame resize (0=skip, use raw resolution)")

    # ── Data ──
    parser.add_argument("--train_metas_path", type=str, required=True,
                        help="Path to training metadata")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--num_frames", type=int, default=1,
                        help="Temporal clip length per sample. 1=single frame (default). "
                             "Set >1 for V-JEPA 2 with real consecutive frames.")
    parser.add_argument("--data_frac", type=float, default=1.0,
                        help="Fraction of training trajectories to use (0.0–1.0). "
                             "Deterministic subsample at trajectory level.")
    parser.add_argument("--data_seed", type=int, default=42,
                        help="Seed for deterministic trajectory subsampling.")

    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--learning_coef", type=float, default=1.0,
                        help="LR multiplier for VLM backbone param group")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Schedule
    parser.add_argument("--iters", type=int, default=1000000)
    parser.add_argument("--freeze_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--use_cosine_decay", action="store_true", default=False)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)

    # Logging / saving
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--log_interval", type=int, default=20)

    # System
    parser.add_argument("--seed", type=int, default=0)

    # Action mode
    parser.add_argument("--action_mode", type=str, default="galaxea_joint")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--norm_stats_path", type=str, default=None)
    parser.add_argument("--num_actions", type=int, default=10)

    # WandB
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_api_key", type=str, default=None)

    # Resume
    parser.add_argument("--resume", action="store_true", default=False)

    # DiT/AdaLN mode
    parser.add_argument("--use_adaln", action="store_true", default=False)

    # Model architecture
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)

    return parser


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True


def build_optimizer(
    model: SmolVLMVLA,
    lr: float,
    weight_decay: float,
    betas=(0.9, 0.95),
    lr_coef_vlm: float = 1.0,
):
    """
    Build AdamW with three param groups:
      - "vlm"              : language-model params (SmolLM text_model, ± SigLIP)
      - "transformer_core" : action transformer + encoder connector
      - "action_heads"     : action encoder/decoder heads

    For non-smolvlm encoders, model.vlm_parameters() returns only text_model
    params; SigLIP is frozen (requires_grad=False) and excluded automatically
    from gradient computation.

    The encoder backbone (DINOv2/I-JEPA/V-JEPA 2) is frozen by default via
    requires_grad=False so it contributes no optimizer state or gradients.
    With encoder_lora=True, LoRA adapter weights fall into "transformer_core".
    """
    vlm_params = model.vlm_parameters()

    if hasattr(model.transformer, "final_layer"):
        action_params = (
            list(model.transformer.final_layer.parameters())
            + list(model.transformer.action_encoder.parameters())
        )
    else:
        action_params = (
            list(model.transformer.action_decoder.parameters())
            + list(model.transformer.action_encoder.parameters())
        )

    exclude_ids = set(id(p) for p in vlm_params + action_params)
    transformer_core_params = [
        p for p in model.parameters()
        if id(p) not in exclude_ids and p.requires_grad
    ]

    param_groups = [
        {"name": "vlm",              "params": vlm_params,              "lr": 0.0, "weight_decay": weight_decay},
        {"name": "transformer_core", "params": transformer_core_params, "lr": 0.0, "weight_decay": weight_decay},
        {"name": "action_heads",     "params": action_params,           "lr": lr,  "weight_decay": weight_decay},
    ]
    return AdamW(param_groups, betas=betas)


def set_group_lr(optim: torch.optim.Optimizer, name: str, lr: float):
    for g in optim.param_groups:
        if g["name"] == name:
            g["lr"] = lr


def get_group_lr(optim: torch.optim.Optimizer, name: str) -> float:
    for g in optim.param_groups:
        if g["name"] == name:
            return g["lr"]
    return 0.0


def linear_warmup_cosine(step, start, warmup, total, base_lr, min_ratio):
    if step < start:
        return 0.0
    progress = step - start
    if progress < warmup:
        return base_lr * (progress / max(1, warmup))
    remain = max(1, total - (start + warmup))
    ratio = 0.5 * (1 + math.cos(math.pi * min(1.0, (progress - warmup) / remain)))
    return base_lr * (min_ratio + (1 - min_ratio) * ratio)


def update_group_lrs(optim, step, args):
    base = {
        "vlm":              args.learning_rate * args.learning_coef,
        "transformer_core": args.learning_rate,
        "action_heads":     args.learning_rate,
    }

    def schedule(step, base_lr):
        return linear_warmup_cosine(
            step, args.freeze_steps, args.warmup_steps,
            args.iters, base_lr, args.min_lr_ratio
        )

    if step < args.freeze_steps:
        set_group_lr(optim, "vlm", 0.0)
        set_group_lr(optim, "transformer_core", 0.0)
        set_group_lr(optim, "action_heads", base["action_heads"])
    else:
        for name, base_lr in base.items():
            new_lr = schedule(step, base_lr) if args.use_cosine_decay else base_lr
            set_group_lr(optim, name, new_lr)


# ============================================================
# Main Training
# ============================================================
def main(args):
    output_dir = Path(args.output_dir)

    wandb_api_key = os.environ.get("WANDB_API_KEY") or args.wandb_api_key
    wandb_project = os.environ.get("WANDB_PROJECT") or args.wandb_project
    use_wandb = WANDB_AVAILABLE and wandb_api_key

    log_with = ["tensorboard"]
    if use_wandb:
        log_with.append("wandb")
        os.environ["WANDB_API_KEY"] = wandb_api_key

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        log_with=log_with,
        project_dir=output_dir,
        kwargs_handlers=[ddp_kwargs],
    )

    tracker_config = {
        "learning_rate":      args.learning_rate,
        "batch_size":         args.batch_size,
        "iters":              args.iters,
        "smolvlm_model_path": args.smolvlm_model_path,
        "encoder_name":       args.encoder_name,
        "encoder_ckpt":       args.encoder_ckpt,
        "encoder_frozen":     args.encoder_frozen,
        "encoder_lora":       args.encoder_lora,
        "num_frames":         args.num_frames,
        "data_frac":          args.data_frac,
        "data_seed":          args.data_seed,
        "freeze_steps":       args.freeze_steps,
        "warmup_steps":       args.warmup_steps,
        "action_mode":        args.action_mode,
        "num_actions":        args.num_actions,
        "image_size":         args.image_size,
        "hidden_size":        args.hidden_size,
        "depth":              args.depth,
        "use_adaln":          args.use_adaln,
        "seed":               args.seed,
    }

    if use_wandb:
        accelerator.init_trackers(
            project_name=wandb_project,
            config=tracker_config,
            init_kwargs={"wandb": {"name": f"{args.encoder_name}-{time.strftime('%Y%m%d-%H%M%S')}"}},
        )
    else:
        accelerator.init_trackers("SmolVLM-VLA-Training", config=tracker_config)

    accelerator.wait_for_everyone()
    logger = get_logger(__name__, output_dir=output_dir, accelerator=accelerator)

    set_seed(args.seed + accelerator.process_index)
    logger.info(f"Args: {args}")

    # ── Model ──
    from models.configuration_smolvlm_vla import SmolVLMVLAConfig
    from models.action_hub import build_action_space

    action_space_kwargs = {}
    if args.norm_stats_path:
        action_space_kwargs["norm_stats_path"] = args.norm_stats_path

    load_path = args.models

    if load_path and os.path.isdir(load_path) and os.path.exists(
        os.path.join(load_path, "model.safetensors")
    ):
        logger.info(f"Loading SmolVLM-VLA from checkpoint: {load_path}")
        model = SmolVLMVLA.from_pretrained(load_path)

        if args.action_mode != model.action_mode:
            model.action_mode = args.action_mode
            model.action_space = build_action_space(args.action_mode, **action_space_kwargs)
        elif action_space_kwargs:
            model.action_space = build_action_space(args.action_mode, **action_space_kwargs)

        if args.num_actions != model.num_actions:
            model.config.num_actions = args.num_actions
            model.num_actions = args.num_actions
    else:
        logger.info("Initializing SmolVLM-VLA from config")
        logger.info(f"  encoder_name:  {args.encoder_name}")
        logger.info(f"  encoder_ckpt:  {args.encoder_ckpt or '(default)'}")
        logger.info(f"  encoder_frozen:{args.encoder_frozen}")
        logger.info(f"  encoder_lora:  {args.encoder_lora}")
        logger.info(f"  num_frames:    {args.num_frames}")
        logger.info(f"  data_frac:     {args.data_frac}  seed={args.data_seed}")

        config = SmolVLMVLAConfig(
            smolvlm_model_path=args.smolvlm_model_path,
            encoder_name=args.encoder_name,
            encoder_ckpt=args.encoder_ckpt,
            encoder_frozen=args.encoder_frozen,
            encoder_lora=args.encoder_lora,
            encoder_arch=args.encoder_arch,
            encoder_t_pad=args.encoder_t_pad,
            encoder_video_size=args.encoder_video_size,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            action_mode=args.action_mode,
            num_actions=args.num_actions,
            use_adaln=args.use_adaln,
            image_size=args.image_size,
        )
        model = SmolVLMVLA(config)

        if action_space_kwargs:
            model.action_space = build_action_space(args.action_mode, **action_space_kwargs)

    # Save git hash for reproducibility
    if accelerator.is_main_process:
        try:
            import subprocess
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "run_info.json", "w") as f:
                json.dump({
                    "git_hash": git_hash,
                    "encoder_name": args.encoder_name,
                    "encoder_ckpt": args.encoder_ckpt,
                    "data_frac": args.data_frac,
                    "data_seed": args.data_seed,
                    "num_frames": args.num_frames,
                    "seed": args.seed,
                }, f, indent=2)
        except Exception:
            pass

    # ── Processor ──
    processor = SmolVLMVLAProcessor.from_pretrained(args.smolvlm_model_path)

    # ── Dataloader ──
    train_dataloader = create_smolvlm_dataloader(
        batch_size=args.batch_size,
        metas_path=args.train_metas_path,
        num_actions=model.num_actions,
        action_mode=model.action_mode,
        training=True,
        num_workers=args.num_workers,
        image_size=args.image_size,
        num_frames=args.num_frames,
        data_frac=args.data_frac,
        data_seed=args.data_seed,
    )

    # ── Optimizer ──
    optim = build_optimizer(
        model=model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
        lr_coef_vlm=args.learning_coef,
    )
    model, optim = accelerator.prepare(model, optim)

    # ── Resume ──
    model.train()
    start_step = 0
    if args.resume and load_path and os.path.isdir(load_path):
        state_json = os.path.join(load_path, "state.json")
        if os.path.exists(state_json):
            try:
                with open(state_json) as f:
                    start_step = int(json.load(f).get("global_step", 0))
                logger.info(f"Resuming from step: {start_step}")
            except Exception:
                pass

    global_step, t0 = start_step, time.time()
    logger.info(
        f"🚀 Training: encoder={args.encoder_name}  "
        f"data_frac={args.data_frac}  num_frames={args.num_frames}  "
        f"iters={args.iters}  world_size={accelerator.num_processes}"
    )

    # ── Training loop ──
    for batch in train_dataloader:
        lang = processor.encode_language(batch["language_instruction"])
        batch.pop("language_instruction", None)
        inputs = {**batch, **lang}
        inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}

        update_group_lrs(optim, global_step, args)

        loss_dict: Dict[str, torch.Tensor] = model(**inputs)
        loss = sum(loss_dict.values())

        accelerator.backward(loss)
        if args.max_grad_norm:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optim.step()
        optim.zero_grad()

        if global_step % args.log_interval == 0:
            logs = {k: v.detach().float().item() for k, v in loss_dict.items()}
            logs["loss_total"] = float(loss.detach().item())
            logs.update({f"lr_{g['name']}": g["lr"] for g in optim.param_groups})
            accelerator.log(logs, step=global_step)

            if accelerator.is_main_process:
                dt = (time.time() - t0) / args.log_interval
                t0 = time.time()
                logger.info(
                    f"[{global_step}/{args.iters}] "
                    f"loss={logs['loss_total']:.4f} "
                    f"lr_core={logs['lr_transformer_core']:.2e} "
                    f"lr_action={logs['lr_action_heads']:.2e} "
                    f"lr_vlm={logs['lr_vlm']:.2e} ({dt:.2f}s/it)"
                )

        global_step += 1
        if accelerator.is_main_process:
            if global_step == args.iters or global_step % args.save_interval == 0:
                save_dir = os.path.join(output_dir, f"ckpt-{global_step}")
                accelerator.print(f"💾 Saving model to {save_dir}")
                accelerator.unwrap_model(model).save_pretrained(
                    save_dir, safe_serialization=True
                )
                with open(os.path.join(save_dir, "state.json"), "w") as f:
                    json.dump({"global_step": global_step}, f)

        if global_step >= args.iters:
            break

    accelerator.end_training()


# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "SmolVLM-VLA training script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
