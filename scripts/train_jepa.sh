#!/bin/bash
# ============================================================
# SimVLA JEPA Experiment — PSC Bridges-2 sbatch template
#
# Usage (direct sbatch):
#   sbatch scripts/train_jepa.sh \
#       smolvlm "" 1.00 0 ./runs/smolvlm_f1.00_s0
#
#   Positional args:
#     $1  ENCODER      : smolvlm | dinov2 | ijepa | vjepa2
#     $2  ENCODER_CKPT : HF repo or local path (empty string → default)
#     $3  DATA_FRAC    : 0.10 | 0.25 | 0.50 | 1.00
#     $4  SEED         : integer seed
#     $5  OUTPUT_DIR   : absolute path under $PROJECT/runs/
#
# Usage (via submit_all.sh — args injected automatically)
# ============================================================

#SBATCH --job-name=simvla_jepa
#SBATCH --partition=GPU-shared
#SBATCH --gpus=l40s-48:4
#SBATCH -t 48:00:00
#SBATCH -A cis260130p
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ── Positional args ──────────────────────────────────────────
ENCODER="${1:-smolvlm}"
ENCODER_CKPT="${2:-}"
DATA_FRAC="${3:-1.00}"
SEED="${4:-0}"
OUTPUT_DIR="${5:-$PROJECT/runs/${ENCODER}_f${DATA_FRAC}_s${SEED}}"

# ── Environment ──────────────────────────────────────────────
module purge
module load cuda

CONDA_BASE="$(conda info --base 2>/dev/null || echo $HOME/miniconda3)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate simvla

export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Cache dirs — keep everything under $PROJECT to avoid $HOME quota
export HF_HOME="$PROJECT/cache/huggingface"
export TORCH_HOME="$PROJECT/cache/torch"
export PIP_CACHE_DIR="$PROJECT/cache/pip"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR"

# ── Derived settings ─────────────────────────────────────────
# Encoder-specific flags
NUM_FRAMES=1
ENCODER_FROZEN_FLAG="--encoder_frozen"
EXTRA_FLAGS=""

case "$ENCODER" in
  vjepa2)
    NUM_FRAMES=4
    # Temporal clip: 4 consecutive frames, resized to 224×224
    EXTRA_FLAGS="--encoder_t_pad 4 --encoder_video_size 224"
    ;;
  ijepa)
    EXTRA_FLAGS="--encoder_arch vith14"
    ;;
  dinov2)
    EXTRA_FLAGS=""
    ;;
  smolvlm)
    EXTRA_FLAGS=""
    ;;
esac

# LoRA rows (B0-ft and E3-ft) are handled via separate submit calls with
# --encoder_lora flag added to EXTRA_FLAGS.

# Effective batch size = 64 (4 GPUs × 16 per GPU)
# If OOM occurs after unfreeze (especially vjepa2), reduce --batch_size
# and increase gradient accumulation to hold effective batch constant.
BATCH_SIZE=16

echo "=================================================="
echo "ENCODER     : $ENCODER"
echo "ENCODER_CKPT: ${ENCODER_CKPT:-(default)}"
echo "DATA_FRAC   : $DATA_FRAC"
echo "SEED        : $SEED"
echo "NUM_FRAMES  : $NUM_FRAMES"
echo "OUTPUT_DIR  : $OUTPUT_DIR"
echo "=================================================="

# ── Checkpoint resume ────────────────────────────────────────
RESUME_FLAG=""
MODELS_FLAG=""
if ls "${OUTPUT_DIR}"/ckpt-*/model.safetensors 2>/dev/null | grep -q .; then
    LATEST_CKPT=$(ls -d "${OUTPUT_DIR}"/ckpt-* 2>/dev/null \
        | sort -t- -k2 -n | tail -1)
    if [ -n "$LATEST_CKPT" ]; then
        echo "Resuming from: $LATEST_CKPT"
        MODELS_FLAG="--models $LATEST_CKPT"
        RESUME_FLAG="--resume"
    fi
fi

# ── Training ─────────────────────────────────────────────────
cd "$SLURM_SUBMIT_DIR" || exit 1

torchrun \
    --nproc_per_node=4 \
    --master_port=$(( 29500 + RANDOM % 1000 )) \
    train_smolvlm.py \
    --output_dir "$OUTPUT_DIR" \
    --train_metas_path "./datasets/metas/libero_train.json" \
    --encoder_name "$ENCODER" \
    --encoder_ckpt "$ENCODER_CKPT" \
    $ENCODER_FROZEN_FLAG \
    --num_frames $NUM_FRAMES \
    --data_frac "$DATA_FRAC" \
    --data_seed 42 \
    --seed "$SEED" \
    --smolvlm_model_path "HuggingFaceTB/SmolVLM-500M-Instruct" \
    --action_mode libero_joint \
    --num_actions 10 \
    --norm_stats_path "./norm_stats/libero_norm.json" \
    --batch_size $BATCH_SIZE \
    --learning_rate 1e-4 \
    --learning_coef 0.1 \
    --betas 0.9 0.95 \
    --weight_decay 0.0 \
    --max_grad_norm 1.0 \
    --iters 40000 \
    --freeze_steps 1000 \
    --warmup_steps 2000 \
    --save_interval 5000 \
    --log_interval 20 \
    --hidden_size 768 \
    --depth 12 \
    --num_heads 12 \
    --image_size 384 \
    --num_workers 4 \
    $MODELS_FLAG \
    $RESUME_FLAG \
    $EXTRA_FLAGS
