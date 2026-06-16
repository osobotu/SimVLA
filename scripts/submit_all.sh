#!/bin/bash
# ============================================================
# SimVLA JEPA — submit full experiment matrix
#
# Minimum publishable core (run first):
#   {B0=smolvlm, E1=dinov2, E3=vjepa2} x {0.10, 1.00} x 3 seeds
#
# Extended matrix (add after core results):
#   E2=ijepa x {0.10, 1.00} x 3 seeds
#   LoRA rows (B0-ft, E3-ft) x {0.10, 1.00} x 3 seeds
#   Data fractions {0.25, 0.50} x all encoders x 3 seeds
#
# Usage:
#   bash scripts/submit_all.sh                   # core only (default)
#   bash scripts/submit_all.sh --full            # full matrix
#   bash scripts/submit_all.sh --dry-run         # print commands, don't submit
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

DRY_RUN=0
FULL=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --full)    FULL=1 ;;
    esac
done

# ── Encoder defaults (HF repo or local path) ─────────────────
declare -A ENCODER_CKPT=(
    [smolvlm]=""
    [dinov2]="facebook/dinov2-base"
    [ijepa]="facebook/ijepa_vith14_1k"
    [vjepa2]="facebook/vjepa2-vitl-16"
)

submit() {
    local encoder="$1"
    local data_frac="$2"
    local seed="$3"
    local lora="${4:-0}"

    local ckpt="${ENCODER_CKPT[$encoder]}"
    local suffix=""
    local extra_flags=""
    if [ "$lora" = "1" ]; then
        suffix="-lora"
        extra_flags="--encoder_lora"
    fi

    local exp_id="${encoder}${suffix}_f${data_frac}_s${seed}"
    local output_dir="$PROJECT/runs/${exp_id}"

    local job_name="simvla_${exp_id}"

    local cmd="sbatch \
        --job-name=${job_name} \
        ${SCRIPT_DIR}/train_jepa.sh \
        ${encoder} \
        \"${ckpt}\" \
        ${data_frac} \
        ${seed} \
        ${output_dir}"

    echo "→ $exp_id"
    if [ "$DRY_RUN" = "1" ]; then
        echo "  [DRY] $cmd"
    else
        eval "$cmd"
        sleep 0.5   # avoid scheduler flood
    fi
}

# Baseline experiments
echo "=== CORE: {smolvlm, dinov2, vjepa2} x {0.10, 1.00} x seeds {0,1,2} ==="
for ENCODER in smolvlm dinov2 vjepa2; do
    for FRAC in 0.10 1.00; do
        for SEED in 0 1 2; do
            submit "$ENCODER" "$FRAC" "$SEED"
        done
    done
done

# Extended experiments
# if [ "$FULL" = "1" ]; then
#     echo ""
#     echo "=== EXTENDED: ijepa x {0.10, 1.00} x seeds {0,1,2} ==="
#     for FRAC in 0.10 1.00; do
#         for SEED in 0 1 2; do
#             submit ijepa "$FRAC" "$SEED"
#         done
#     done

#     echo ""
#     echo "=== LoRA rows: smolvlm-lora, vjepa2-lora x {0.10, 1.00} x seeds {0,1,2} ==="
#     for ENCODER in smolvlm vjepa2; do
#         for FRAC in 0.10 1.00; do
#             for SEED in 0 1 2; do
#                 submit "$ENCODER" "$FRAC" "$SEED" 1
#             done
#         done
#     done

#     echo ""
#     echo "=== Intermediate fractions: {0.25, 0.50} x all encoders x seeds {0,1,2} ==="
#     for ENCODER in smolvlm dinov2 ijepa vjepa2; do
#         for FRAC in 0.25 0.50; do
#             for SEED in 0 1 2; do
#                 submit "$ENCODER" "$FRAC" "$SEED"
#             done
#         done
#     done
# fi

echo ""
echo "Done. Monitor with: squeue -u \$USER"
