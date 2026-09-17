#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Generic multi-node Slurm template for ArtiFixer training.
#
# Usage:
#   SPLIT_PATH=/path/to/trainval_test_split.json \
#   DL3DV_DIR=/path/to/DL3DV-ALL-960P \
#   PROMPT_DIR=/path/to/artifixer-data/DL3DV-ALL-960P-captions \
#   sbatch model_training/slurm/sample-slurm-submit.sh
#
# Set your cluster's account/partition with sbatch flags, or uncomment and edit:
##SBATCH --account=<account>
##SBATCH --partition=<partition>

#SBATCH --job-name=artifixer-train
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/artifixer-train-%j.out
#SBATCH --error=slurm-logs/artifixer-train-%j.err
#SBATCH --signal=B:SIGUSR1@600

set -euo pipefail

: "${SPLIT_PATH:?Set SPLIT_PATH to trainval_test_split.json}"
: "${DL3DV_DIR:?Set DL3DV_DIR to the DL3DV archive root}"
: "${PROMPT_DIR:?Set PROMPT_DIR to the prompt HDF5 root}"

REPO_DIR="${REPO_DIR:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)}"
if [[ -z "${REPO_DIR}" ]]; then
    echo "Could not determine REPO_DIR; set REPO_DIR or submit from inside the repo." >&2
    exit 1
fi

RUN_NAME="${RUN_NAME:-artifixer-s1-14b}"
PROJECT_DIR="${PROJECT_DIR:-${REPO_DIR}/runs/${RUN_NAME}}"
MODEL_ID="${MODEL_ID:-Wan-AI/Wan2.1-T2V-14B-Diffusers}"
LOG_WITH="${LOG_WITH:-wandb}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-8}}}"
GPUS_PER_NODE="${GPUS_PER_NODE##*:}"
GPUS_PER_NODE="${GPUS_PER_NODE%%(*}"
NUM_MACHINES="${SLURM_NNODES:-1}"
NUM_PROCESSES=$((NUM_MACHINES * GPUS_PER_NODE))
TARGET_EFFECTIVE_GPUS="${TARGET_EFFECTIVE_GPUS:-128}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-$(((TARGET_EFFECTIVE_GPUS + NUM_PROCESSES - 1) / NUM_PROCESSES))}"
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-29501}"
SHOULD_CHECKPOINT_FLAG="${SHOULD_CHECKPOINT_FLAG:-${PROJECT_DIR}/should_checkpoint}"
child=
preempt_signal=0

term_handler() {
    preempt_signal=1
    mkdir -p "${PROJECT_DIR}"
    touch "${SHOULD_CHECKPOINT_FLAG}"
}

read -r -d '' COMMAND <<EOF || true
set -euo pipefail
mkdir -p "${PROJECT_DIR}"
rm -f "${SHOULD_CHECKPOINT_FLAG}"
cd "${REPO_DIR}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ -n "\${HUGGINGFACE_HUB_CACHE:-}" ]]; then export HUGGINGFACE_HUB_CACHE; fi
accelerate launch \
  --rdzv_backend c10d \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  --multi_gpu \
  --num_machines "${NUM_MACHINES}" \
  --num_processes "${NUM_PROCESSES}" \
  --machine_rank \${SLURM_PROCID} \
  --module model_training.train \
  --project_dir "${PROJECT_DIR}" \
  --split_path "${SPLIT_PATH}" \
  --dl3dv_dir "${DL3DV_DIR}" \
  --prompt_dir "${PROMPT_DIR}" \
  --model_id "${MODEL_ID}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --log_with "${LOG_WITH}" \
  --tracker_run_name "${RUN_NAME}" \
  --resume_from_checkpoint auto \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}" \
  --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR:-2}" \
  --should_checkpoint_flag "${SHOULD_CHECKPOINT_FLAG}"
EOF

trap term_handler SIGUSR1 SIGTERM

mkdir -p "${SLURM_SUBMIT_DIR:-$PWD}/slurm-logs"

srun_args=(
    --nodes="${NUM_MACHINES}"
    --ntasks="${NUM_MACHINES}"
    --ntasks-per-node=1
)

# Optional Pyxis/Enroot-style container support. Leave CONTAINER_IMAGE unset if
# your Slurm environment launches directly into an activated Python environment.
if [[ -n "${CONTAINER_IMAGE:-}" ]]; then
    srun_args+=(
        --container-image="${CONTAINER_IMAGE}"
        --container-mounts="${CONTAINER_MOUNTS:-${HOME}:${HOME}}"
        --container-workdir="${REPO_DIR}"
    )
fi

srun "${srun_args[@]}" bash --noprofile --norc -c "${COMMAND}" &
child=$!
set +e
while true; do
    wait "${child}"
    child_status=$?
    if [[ "${child_status}" -gt 128 ]] && kill -0 "${child}" 2>/dev/null; then
        continue
    fi
    break
done
set -e

if [[ "${preempt_signal}" -eq 1 && -f "${SHOULD_CHECKPOINT_FLAG}" && "${child_status}" -eq 124 ]]; then
    echo "Preemption signal received, requeueing job ${SLURM_JOB_ID}"
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
fi

exit "${child_status}"
