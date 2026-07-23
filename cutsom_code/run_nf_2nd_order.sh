#!/bin/bash
set -eo pipefail

PROJECT=/home/causalml26_team000/biscuit_project2/BISCUIT
ENV=/home/causalml26_team000/.conda/envs/biscuit_env
PYTHON="$ENV/bin/python"

cd "$PROJECT"
export PYTHONPATH="$PROJECT"
export PYTHONUNBUFFERED=1

"$PYTHON" experiments/train_nf_2nd_order.py \
    --data_dir data/causal_world \
    --autoencoder_checkpoint data/causal_world/models/AE_32l_128hid.ckpt \
    --seq_len 3 \
    --num_latents 32 \
    --c_hid 64 \
    --num_flows 6 \
    --lr 1e-3 \
    --num_samples 2 \
    --batch_size 1024 \
    --warmup 100 \
    --seed 42 \
    --cluster
