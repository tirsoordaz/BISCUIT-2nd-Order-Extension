#!/bin/bash
set -eo pipefail

# Assumes the user has already activated their Conda environment 
# and is running this script from the root of the BISCUIT repository.

export PYTHONPATH="."
export PYTHONUNBUFFERED=1

python experiments/train_nf_2nd_order.py \
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
    --seed 42