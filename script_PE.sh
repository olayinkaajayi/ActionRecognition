#!/bin/bash


############
# Usage
############

# bash script_NAPE.sh



############
# NAPE
############


############
# script_NAPE - 4 seeds
############

seed0=0
seed1=1
seed8=8
seed11=11
seed19=19
code=~/codes/DHCS_implement/run_algo_pos_encode.py
tmux new -s NAPE_PE -d
tmux send-keys -t NAPE_PE "cd codes" C-m
tmux send-keys "conda activate yinka_env" C-m

tmux send-keys "
CUDA_VISIBLE_DEVICES=2 python3 $code --checkpoint NAPE_SigLin2.pt \
--lr 0.1 --many-gpu --epochs 400
" C-m

# tmux send-keys "
# CUDA_VISIBLE_DEVICES=0 python3 $code --checkpoint checkpoint_pos_encode_default.pt --lr 0.1 --many-gpu
# wait" C-m

# tmux send-keys "
# CUDA_VISIBLE_DEVICES=0,1,2 python3 $code --seed $seed0 --checkpoint checkpoint_pos_encode_seed0.pt --lr 0.1 --many-gpu
# wait" C-m
# tmux send-keys "
# CUDA_VISIBLE_DEVICES=0,1,2 python3 $code --seed $seed8 --checkpoint checkpoint_pos_encode_seed8.pt --lr 0.1 --many-gpu
# wait" C-m
# tmux send-keys "
# CUDA_VISIBLE_DEVICES=0,1,2 python3 $code --seed $seed11 --checkpoint checkpoint_pos_encode_seed11.pt --lr 0.1 --many-gpu
# wait" C-m
# tmux send-keys "
# CUDA_VISIBLE_DEVICES=0,1,2 python3 $code --seed $seed19 --checkpoint checkpoint_pos_encode_seed19.pt --lr 0.1 --many-gpu
# wait" C-m
# tmux send-keys "
# tmux kill-session -t NAPE_PE" C-m


# Use CPU
# tmux send-keys "
# CUDA_VISIBLE_DEVICES=0,1 python3 $code --seed $seed0 --checkpoint checkpoint_pos_encode_seed0_cudax.pt --lr 0.1 --many-gpu
# wait" C-m
# tmux send-keys "
# CUDA_VISIBLE_DEVICES=0,1 python3 $code --seed $seed1 --checkpoint checkpoint_pos_encode_seed1_cudax.pt --lr 0.1 --many-gpu
# wait" C-m
# tmux send-keys "
# CUDA_VISIBLE_DEVICES=0,1 python3 $code --seed $seed8 --checkpoint checkpoint_pos_encode_seed8_cudax.pt --lr 0.1 --many-gpu
# wait" C-m
# tmux send-keys "
# CUDA_VISIBLE_DEVICES=0,1 python3 $code --seed $seed11 --checkpoint checkpoint_pos_encode_seed11_cudax.pt --lr 0.1 --many-gpu
# wait" C-m
# tmux send-keys "
# CUDA_VISIBLE_DEVICES=0,1 python3 $code --seed $seed19 --checkpoint checkpoint_pos_encode_seed19_cudax.pt --lr 0.1 --many-gpu
# " C-m
# tmux send-keys "
# tmux kill-session -t NAPE_PE" C-m
