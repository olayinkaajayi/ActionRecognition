#!/bin/bash


############
# Usage
############

# bash script_action_recog.sh



############
# Action Recognition model
############


############
# Action Recognition scripts for 4 seed versions of NAPE
############

code=~/codes/DHCS_implement/run_algo_improved.py
# GPU
# checkpoint_seed0=checkpoint_multistream_view60_seed0_PE.pt
# checkpoint_seed8=checkpoint_multistream_view60_seed8_PE.pt
# checkpoint_seed11=checkpoint_multistream_view60_seed11_PE.pt
# checkpoint_seed19=checkpoint_multistream_view60_seed19_PE.pt

checkpoint_infodata=checkpoint_multistream_view60_infodata.pt
# CPU
checkpoint_seed0=checkpoint_multistream_view60_seed0_PE_CPU_new_preprocess_ls02.pt
checkpoint_seed8=checkpoint_multistream_view60_seed8_PE_CPU_new_preprocess_ls02.pt
checkpoint_seed11=checkpoint_multistream_view60_seed11_PE_CPU_new_preprocess_ls02.pt
checkpoint_seed19=checkpoint_multistream_view60_seed19_PE_CPU_new_preprocess_ls02.pt

checkpoint_pos_encode_seed0=checkpoint_pos_encode_seed0.pt
checkpoint_pos_encode_seed8=checkpoint_pos_encode_seed8.pt
checkpoint_pos_encode_seed11=checkpoint_pos_encode_seed11.pt
checkpoint_pos_encode_seed19=checkpoint_pos_encode_seed19.pt


code=/dcs/pg20/u2034358/codes/DHCS_implement/run_algo_improved.py

learning_rate=0.001
batch_size=384

seed0=0
seed8=8
seed11=11
seed19=19

cv60=CV
sub60=CS
set120=CSet
sub120=CSub

tmux new -s multistream_seeds_p -d
tmux send-keys -t multistream_seeds_p "cd codes" C-m
tmux send-keys -t multistream_seeds_p "conda activate yinka_env" C-m


#cross-view 60 classes
tmux send-keys -t multistream_seeds_p "
CUDA_VISIBLE_DEVICES=0,1,2 python3 $code --checkpoint checkpoint_multistream_view60_infodata_seed0_no_PE.pt \
--seed=$seed0 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$cv60 --num_class 60 --epochs 140 --no-PE &

CUDA_VISIBLE_DEVICES=0,1,2 python3 $code --checkpoint checkpoint_multistream_view60_infodata_seed0_zeros.pt \
--seed=$seed0 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$cv60 --num_class 60 --epochs 140 --zeros &

CUDA_VISIBLE_DEVICES=0,1,2 python3 $code --checkpoint checkpoint_multistream_view60_infodata_seed0_no_interact.pt \
--seed=$seed0 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$cv60 --num_class 60 --epochs 140 --no-interact
wait" C-m

# # compute average of the seeds
# tmux send-keys -t multistream_seeds_p "
# python3 $code --datacase=$cv60 --num_class 60 --avg_best_acc --no-PE
# wait" C-m
#
# #cross-subject 60 classes
# tmux send-keys -t multistream_seeds_p "
# CUDA_VISIBLE_DEVICES=0,1,2 python3 $code --checkpoint checkpoint_multistream_sub60_infodata_seed0_no_PE.pt \
# --seed=$seed0 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$sub60 --num_class 60 --epochs 140 --no-PE &
#
# CUDA_VISIBLE_DEVICES=0,1,2 python3 $code --checkpoint checkpoint_multistream_sub60_infodata_seed8_no_PE.pt \
# --seed=$seed8 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$sub60 --num_class 60 --epochs 140 --no-PE &
#
# CUDA_VISIBLE_DEVICES=0,1,2 python3 $code --checkpoint checkpoint_multistream_sub60_infodata_seed11_no_PE.pt \
# --seed=$seed11 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$sub60 --num_class 60 --epochs 140 --no-PE &
#
# CUDA_VISIBLE_DEVICES=0,1,2 python3 $code --checkpoint checkpoint_multistream_sub60_infodata_seed19_no_PE.pt \
# --seed=$seed19 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$sub60 --num_class 60 --epochs 140 --no-PE
# wait" C-m
#
# # compute average of the seeds
# tmux send-keys -t multistream_seeds_p "
# python3 $code --datacase=$sub60 --num_class 60 --avg_best_acc --no-PE
# wait" C-m
#
#
# #cross-setup 120 classes
# tmux send-keys -t multistream_seeds_p "
# CUDA_VISIBLE_DEVICES=0,1,2 python3 $code --checkpoint checkpoint_multistream_setup120_infodata_seed0_no_PE.pt \
# --seed=$seed0 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$set120 --num_class 120 --epochs 140 --no-PE &
#
# CUDA_VISIBLE_DEVICES=0,1,2 python3 $code --checkpoint checkpoint_multistream_setup120_infodata_seed8_no_PE.pt \
# --seed=$seed8 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$set120 --num_class 120 --epochs 140 --no-PE &
#
# CUDA_VISIBLE_DEVICES=0,1,2 python3 $code --checkpoint checkpoint_multistream_setup120_infodata_seed11_no_PE.pt \
# --seed=$seed11 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$set120 --num_class 120 --epochs 140 --no-PE &
#
# CUDA_VISIBLE_DEVICES=0,1,2 python3 $code --checkpoint checkpoint_multistream_setup120_infodata_seed19_no_PE.pt \
# --seed=$seed19 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$set120 --num_class 120 --epochs 140 --no-PE
# wait" C-m
#
# compute average of the seeds
# tmux send-keys -t multistream_seeds_p "
# python3 $code --datacase=$set120 --num_class 120 --avg_best_acc --no-PE
# wait" C-m
#
#
# #cross-subject 120 classes
# tmux send-keys -t multistream_seeds_p "
# CUDA_VISIBLE_DEVICES=0,1,2 python3 $code --checkpoint checkpoint_multistream_sub120_infodata_seed0_no_PE.pt \
# --seed=$seed0 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$sub120 --num_class 120 --epochs 140 --no-PE &
#
# CUDA_VISIBLE_DEVICES=0,1,2 python3 $code --checkpoint checkpoint_multistream_sub120_infodata_seed8_no_PE.pt \
# --seed=$seed8 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$sub120 --num_class 120 --epochs 140 --no-PE &
#
# CUDA_VISIBLE_DEVICES=0,1,2 python3 $code --checkpoint checkpoint_multistream_sub120_infodata_seed11_no_PE.pt \
# --seed=$seed11 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$sub120 --num_class 120 --epochs 140 --no-PE &
#
# CUDA_VISIBLE_DEVICES=0,1,2 python3 $code --checkpoint checkpoint_multistream_sub120_infodata_seed19_no_PE.pt \
# --seed=$seed19 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$sub120 --num_class 120 --epochs 140 --no-PE
# wait" C-m
#
# # compute average of the seeds
# tmux send-keys -t multistream_seeds_p "
# python3 $code --datacase=$sub120 --num_class 120 --avg_best_acc --no-PE
# wait" C-m


# tmux send-keys "
# tmux kill-session -t multistream_seeds" C-m
