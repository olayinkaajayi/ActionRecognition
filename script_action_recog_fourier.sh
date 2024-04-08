#!/bin/bash


############
# Usage
############

# bash script_action_recog_fourier.sh



############
# Action Recognition model using Fourier Resistors as Temporal module
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

# chkpt=chkpt_mulstrm_fourier.pt
# tenXbrd=multisteam_fourier
#
# tmux new -s multistream_fourier -d
# tmux send-keys -t multistream_fourier "cd codes" C-m
# tmux send-keys -t multistream_fourier_mlp_skip "conda activate yinka_env" C-m
#
# tmux send-keys -t multistream_fourier "
# CUDA_VISIBLE_DEVICES=0,1,2 python3 $code --checkpoint $chkpt --seed=$seed0 --lr=$learning_rate \
# --bs=$batch_size --many-gpu --info_data --datacase=$cv60 --num_class 60 -t $tenXbrd
# wait" C-m


# chkpt=chkpt_mulstrm_fourier_mlp.pt
# tenXbrd=multisteam_fourier_mlp
#
# tmux new -s multistream_fourier_mlp -d
# tmux send-keys -t multistream_fourier_mlp "cd codes" C-m
# tmux send-keys -t multistream_fourier_mlp "conda activate yinka_env" C-m
#
# tmux send-keys -t multistream_fourier_mlp "
# CUDA_VISIBLE_DEVICES=0,1,2 python3 $code --checkpoint $chkpt --seed=$seed0 --lr=$learning_rate \
# --bs=$batch_size --many-gpu --info_data --datacase=$cv60 --num_class 60 -t $tenXbrd
# wait" C-m

# chkpt=chkpt_mulstrm_fourier_mlp_skip.pt
# tenXbrd=multisteam_fourier_mlp_skip
#
# tmux new -s multistream_fourier_mlp_skip -d
# tmux send-keys -t multistream_fourier_mlp_skip "cd codes" C-m
# tmux send-keys -t multistream_fourier_mlp_skip "conda activate yinka_env" C-m

# chkpt=chkpt_mulstrm_fourier_mlp_norm.pt
# tenXbrd=multisteam_fourier_mlp_norm
#
# tmux new -s multistream_fourier_mlp_norm -d
# tmux send-keys -t multistream_fourier_mlp_norm "cd codes" C-m
# tmux send-keys -t multistream_fourier_mlp_norm "conda activate yinka_env" C-m
#
# tmux send-keys -t multistream_fourier_mlp_norm "
# CUDA_VISIBLE_DEVICES=0,1,2 python3 $code --checkpoint $chkpt --seed=$seed0 --lr=$learning_rate \
# --bs=$batch_size --many-gpu --info_data --datacase=$cv60 --num_class 60 -t $tenXbrd
# wait" C-m

# chkpt=chkpt_mulstrm_fourier_mlp_norm_pos.pt
# tenXbrd=multisteam_fourier_mlp_norm_pos
#
# tmux new -s multistream_fourier_mlp_norm_pos -d
# tmux send-keys -t multistream_fourier_mlp_norm_pos "cd codes" C-m
# tmux send-keys -t multistream_fourier_mlp_norm_pos "conda activate yinka_env" C-m
#
# tmux send-keys -t multistream_fourier_mlp_norm_pos "
# CUDA_VISIBLE_DEVICES=0,1,2 python3 $code --checkpoint $chkpt --seed=$seed0 --lr=$learning_rate \
# --bs=$batch_size --many-gpu --info_data --datacase=$cv60 --num_class 60 -t $tenXbrd
# wait" C-m

# chkpt=chkpt_mulstrm_fourier_mlp_norm_addPos.pt
# tenXbrd=multisteam_fourier_mlp_norm_addPos
#
# tmux new -s multistream_fourier_mlp_norm_addPos -d
# tmux send-keys -t multistream_fourier_mlp_norm_addPos "cd codes" C-m
# tmux send-keys -t multistream_fourier_mlp_norm_addPos "conda activate yinka_env" C-m
#
# tmux send-keys -t multistream_fourier_mlp_norm_addPos "
# CUDA_VISIBLE_DEVICES=0,1,2 python3 $code --checkpoint $chkpt --seed=$seed0 --lr=$learning_rate \
# --bs=$batch_size --many-gpu --info_data --datacase=$cv60 --num_class 60 -t $tenXbrd
# wait" C-m

# chkpt=chkpt_mulstrm_fourier_mlp_norm_pos_flat.pt
# tenXbrd=multisteam_fourier_mlp_norm_pos_flat
#
# tmux new -s multistream_fourier_mlp_norm_pos_flat -d
# tmux send-keys -t multistream_fourier_mlp_norm_pos_flat "cd codes" C-m
# tmux send-keys -t multistream_fourier_mlp_norm_pos_flat "conda activate yinka_env" C-m
#
# tmux send-keys -t multistream_fourier_mlp_norm_pos_flat "
# CUDA_VISIBLE_DEVICES=0,1,2 python3 $code --checkpoint $chkpt --seed=$seed0 --lr=$learning_rate \
# --bs=$batch_size --many-gpu --info_data --datacase=$cv60 --num_class 60 -t $tenXbrd
# wait" C-m

chkpt=chkpt_mulstrm_fourier_mlp_norm_pos_mlpHEAD.pt
tenXbrd=multisteam_fourier_mlp_norm_pos_mlpHEAD

tmux new -s multistream_fourier_mlp_norm_pos_mlpHEAD -d
tmux send-keys -t multistream_fourier_mlp_norm_pos_mlpHEAD "cd codes" C-m
tmux send-keys -t multistream_fourier_mlp_norm_pos_mlpHEAD "conda activate yinka_env" C-m

tmux send-keys -t multistream_fourier_mlp_norm_pos_mlpHEAD "
CUDA_VISIBLE_DEVICES=0,1,2 python3 $code --checkpoint $chkpt --seed=$seed0 --lr=$learning_rate \
--bs=$batch_size --many-gpu --info_data --datacase=$cv60 --num_class 60 -t $tenXbrd
wait" C-m

# tmux send-keys "
# tmux kill-session -t multistream_seeds" C-m
