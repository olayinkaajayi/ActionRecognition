#!/bin/bash
#
#SBATCH --job-name=stgcn # Job name for tracking
#SBATCH --partition=sip-ampere  # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8      # Number of CPU threads used by your job
#SBATCH --gres=gpu:2           # Number of GPUs to use 
# 60GB of system RAM will be allocated with 1 GPU specified
###  #### SBATCH --array 0-3 # T allow for simultaneous running of tasks

#SBATCH --time=10:00:00      # Job time limit set to 2 days (48 hours)
#
#SBATCH --output=joboutput_%j.out # Standard out from your job
#SBATCH --error=joboutput_%j.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

## Execute your program(s) ##
module load CUDA/12.2


############
# Usage
############

## sbatch codes/DHCS_implement/script_stgcn-sb.sbatch



############
# STGCN
############

# position encoding model
checkpoint_pos_encode=checkpoint_betaNAPE_ntu-sb.pt


code=/dcs/pg20/u2034358/codes/DHCS_implement/run_algo_stgcn.py

learning_rate=0.001
batch_size=64

seed0=0
seed8=8
seed11=11
seed19=19

cv60=CV
sub60=CS
set120=CSet
sub120=CSub

cd codes
conda activate yinka_env

#cross-view 60 classes
python3 $code --checkpoint checkpoint_stgcn_view60_infodata_seed0.pt --checkpoint_PE $checkpoint_pos_encode  \
--seed=$seed0 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$cv60 --num_class 60 

# python3 $code --checkpoint checkpoint_multistream_view60_infodata_seed8.pt --checkpoint_PE $checkpoint_pos_encode \
# --seed=$seed8 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$cv60 --num_class 60 &

# python3 $code --checkpoint checkpoint_multistream_view60_infodata_seed11.pt --checkpoint_PE $checkpoint_pos_encode \
# --seed=$seed11 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$cv60 --num_class 60 &

# python3 $code --checkpoint checkpoint_multistream_view60_infodata_seed19.pt --checkpoint_PE $checkpoint_pos_encode \
# --seed=$seed19 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$cv60 --num_class 60 &

# # compute average of the seeds
# python3 $code --datacase=$cv60 --num_class 60 --avg_best_acc


# #cross-subject 60 classes
# python3 $code --checkpoint checkpoint_multistream_sub60_infodata_seed0.pt --checkpoint_PE $checkpoint_pos_encode \
# --seed=$seed0 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$sub60 --num_class 60 &

# python3 $code --checkpoint checkpoint_multistream_sub60_infodata_seed8.pt --checkpoint_PE $checkpoint_pos_encode \
# --seed=$seed8 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$sub60 --num_class 60 &

# python3 $code --checkpoint checkpoint_multistream_sub60_infodata_seed11.pt --checkpoint_PE $checkpoint_pos_encode \
# --seed=$seed11 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$sub60 --num_class 60 &

# python3 $code --checkpoint checkpoint_multistream_sub60_infodata_seed19.pt --checkpoint_PE $checkpoint_pos_encode \
# --seed=$seed19 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$sub60 --num_class 60 &

# # compute average of the seeds
# python3 $code --datacase=$sub60 --num_class 60 --avg_best_acc


# #cross-setup 120 classes
# python3 $code --checkpoint checkpoint_multistream_setup120_infodata_seed0.pt --checkpoint_PE $checkpoint_pos_encode \
# --seed=$seed0 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$set120 --num_class 120 &

# python3 $code --checkpoint checkpoint_multistream_setup120_infodata_seed8.pt --checkpoint_PE $checkpoint_pos_encode \
# --seed=$seed8 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$set120 --num_class 120 &

# python3 $code --checkpoint checkpoint_multistream_setup120_infodata_seed11.pt --checkpoint_PE $checkpoint_pos_encode \
# --seed=$seed11 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$set120 --num_class 120 &

# python3 $code --checkpoint checkpoint_multistream_setup120_infodata_seed19.pt --checkpoint_PE $checkpoint_pos_encode \
# --seed=$seed19 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$set120 --num_class 120 &

# # compute average of the seeds
# python3 $code --datacase=$set120 --num_class 120 --avg_best_acc


# #cross-ssubject 120 classes
# python3 $code --checkpoint checkpoint_multistream_sub120_infodata_seed0.pt --checkpoint_PE $checkpoint_pos_encode \
# --seed=$seed0 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$sub120 --num_class 120 &

# python3 $code --checkpoint checkpoint_multistream_sub120_infodata_seed8.pt --checkpoint_PE $checkpoint_pos_encode \
# --seed=$seed8 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$sub120 --num_class 120 &

# python3 $code --checkpoint checkpoint_multistream_sub120_infodata_seed11.pt --checkpoint_PE $checkpoint_pos_encode \
# --seed=$seed11 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$sub120 --num_class 120 &

# python3 $code --checkpoint checkpoint_multistream_sub120_infodata_seed19.pt --checkpoint_PE $checkpoint_pos_encode \
# --seed=$seed19 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$sub120 --num_class 120 &

# # compute average of the seeds
# python3 $code --datacase=$sub120 --num_class 120 --avg_best_acc

