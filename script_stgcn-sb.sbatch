#!/bin/bash
#
#SBATCH --job-name=stgcn # Job name for tracking
#SBATCH --partition=sip-ampere  # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16      # Number of CPU threads used by your job
#SBATCH --gres=gpu:2           # Number of GPUs to use 
# 60GB of system RAM will be allocated with 1 GPU specified

#SBATCH --time=2-00:00:00      # Job time limit set to 2 days (48 hours)
#SBATCH --mail-type=ALL # Events to send email on
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

## sbatch script_stgcn-sb.sbatch



############
# STGCN
############

# position encoding model
checkpoint_pos_encode=checkpoint_betaNAPE_ntu-sb.pt


code=run_algo_stgcn.py

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
python3 $code --checkpoint checkpoint_stgcn_view60_seed${seed0}.pt --checkpoint_PE $checkpoint_pos_encode  \
--seed=$seed0 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$cv60 --num_class 60 --no-PE &

#cross-view 60 classes: projection only
python3 $code --checkpoint checkpoint_stgcn_view60-proj_seed${seed0}.pt --checkpoint_PE $checkpoint_pos_encode  \
--seed=$seed0 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$cv60 --num_class 60 --just-project &

wait

#cross-view 60 classes: NAPE
python3 $code --checkpoint checkpoint_stgcn_view60-NAPE_seed${seed0}.pt --checkpoint_PE $checkpoint_pos_encode  \
--seed=$seed0 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$cv60 --num_class 60 &


#cross-subject 60 classes
python3 $code --checkpoint checkpoint_stgcn_sub60_seed${seed0}.pt --checkpoint_PE $checkpoint_pos_encode  \
--seed=$seed0 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$sub60 --num_class 60 --no-PE &

wait

#cross-subject 60 classes: projection only
python3 $code --checkpoint checkpoint_stgcn_sub60-proj_seed${seed0}.pt --checkpoint_PE $checkpoint_pos_encode  \
--seed=$seed0 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$sub60 --num_class 60 --just-project &

#cross-subject 60 classes: NAPE
python3 $code --checkpoint checkpoint_stgcn_sub60-NAPE_seed${seed0}.pt --checkpoint_PE $checkpoint_pos_encode  \
--seed=$seed0 --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$sub60 --num_class 60 &
