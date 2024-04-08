#!/bin/bash
#
#SBATCH --job-name=stgcn-arr # Job name for tracking
#SBATCH --partition=sip-ampere  # Partition you wish to use
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8      # Number of CPU threads used by your job
#SBATCH --gres=gpu:2           # Number of GPUs to use 

#SBATCH --array=0-3  # One job for each of the 4 data cases (modify if needed)
#                    # 0 - cv60 (cross-view)
#                    # 1 - sub60 (cross-subject)
#                    # 2 - set120 (cross-setup)
#                    # 3 - sub120 (cross-subjet)

# 120GB of system RAM will be requested with 2 GPUs specified

#SBATCH --time=2-00:00:00      # Job time limit set to 2 days (48 hours)
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on
#SBATCH --output=joboutput_%A_%a.out # Standard out from your job
#SBATCH --error=joboutput_%A_%a.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

## Execute your program(s) ##
module load CUDA/12.2


############
# Usage
############

## sbatch codes/DHCS_implement/script_stgcn-sb-array.sbatch



############
# Spatiotemporal GCN model
############

# position encoding model
checkpoint_pos_encode=checkpoint_betaNAPE_ntu-sb.pt
# checkpoint_pos_encode=checkpoint_pos_encode_default.pt
# checkpoint_pos_encode=checkpoint_pos_encode.pt


code=/dcs/pg20/u2034358/codes/DHCS_implement/run_algo_stgcn.py

# Parameters for training
learning_rate=0.001
batch_size=64

# Seed values
seed0=0
seed8=8
seed11=11
seed19=19

# Get the data case based on the SLURM array task ID
data_case=(CV CS CSet CSub)
current_case=${data_case[$SLURM_ARRAY_TASK_ID]}

# Determine the number of classes based on the data case
if [[ "$current_case" == "CV" || "$current_case" == "CS" ]]; then
  num_class=60
else
  num_class=120
fi


cd codes
conda activate yinka_env


python3 $code --checkpoint checkpoint_stgcn_${current_case}${num_class}_infodata_seed${seed0}.pt --checkpoint_PE $checkpoint_pos_encode  \
--seed=$seed0 --lr=$learning_rate --bs=$batch_size  --info_data --datacase=$current_case --num_class=$num_class --no-PE &

python3 $code --checkpoint checkpoint_stgcn_${current_case}${num_class}_infodata_seed${seed8}.pt --checkpoint_PE $checkpoint_pos_encode \
--seed=$seed8 --lr=$learning_rate --bs=$batch_size  --info_data --datacase=$current_case --num_class=$num_class --no-PE &

wait

python3 $code --checkpoint checkpoint_stgcn_${current_case}${num_class}_infodata_seed${seed11}.pt --checkpoint_PE $checkpoint_pos_encode \
--seed=$seed11 --lr=$learning_rate --bs=$batch_size  --info_data --datacase=$current_case --num_class $num_class --no-PE &

python3 $code --checkpoint checkpoint_stgcn_${current_case}${num_class}_infodata_seed${seed19}.pt --checkpoint_PE $checkpoint_pos_encode \
--seed=$seed19 --lr=$learning_rate --bs=$batch_size  --info_data --datacase=$current_case --num_class=$num_class --no-PE &

wait

# compute average of the seeds (assuming same command for all data cases)
python3 $code --datacase=$current_case --num_class $num_class --avg_best_acc

