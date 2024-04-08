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


chkpt_5p44=FL_5p44_FS_0_Sd11_Dv_c.pt
chkpt_5p48=FL_5p48_FS_0_Sd19_Dv_cu2.pt
chkpt_5p51=FL_5p51_FS_2_Sd0_Dv_c.pt
# chkpt_5p51_ex=FL_5p51_FS_0_Sd11_Dv_cu1.pt
# chkpt_5p55=FL_5p55_FS_2_Sd0_Dv_cu1.pt
# chkpt_5p60=FL_5p60_FS_0_Sd11_Dv_cu2.pt
chkpt_5p62=FL_5p62_FS_0_Sd8_Dv_cu2.pt
chkpt_5p82=FL_5p82_FS_0_Sd1_Dv_c.pt
# chkpt_5p82_ex=FL_5p82_FS_0_Sd1_Dv_cu1.pt
# chkpt_6p22=FL_6p22_FS_0_Sd19_Dv_cu1.pt
# chkpt_6p22_ex=FL_6p22_FS_0_Sd19_Dv_c.pt
# chkpt_6p32=FL_6p32_FS_0_Sd8_Dv_c.pt
# chkpt_6p32_ex=FL_6p32_FS_0_Sd8_Dv_cu1.pt
# chkpt_6p41=FL_6p41_FS_0_Sd1_Dv_cu2.pt
chkpt_10p52=FL_10p52_FS_0_Sd0_Dv_cu2.pt


learning_rate=0.001
batch_size=384

seed=0
cv60=CV
sub60=CS
set120=CSet
sub120=CSub

case=$cv60
num_cls=60

tmux new -s multistream_seeds_sub -d
tmux send-keys -t multistream_seeds_sub "cd codes" C-m
tmux send-keys -t multistream_seeds_sub "conda activate yinka_env" C-m


#cross-view 60 classes
tmux send-keys -t multistream_seeds_sub "
CUDA_VISIBLE_DEVICES=0,1,2 python3 $code --checkpoint chkpt_ms_${case}${num_cls}_seed${seed}_${chkpt_5p44} --checkpoint_PE=$chkpt_5p44 \
--seed=$seed --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$case --num_class=$num_cls --epochs 140 &

CUDA_VISIBLE_DEVICES=0,1,2 python3 $code --checkpoint chkpt_ms_${case}${num_cls}_seed${seed}_${chkpt_5p48} --checkpoint_PE=$chkpt_5p48 \
--seed=$seed --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$case --num_class=$num_cls --epochs 140 &

CUDA_VISIBLE_DEVICES=0,1,2 python3 $code --checkpoint chkpt_ms_${case}${num_cls}_seed${seed}_${chkpt_5p51} --checkpoint_PE=$chkpt_5p51 \
--seed=$seed --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$case --num_class=$num_cls --epochs 140 &

CUDA_VISIBLE_DEVICES=0,1,2 python3 $code --checkpoint chkpt_ms_${case}${num_cls}_seed${seed}_${chkpt_5p62} --checkpoint_PE=$chkpt_5p62 \
--seed=$seed --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$case --num_class=$num_cls --epochs 140
wait" C-m


tmux send-keys -t multistream_seeds_sub "
CUDA_VISIBLE_DEVICES=0,1,2 python3 $code --checkpoint chkpt_ms_${case}${num_cls}_seed${seed}_${chkpt_5p82} --checkpoint_PE=$chkpt_5p82 \
--seed=$seed --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$case --num_class=$num_cls &

CUDA_VISIBLE_DEVICES=0,1,2 python3 $code --checkpoint chkpt_ms_${case}${num_cls}_seed${seed}_${chkpt_6p32} --checkpoint_PE=$chkpt_6p32 \
--seed=$seed --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$case --num_class=$num_cls &

CUDA_VISIBLE_DEVICES=0,1,2 python3 $code --checkpoint chkpt_ms_${case}${num_cls}_seed${seed}_${chkpt_10p52} --checkpoint_PE=$chkpt_10p52 \
--seed=$seed --lr=$learning_rate --bs=$batch_size --many-gpu --info_data --datacase=$case --num_class=$num_cls
wait" C-m
