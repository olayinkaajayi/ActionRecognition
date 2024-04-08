#!/bin/bash


# bash script_tensorboard.sh

log_path=~/codes/runs/
# log_path=~/codes/infogcn_tenX/

tmux new -s tensorboard_infogcn -d
tmux send-keys "conda activate yinka_env
" C-m
tmux send-keys "python3 -m tensorboard.main --logdir $log_path --port 8889
" C-m
