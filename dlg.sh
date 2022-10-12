#!/bin/bash
array=( $@ )
len=${#array[@]}
last_args=${array[@]:3:$len}

python3 attack.py --baseline --dataset $2 --split test --loss dlg --n_inputs 100 -b $3 --lr 0.1 --lr_decay 1 --bert_path $1 --n_steps 2500 $last_args
