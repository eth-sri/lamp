#!/bin/bash
array=( $@ )
len=${#array[@]}
last_args=${array[@]:3:$len}

python3 attack.py --dataset $2 --split test --loss cos --n_inputs 100 -b $3 --coeff_perplexity 0.2 --coeff_reg 1 --lr 0.01 --lr_decay 0.89 --bert_path $1 --n_steps 2000 $last_args
