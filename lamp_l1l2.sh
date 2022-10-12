#!/bin/bash
array=( $@ )
len=${#array[@]}
last_args=${array[@]:3:$len}

python3 attack.py --dataset $2 --split test --loss tag --n_inputs 100 -b $3 --coeff_perplexity 60 --coeff_reg 25 --lr 0.01 --lr_decay 0.89 --tag_factor 0.01 --bert_path $1 --n_steps 2000 $last_args

