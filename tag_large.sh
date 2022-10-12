#!/bin/bash
array=( $@ )
len=${#array[@]}
last_args=${array[@]:2:$len}

python3 attack.py --baseline --dataset $1 --split test --loss tag --n_inputs 100 -b $2 --tag_factor 0.01 --lr 0.03 --lr_decay_type LambdaLR --grad_clip 1.0 --bert_path bert-large-uncased --n_steps 10000 --opt_alg bert-adam --lr_max_it 10000 $last_args
