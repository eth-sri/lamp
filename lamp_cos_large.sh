#!/bin/bash
array=( $@ )
len=${#array[@]}
last_args=${array[@]:2:$len}

python3 attack.py --dataset $1 --split test --loss cos --n_inputs 100 -b $2 --swap_burnin 0.1 --swap_every 200 --coeff_perplexity 0.2 --coeff_reg 1 --lr 0.01 --lr_decay_type LambdaLR --grad_clip 0.5 --bert_path bert-large-uncased --n_steps 5000 --opt_alg bert-adam --lr_max_it 10000 $last_args
