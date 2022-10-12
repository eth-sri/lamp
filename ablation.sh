#!/bin/bash
echo "LAMP with cosine loss"
python3 attack.py --dataset $1 --split test --loss cos --n_inputs 100 -b 1 --coeff_perplexity 0.2 --coeff_reg 1 --lr 0.01 --lr_decay 0.89 --bert_path bert-base-uncased

echo "LAMP with L1+L2 loss"
python3 attack.py --dataset $1 --split test --loss tag --n_inputs 100 -b 1 --coeff_perplexity 60 --coeff_reg 25 --lr 0.01 --lr_decay 0.89 --tag_factor 0.01 --bert_path bert-base-uncased

echo "LAMP with L2 loss"
python3 attack.py --dataset $1 --split test --loss dlg --n_inputs 100 -b 1 --coeff_perplexity 60 --coeff_reg 25 --lr 0.01 --lr_decay 0.89 --bert_path bert-base-uncased

echo "LAMP without perplexity"
python3 attack.py --dataset $1 --split test --loss cos --n_inputs 100 -b 1 --coeff_perplexity 0 --coeff_reg 1 --lr 0.01 --lr_decay 0.89 --bert_path bert-base-uncased

echo "LAMP without regularisation"
python3 attack.py --dataset $1 --split test --loss cos --n_inputs 100 -b 1 --coeff_perplexity 0.2 --coeff_reg 0 --lr 0.01 --lr_decay 0.89 --bert_path bert-base-uncased

echo "LAMP with discrete optimisation at end"
python3 attack.py --dataset $1 --split test --loss cos --n_inputs 100 -b 1 --coeff_perplexity 0.2 --coeff_reg 1 --lr 0.01 --lr_decay 0.89 --use_swaps_at_end --bert_path bert-base-uncased

echo "LAMP without discrete optimisation"
python3 attack.py --dataset $1 --split test --loss cos --n_inputs 100 -b 1 --coeff_perplexity 0.2 --coeff_reg 1 --lr 0.01 --lr_decay 0.89 --no-use_swaps --bert_path bert-base-uncased
