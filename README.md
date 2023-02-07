# LAMP: Extracting Text from Gradients with <br/> Language Model Priors <a href="https://www.sri.inf.ethz.ch/"><img width="100" alt="portfolio_view" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg"></a>

The code accompanying our NeurIPS 2022 paper: [**LAMP: Extracting Text from Gradients with Language Model Priors**](https://openreview.net/forum?id=6iqd9JAVR1z).

For a brief overview, check out our **[blogpost](https://www.sri.inf.ethz.ch/blog/lamp)**.

## Prerequisites
- Install Anaconda. 
- Create the conda environment:<br>
> conda env create -f environment.yml
- Enable the created environment:<br>
> conda activate lamp
- Download required files:<br>
> wget -r -np -R "index.html*" https://files.sri.inf.ethz.ch/lamp/  
> mv files.sri.inf.ethz.ch/lamp/* ./    
> rm -rf files.sri.inf.ethz.ch

## Main experiments (Table 1)

### Parameters
- *DATASET* - the dataset to use. Must be one of **cola**, **sst2**, **rotten_tomatoes**.
- *BERT\_PATH* - the language model to attack. Must be one of **bert-base-uncased**, **huawei-noah/TinyBERT_General_6L_768D**, **models/bert-base-finetuned-cola**, **models/bert-base-finetuned-sst2**, **models/bert-base-finetuned-rottentomatoes** for BERT<sub>BASE</sub>, TinyBERT<sub>6</sub>, and each of the three fine-tuned BERT<sub>BASE</sub>-FT models (on each of the datasets).

### Commands
- To run the experiment on LAMP with cosine loss:<br>
> ./lamp_cos.sh BERT\_PATH DATASET 1
- To run the experiment on LAMP with cosine loss on BERT<sub>LARGE</sub>:<br>
> ./lamp_cos_large.sh DATASET 1
- To run the experiment on LAMP with L1+L2 loss:<br>
> ./lamp_l1l2.sh BERT\_PATH DATASET 1
- To run the experiment on LAMP with L1+L2 loss on BERT<sub>LARGE</sub>:<br>
> ./lamp_l1l2_large.sh DATASET 1
- To run the experiment on TAG:<br>
> ./tag.sh BERT\_PATH DATASET 1
- To run the experiment on TAG on BERT<sub>LARGE</sub>:<br>
> ./tag_large.sh DATASET 1
- To run the experiment on DLG:<br>
> ./dlg.sh BERT\_PATH DATASET 1
- To run the experiment on DLG on BERT<sub>LARGE</sub>:<br>
> ./dlg_large.sh DATASET 1

## Batch size experiments (Table 2)

### Parameters
- *DATASET* - the dataset to use. Must be one of **cola**, **sst2**, **rotten_tomatoes**.
- *BATCH\_SIZE* - the batch size to use e.g **4**.

### Commands
- To run the experiment on LAMP with cosine loss:<br>
> ./lamp_cos.sh bert-base-uncased DATASET BATCH\_SIZE
- To run the experiment on LAMP with L1+L2 loss:<br>
> ./lamp_l1l2.sh bert-base-uncased DATASET BATCH\_SIZE
- To run the experiment on TAG:<br>
> ./tag.sh bert-base-uncased DATASET BATCH\_SIZE
- To run the experiment on DLG:<br>
> ./dlg.sh bert-base-uncased DATASET BATCH\_SIZE

## Ablation study (Table 4)

### Parameters
- *DATASET* - the dataset to use. Must be one of **cola**, **sst2**, **rotten_tomatoes**.

### Commands
- To run the ablation experiments in Table 4:<br>
> ./ablation.sh DATASET

## Gaussian noise defense (Table 5)

### Parameters
- *SIGMA* - the amount of Gaussian noise with which to defend e.g **0.001**.

### Commands
- To run the experiment on LAMP with cosine loss:<br>
> ./lamp_cos.sh bert-base-uncased cola 1 --defense_noise SIGMA
- To run the experiment on LAMP with L1+L2 loss:<br>
> ./lamp_l1l2.sh bert-base-uncased cola 1 --defense_noise SIGMA
- To run the experiment on TAG:<br>
> ./tag.sh bert-base-uncased cola 1 --defense_noise SIGMA
- To run the experiment on DLG:<br>
> ./dlg.sh bert-base-uncased cola 1 --defense_noise SIGMA

## Zeroed-out gradient entries defense (Table 8)

### Parameters
- *ZEROED* - the ratio of zeroed out gradient entries e.g **0.75**.

### Commands
- To run the experiment on LAMP with cosine loss:<br>
> ./lamp_cos.sh bert-base-uncased cola 1 --defense_pct_mask ZEROED
- To run the experiment on LAMP with L1+L2 loss:<br>
> ./lamp_l1l2.sh bert-base-uncased cola 1 --defense_pct_mask ZEROED
- To run the experiment on TAG:<br>
> ./tag.sh bert-base-uncased cola 1 --defense_pct_mask ZEROED
- To run the experiment on DLG:<br>
> ./dlg.sh bert-base-uncased cola 1 --defense_pct_mask ZEROED



## Fine-tuning BERT with and without defended gradients
### Parameters
- *DATASET* - the dataset to use. Must be one of **cola**, **sst2**, **rotten_tomatoes**.
- *SIGMA* - the amount of Gaussian noise with which to train e.g **0.001**. To train without defense set to **0.0**.
- *NUM_EPOCHS* - for how many epochs to train e.g **2**.

### Commands

- To train your own network:<br>
> python3 train.py --dataset DATASET --batch_size 32 --noise SIGMA --num_epochs NUM_EPOCHS --save_every 100

The models are stored under `finetune/DATASET/noise_SIGMA/STEPS`

## Citation

```
@inproceedings{
    balunovic2022lamp,
    title={{LAMP}: Extracting Text from Gradients with Language Model Priors},
    author={Mislav Balunovic and Dimitar Iliev Dimitrov and Nikola Jovanovi{\'c} and Martin Vechev},
    booktitle={Advances in Neural Information Processing Systems},
    editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
    year={2022},
    url={https://openreview.net/forum?id=6iqd9JAVR1z}
}
```

