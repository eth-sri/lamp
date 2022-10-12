import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AdamW, get_scheduler
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import matthews_corrcoef

np.random.seed(100)
torch.manual_seed(100)
device = 'cuda'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cola', 'sst2', 'rotten_tomatoes'], default='cola')
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--noise', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=1)
    args = parser.parse_args()

    seq_key = 'text' if args.dataset == 'rotten_tomatoes' else 'sentence'
    num_labels = 2
    
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels).to(device)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    tokenizer.model_max_length = 512
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if args.dataset == 'cola':
        metric = load_metric('matthews_correlation')
        train_metric = load_metric('matthews_correlation')
    else:
        metric = load_metric('accuracy')
        train_metric = load_metric('accuracy')

    def tokenize_function(examples):
        return tokenizer(examples[seq_key], truncation=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    if args.dataset in ['cola', 'sst2', 'rte']:
        datasets = load_dataset('glue', args.dataset)
    else:
        datasets = load_dataset(args.dataset)
    
    tokenized_datasets = datasets.map(tokenize_function, batched=True)
    if args.dataset == 'cola' or args.dataset == 'sst2':
        tokenized_datasets = tokenized_datasets.remove_columns(['idx', 'sentence'])
    elif args.dataset == 'rotten_tomatoes':
        tokenized_datasets = tokenized_datasets.remove_columns(['text'])
    else:
        assert False
    tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
    tokenized_datasets.set_format('torch')

    train_dataset = tokenized_datasets['train']
    eval_dataset = tokenized_datasets['validation']
    
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator)
    eval_loader = DataLoader(eval_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator)

    opt = AdamW(model.parameters(), lr=5e-5)

    num_training_steps = args.num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        'linear',
        optimizer=opt,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))
    model.train()
    n_steps = 0
    train_loss = 0
    
    for epoch in range(args.num_epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            train_metric.add_batch(predictions=predictions, references=batch['labels'])
            
            loss = outputs.loss
            train_loss += loss.item()
            loss.backward()

            if args.noise is not None:
                for param in model.parameters():
                    param.grad.data = param.grad.data + torch.randn(param.grad.shape).to(device) * args.noise

            opt.step()
            lr_scheduler.step()
            opt.zero_grad()
            progress_bar.update(1)

            n_steps += 1
            if n_steps % args.save_every == 0:
                model.save_pretrained(f'finetune/{args.dataset}/noise_{args.noise}/{n_steps}')
                print('metric train: ', train_metric.compute())
                print('loss train: ', train_loss/n_steps)
                train_loss = 0.0

        model.eval()
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            metric.add_batch(predictions=predictions, references=batch['labels'])
        with open(f'finetune/{args.dataset}/noise_{args.noise}/metric.txt', 'w') as fou:
            print('metric eval: ', metric.compute(), file=fou)
    model.save_pretrained(f'finetune/{args.dataset}/noise_{args.noise}/{n_steps}')
    

if __name__ == '__main__':
    main()
