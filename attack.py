import sys, argparse
import os
import datetime
import itertools
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_metric
from nlp_utils import load_gpt2_from_dict
from transformers import AdamW, AutoConfig, AutoModel, AutoTokenizer, AutoModelForSequenceClassification, LogitsProcessor, BeamSearchScorer
from init import get_init
from constants import BERT_CLS_TOKEN, BERT_SEP_TOKEN, BERT_PAD_TOKEN
from utilities import compute_grads, get_closest_tokens, get_reconstruction_loss, get_perplexity, fix_special_tokens, remove_padding
from data_utils import TextDataset
from args_factory import get_args
import time

from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from scipy.optimize import linear_sum_assignment

args = get_args()
np.random.seed(args.rng_seed)
torch.manual_seed(args.rng_seed)

if args.neptune:
    import neptune
    neptune.init(api_token=os.getenv('NEPTUNE_API_KEY'), project_qualified_name=args.neptune)
    neptune.create_experiment(args.neptune_label, params=vars(args))

def get_loss(args, lm, model, ids, x_embeds, true_labels, true_grads, create_graph=False):
    perplexity = lm(input_ids=ids, labels=ids).loss
    rec_loss = get_reconstruction_loss(model, x_embeds, true_labels, true_grads, args, create_graph=create_graph)
    return perplexity, rec_loss, rec_loss + args.coeff_perplexity * perplexity

def swap_tokens(args, x_embeds, max_len, cos_ids, lm, model, true_labels, true_grads ):
    print('Attempt swap', flush=True)
    best_x_embeds, best_tot_loss = None, None
    changed = None
    for sen_id in range(x_embeds.data.shape[0]):
        for sample_idx in range(200):
            perm_ids = np.arange(x_embeds.shape[1])
            
            if sample_idx != 0:
                if sample_idx % 4 == 0: # swap two tokens
                    i, j = 1 + np.random.randint(max_len[sen_id] - 2), 1 + np.random.randint(max_len[sen_id] - 2)
                    perm_ids[i], perm_ids[j] = perm_ids[j], perm_ids[i]
                elif sample_idx % 4 == 1: # move a token to another place
                    i = 1 + np.random.randint(max_len[sen_id] - 2)
                    j = 1 + np.random.randint(max_len[sen_id] - 1)
                    if i < j:
                        perm_ids = np.concatenate([perm_ids[:i], perm_ids[i+1:j], perm_ids[i:i+1], perm_ids[j:]])
                    else:
                        perm_ids = np.concatenate([perm_ids[:j], perm_ids[i:i+1], perm_ids[j:i], perm_ids[i+1:]])
                elif sample_idx % 4 == 2: # move a sequence to another place
                    b = 1 + np.random.randint(max_len[sen_id] - 1)
                    e = 1 + np.random.randint(max_len[sen_id] - 1)
                    if b > e:
                        b, e = e, b
                    p = 1 + np.random.randint(max_len[sen_id] - 1 - (e-b))
                    if p >= b:
                        p += e-b
                    if p < b:
                        perm_ids = np.concatenate([perm_ids[:p], perm_ids[b:e], perm_ids[p:b], perm_ids[e:]])
                    elif p >= e:
                        perm_ids = np.concatenate([perm_ids[:b], perm_ids[e:p], perm_ids[b:e], perm_ids[p:]])
                    else:
                        assert False
                elif sample_idx % 4 == 3: # take some prefix and put it at the end
                    i = 1 + np.random.randint(max_len[sen_id] - 2)
                    perm_ids = np.concatenate([perm_ids[:1], perm_ids[i:-1], perm_ids[1:i], perm_ids[-1:]])
                
            new_ids = cos_ids.clone()
            new_ids[sen_id] = cos_ids[sen_id, perm_ids]
            new_x_embeds = x_embeds.clone()
            new_x_embeds[sen_id] = x_embeds[sen_id, perm_ids, :]
            
            _, _, new_tot_loss = get_loss(args, lm, model, new_ids, new_x_embeds, true_labels, true_grads)

            if (best_tot_loss is None) or (new_tot_loss < best_tot_loss):
                best_x_embeds = new_x_embeds
                best_tot_loss = new_tot_loss
                if sample_idx != 0:
                    changed = sample_idx % 4
        if not( changed is None ):
            change = ['Swapped tokens', 'Moved token', 'Moved sequence', 'Put prefix at the end'][changed]
            print( change , flush=True)
        x_embeds.data = best_x_embeds
        
def reconstruct(args, device, sample, metric, tokenizer, lm, model):
    sequences, true_labels = sample
    
    lm_tokenizer = tokenizer

    gpt2_embeddings = lm.get_input_embeddings()
    gpt2_embeddings_weight = gpt2_embeddings.weight.unsqueeze(0)

    bert_embeddings = model.get_input_embeddings()
    bert_embeddings_weight = bert_embeddings.weight.unsqueeze(0)

    orig_batch = tokenizer(sequences, padding=True, truncation=True, return_tensors='pt').to(device)
    true_embeds = bert_embeddings(orig_batch['input_ids'])
    true_grads = compute_grads(model, true_embeds, true_labels)

    if args.defense_pct_mask is not None:
        for grad in true_grads:
            grad.data = grad.data * (torch.rand(grad.shape).to(device) > args.defense_pct_mask).float()
    if args.defense_noise is not None:
        for grad in true_grads:
            grad.data = grad.data + torch.randn(grad.shape).to(device) * args.defense_noise

    # BERT special tokens (0-999) are never part of the sentence
    unused_tokens = []
    if args.use_embedding:
        for i in range(tokenizer.vocab_size):
            if true_grads[0][i].abs().sum() < 1e-9 and i != BERT_PAD_TOKEN:
                unused_tokens += [i]
    else:
        unused_tokens += list(range(1, 100))
        unused_tokens += list(range(104, 999))
    unused_tokens = np.array(unused_tokens)

    # If length of sentences is known to attacker keep padding fixed
    pads = None
    if args.know_padding:
        pads = [orig_batch['input_ids'].shape[1]]*orig_batch['input_ids'].shape[0]
        for sen_id in range(orig_batch['input_ids'].shape[0]):
            for i in range(orig_batch['input_ids'].shape[1]-1, 0, -1):
                if orig_batch['input_ids'][sen_id][i] == BERT_PAD_TOKEN:
                    pads[sen_id] = i
                else:
                    break
    print(f'Debug: ids_shape = {orig_batch["input_ids"].shape[1]}, pads = {pads}', flush=True)
    print(f'Debug: input ids = {orig_batch["input_ids"]}', flush=True)
    print(f'Debug: ref = {tokenizer.batch_decode(orig_batch["input_ids"])}', flush=True)

    # Get initial embeddings + set up opt
    x_embeds = get_init(args, model, unused_tokens, true_embeds.shape, true_labels, true_grads, bert_embeddings,  bert_embeddings_weight, tokenizer, lm, lm_tokenizer, orig_batch['input_ids'], pads)

    bert_embeddings_weight = bert_embeddings.weight.unsqueeze(0)
    if args.opt_alg == 'adam':
        opt = optim.Adam([x_embeds], lr=args.lr)
    elif args.opt_alg == 'bfgs':
        opt = optim.LBFGS([x_embeds], lr=args.lr)
    elif args.opt_alg == 'bert-adam':
        opt = torch.optim.AdamW([x_embeds], lr=args.lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)

    if args.lr_decay_type == 'StepLR':
        lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=50, gamma=args.lr_decay)
    elif args.lr_decay_type == 'LambdaLR':
        def lr_lambda(current_step: int):
            return max(0.0, float(args.lr_max_it - current_step) / float(max(1, args.lr_max_it)))
        lr_scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    print('Nsteps:',args.n_steps, flush=True)

    if pads is None:
        max_len = [x_embeds.shape[1]]*x_embeds.shape[0]
    else:
        max_len = pads
    
    # Main loop
    best_final_error, best_final_x = None, x_embeds.detach().clone()
    for it in range(args.n_steps):
        t_start = time.time()

        def closure():
            opt.zero_grad()
            rec_loss = get_reconstruction_loss(model, x_embeds, true_labels, true_grads, args, create_graph=True)
            reg_loss = ( x_embeds.norm(p=2,dim=2).mean() - args.init_size ).square() 
            tot_loss = rec_loss + args.coeff_reg * reg_loss
            tot_loss.backward(retain_graph=True)
            with torch.no_grad():
                if args.grad_clip is not None:
                    grad_norm = x_embeds.grad.norm()
                    if grad_norm > args.grad_clip:
                        x_embeds.grad.mul_(args.grad_clip / (grad_norm + 1e-6))
            return tot_loss

        error = opt.step(closure)
        if best_final_error is None or error <= best_final_error:
            best_final_error = error.item()
            best_final_x.data[:] = x_embeds.data[:]
        del error

        lr_scheduler.step()

        fix_special_tokens(x_embeds, bert_embeddings.weight, pads)
        
        _, cos_ids = get_closest_tokens(x_embeds, unused_tokens, bert_embeddings_weight)
        
        # Trying swaps
        if args.use_swaps and it >= args.swap_burnin * args.n_steps and it % args.swap_every == 1:
            swap_tokens(args, x_embeds, max_len, cos_ids, lm, model, true_labels, true_grads)

        steps_done = it+1
        if steps_done % args.print_every == 0:
            _, cos_ids = get_closest_tokens(x_embeds, unused_tokens, bert_embeddings_weight)
            x_embeds_proj = bert_embeddings(cos_ids) * x_embeds.norm(dim=2, p=2, keepdim=True) / bert_embeddings(cos_ids).norm(dim=2, p=2, keepdim=True)
            _, _, tot_loss_proj = get_loss(args, lm, model, cos_ids, x_embeds_proj, true_labels, true_grads)
            perplexity, rec_loss, tot_loss = get_loss(args, lm, model, cos_ids, x_embeds, true_labels, true_grads)

            step_time = time.time() - t_start
            
            print('[%4d/%4d] tot_loss=%.3f (perp=%.3f, rec=%.3f), tot_loss_proj:%.3f [t=%.2fs]' % (
                steps_done, args.n_steps, tot_loss.item(), perplexity.item(), rec_loss.item(), tot_loss_proj.item(), step_time), flush=True)
            print('prediction: %s'% (tokenizer.batch_decode(cos_ids)), flush=True)
            
            tokenizer.batch_decode(cos_ids)

    
    # Swaps in the end for ablation
    if args.use_swaps_at_end:
        swap_at_end_it = int( (1 - args.swap_burnin) * args.n_steps // args.swap_every )
        print('Trying %i swaps' % swap_at_end_it, flush=True )
        for i in range( swap_at_end_it ):
            swap_tokens(args, x_embeds, max_len, cos_ids, lm, model, true_labels, true_grads)
        
    # Postprocess
    x_embeds.data = best_final_x
    fix_special_tokens(x_embeds, bert_embeddings.weight, pads)
    m = 5
    d, cos_ids = get_closest_tokens(x_embeds, unused_tokens, bert_embeddings_weight, metric='cos')
    x_embeds_proj = bert_embeddings(cos_ids) * x_embeds.norm(dim=2, p=2, keepdim=True) / bert_embeddings(cos_ids).norm(dim=2, p=2, keepdim=True)
    _, _, best_tot_loss = get_loss(args, lm, model, cos_ids, x_embeds_proj, true_labels, true_grads)
    best_ids = cos_ids
    best_x_embeds_proj = x_embeds_proj
    
    prediction, reference = [], []
    for i in range(best_ids.shape[0]):
        prediction += [remove_padding(tokenizer, best_ids[i])]
        reference += [remove_padding(tokenizer, orig_batch['input_ids'][i])]
    
    # Matching
    cost = np.zeros((x_embeds.shape[0], x_embeds.shape[0]))
    for i in range(x_embeds.shape[0]):
        for j in range(x_embeds.shape[0]):
            fm = metric.compute(predictions=[prediction[i]], references=[reference[j]])['rouge1'].mid.fmeasure
            cost[i, j] = 1.0 - fm
    row_ind, col_ind = linear_sum_assignment(cost)

    ids = list(range(x_embeds.shape[0]))
    ids.sort(key=lambda i: col_ind[i])
    new_prediction = []
    for i in range(x_embeds.shape[0]):
        new_prediction += [prediction[ids[i]]]
    prediction = new_prediction

    return prediction, reference

def print_metrics(res, suffix, use_neptune):
    for metric in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']:
        curr = res[metric].mid
        print(f'{metric:10} | fm: {curr.fmeasure*100:.3f} | p: {curr.precision*100:.3f} | r: {curr.recall*100:.3f}', flush=True)
        if use_neptune:
            neptune.log_metric(f'{metric}-fm_{suffix}', curr.fmeasure*100)
            neptune.log_metric(f'{metric}-p_{suffix}', curr.precision*100)
            neptune.log_metric(f'{metric}-r_{suffix}', curr.recall*100)
    sum_12_fm = res['rouge1'].mid.fmeasure + res['rouge2'].mid.fmeasure
    if use_neptune:
        neptune.log_metric(f'r1fm+r2fm_{suffix}', sum_12_fm*100)
    print(f'r1fm+r2fm = {sum_12_fm*100:.3f}\n', flush=True)

def main():
    print( '\n\n\nCommand:', ' '.join( sys.argv ), '\n\n\n', flush=True)

    device = torch.device(args.device)
    metric = load_metric('rouge')
    dataset = TextDataset(args.device, args.dataset, args.split, args.n_inputs, args.batch_size)

    lm = load_gpt2_from_dict("transformer_wikitext-103.pth", device, output_hidden_states=True).to(device)
    lm.eval()

    model = AutoModelForSequenceClassification.from_pretrained(args.bert_path).to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    tokenizer.model_max_length = 512

    print('\n\nAttacking..\n', flush=True)
    predictions, references = [], []
    t_start = time.time()
    for i in range(args.n_inputs):
        t_input_start = time.time()
        sample = dataset[i] # (seqs, labels)

        print(f'Running input #{i} of {args.n_inputs}.')
        if args.neptune:
            neptune.log_metric('curr_input', i)

        print('reference: ')
        for seq in sample[0]:
            print('========================')
            print(seq)

        print('========================', flush=True)

        prediction, reference = reconstruct(args, device, sample, metric, tokenizer, lm, model)
        predictions += prediction
        references += reference

        print(f'Done with input #{i} of {args.n_inputs}.')
        print('reference: ')
        for seq in reference:
            print('========================')
            print(seq)
        print('========================')

        print('predicted: ')
        for seq in prediction:
            print('========================')
            print(seq)
        print('========================', flush=True)

        print('[Curr input metrics]:')
        res = metric.compute(predictions=prediction, references=reference)
        print_metrics(res, suffix='curr', use_neptune=args.neptune is not None)

        print('[Aggregate metrics]:')
        res = metric.compute(predictions=predictions, references=references)
        print_metrics(res, suffix='agg', use_neptune=args.neptune is not None)

        input_time = str(datetime.timedelta(seconds=time.time() - t_input_start)).split(".")[0]
        total_time = str(datetime.timedelta(seconds=time.time() - t_start)).split(".")[0]
        print(f'input #{i} time: {input_time} | total time: {total_time}\n\n', flush=True)

    print('Done with all.', flush=True)
    if args.neptune:
        neptune.log_metric('curr_input', args.n_inputs)

if __name__ == '__main__':
    main()
