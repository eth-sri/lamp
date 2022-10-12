import torch
from utilities import get_reconstruction_loss, get_closest_tokens, fix_special_tokens


def get_init(args, model, unused_tokens, shape, true_labels, true_grads, bert_embeddings, bert_embeddings_weight, tokenizer, lm, lm_tokenizer, ids, pads):
    device = lm.device
    num_inits = shape[0]
    
    # Generate candidates from language model / random
    if args.init == 'lm':
        sentence = 'the'
        input_ids = lm_tokenizer.encode(sentence, return_tensors='pt').to(device)[:,1:-1]
        init_len = 10
        gen_outs = lm.generate(
            input_ids,
            no_repeat_ngram_size=2,
            num_return_sequences= args.init_candidates*num_inits, 
            do_sample=True,
            max_length= shape[1] + init_len,
        )
        gen_outs = gen_outs[:, init_len:]
        all_candidates = lm_tokenizer.batch_decode(gen_outs)
        embeds = tokenizer(all_candidates, padding=True, truncation=True, return_tensors='pt')['input_ids'].to(device)
        embeds = bert_embeddings(embeds)[:, :shape[1], :]
    elif args.init == 'random':
        new_shape = [args.init_candidates*num_inits] + list(shape[1:])
        embeds = torch.randn(new_shape).to(device)

    # Pick candidates based on rec loss
    best_x_embeds, best_rec_loss = None, None
    for i in range(args.init_candidates):
        tmp_embeds = embeds[i*num_inits:(i+1)*num_inits]
        fix_special_tokens(tmp_embeds, bert_embeddings.weight, pads)
        
        rec_loss = get_reconstruction_loss(model, tmp_embeds, true_labels, true_grads, args)
        if (best_rec_loss is None) or (rec_loss < best_rec_loss):
            best_rec_loss = rec_loss
            best_x_embeds = tmp_embeds
            _, cos_ids = get_closest_tokens(tmp_embeds, unused_tokens, bert_embeddings_weight, metric='cos')
            sen = tokenizer.batch_decode(cos_ids)
            print(f'[Init] best rec loss: {best_rec_loss.item()} for {sen}', flush=True)
    
    # Pick best permutation of candidates
    for i in range(args.init_candidates):
        idx = torch.cat((torch.tensor([0], dtype=torch.int32), torch.randperm(shape[1]-2)+1, torch.tensor([shape[1]-1], dtype=torch.int32) ))
        tmp_embeds = best_x_embeds[:, idx].detach()
        rec_loss = get_reconstruction_loss(model, tmp_embeds, true_labels, true_grads, args)
        if (rec_loss < best_rec_loss):
            best_rec_loss = rec_loss
            best_x_embeds = tmp_embeds
            _, cos_ids = get_closest_tokens(tmp_embeds, unused_tokens, bert_embeddings_weight, metric='cos')
            sen = tokenizer.batch_decode(cos_ids)
            print(f'[Init] best perm rec loss: {best_rec_loss.item()} for {sen}', flush=True)
    
    # Scale inital embeddings to args.init_size (e.g., avg of BERT embeddings ~1.4)
    if args.init_size >= 0:
        best_x_embeds /= best_x_embeds.norm(dim=2,keepdim=True)
        best_x_embeds *= args.init_size

    x_embeds = best_x_embeds.detach().clone()
    x_embeds = x_embeds.requires_grad_(True)
    
    return x_embeds
