import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config


def embedding_from_weights(w):
    layer = nn.Embedding(w.size(0), w.size(1))
    layer.weight.data = w
    return layer


# https://github.com/facebookresearch/text-adversarial-attack/blob/main/src/utils.py
def load_gpt2_from_dict(dict_path, device, output_hidden_states=False):
    state_dict = torch.load(dict_path, map_location=device)['model']

    config = GPT2Config(
        vocab_size=30522,
        n_embd=1024,
        n_head=8,
        activation_function='relu',
        n_layer=24,
        output_hidden_states=output_hidden_states
    )
    model = GPT2LMHeadModel(config)
    model.load_state_dict(state_dict)
    # The input embedding is not loaded automatically
    model.set_input_embeddings(embedding_from_weights(state_dict['transformer.wte.weight'].cpu()))

    return model
