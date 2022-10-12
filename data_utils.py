from datasets import load_dataset
import numpy as np
import torch 

class TextDataset:
    def __init__(self, device, dataset, split, n_inputs, batch_size):
        
        seq_keys = {
            'cola': 'sentence',
            'sst2': 'sentence',
            'rotten_tomatoes': 'text'
        }
        seq_key = seq_keys[dataset]

        if dataset in ['cola', 'sst2']:
            full = load_dataset('glue', dataset)['train']
        else:
            full = load_dataset(dataset)['train']
            
        idxs = list(range(len(full)))
        np.random.shuffle(idxs)
        if dataset == 'cola':
            assert idxs[0] == 2310 # with seed 101

        n_samples = n_inputs * batch_size

        if split == 'test':
            assert n_samples <= 1000
            idxs = idxs[:n_samples]
        elif split == 'val':
            idxs = idxs[1000:] # first 1000 saved for testing
            assert len(idxs) >= n_samples

            zipped = [(idx, len(full[idx][seq_key])) for idx in idxs]
            zipped = sorted(zipped, key=lambda x: x[1])
            chunk_sz = len(zipped) // n_samples 
            idxs = []
            for i in range(n_samples):
                tmp = chunk_sz*i + np.random.randint(0, chunk_sz)
                idxs.append(zipped[tmp][0])
            np.random.shuffle(idxs)

        # Slice
        self.seqs = []
        self.labels = []
        for i in range(n_inputs):
            seqs = []
            for j in range(batch_size):
                seqs.append(full[idxs[i*batch_size+j]][seq_key])
            labels = torch.tensor([full[idxs[i*batch_size : (i+1)*batch_size]]['label']], device=device)
            self.seqs.append(seqs)
            self.labels.append(labels)
        assert len(self.seqs) == n_inputs
        assert len(self.labels) == n_inputs


    def __getitem__(self, idx):
        return (self.seqs[idx], self.labels[idx])
