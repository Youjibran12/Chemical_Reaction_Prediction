import torch
from torch.utils.data import Dataset

class ReactionDataset(Dataset):
    def __init__(self, reactions, tokenizer, max_len=128):
        self.reactions = reactions
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.reactions)

    def __getitem__(self, idx):
        rxn = self.reactions[idx]
        src = torch.tensor(self.tokenizer.encode(rxn['input'],  max_len=self.max_len))
        tgt = torch.tensor(self.tokenizer.encode(rxn['output'], max_len=self.max_len))
        return src, tgt
