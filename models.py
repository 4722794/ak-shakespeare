import torch

import torch.nn as nn
import torch.nn.functional as F

#%% Simple bigram language model

class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.W = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        vocab_size = self.W.weight.size(0)
        logits = self.W(idx)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=-1)
        return idx
    
class TransformerModel(nn.Module):

    def __init__(self, vocab_size,n_embd,block_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # B,T --> B,T,E
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # B,T --> B,T,E
        self.lm_head = nn.Linear(n_embd, vocab_size) # B,T,E --> B,T,V
        
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size

    def forward(self, idx, targets=None):
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(idx.shape[1],device=idx.device))
        x = token_emb + pos_emb
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=-1)
        return idx
    

