#%%
import torch

import torch.nn as nn
import torch.nn.functional as F

class MultiHead(nn.Module):
    def __init__(self,num_heads, block_size,embd_size):
        super().__init__()
        head_size = embd_size // num_heads
        self.heads = nn.ModuleList([Head(block_size,embd_size,head_size) for _ in range(num_heads)])

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        # projection
        return out


class Head(nn.Module):
    def __init__(self,block_size,embd_size,num_heads):
        super().__init__()
        # Initialize your layers here
        head_size = embd_size // num_heads
        self.key = nn.Linear(embd_size, head_size, bias=False)
        self.query = nn.Linear(embd_size, head_size, bias=False)
        self.value = nn.Linear(embd_size, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # Define the forward pass here
        B,T,E = x.size()
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        attn_scores = torch.matmul(q, k.transpose(-2,-1))* (E ** -0.5)
        masked_attn_scores = attn_scores.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        attn_probs = F.softmax(masked_attn_scores, dim=-1)
        out = torch.matmul(attn_probs, v)
        return out

class CasualAttentionHead(nn.Module):
    def __init__(self,block_size,embd_size,num_heads):
        super().__init__()
        # Initialize your layers here
        self.num_heads = num_heads
        self.key = nn.Linear(embd_size, embd_size, bias=False)
        self.query = nn.Linear(embd_size, embd_size, bias=False)
        self.value = nn.Linear(embd_size, embd_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # Define the forward pass here
        B,T,E = x.size()
        head_size = E // self.num_heads
        k = self.key(x).view(B,T,num_heads,-1).transpose(1,2)
        q = self.query(x).view(B,T,num_heads,-1).transpose(1,2)
        v = self.value(x).view(B,T,num_heads,-1).transpose(1,2)
        attn_scores = torch.matmul(q, k.transpose(-2,-1))* (head_size ** -0.5)
        masked_attn_scores = attn_scores.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        attn_probs = F.softmax(masked_attn_scores, dim=-1)
        out = torch.matmul(attn_probs, v)
        out = out.transpose(1,2).contiguous().view(B,T,E)
        return out
#%%
B = 1
T = block_size = 4
E = embd_size = 8
H = head_size = 2
num_heads = 4
x = torch.randn(B,T,E)
head = Head(block_size, embd_size, head_size)
out = head(x)
multihead = MultiHead(num_heads, block_size, embd_size)
# %%
