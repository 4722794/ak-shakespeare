#! python3
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocess import read_file, extract_unique_characters, create_character_mapping, split_dataset,encode,decode
from pathlib import Path
#%%
# get words
text = read_file('input.txt')
chars = extract_unique_characters(text)
stoi,itos = create_character_mapping(chars)
train_data, val_data = split_dataset(text,train_size=0.9)

# %% hyper parameters

block_size = 8
batch_size = 4
num_heads = 6
hidden_dim = 128
vocab_size = len(stoi)
# %% get batch

def get_batch(data,block_size,batch_size):
    n = len(data)
    # get random index
    idx = torch.randint(0,n-block_size,(batch_size,))
    # get batch
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    return x,y
 #%%
xb,yb = get_batch(train_data,block_size,batch_size)

#%% Simple bigram language model

W = torch.randn(len(stoi),len(stoi),requires_grad=True)

# training loop

for k in range(100000):
    xb,yb = get_batch(train_data,block_size,batch_size)
    xenc = F.one_hot(xb,num_classes=vocab_size).float()
    logits = xenc @ W
    loss = F.cross_entropy(logits.view(-1,vocab_size),yb.view(-1))
    
    # update step
    W.grad = None
    loss.backward()
    with torch.no_grad():
        W -= 0.001*W.grad

    # print progress
    if k%100 == 0:
        print(f'iteration: {k} loss: {loss.item()}')
#%% Sampling

g = torch.Generator().manual_seed(2147483647)

for i in range(10):
    out = []
    ix = 0 
    while True:
        xenc = F.one_hot(torch.tensor([ix]),vocab_size).float()
        logits = xenc @ W
        p = F.softmax(logits,dim=1).squeeze()
        ix = torch.multinomial(p,num_samples=1,replacement=True,generator=g).item()
        out.append(itos[ix])
        if ix==0:
            break

    print(''.join(out))

#%%

### Rough

for b in range(batch_size):     
    for t in range(block_size):
        context = xb[b,:t+1]
        target = yb[b,t]
        print(f'context: {context} target: {target}')
#%%

x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f'context: {context} target: {target}')
# %%
