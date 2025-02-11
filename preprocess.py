# imports
import torch

#%% get words

def read_file(filepath):
    with open(filepath, 'r') as f:
        text = f.read()
    return text

text = read_file('input.txt')

#%%
def extract_unique_characters(text):
    characters = sorted(set(text))
    return characters

chars = extract_unique_characters(text)
vocab_size = len(chars)

def create_character_mapping(characters):
    stoi = {s:i for i,s in enumerate(characters)}
    itos = {i:s for s,i in stoi.items()}
    return stoi, itos

stoi, itos = create_character_mapping(chars)
encode = lambda x: [stoi[c] for c in x]
decode = lambda x: ''.join([itos[i] for i in x]) 
#%% get train and validation set from the dataset
def split_dataset(text,train_size=0.8):
    data_tensor = torch.tensor(encode(text))
    split = int(data_tensor.shape[0]*(train_size))
    return data_tensor[:split], data_tensor[split:]
 
# %% 
@torch.no_grad()
def split_loss(X,y,state_dict,hyperparams):
    """
    The state_dict dictionary stores neural network parameters: 
    # 'C' for the word embeddings, 'w1', 'b1' for the first layer's weights and bias,
    # and 'w2', 'b2' for the second layer's weights and bias, all initialized randomly.
    """
    # forward pass
    z = state_dict['C'][X].view(-1, hyperparams['embedding_size']*hyperparams['block_size'])
    hpreact = z@state_dict['w1']
    hpreact = (hpreact - hpreact.mean(0,keepdim=True)) / hpreact.std(0,keepdim=True) # modify this to somehow include the running mean instead of the mean of the batch
    hpreact = hpreact*state_dict['bngain'] + state_dict['bnbias']
    logits = hpreact@state_dict['w2'] + state_dict['b2']
    loss = torch.nn.CrossEntropyLoss()(logits,y)
    return loss