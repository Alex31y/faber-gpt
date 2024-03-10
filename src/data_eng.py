import torch
from config import Config

cfg = Config()
with open('../model/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
cfg.set_vocab_size(vocab_size)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    # Randomly select indices for the batch, ensuring that a complete block of data can be formed for each index.
    ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))

    # Gather input data (x) by selecting blocks of data starting from the randomly chosen indices.
    x = torch.stack([data[i:i + cfg.block_size] for i in ix])

    # Gather target data (y) which is offset by one from the inputs, to predict the next item in the sequence.
    y = torch.stack([data[i + 1:i + cfg.block_size + 1] for i in ix])

    x, y = x.to(cfg.device), y.to(cfg.device)
    return x, y