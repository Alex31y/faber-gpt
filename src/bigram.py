import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('../model/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
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
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Gather input data (x) by selecting blocks of data starting from the randomly chosen indices.
    x = torch.stack([data[i:i + block_size] for i in ix])

    # Gather target data (y) which is offset by one from the inputs, to predict the next item in the sequence.
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

    x, y = x.to(device), y.to(device)
    return x, y


# The decorator @torch.no_grad() is used to disable gradient calculation,
# which is not needed for loss estimation and can reduce memory consumption and speed up computations.
@torch.no_grad()
def estimate_loss():
    """
    Estimate the loss of the model on both training and validation datasets without updating the model's weights.

    Returns:
        dict: A dictionary containing the mean loss for both the training ('train') and validation ('val') datasets.
    """

    # Initialize a dictionary to store the mean loss for training and validation data.
    out = {}

    # Put the model in evaluation mode. This affects layers like dropout layers and batch normalization layers
    # which behave differently during training vs. evaluation.
    model.eval()

    # Iterate over both the training and validation splits to calculate the losses.
    for split in ['train', 'val']:
        # Initialize a tensor to store the losses for each evaluation iteration.
        # This tensor has a size equal to eval_iters and is filled with zeros.
        losses = torch.zeros(eval_iters)

        # Perform several iterations to estimate the loss, defined by eval_iters.
        for k in range(eval_iters):
            # Get a batch of data for the current split.
            X, Y = get_batch(split)

            # Pass the batch through the model to obtain the logits and loss for this batch.
            logits, loss = model(X, Y)

            # Store the loss of the current iteration.
            losses[k] = loss.item()  # .item() converts a one-element tensor to a Python scalar.

        # Calculate the mean loss for the current split and store it in the output dictionary.
        out[split] = losses.mean()

    # After evaluating the model, put it back in training mode.
    # This re-enables the normal behavior of layers like dropout and batch normalization for subsequent training.
    model.train()

    # Return the dictionary containing the mean losses for training and validation.
    return out


# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Ciclo di training
for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    # sample a batch of data
    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)

    # Azzera i gradienti dell'ottimizzatore per evitare accumuli indesiderati durante le iterazioni successive.
    # `set_to_none=True` può offrire leggeri miglioramenti alle prestazioni rispetto all'impostazione predefinita.
    optimizer.zero_grad(set_to_none=True)

    # Esegue la backpropagation a partire dalla perdita calcolata.
    loss.backward()

    # Aggiorna i pesi del modello basandosi sui gradienti calcolati durante la backpropagation.
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

# Save the model's state dictionary
model_state_dict_path = "bigram/bigramLM_state_dict.pth"
torch.save(model.state_dict(), model_state_dict_path)

# It's often useful to save the optimizer's state as well, especially if you plan to resume training later.
# This includes information about the optimizer's current state and hyperparameters.
# optimizer_state_dict_path = "optimizer_state_dict.pth"
# torch.save(optimizer.state_dict(), optimizer_state_dict_path)