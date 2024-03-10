import torch
from src.faberGPT import FaberGPT
from config import Config
import data_eng as deng

cfg = Config()
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
        losses = torch.zeros(cfg.eval_iters)

        # Perform several iterations to estimate the loss, defined by eval_iters.
        for k in range(cfg.eval_iters):
            # Get a batch of data for the current split.
            X, Y = deng.get_batch(split)

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


model = FaberGPT(cfg.vocab_size)
m = model.to(cfg.device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

# Ciclo di training
for iter in range(cfg.max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % cfg.eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    # sample a batch of data
    xb, yb = deng.get_batch('train')

    logits, loss = model(xb, yb)

    # Azzera i gradienti dell'ottimizzatore per evitare accumuli indesiderati durante le iterazioni successive.
    # `set_to_none=True` può offrire leggeri miglioramenti alle prestazioni rispetto all'impostazione predefinita.
    optimizer.zero_grad(set_to_none=True)

    # Esegue la backpropagation a partire dalla perdita calcolata.
    loss.backward()

    # Aggiorna i pesi del modello basandosi sui gradienti calcolati durante la backpropagation.
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)
print(deng.decode(m.generate(context, max_new_tokens=500)[0].tolist()))

# Save the model's state dictionary
model_state_dict_path = "../model/faberGPT.pth"
torch.save(model.state_dict(), model_state_dict_path)

# It's often useful to save the optimizer's state as well, especially if you plan to resume training later.
# This includes information about the optimizer's current state and hyperparameters.
# optimizer_state_dict_path = "optimizer_state_dict.pth"
# torch.save(optimizer.state_dict(), optimizer_state_dict_path)
