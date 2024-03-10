import torch
from config import Config
from src.faberGPT import FaberGPT
import data_eng as deng

cfg = Config()
# Create the model and optimizer instances
model = FaberGPT(cfg.vocab_size).to(cfg.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

# Load the model's state dictionary
model.load_state_dict(torch.load("../model/faberGPT.pth"))
m = model.to(cfg.device)
# If you also saved the optimizer's state, load it as well
# optimizer.load_state_dict(torch.load(optimizer_state_dict_path))


context = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)
print(deng.decode(m.generate(context, max_new_tokens=500)[0].tolist()))
