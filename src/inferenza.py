import torch
from config import Config
from src.faberGPT import FaberGPT
import data_eng as deng

cfg = Config()
model = FaberGPT().to(cfg.device)

# Load the model's state dictionary
model.load_state_dict(torch.load("../model/faberGPT.pth"))
m = model.to(cfg.device)



context = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)
print(deng.decode(m.generate(context, max_new_tokens=500)[0].tolist()))


# optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
# If you also saved the optimizer's state, load it as well
# optimizer.load_state_dict(torch.load(optimizer_state_dict_path))