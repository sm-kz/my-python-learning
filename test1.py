import torch
import pandas as pd

x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([1, 2])
print(torch.mv(x, y))