import torch
import sys
import os
import numpy as np

sys.path.append("./model.py")
from model import Lstm, Decoder, Attention

# a = torch.ones([7, 5, 3])
# b = torch.ones([3])
# c = torch.matmul(a, b)
# print(c)
# print(c.size())
# print(a.size(2))
input = np.arange(10)
input = torch.tensor(input)
input.unsqueeze(0)
a = Lstm(10, 10, 10, 10, 0.5)
a.forward(input)
