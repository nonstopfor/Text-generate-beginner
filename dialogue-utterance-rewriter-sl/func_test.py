import torch

a=torch.ones([7,5,3])
b=torch.ones([3])
c=torch.matmul(a,b)
print(c)
print(c.size())