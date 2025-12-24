import torch

b = 32
w1 = torch.tensor(1.8)
X1 = torch.tensor([10, 20, 30, 40, 50])
y_pred = 1*b + X1*w1 # if anyone is a tensor, the result is a tensor. here b is not defined as a tensor.
print(y_pred[0].item())