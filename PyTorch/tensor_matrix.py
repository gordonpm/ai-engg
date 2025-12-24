import torch
X1 = torch.tensor([
    [10, 1], 
    [20, 2], 
    [30, 3], 
    [40, 1]
])

print(X1.shape)
print(X1[2])
print(X1[0][0].item())
