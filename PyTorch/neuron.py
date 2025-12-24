"""A simple example to convert temperature from Celsius to Fahrenheit using a single neuron."""

import torch
from torch import nn

X = torch.tensor([
    [10.0],
    [20.0],
    [30.0],
    [40.0]
])

model = nn.Linear(1, 1)

model.bias = nn.Parameter(
    torch.tensor([32.0])
)

model.weight = nn.Parameter(
    torch.tensor([[1.8]])
)

print(model.bias)
print(model.weight)

y_pred = model(X)
print(y_pred)