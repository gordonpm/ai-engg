import torch
from torch import nn
from torch import float32
from torch import tensor
from torch import optim

X = tensor([
    [10.0],
    [37.78]], dtype=float32)

y = tensor([
    [50.0],
    [100.0]], dtype=float32)

model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001)

# Training
for i in range(0, 150000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()

    if i % 1000 == 0:
        print("weight: ", model.weight)
        print("bias: ", model.bias)

# Inference
measurements = tensor([
    [37.5]
], dtype=float32)        

model.eval()
with torch.no_grad():
    predictions = model(measurements)
    print("predictions: ", predictions) 