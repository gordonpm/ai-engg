from torch import nn
from torch import float32
from torch import tensor
from torch import optim

X1 = tensor([[10.0]], dtype=float32)
y1 = tensor([[50.0]], dtype=float32)

X2 = tensor([[37.78]], dtype=float32)
y2 = tensor([[100.0]], dtype=float32)

model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001)

# Training
for i in range(0, 100000):
    optimizer.zero_grad()
    outputs = model(X1)
    loss = loss_fn(outputs, y1)
    loss.backward()
    optimizer.step()

    optimizer.zero_grad()
    outputs = model(X2)
    loss = loss_fn(outputs, y2)
    loss.backward()
    optimizer.step()

    if i % 1000 == 0:
        print("weight: ", model.weight)
        print("bias: ", model.bias)

y1_pred = model(X1)
print("y1_pred: ", y1_pred)
