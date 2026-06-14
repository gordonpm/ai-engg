import torch
from torch import nn

class CircleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
            return self.layer_2(self.layer_1(x))

"""" 
install cuda enabled pytorch using: 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
""" 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device is: {device}")
print(f"cuda version: {torch.version.cuda}")
model_0 = CircleModel().to(device)
print(model_0)
print(next(model_0.parameters()).device)