import torch
from torch import nn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

n_samples = 1000
X, y = make_circles(n_samples,
                    noise=0.03, # a little bit of noise to the dots
                    random_state=42) # keep random state so we get the same values

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, # 20% test, 80% train
                                                    random_state=42) # make the random split reproducible

print(f"First 5 X features:\n{X[:5]}")
print(f"\nFirst 5 y labels:\n{y[:5]}")
print(f"X_train: {len(X_train)}, X_test: {len(X_test)}, y_train: {len(y_train)}, y_test: {len(y_test)}")

print(f"X shape: {X.shape}, y shape: {y.shape}")

device = "cuda" if torch.cuda.is_available() else "cpu"
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)

print(model_0)
print(model_0.state_dict())

with torch.inference_mode():
    untrained_preds = model_0(X_test.to(device))
print(f"untrained preds: {len(untrained_preds)}")
print(f"shape: {untrained_preds.shape}")
print(f"Length of test samples: {len(y_test)}, Shape: {y_test.shape}")
print(f"\nFirst 10 predictions:\n{untrained_preds[:10]}")
print(f"\nFirst 10 test labels:\n{y_test[:10]}")

y_pred_probs = torch.sigmoid(untrained_preds)
print(f"\nFirst 10 prediction probabilities:\n{y_pred_probs[:10]}")