import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# ── Load Cleveland data ──
df = pd.read_csv("../data/cleveland_clean.csv")
X = df.drop(columns=['target']).values
y = df['target'].values

# ── Train/Test split ──
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# ── Define DNN Model ──
class HeartDiseaseDNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = HeartDiseaseDNN(input_dim=X_train.shape[1])

# ── Train ──
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training started...")
for epoch in range(100):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/100, Loss: {loss.item():.4f}")

# ── Evaluate ──
model.eval()
with torch.no_grad():
    preds = model(X_test)
    preds_binary = (preds > 0.5).float()
    accuracy = (preds_binary == y_test).float().mean()

print(f"\nTest Accuracy: {accuracy.item()*100:.2f}%")