import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import flwr as fl
from model import HeartDiseaseDNN

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['target']).values
    y = df['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1),
    )

class HeartClient(fl.client.NumPyClient):
    def __init__(self, csv_path, mu=0.01):
        self.model = HeartDiseaseDNN(input_dim=10)
        self.X_train, self.y_train, self.X_test, self.y_test = load_data(csv_path)
        self.criterion = nn.BCELoss()
        self.mu = mu  # FedProx proximal term strength

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # Save a copy of the GLOBAL model parameters (before local training)
        global_params = [torch.tensor(p.copy()) for p in parameters]

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.model.train()
        for epoch in range(5):  # 5 local epochs
            optimizer.zero_grad()
            output = self.model(self.X_train)
            loss = self.criterion(output, self.y_train)

            # ── FedProx Proximal Term ──
            proximal_term = 0.0
            for local_param, global_param in zip(self.model.parameters(), global_params):
                proximal_term += (local_param - global_param).norm(2) ** 2

            loss = loss + (self.mu / 2) * proximal_term
            # ────────────────────────────

            loss.backward()
            optimizer.step()

        return self.get_parameters(config={}), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.X_test)
            loss = self.criterion(output, self.y_test).item()
            preds = (output > 0.5).float()
            accuracy = (preds == self.y_test).float().mean().item()
        return loss, len(self.X_test), {"accuracy": accuracy}