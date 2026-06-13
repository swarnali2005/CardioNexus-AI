import pandas as pd
import torch
import shap
import numpy as np
from model import HeartDiseaseDNN

feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak']

# ── Load the trained global model ──
model = HeartDiseaseDNN(input_dim=10)
model.load_state_dict(torch.load("global_model.pt"))
model.eval()

def predict_fn(x):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        return model(x_tensor).numpy()

def compute_local_shap(csv_path, hospital_name):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['target']).values

    # Use 50 background samples (as in your VitaFed AI work)
    background = shap.sample(X, min(50, len(X)))
    explainer = shap.KernelExplainer(predict_fn, background)

    # Compute SHAP values on a subset for speed
    sample = shap.sample(X, min(30, len(X)))
    shap_values = explainer.shap_values(sample, nsamples=100)

    mean_abs_shap = np.abs(shap_values).mean(axis=0).flatten()

    print(f"\n{hospital_name} - SHAP Feature Importance:")
    for fname, val in sorted(zip(feature_names, mean_abs_shap), key=lambda x: -x[1]):
        print(f"  {fname:12s}: {val:.4f}")

    return mean_abs_shap, len(X)

# ── Compute local SHAP for each hospital ──
print("Computing SHAP for Hospital 1 (Cleveland)...")
shap_cleveland, n_cleveland = compute_local_shap("../data/cleveland_clean.csv", "Cleveland")

print("\nComputing SHAP for Hospital 2 (Hungarian)...")
shap_hungarian, n_hungarian = compute_local_shap("../data/hungarian_clean.csv", "Hungarian")

# ── Federated SHAP Aggregation (weighted by number of samples) ──
total = n_cleveland + n_hungarian
global_shap = (shap_cleveland * n_cleveland + shap_hungarian * n_hungarian) / total

print("\n" + "="*50)
print("GLOBAL FEDERATED SHAP - Feature Importance Ranking")
print("="*50)
for fname, val in sorted(zip(feature_names, global_shap), key=lambda x: -x[1]):
    print(f"  {fname:12s}: {val:.4f}")