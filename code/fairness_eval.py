import pandas as pd
import torch
import numpy as np
from model import HeartDiseaseDNN
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score, recall_score

# ── Load the trained global model ──
model = HeartDiseaseDNN(input_dim=10)
model.load_state_dict(torch.load("global_model.pt"))
model.eval()

def predict(X):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        preds = model(X_tensor).numpy()
    return (preds > 0.5).astype(int).flatten()

def evaluate_fairness(csv_path, hospital_name):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['target']).values
    y_true = df['target'].values
    y_pred = predict(X)

    print(f"\n{'='*50}")
    print(f"{hospital_name} - Fairness Evaluation")
    print(f"{'='*50}")

    # ── Fairness by Sex (sex column is normalized 0-1, originally 0=female, 1=male) ──
    sex_raw = df['sex'].values
    sex_group = np.where(sex_raw > 0.5, "Male", "Female")

    mf_sex = MetricFrame(
        metrics={"accuracy": accuracy_score, "recall": recall_score},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sex_group
    )
    print("\n--- By Gender ---")
    print(mf_sex.by_group)
    print(f"\nAccuracy Gap (max-min): {mf_sex.difference()['accuracy']:.4f}")

    # ── Fairness by Age Group (age column normalized 0-1) ──
    age_raw = df['age'].values
    age_min, age_max = age_raw.min(), age_raw.max()
    age_actual = age_raw * (age_max - age_min) + age_min  # approx denormalize

    age_bins = pd.cut(age_raw, bins=3, labels=["Young", "Middle", "Old"])

    mf_age = MetricFrame(
        metrics={"accuracy": accuracy_score, "recall": recall_score},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=age_bins
    )
    print("\n--- By Age Group ---")
    print(mf_age.by_group)
    print(f"\nAccuracy Gap (max-min): {mf_age.difference()['accuracy']:.4f}")

# ── Evaluate both hospitals ──
evaluate_fairness("../data/cleveland_clean.csv", "Cleveland")
evaluate_fairness("../data/hungarian_clean.csv", "Hungarian")