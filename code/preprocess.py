import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ── Load datasets ──
cleveland = pd.read_csv("../data/cleveland.csv")
hungarian = pd.read_csv("../data/hungarian.csv")

# ── Fix missing values ──
# Cleveland: fill 'ca' and 'thal' with median
cleveland['ca'] = cleveland['ca'].fillna(cleveland['ca'].median())
cleveland['thal'] = cleveland['thal'].fillna(cleveland['thal'].median())

# Hungarian: too many missing in slope/ca/thal -> drop these columns
hungarian = hungarian.drop(columns=['slope', 'ca', 'thal'])
cleveland_matched = cleveland.drop(columns=['slope', 'ca', 'thal'])

# Hungarian: fill remaining missing values with median
for col in hungarian.columns:
    if hungarian[col].isnull().sum() > 0:
        hungarian[col] = hungarian[col].fillna(hungarian[col].median())

# ── Convert target to binary (0 = no disease, 1 = disease) ──
cleveland_matched['target'] = (cleveland_matched['target'] > 0).astype(int)
hungarian['target'] = (hungarian['target'] > 0).astype(int)

# ── Normalize features (0 to 1) ──
feature_cols = [c for c in cleveland_matched.columns if c != 'target']

scaler1 = MinMaxScaler()
cleveland_matched[feature_cols] = scaler1.fit_transform(cleveland_matched[feature_cols])

scaler2 = MinMaxScaler()
hungarian[feature_cols] = scaler2.fit_transform(hungarian[feature_cols])

# ── Save cleaned data ──
cleveland_matched.to_csv("../data/cleveland_clean.csv", index=False)
hungarian.to_csv("../data/hungarian_clean.csv", index=False)

print("Cleveland cleaned shape:", cleveland_matched.shape)
print("Hungarian cleaned shape:", hungarian.shape)
print("\nCleveland target distribution:\n", cleveland_matched['target'].value_counts())
print("\nHungarian target distribution:\n", hungarian['target'].value_counts())
print("\nFeature columns used:", feature_cols)
print("\nSaved: cleveland_clean.csv and hungarian_clean.csv")