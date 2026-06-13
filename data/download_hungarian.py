import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data"

columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]

df = pd.read_csv(url, names=columns, na_values="?")

print("Shape:", df.shape)
print(df.head())

df.to_csv("hungarian.csv", index=False)
print("Saved as hungarian.csv")