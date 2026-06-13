import os
os.environ["KAGGLE_KEY"] = os.environ.get("KAGGLE_API_TOKEN")
os.environ["KAGGLE_USERNAME"] = "swarn"

import kagglehub

path = kagglehub.dataset_download("andrewmvd/heart-failure-clinical-data")

print("Dataset downloaded to:", path)

for f in os.listdir(path):
    print(f)