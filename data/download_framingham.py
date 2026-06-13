import os
os.environ["KAGGLE_KEY"] = os.environ.get("KAGGLE_API_TOKEN")
os.environ["KAGGLE_USERNAME"] = "swarn"

import kagglehub

path = kagglehub.dataset_download("aasheesh200/framingham-heart-study-dataset")

print("Dataset downloaded to:", path)

import os
for f in os.listdir(path):
    print(f)