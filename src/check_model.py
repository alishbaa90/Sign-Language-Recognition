import pandas as pd
import glob
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

all_data = []
all_labels = []

for file in glob.glob("../data/*.csv"):
    label = file.split("\\")[-1].replace(".csv", "")
    df = pd.read_csv(file, header=None)
    all_data.append(df)
    all_labels.extend([label] * len(df))

X = pd.concat(all_data, ignore_index=True)
y = all_labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

with open("../models/gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

preds = model.predict(X_test)
print("Classes:", model.classes_)
print("Confusion Matrix:\n", confusion_matrix(y_test, preds, labels=model.classes_))
print("\n", classification_report(y_test, preds))