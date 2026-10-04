import pandas as pd
import glob
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

with open("../models/gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to models/gesture_model.pkl")