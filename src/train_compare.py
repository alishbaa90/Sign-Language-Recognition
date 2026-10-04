import pandas as pd
import glob
import pickle
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Load all data
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

# Scale features for SVM and Neural Net
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "RandomForest": (RandomForestClassifier(n_estimators=200, random_state=42), X_train, X_test),
    "SVM": (SVC(kernel='rbf', probability=True, random_state=42), X_train_scaled, X_test_scaled),
    "NeuralNet (MLP)": (MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42), X_train_scaled, X_test_scaled),
}

results = []
report_lines = []

for name, (model, Xtr, Xte) in models.items():
    start = time.time()
    model.fit(Xtr, y_train)
    train_time = time.time() - start

    preds = model.predict(Xte)
    acc = accuracy_score(y_test, preds)

    results.append((name, acc, train_time))
    report_lines.append(f"\n{'='*50}\n{name} — Accuracy: {acc:.4f} — Train time: {train_time:.2f}s\n{'='*50}")
    report_lines.append(str(confusion_matrix(y_test, preds)))
    report_lines.append(classification_report(y_test, preds))

    # Save the best-performing one separately too
    with open(f"../models/{name.replace(' ', '_').replace('(', '').replace(')', '')}.pkl", "wb") as f:
        pickle.dump(model, f)

# Print summary
print("\n\n=== MODEL COMPARISON SUMMARY ===")
for name, acc, t in sorted(results, key=lambda r: -r[1]):
    print(f"{name:20s} | Accuracy: {acc:.4f} | Train time: {t:.2f}s")

# Save full report to file
with open("../outputs/model_comparison_report.txt", "w") as f:
    f.write("\n".join(report_lines))

production_model_path = "../models/RandomForest.pkl"
with open(production_model_path, "rb") as src, open("../models/gesture_model.pkl", "wb") as dst:
    dst.write(src.read())

best_name_for_report = max(results, key=lambda r: r[1])[0]
print(f"\nBest accuracy in comparison: {best_name_for_report}")
print("Production model set to: RandomForest (avoids scaling mismatch at inference)")
print("Full report saved to outputs/model_comparison_report.txt")