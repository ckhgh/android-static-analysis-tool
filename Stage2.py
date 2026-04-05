import numpy as np
from scipy.sparse import load_npz, csr_matrix
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from tqdm import tqdm
import time
import warnings
from joblib import dump, load   # ← NEW: for saving/loading models

warnings.filterwarnings("ignore")

# ================== CONFIGURATION ==================
DATA_DIR = r"C:\Users\Samuel\Desktop\final year project\code\SparseMatrix"
SAVE_DIR = r"C:\Users\Samuel\Desktop\final year project\code\TrainedModels"

feature_path = os.path.join(DATA_DIR, "feature_names.json")
labels_path  = os.path.join(DATA_DIR, "labels.npy")
sparse_path  = os.path.join(DATA_DIR, "features_sparse.npz")


print("Loading APK malware dataset...")

with open(feature_path, "r") as f:
    feature_names = json.load(f)

X = load_npz(sparse_path)
y = np.load(labels_path)


print(f"Loaded {len(feature_names)} features")
print(f"Features shape: {X.shape}")
print(f"Labels: {np.sum(y == 0)} benign + {np.sum(y == 1)} malicious\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================== MODELS ==================
models = {
    "SVM (Linear)": LinearSVC(random_state=42, max_iter=10000),
    "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
        random_state=42, eval_metric="logloss", verbosity=0
    ),
}

results = {}
print("Training models...\n")

for name, model in tqdm(models.items(), desc="Training", unit="model"):
    start = time.time()
    print(f"\n→ Training {name}...")
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A",
        "Train Time (s)": round(train_time, 2),
    }

    print(f"{name} done in {train_time:.2f}s")
    print(classification_report(y_test, y_pred))
    print("-" * 70)

# ================== SAVE TRAINED MODELS ==================
print("\nSaving trained models to disk...")
for name, model in models.items():
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
    model_path = os.path.join(SAVE_DIR, f"{safe_name}_model.joblib")
    dump(model, model_path)
    print(f"Saved {name} → {model_path}")

print("\nTRAINING COMPLETE!")


