import os
import numpy as np
from scipy import sparse
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import joblib


DATA_DIR = r"C:\Users\Samuel\Desktop\final year project\code\SparseMatrix"  
MODEL_DIR = r"C:\Users\Samuel\Desktop\final year project\code\TrainedModels"
MODEL_FILENAME = "xgboost.joblib"


# load sparse matrix and label
x = sparse.load_npz(os.path.join(DATA_DIR, "features.npz"))
y = np.load(os.path.join(DATA_DIR, "labels.npy"))

# Compute weight to balance benign and malicious
neg, pos = np.bincount(y.astype(int))
scale_pos_weight = neg / pos if pos > 0 else 1.0


# XGBoost settings
xgb_clf = xgb.XGBClassifier(
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss',
)


# XGBoost hyperparameters, 128 combo
param_grid = {
    'n_estimators': [100, 200], # num of trees
    'max_depth': [6, 10], # max depth of each tree
    'learning_rate': [0.05, 0.1],
    'min_child_weight': [1, 5], # min sum of weight in a leaf node
    'subsample': [0.8, 1.0], # APK used per tree
    'colsample_bytree': [0.8, 1.0], # features used per tree
    'scale_pos_weight': [1.0, scale_pos_weight] # apply weight on benign and malicious to balance or not
}


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # shuffle malicious and benign order before doing 5-fold split


# finding optimal hyperparameter by looking for combination with best f1 score
grid_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    cv=cv,
    scoring='f1',
    n_jobs=-1,
    verbose=2,
)
grid_search.fit(x, y)  # training


best_model = grid_search.best_estimator_  # save best model

print(f"Best hyperparameters combination: {grid_search.best_params_}")

# do 5-fold cv with best model hyperparameter again
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='binary', zero_division=0),
    'recall': make_scorer(recall_score, average='binary', zero_division=0),
    'f1': make_scorer(f1_score, average='binary', zero_division=0)
}
cv_results = cross_validate(
    best_model, x, y,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring=scoring,
    n_jobs=-1
)


# Print metrics of best model
print(f"Accuracy  : {cv_results['test_accuracy'].mean():.4f}")
print(f"Precision : {cv_results['test_precision'].mean():.4f}")
print(f"Recall    : {cv_results['test_recall'].mean():.4f}")
print(f"F1-score  : {cv_results['test_f1'].mean():.4f}")


# save the model
model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
joblib.dump(best_model, model_path)

print("\nDone")