import os
import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib


DATA_DIR = "SparseMatrix"  
MODEL_DIR = "TrainedModels"
MODEL_FILENAME = "random_forest.joblib"


# load sparse matrix and label
x = sparse.load_npz(os.path.join(DATA_DIR, "features.npz"))
y = np.load(os.path.join(DATA_DIR, "labels.npy"))


# rf settings
rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
)

# rf hyperparameters combination, 72 combo
param_grid = {
    'n_estimators': [100, 200, 300], # num of tree
    'max_depth': [None, 20, 30], # max depth of tree
    'min_samples_split': [2, 5], # min num of sample to split a node
    'min_samples_leaf': [1, 2], # min num of sample in a node to be leaf
    'class_weight': [None, 'balanced'] # apply weight on benign and malicious to balance or not
}


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # shuffle maliciious and benign order before doing 5 fold split


# finding optimal hyperparameter by look for combination with best f1 score
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=cv,
    scoring='f1',
    n_jobs=-1,
    verbose=2,
)
grid_search.fit(x, y) # training


best_model = grid_search.best_estimator_ # save best model

print(f"Best hyperparameters combination: {grid_search.best_params_}")

# do 5 fold cv with best model hyperparameter again
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


# print metrics of best model
print(f"Accuracy  : {cv_results['test_accuracy'].mean():.6f}")
print(f"Precision : {cv_results['test_precision'].mean():.6f}")
print(f"Recall    : {cv_results['test_recall'].mean():.6f}")
print(f"F1-score  : {cv_results['test_f1'].mean():.6f}")


# print confusion matrix
y_pred_cv = cross_val_predict(
    best_model, x, y,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1
)
tn, fp, fn, tp = confusion_matrix(y, y_pred_cv).ravel()

print(f"True Positives  : {tp}")
print(f"True Negatives  : {tn}")
print(f"False Positives : {fp}")
print(f"False Negatives : {fn}")


# save the model
model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
joblib.dump(best_model, model_path)

print("\nDone")