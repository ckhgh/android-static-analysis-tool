import os
import numpy as np
from scipy import sparse
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import joblib


DATA_DIR = r"C:\Users\Samuel\Desktop\final year project\code\SparseMatrix"  
MODEL_DIR = r"C:\Users\Samuel\Desktop\final year project\code\TrainedModels"
MODEL_FILENAME = "svm.joblib"


# load sparse matrix and label
x = sparse.load_npz(os.path.join(DATA_DIR, "features.npz"))
y = np.load(os.path.join(DATA_DIR, "labels.npy"))


# SVM settings 
svm = LinearSVC(
    random_state=42,
    max_iter=2000,
    dual='auto',
)

# SVM hyperparameters, 10 combo
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100], # regularization strength
    'class_weight': [None, 'balanced'] # apply weight on benign and malicious to balance or not
}


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# finding optimal hyperparameter by looking for combination with best f1 score
grid_search = GridSearchCV(
    estimator=svm,
    param_grid=param_grid,
    cv=cv,
    scoring='f1',
    n_jobs=-1,
    verbose=2,
)
grid_search.fit(x, y)  # training


best_model = grid_search.best_estimator_  # save best model

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


# Print metrics of best model
print(f"Accuracy  : {cv_results['test_accuracy'].mean():.4f}")
print(f"Precision : {cv_results['test_precision'].mean():.4f}")
print(f"Recall    : {cv_results['test_recall'].mean():.4f}")
print(f"F1-score  : {cv_results['test_f1'].mean():.4f}")


# save the model
model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
joblib.dump(best_model, model_path)

print("\nDone")