# Android malware detection (static analysis + ML)

Python pipeline that extracts static features from Android APKs with Androguard, builds a sparse feature matrix with TF–IDF and mutual-information selection, trains classifiers, and scores unknown APKs with a trained model (default: XGBoost).

## What gets extracted per APK

- **File hash** (SHA-256)
- **Manifest**: permissions, activities, services, receivers
- **Intent filters** (actions and categories per component)
- **External API calls** (class/method signatures and callers)
- **Dalvik opcode counts** (internal methods only)

## Folders

| Path | Purpose |
|------|---------|
| `Malicious/`, `Benign/` | Training APKs (place `.apk` files here before extraction) |
| `MaliciousExtracted/`, `BenignExtracted/` | One JSON file per training APK: extracted features saved by `feature_extraction.py` |
| `SparseMatrix/` | `features.npz`, `labels.npy`, `feature_names.json` after matrix build |
| `TrainedModels/` | Saved models (`xgboost.joblib`, `random_forest.joblib`, `svm.joblib`) |
| `StaticAnalysis/` | APKs you want to classify with `static_analysis_tool.py` |
| `StaticAnalysisExtracted/` | One JSON file per APK in `StaticAnalysis/`, created the first time that APK is analyzed |

## Instructions

Install dependencies once (from the project root):

```bash
pip install -r requirements.txt
```

### Training with your own malicious and benign APKs

Place your training APKs in `Malicious/` and `Benign/`, then run:

```bash
python feature_extraction.py
python sparse_matrix.py
python train_xgboost.py
```

You can use `train_random_forest.py` or `train_support_vector_machine.py` instead of `train_xgboost.py`. Try each trainer and compare their cross-validation metrics and confusion matrix output to see how the models perform on your dataset.

### Using the bundled trained model only

If you are not training your own model, you can simply use the provided trained model. Put the APKs you want to scan in `StaticAnalysis/`, then run:

```bash
python static_analysis_tool.py
```

