import os
import json
import numpy as np
from scipy import sparse
from joblib import load
from tqdm import tqdm
import sys
import shutil
from androguard.misc import AnalyzeAPK
from feature_extraction import extract_manifest, extract_intent_filters, extract_apis, extract_opcodes, extract_features

# ====================== CONFIGURATION - CHANGE PATHS HERE ======================
BASE_DIR = r"C:\Users\Samuel\Desktop\final year project\code"

MODELS_DIR          = os.path.join(BASE_DIR, "TrainedModels")
STATIC_ANALYSIS_DIR = os.path.join(BASE_DIR, "StaticAnalysis")
FEATURE_NAMES_DIR   = os.path.join(BASE_DIR, "SparseMatrix")

# =============================================================================
MODEL_OPTIONS = {
    "1": ("SVM_Linear_model.joblib",      "SVM (Linear)"),
    "2": ("Random_Forest_model.joblib",   "Random Forest"),
    "3": ("XGBoost_model.joblib",         "XGBoost"),
}
# =============================================================================

def load_feature_mapping():
    feature_path = os.path.join(FEATURE_NAMES_DIR, "feature_names.json")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"feature_names.json not found at {feature_path}")
    
    with open(feature_path, "r", encoding="utf-8") as f:
        selected_names = json.load(f)
    
    feature_to_idx = {name: idx for idx, name in enumerate(selected_names)}
    print(f"Loaded {len(selected_names):,} selected features (clean).")
    return feature_to_idx, len(selected_names)

def analyze_single_apk(original_path: str, feature_to_idx: dict, model, n_features: int):
    filename = os.path.basename(original_path)
    analyze_path = original_path
    temp_path = None

    try:
        if not filename.lower().endswith('.apk'):
            temp_path = original_path + ".apk"
            shutil.copy2(original_path, temp_path)
            analyze_path = temp_path

        a, d, dx = AnalyzeAPK(analyze_path)
        
        raw_data = {
            "manifest": extract_manifest(a),
            "intentFilters": extract_intent_filters(a),
            "apiCalls": extract_apis(dx),
            "opcodes": extract_opcodes(dx)
        }
        
        # CHANGED: now uses weighted features (opcode counts)
        feature_dict = extract_features(raw_data)
        
        indices = []
        values = []
        for feat, val in feature_dict.items():
            if feat in feature_to_idx:
                indices.append(feature_to_idx[feat])
                values.append(float(val))

        if indices:
            X_new = sparse.coo_matrix(
                (values, ([0] * len(indices), indices)),
                shape=(1, n_features), dtype=np.float32
            ).tocsr()
        else:
            X_new = sparse.csr_matrix((1, n_features), dtype=np.float32)
        
        prediction = model.predict(X_new)[0]
        label = "MALICIOUS" if prediction == 1 else "BENIGN"
        
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_new)[0]
            confidence = proba[1] if prediction == 1 else proba[0]
            conf_str = f"{confidence:.4f}"
        else:
            conf_str = "N/A (Linear SVM)"
        
        return {
            "filename": filename,
            "prediction": label,
            "confidence": conf_str,
            "raw_pred": int(prediction)
        }

    except Exception as e:
        error_msg = str(e)
        if "End of central directory record (EOCD) signature not found" in error_msg:
            prediction_label = "INVALID APK"
            confidence_str = "Not a valid APK file"
        else:
            prediction_label = "ERROR"
            confidence_str = error_msg[:80]
        
        return {
            "filename": filename,
            "prediction": prediction_label,
            "confidence": confidence_str,
            "raw_pred": None
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

# ====================== MAIN ======================
if __name__ == "__main__":
    print("=" * 70)
    print("   Android APK Malware Static Analyzer (Clean Version)")
    print("=" * 70)
    
    print("\nAvailable Models:")
    for key, (file, name) in MODEL_OPTIONS.items():
        print(f"  {key}. {name} ({file})")
    
    while True:
        choice = input("\nSelect model (1/2/3): ").strip()
        if choice in MODEL_OPTIONS:
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    
    model_filename, model_name = MODEL_OPTIONS[choice]
    model_path = os.path.join(MODELS_DIR, model_filename)
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    
    print(f"\nLoading {model_name} model...")
    model = load(model_path)
    print("Model loaded successfully!")
    
    feature_to_idx, n_features = load_feature_mapping()
    
    if not os.path.exists(STATIC_ANALYSIS_DIR):
        print(f"Error: StaticAnalysis folder not found.")
        sys.exit(1)
    
    all_files = [f for f in os.listdir(STATIC_ANALYSIS_DIR) 
                 if os.path.isfile(os.path.join(STATIC_ANALYSIS_DIR, f))]
    
    if not all_files:
        print("No files found in StaticAnalysis folder.")
        sys.exit(0)
    
    print(f"\nFound {len(all_files)} file(s) to analyze.\n")
    
    results = []
    for f in tqdm(all_files, desc="Analyzing files", unit="file"):
        original_path = os.path.join(STATIC_ANALYSIS_DIR, f)
        result = analyze_single_apk(original_path, feature_to_idx, model, n_features)
        results.append(result)
        print(f"{result['filename']:<40} {result['prediction']:>12}  Confidence: {result['confidence']}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    malicious_count = sum(1 for r in results if r.get("raw_pred") == 1)
    benign_count    = sum(1 for r in results if r.get("raw_pred") == 0)
    invalid_count   = len(results) - malicious_count - benign_count
    
    print(f"Total files analyzed   : {len(results)}")
    print(f"Detected Malicious     : {malicious_count}")
    print(f"Detected Benign        : {benign_count}")
    print(f"Invalid / Skipped      : {invalid_count}")
    
    results_path = os.path.join(STATIC_ANALYSIS_DIR, "analysis_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {results_path}")
    print("\nDone!")