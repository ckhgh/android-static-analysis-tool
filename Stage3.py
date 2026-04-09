import os
import json
import numpy as np
from scipy import sparse
import joblib
from tqdm import tqdm
from androguard.misc import AnalyzeAPK
from feature_extraction import (
    calculate_file_hash,
    extract_manifest,
    extract_intent_filters,
    extract_apis,
    extract_opcodes,
    extract_features
)


APK_DIR = r"C:\Users\Samuel\Desktop\final year project\code\StaticAnalysis"
EXTRACTED_DIR = r"C:\Users\Samuel\Desktop\final year project\code\StaticAnalysisExtracted"
SPARSE_DIR = r"C:\Users\Samuel\Desktop\final year project\code\SparseMatrix"
MODEL_PATH = r"C:\Users\Samuel\Desktop\final year project\code\TrainedModels\xgboost.joblib"
MALICIOUS_THRESHOLD = 0.70
SUSPICIOUS_THRESHOLD = 0.30


#loading the model and features name correspond to the 400 features used to train XGB
model = joblib.load(MODEL_PATH)
with open(os.path.join(SPARSE_DIR, "feature_names.json"), "r", encoding="utf-8") as f:
    final_feature_names = json.load(f)
feature_to_col = {name: idx for idx, name in enumerate(final_feature_names)}


#same feature extraction function as stage 1
def analyze_single_apk(apk_path: str):
    filename = os.path.basename(apk_path)
    output_file = os.path.join(EXTRACTED_DIR, filename + "_analysis.json")

    if os.path.exists(output_file):
        print(f"Skipping {filename}")
        return output_file

    try:
        a, d, dx = AnalyzeAPK(apk_path)
        extracted_features = {
            "hash": calculate_file_hash(apk_path),
            "manifest": extract_manifest(a),
            "intentFilters": extract_intent_filters(a),
            "apiCalls": extract_apis(dx),
            "opcodes": extract_opcodes(dx),
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(extracted_features, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error analyzing {filename} {e}")
        return output_file


# get list of all apk
apk_files = [
    os.path.join(APK_DIR, f)
    for f in os.listdir(APK_DIR)
    if f.lower().endswith(".apk") and os.path.isfile(os.path.join(APK_DIR, f))
]


#same feature extraction to JSON as stage 1
analysis_files = []
for apk_path in tqdm(apk_files, desc=f"Extracting APKs", unit="apk"):
    json_path = analyze_single_apk(apk_path)
    if json_path:
        analysis_files.append(json_path)


#load the JSON for classification
results = []
for json_path in tqdm(analysis_files, desc="Analyzing and classifying APKs with XGBoost", unit="apk"):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        #Build sparse matrix with the same 400 features as stage 1.5
        feat_dict = extract_features(data)
        vector = np.zeros(len(final_feature_names), dtype=np.float32)
        for feat_name, value in feat_dict.items():
            if feat_name in feature_to_col:
                vector[feature_to_col[feat_name]] = value
        X_new = sparse.csr_matrix([vector])

        # do classification and confidence score
        proba = model.predict_proba(X_new)[0][1]

        #save apk name and probability (we determine label later based on confidence)
        apk_name = os.path.basename(json_path).replace("_analysis.json", "")
        results.append({
            "apk": apk_name,
            "malicious_prob": float(proba)
        })
    except Exception as e:
        print(f"Error classifying {os.path.basename(json_path)}: {e}")


#print results
print("\n" + "="*85)
print(" " * 28 + "CLASSIFICATION RESULTS")
print("="*85)

safe_count = 0
suspicious_count = 0
malicious_count = 0

for r in results:
    proba = r["malicious_prob"]
    prob_percent = proba * 100

    if proba >= MALICIOUS_THRESHOLD:
        label = "Malicious"
        rec_action = "Avoid installing this APK and consider deleting it."
        malicious_count += 1
    elif proba >= SUSPICIOUS_THRESHOLD:
        label = "Suspicious"
        rec_action = "Further analysis or testing is recommended before installing the APK."
        suspicious_count += 1
    else:
        label = "Safe"
        rec_action = "The APK is safe to install."
        safe_count += 1


    # add color to the label
    color = {
        "Malicious": "\033[91m",
        "Suspicious": "\033[93m",
        "Safe": "\033[92m"
    }
    status = f"{color[label]}{label}\033[0m"


    print(f"APK                    : {r['apk']}")
    print(f"Result                 : {status}")
    print(f"Malicious probability  : {prob_percent:.1f}%")
    print(f"Recommended action     : {rec_action}")
    print("-" * 85)

print("="*85)
print(f"Total APKs processed : {len(results)}")
print(f"Benign count         : {safe_count}")
print(f"Suspicious count     : {suspicious_count}")
print(f"Malicious count      : {malicious_count}")
print("="*85)