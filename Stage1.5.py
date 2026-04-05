import os
import json
import numpy as np
from scipy import sparse
from sklearn.feature_selection import SelectKBest, chi2
from feature_extraction import extract_features
from tqdm import tqdm


MAL_DIR = "MaliciousExtracted"
BEN_DIR = "BenignExtracted"
OUTPUT_DIR = "SparseMatrix"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MIN_DF = 5  # minimum appearance to be used for feature
MAX_DF_RATIO = 0.99  # remove feature appeares 99% of time
CHI2_K = 2000  # number of feature selected


mal_files = [os.path.join(MAL_DIR, f) for f in os.listdir(MAL_DIR) if f.endswith(".apk_analysis.json")]
ben_files = [os.path.join(BEN_DIR, f) for f in os.listdir(BEN_DIR) if f.endswith(".apk_analysis.json")]
all_files = mal_files + ben_files
labels = np.array([1] * len(mal_files) + [0] * len(ben_files), dtype=np.int8)  # create label array

print(f"{len(mal_files):,} malicious and {len(ben_files):,} benign JSONs, total: {len(mal_files)+len(ben_files):,}")


# go through all json and map all feature to index for sparse matrix
feature_to_idx = {}
idx_counter = 0
for file_path in tqdm(all_files, desc="collecting features", unit="file"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for feat in extract_features(data).keys():
        if feat not in feature_to_idx:
            feature_to_idx[feat] = idx_counter
            idx_counter += 1

print(f"total features: {len(feature_to_idx):,}")


# build sparse matrix
rows, cols, data_vals = [], [], []
for row_idx, file_path in tqdm(enumerate(all_files), total=len(all_files), desc="building sparse matrix", unit="file"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    feature_dict = extract_features(data)
    for feat, value in feature_dict.items():
        if feat in feature_to_idx:
            rows.append(row_idx)  # row index of the APK
            cols.append(feature_to_idx[feat])  # column index of the feature
            data_vals.append(value)  # 1 or 0 for prsent or not, frequency for opcode


# compress to sparse row matrix
X_sparse = sparse.coo_matrix(
    (data_vals, (rows, cols)),
    shape=(len(all_files), len(feature_to_idx)),
    dtype=np.float32,
).tocsr()

print(f"sparse matrix: {X_sparse.shape}")


# remove freqeuntly appeared feature and feature that does not fulfill minimum appearance
df = np.array((X_sparse > 0).sum(axis=0)).flatten()
n_samples = X_sparse.shape[0]
mask_cleanup = (df >= MIN_DF) & (df <= MAX_DF_RATIO * n_samples)
X_clean = X_sparse[:, mask_cleanup]
clean_feature_names = [name for i, name in enumerate(feature_to_idx.keys()) if mask_cleanup[i]]

print(f"cleaned sparse matrix: ({X_clean.shape[1]:,})")


# do chi2 to select meaningful features
selector = SelectKBest(chi2, k=CHI2_K)
X_selected = selector.fit_transform(X_clean, labels)
support = selector.get_support()
selected_feature_names = [clean_feature_names[i] for i in range(len(clean_feature_names)) if support[i]]

print(f"chi2ed cleaned sparse matrix: {X_selected.shape}")


# save the sparse matrix
sparse.save_npz(os.path.join(OUTPUT_DIR, "features_sparse.npz"), X_selected)  # save sparse matrix in npz
with open(
    os.path.join(OUTPUT_DIR, "feature_names.json"), "w", encoding="utf-8"
) as f:  # save feature names of sparse matrix in json
    json.dump(selected_feature_names, f, ensure_ascii=False, indent=2)
np.save(os.path.join(OUTPUT_DIR, "labels.npy"), labels)  # save malicious label in npy

print(f"\nDone")
