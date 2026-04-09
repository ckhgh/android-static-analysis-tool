import os
import json
import numpy as np
import warnings
from scipy import sparse
from feature_extraction import extract_features
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import mutual_info_classif

MAL_DIR = "MaliciousExtracted"
BEN_DIR = "BenignExtracted"
OUTPUT_DIR = "SparseMatrix"
MIN_DF = 5  # minimum appearance to be used for feature
MAX_DF_RATIO = 0.99  # remove feature appeares 99% of time
N_IG_FEATURES = 400 # how many features keep after information gain

warnings.filterwarnings(
    "ignore",
    message=r".*Clustering metrics expects discrete values.*",
    category=UserWarning,
)

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
            data_vals.append(value)  #  frequency of feature apperance


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

print(f"cleaned sparse matrix: ({X_clean.shape})")


# tfidf
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_clean)

print(f"tfidfed cleaned sparse matrix: {X_tfidf.shape}")


# ig
mi_scores = mutual_info_classif(X_tfidf, labels)
k = min(N_IG_FEATURES, X_tfidf.shape[1])
top_indices = np.argsort(mi_scores)[-k:]
X_final = X_tfidf[:, top_indices]
final_feature_names = [clean_feature_names[i] for i in top_indices]

print(f"iged tfidfed cleaned sparse matrix: {X_final.shape}")


# save the processed matrix
os.makedirs(OUTPUT_DIR, exist_ok=True)
sparse.save_npz(os.path.join(OUTPUT_DIR, "features.npz"), X_final)
with open(
    os.path.join(OUTPUT_DIR, "feature_names.json"), "w", encoding="utf-8"
) as f:
    json.dump(final_feature_names, f, ensure_ascii=False, indent=2)
np.save(os.path.join(OUTPUT_DIR, "labels.npy"), labels)

print(f"\nDone")
