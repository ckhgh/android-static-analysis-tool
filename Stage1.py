from androguard.core.apk import APK
from androguard.misc import AnalyzeAPK
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import json
import os
import multiprocessing
from feature_extraction import (
    calculate_file_hash,
    extract_manifest,
    extract_intent_filters,
    extract_apis,
    extract_opcodes,
)


def analyze_single_apk(apk_path: str, output_folder: str) -> None:

    filename = os.path.basename(apk_path)
    output_file = os.path.join(output_folder, filename + "_analysis.json")

    if os.path.exists(output_file):
        print(f"Skipping {filename}")
        return

    try:
        a, d, dx = AnalyzeAPK(apk_path)

        apis = extract_apis(dx)
        opcodes = extract_opcodes(dx)
        file_hash = calculate_file_hash(apk_path)
        manifest = extract_manifest(a)
        intent_filters = extract_intent_filters(a)

        extracted_features = {
            "hash": file_hash,
            "manifest": manifest,
            "intentFilters": intent_filters,
            "apiCalls": apis,
            "opcodes": opcodes,
        }

        with open(output_file, "w") as f:
            json.dump(extracted_features, f, indent=4)

    except Exception as e:
        print(f"Error analyzing {filename} {e}")


if __name__ == "__main__":
    folder_pairs = [("Malicious", "MaliciousExtracted"), ("Benign", "BenignExtracted")]

    for input_folder, output_folder in folder_pairs:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        apk_files = [
            os.path.join(input_folder, f)
            for f in os.listdir(input_folder)
            if f.endswith(".apk") and os.path.isfile(os.path.join(input_folder, f))
        ]

        print(f"\nProcessing {input_folder} folder with{len(apk_files)} APKs ")

        max_workers = multiprocessing.cpu_count() - 5

        print(f"parallel process with {max_workers} core")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(analyze_single_apk, apk, output_folder) for apk in apk_files]

            for future in tqdm(
                as_completed(futures),
                total=len(apk_files),
                desc=f"Analyzing {input_folder}",
                unit="apk",
            ):
                future.result()

    print("\nDone")
