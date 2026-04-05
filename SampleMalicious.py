import os
import random
import shutil
from pathlib import Path

def select_and_copy_files(source_dir: str, dest_dir: str, num_files: int):
    """
    Randomly select 'num_files' from source_dir and copy them to dest_dir.
    Automatically adds .apk extension to any file that doesn't already have it.
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    # Get only files (ignore any subfolders)
    all_files = [f for f in source_path.iterdir() if f.is_file()]
    
    if len(all_files) < num_files:
        print(f"Warning: {source_dir} has only {len(all_files)} files. Taking ALL of them.")
        selected_files = all_files
    else:
        selected_files = random.sample(all_files, num_files)
    
    # Create destination folder if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)
    
    copied = 0
    for file_path in selected_files:
        try:
            # Ensure the file has .apk extension in the destination
            dest_name = file_path.name
            if not dest_name.lower().endswith('.apk'):
                dest_name += '.apk'
            
            shutil.copy2(file_path, dest_path / dest_name)  # copy2 preserves metadata
            copied += 1
        except Exception as e:
            print(f"Error copying {file_path.name}: {e}")
    
    print(f"Copied {copied} files from {source_path.name} → {dest_path}")


if __name__ == "__main__":
    # ================== CONFIGURATION ==================
    # ←←←←←←←← CHANGE THESE PATHS TO YOUR ACTUAL FOLDERS ←←←←←←←←
    ADWARE_PATH  = r"C:\Users\Samuel\Desktop\final year project\MalDroid2020\Adware"     # Folder with 1515 files
    BANKING_PATH = r"C:\Users\Samuel\Desktop\final year project\MalDroid2020\Banking"    # Folder with 2506 files
    SMS_PATH     = r"C:\Users\Samuel\Desktop\final year project\MalDroid2020\SMS"        # Folder with 4822 files
    
    # Destination folder (will be created automatically)
    OUTPUT_BASE = r"C:\Users\Samuel\Desktop\final year project\code\Malicious"   # Change this too!
    # ==================================================

    # We want exactly 4000 files total with equal ratio (~1333-1334 per class)
    NUM_ADWARE  = 1333
    NUM_BANKING = 1333
    NUM_SMS     = 1333   # 1333+1333+1334 = 4000

    # Optional: set a seed so you get the same selection every time you run the script
    random.seed(42)

    print("Starting balanced file extraction...")
    print(f"Target: {NUM_ADWARE} from Adware + {NUM_BANKING} from Banking + {NUM_SMS} from SMS = 4000 files\n")

    select_and_copy_files(ADWARE_PATH,  os.path.join(OUTPUT_BASE, "Adware"),  NUM_ADWARE)
    select_and_copy_files(BANKING_PATH, os.path.join(OUTPUT_BASE, "Banking"), NUM_BANKING)
    select_and_copy_files(SMS_PATH,     os.path.join(OUTPUT_BASE, "SMS"),     NUM_SMS)

    print("\nDone! 4000 files have been extracted and organized.")
    print(f"Check the folder: {OUTPUT_BASE}")
    print("Each subfolder (Adware / Banking / SMS) now has roughly equal numbers of files.")
    print("All copied files now end with .apk (added automatically if missing).")