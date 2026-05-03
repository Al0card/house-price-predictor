import kagglehub
import shutil
import os



# Download latest version
downlaod_path = kagglehub.competition_download('house-prices-advanced-regression-techniques')

target_dir = "data/raw"
os.makedirs(target_dir, exist_ok = True)

for filename in os.listdir(downlaod_path):
    src = os.path.join(downlaod_path, filename)
    dst = os.path.join(target_dir, filename)

    if os.path.isfile(src):
        shutil.copy(src, dst)
        print(f"{filename} saved.")
print("All files saved.")
