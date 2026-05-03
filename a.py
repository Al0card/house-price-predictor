import kagglehub
import shutil
import os

download_path = kagglehub.competition_download(
    "house-prices-advanced-regression-techniques"
)

target_dir = "data/raw"
os.makedirs(target_dir, exist_ok=True)

for filename in os.listdir(download_path):
    src = os.path.join(download_path, filename)
    dst = os.path.join(target_dir, filename)

    if os.path.isfile(src):
        shutil.copy(src, dst)

print("Files saved to:", target_dir)