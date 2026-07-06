import os

# Konfigurasi Path Dataset
DATA_DIR = "dataset"
IMAGES_DIR = os.path.join(DATA_DIR, "images", "small")
METADATA_PATH = os.path.join(DATA_DIR, "images", "metadata", "images.csv.gz")

# Konfigurasi Model
# Menggunakan model pre-trained (tanpa training ulang)
MODEL_NAME = "openai/clip-vit-base-patch32" 

# Batasan Skala Dataset
MAX_SAMPLES = 9000
