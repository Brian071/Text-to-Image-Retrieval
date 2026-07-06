import pandas as pd
import os
import gzip
import json
from config import IMAGES_DIR, METADATA_PATH, MAX_SAMPLES

def load_data():
    """Memuat dan memfilter ketat hanya data kategori SHOES."""
    try:
        print("Memuat dan menyinkronkan metadata produk dengan gambar...")
        
        # 1. Baca metadata dari listings
        listings_path = 'dataset/listings/metadata/listings_0.json.gz'
        if not os.path.exists(listings_path):
            listings_path = 'dataset/listings/listings/metadata/listings_0.json.gz'
            
        data = []
        with gzip.open(listings_path, 'rt') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except:
                    continue
                    
        df_listings = pd.DataFrame(data)
        df_listings['category'] = df_listings['product_type'].apply(
            lambda x: x[0]['value'] if isinstance(x, list) and len(x) > 0 else None
        )
        
        # 2. Baca metadata gambar
        df_images = pd.read_csv(METADATA_PATH)
        
        # 3. Gabungkan dan filter HANYA sepatu (Perbaikan Fatal!)
        df_merged = pd.merge(df_listings, df_images, left_on='main_image_id', right_on='image_id', how='inner')
        df_shoes = df_merged[df_merged['category'].astype(str).str.upper() == 'SHOES'].copy()
        
        # 4. Potong sampel sesuai batas memori dan cari path aslinya
        df_shoes = df_shoes.head(MAX_SAMPLES)
        
        def get_path(p):
            p1 = os.path.join(IMAGES_DIR, str(p))
            p2 = os.path.join("dataset/images/", str(p))
            return p1 if os.path.exists(p1) else p2

        df_shoes['absolute_path'] = df_shoes['path'].apply(get_path)
        df_shoes = df_shoes[df_shoes['absolute_path'].apply(os.path.exists)]
        
        print(f"Berhasil memuat {len(df_shoes)} data sepatu murni.")
        return df_shoes
    except Exception as e:
        print(f"Error memuat dataset: {e}")
        return pd.DataFrame()
