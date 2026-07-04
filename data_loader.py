import pandas as pd
import os
from config import IMAGES_DIR, METADATA_PATH, MAX_SAMPLES

def load_data():
    """Memuat data metadata gambar dan mengembalikan dataframe dengan path gambar absolut."""
    try:
        df = pd.read_csv(METADATA_PATH)
        # Membatasi sampel berdasarkan batasan penelitian (misal fokus ke shoes)
        df = df.head(MAX_SAMPLES)
        
        # Membuat kolom path gambar absolut
        df['absolute_path'] = df['path'].apply(lambda x: os.path.join(IMAGES_DIR, x))
        
        # Filter hanya gambar yang eksis di penyimpanan
        df = df[df['absolute_path'].apply(os.path.exists)]
        
        print(f"Berhasil memuat {len(df)} data gambar.")
        return df
    except Exception as e:
        print(f"Error memuat dataset: {e}")
        return pd.DataFrame()
