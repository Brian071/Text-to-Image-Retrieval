import torch
from model_handler import CLIPHandler
from data_loader import load_data

class ImageRetriever:
    def __init__(self):
        self.clip = CLIPHandler()
        self.data = load_data()
        self.image_paths = self.data['absolute_path'].tolist()
        
        print("Mengekstraksi embedding murni untuk seluruh galeri sepatu...")
        if self.image_paths:
            self.image_embeddings = self.clip.get_image_embeddings(self.image_paths)
        else:
            self.image_embeddings = None

    def search(self, query, top_k=4):
        if self.image_embeddings is None or len(self.image_paths) == 0:
            return []
            
        # PERBAIKAN 1: Hapus kata-kata berlebihan yang menenggelamkan query
        enhanced_query = f"a photo of {query}"
        text_emb = self.clip.get_text_embedding(enhanced_query)
        
        # PERBAIKAN 2: Gunakan Matrix Multiplication
        similarities = (self.image_embeddings @ text_emb.T).squeeze()
        
        # Ambil top K
        top_indices = similarities.argsort(descending=True)[:top_k].cpu().numpy()
        
        return [self.image_paths[i] for i in top_indices]
