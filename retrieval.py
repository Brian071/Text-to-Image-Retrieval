import torch
from model_handler import CLIPHandler
from data_loader import load_data

class ImageRetriever:
    def __init__(self):
        self.clip = CLIPHandler()
        self.data = load_data()
        self.image_paths = self.data['absolute_path'].tolist()
        
        print("Mengekstraksi embedding untuk seluruh galeri gambar...")
        # (Catatan: Untuk produksi skala besar, lakukan proses ini dalam *batch*)
        if self.image_paths:
            self.image_embeddings = self.clip.get_image_embeddings(self.image_paths)
        else:
            self.image_embeddings = None

    def search(self, query, top_k=4):
        if self.image_embeddings is None:
            return []
            
        text_emb = self.clip.get_text_embedding(query)
        
        # Hitung skor kesamaan (Cosine Similarity)
        similarities = torch.cosine_similarity(text_emb, self.image_embeddings)
        
        # Ambil indeks 'top_k' dari nilai tertinggi
        top_indices = similarities.argsort(descending=True)[:top_k].cpu().numpy()
        
        return [self.image_paths[i] for i in top_indices]
