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
            
        # Menggunakan format prompt Anda untuk menetralisir white background
        enhanced_query = f"a clear photo of a {query}, shoe product on a white background"
        text_emb = self.clip.get_text_embedding(enhanced_query)
        
        similarities = torch.cosine_similarity(text_emb, self.image_embeddings)
        
        if similarities.dim() == 0: similarities = similarities.unsqueeze(0)
        top_indices = similarities.argsort(descending=True)[:top_k].cpu().numpy()
        
        return [self.image_paths[i] for i in top_indices]
