import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from config import MODEL_NAME

class CLIPHandler:
    def __init__(self):
        # Otomatis menggunakan GPU jika tersedia
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Memuat model {MODEL_NAME} di {self.device}...")
        
        self.model = CLIPModel.from_pretrained(MODEL_NAME).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    def get_image_embeddings(self, image_paths):
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            embeddings = self.model.get_image_features(**inputs)
            
        # Normalisasi (L2) untuk keperluan similarity
        return embeddings / embeddings.norm(dim=-1, keepdim=True)

    def get_text_embedding(self, text):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            embedding = self.model.get_text_features(**inputs)
            
        return embedding / embedding.norm(dim=-1, keepdim=True)
