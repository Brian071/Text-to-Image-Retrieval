import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from config import MODEL_NAME
import gc

class CLIPHandler:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Memuat model {MODEL_NAME} di {self.device}...")
        
        self.model = CLIPModel.from_pretrained(MODEL_NAME).to(self.device)
        self.model.eval()  # PERBAIKAN: Kunci model agar memori tidak berubah-ubah (no dropout)
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        self.batch_size = 64 # PERBAIKAN: Mencegah OOM di RAM/VRAM

    def get_image_embeddings(self, image_paths):
        all_embeddings = []
        
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i+self.batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.model.get_image_features(**inputs)
                # PERBAIKAN: Ambil vektor hasil proyeksi 512, bukan pooler_output mentah!
                embeddings = out if isinstance(out, torch.Tensor) else out.image_embeds
                
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                all_embeddings.append(embeddings.cpu())
                
            del images, inputs, out, embeddings
            gc.collect()
            
        return torch.cat(all_embeddings, dim=0).to(self.device)

    def get_text_embedding(self, text):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            out = self.model.get_text_features(**inputs)
            # PERBAIKAN: Ambil vektor hasil proyeksi 512, bukan pooler_output mentah!
            embeddings = out if isinstance(out, torch.Tensor) else out.text_embeds
            
        return embeddings / embeddings.norm(dim=-1, keepdim=True)
