"""
Safe split of MM-SafetyBench.
558 safe queries on the same sensitive images as the unsafe split —
much harder safe data than MM-Vet (benign captioning).
"""
import json
import os
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_PATH  = "/data/mmsafety/safe_input"
JSON_PATH  = f"{BASE_PATH}/safe_input_sample.json"


class SafeVLMDataset_MMsafety(Dataset):
    def __init__(self, processor, device="cuda"):
        self.device = device
        self.processor = processor
        self.data = []

        items = json.load(open(JSON_PATH))
        for item in items:
            image_path = f"{BASE_PATH}/{item['image']}"
            if not os.path.exists(image_path):
                continue
            text = item["instr-resp"][0]["safe_instruction"] if item["instr-resp"] else ""
            self.data.append((image_path, text))

        print(f"Loaded SafeVLMDataset_MMsafety: {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, text = self.data[idx]
        image = Image.open(image_path).convert("RGB").resize((224, 224))
        dummy_embedding = torch.zeros(4096, dtype=torch.float32)
        # category=0 marks safe
        return image, text, dummy_embedding, torch.tensor(0, dtype=torch.int64)


def collate_fn(batch):
    images, texts, embeddings, categories = zip(*batch)
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = proc(
        text=list(texts), images=list(images),
        return_tensors="pt", padding="longest",
        truncation=True, max_length=77,
    )
    inputs["safe"]      = torch.ones(len(batch), dtype=torch.int64, device=device)  # safe=1
    inputs["embedding"] = torch.stack(embeddings).to(device)
    inputs["category"]  = torch.tensor(categories, dtype=torch.int64, device=device)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    return inputs


def main(model, processor):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset    = SafeVLMDataset_MMsafety(processor, device)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    return dataset, dataloader
