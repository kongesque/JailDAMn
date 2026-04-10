import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor


class VLMDataset_mmvet(Dataset):
    def __init__(self, json_path, image_base_dir, processor, device="cuda"):
        self.device = device
        self.processor = processor
        self.image_base_dir = image_base_dir
        self.data = []

        with open(json_path, "r") as f:
            json_data = json.load(f)

        for item in json_data:
            image_path = os.path.join(self.image_base_dir, item["image_path"].split("images/")[1])

            if os.path.exists(image_path):
                safe_instruction = item["question"]
                embedding = torch.tensor(np.random.rand(4096), dtype=torch.float32)
                self.data.append((image_path, safe_instruction, embedding, 0))
            else:
                print(f"Warning: Image {image_path} not found, skipping.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, safe_instruction, embedding, category = self.data[idx]
        image = Image.open(image_path).convert("RGB").resize((224, 224))
        return image, safe_instruction, embedding, torch.tensor(category, dtype=torch.int64)


def main(model, processor):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_base_dir = "/data/mm-vet/images"
    json_path = "/data/mm-vet/sample.json"
    safe_dataset = VLMDataset_mmvet(json_path, image_base_dir, processor, device)
    return safe_dataset, None
