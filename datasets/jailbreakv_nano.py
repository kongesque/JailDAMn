import os
import json
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_PATH = "/data/jailbreakv_nano"


class UnsafeVLMDataset_jailbreakv_nano(Dataset):
    def __init__(self, base_path, processor, device="cuda"):
        self.device = device
        self.processor = processor
        self.base_path = base_path
        self.data = []

        json_files = sorted([f for f in os.listdir(base_path) if f.endswith(".json")])
        self.category_map = {filename: i + 1 for i, filename in enumerate(json_files)}
        print("Loaded Categories:", self.category_map)

        for json_file in json_files:
            category_label = self.category_map[json_file]
            json_path = os.path.join(base_path, json_file)

            with open(json_path, "r") as f:
                category_data = json.load(f)

            for item in category_data:
                if "image_path" not in item:
                    print(f"Warning: Missing image_path in {json_file}, skipping entry.")
                    continue

                image_path = os.path.join(base_path, item["image_path"])
                question_text = item.get("jailbreak_query", item.get("redteam_query", ""))

                if os.path.exists(image_path):
                    self.data.append((image_path, question_text, category_label))
                else:
                    print(f"Warning: Image {image_path} not found, skipping.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, question_text, category = self.data[idx]
        image = Image.open(image_path).convert("RGB").resize((224, 224))
        dummy_embedding = torch.zeros(4096, dtype=torch.float32)
        return image, question_text, dummy_embedding, torch.tensor(category, dtype=torch.int64)


def main(model, processor):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    unsafe_dataset = UnsafeVLMDataset_jailbreakv_nano(BASE_PATH, processor, device)
    return unsafe_dataset, None
