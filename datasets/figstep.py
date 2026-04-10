import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor


class UnsafeVLMDataset_fig_step(Dataset):
    def __init__(self, base_path, embedding_base_path, processor, device="cuda"):
        self.device = device
        self.processor = processor
        self.base_path = base_path
        self.embedding_base_path = embedding_base_path
        self.data = []

        json_files = sorted([f for f in os.listdir(base_path) if f.endswith(".json")])
        self.category_map = {filename: i + 1 for i, filename in enumerate(json_files)}
        print("Loaded Categories:", self.category_map)

        for json_file in json_files:
            category_label = self.category_map[json_file]
            json_path = os.path.join(base_path, json_file)

            if not os.path.exists(json_path):
                print(f"Warning: {json_path} not found. Skipping.")
                continue

            with open(json_path, "r") as f:
                category_data = json.load(f)

            embedding_file = os.path.join(embedding_base_path, json_file.replace(".json", ".pt"))
            embeddings = self.load_embeddings(embedding_file)

            embedding_dict = {
                str(item["id"]): item["Rephrased_Question_hidden_states"].mean(dim=1).squeeze().numpy()
                for item in embeddings
                if "Rephrased_Question_hidden_states" in item
            }

            for item in category_data:
                item_id = str(item["id"])

                if "image_path" not in item:
                    print(f"Warning: Missing image_path in JSON entry. Skipping entry: {item}")
                    continue

                image_path = os.path.join(self.base_path, item["image_path"])
                question_text = item.get("Question", "")

                if os.path.exists(image_path):
                    embedding = embedding_dict.get(item_id, torch.zeros(4096))
                    self.data.append((image_path, question_text, embedding, category_label))
                else:
                    print(f"Warning: Image {image_path} not found, skipping.")

    def load_embeddings(self, file_path):
        if not os.path.exists(file_path):
            print(f"Warning: Embedding file {file_path} not found.")
            return []
        try:
            return torch.load(file_path, map_location="cpu")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, question_text, embedding, category = self.data[idx]
        image = Image.open(image_path).convert("RGB").resize((224, 224))
        return image, question_text, torch.tensor(embedding, dtype=torch.float32), torch.tensor(category, dtype=torch.int64)


def main(model, processor):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_path = "/data/fig_step"
    embedding_base_path = "/data/mmsafety/unsafe_input/weights"
    unsafe_dataset = UnsafeVLMDataset_fig_step(base_path, embedding_base_path, processor, device)
    return unsafe_dataset, None
