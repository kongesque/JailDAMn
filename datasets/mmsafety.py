import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor

CATEGORIES = [
    "01-Illegal_Activity",
    "02-HateSpeech",
    "03-Malware_Generation",
    "04-Physical_Harm",
    "05-EconomicHarm",
    "06-Fraud",
    "07-Sex",
    "08-Political_Lobbying",
    "09-Privacy_Violence",
    "10-Legal_Opinion",
    "11-Financial_Advice",
    "12-Health_Consultation",
    "13-Gov_Decision",
]


class UnsafeVLMDataset_MMsafety(Dataset):
    def __init__(self, base_path, embedding_base_path, categories, processor, device="cuda"):
        self.device = device
        self.processor = processor
        self.base_path = base_path
        self.embedding_base_path = embedding_base_path
        self.image_base_path = "/data/mmsafety/imgs"
        self.data = []

        category_map = {category: i + 1 for i, category in enumerate(categories)}

        for category in categories:
            category_label = category_map[category]

            for attack_type in ["attack_failure.json", "attack_success.json"]:
                json_path = os.path.join(base_path, "unsafe_input", "sample", category, attack_type)

                if not os.path.exists(json_path):
                    print(f"Warning: {json_path} not found. Skipping {attack_type}.")
                    continue

                with open(json_path, "r") as f:
                    category_data = json.load(f)

                embedding_file = os.path.join(embedding_base_path, category, attack_type.replace(".json", ".pt"))
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

                    image_path = os.path.join(self.image_base_path, item["image_path"].split("images/")[1])
                    question_text = item.get("Rephrased Question(SD)", "")

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
    base_path = "/data/mmsafety"
    embedding_base_path = "/data/mmsafety/unsafe_input/weights"
    unsafe_dataset = UnsafeVLMDataset_MMsafety(base_path, embedding_base_path, CATEGORIES, processor, device)
    return unsafe_dataset, None
