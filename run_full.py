"""
Full JailDAM evaluation mirroring demo.ipynb structure, with:
  - Safe:   MM-Vet (517 samples)
  - Unsafe: MM-SafetyBench + FigStep + JailbreakV-Nano (combined)
Fixed seed=42, all unsafe samples used for OOD evaluation.
"""

import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
)
from transformers import (
    CLIPModel,
    CLIPProcessor,
    AutoProcessor,
    AutoModelForZeroShotImageClassification,
)

from datasets.mmsafety import main as main_mmsafety
from datasets.figstep import main as main_figstep
from datasets.jailbreakv_nano import main as main_nano
from datasets.mmvet import main as main_safe_mmvet
from memory_network import MemoryNetwork

# -------------------------
# Fixed seed
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# -------------------------
# Concept embedding generation
# -------------------------
def build_concept_embeddings(concept_numbers, device):
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    with open("concept.json", "r") as f:
        concept_dict = json.load(f)

    merged_list = [
        item.strip("[]")
        for key, value in concept_dict.items()
        for item in value.split(", ")
    ]
    sampled = random.sample(merged_list, concept_numbers)
    merged_list = [random.choice(sampled) for _ in range(concept_numbers)]

    text_inputs = processor(text=merged_list, return_tensors="pt", padding=True).to(device)
    text_embeddings = clip_model.get_text_features(**text_inputs)
    if not isinstance(text_embeddings, torch.Tensor):
        text_embeddings = text_embeddings.pooler_output

    concept_embeddings = F.normalize(text_embeddings, p=2, dim=1).detach().cpu()
    del clip_model
    torch.cuda.empty_cache()
    return concept_embeddings


# -------------------------
# Collate function
# -------------------------
def collate_fn(batch):
    images, texts, embeddings, categories = zip(*batch)
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = proc(
        text=list(texts),
        images=list(images),
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=77,
    )
    inputs["safe"] = torch.zeros(len(batch), dtype=torch.int64, device=device)
    inputs["embedding"] = torch.stack(embeddings).to(device)
    inputs["category"] = torch.tensor(categories, dtype=torch.int64, device=device)
    inputs["texts"] = list(texts)
    inputs["images"] = list(images)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    return inputs


# -------------------------
# Autoencoder
# -------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# -------------------------
# Concept embedding updater
# -------------------------
def update_concept_embeddings(
    concept_embeddings, concept_frequency_total,
    attention_features_text, attention_features_image,
    threshold, top_k=5, device="cuda",
):
    def _update(features, col_slice):
        softmax_probs = torch.nn.functional.softmax(features, dim=-1)
        max_probs, indices = torch.topk(softmax_probs, top_k, dim=-1)
        updates = []
        for i in range(features.shape[0]):
            if max_probs[i, 0] > threshold:
                freq_tensor = torch.tensor(list(concept_frequency_total.values()), device=device)
                min_idx = torch.argmin(freq_tensor).item()
                top_concepts = concept_embeddings[indices[i], col_slice.start:col_slice.stop]
                weighted = (max_probs[i].unsqueeze(-1) * top_concepts).sum(dim=0)
                new_concept = features[i] - weighted
                updates.append((min_idx, col_slice.start, col_slice.stop, new_concept.detach().clone()))
                concept_frequency_total[min_idx] = torch.max(freq_tensor).item() + 1
        for min_idx, start, stop, new_concept in updates:
            concept_embeddings[min_idx, start:stop] = new_concept

    _update(attention_features_text, slice(0, 768))
    _update(attention_features_image, slice(768, 1536))
    return concept_embeddings, concept_frequency_total


# -------------------------
# Evaluation (overall + per-dataset)
# -------------------------
def evaluate(dataloaders, dataset_names, concept_embeddings, autoencoder, memory_network, device, concept_frequency_total):
    autoencoder.eval()
    concept_embeddings = concept_embeddings.to(device)

    all_labels, all_scores = [], []
    per_dataset = {}
    total_inputs, total_time = 0, 0.0

    with torch.no_grad():
        for dataloader, name in zip(dataloaders, dataset_names):
            t_dl = time.time()
            ds_labels, ds_scores = [], []

            for batch in dataloader:
                t0 = time.time()
                pixel_values  = batch["pixel_values"].to(device)
                input_ids     = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["category"].cpu().numpy()

                _, _, text_emb, img_emb, _, _ = memory_network.forward(
                    text_input_ids=input_ids,
                    text_attention_mask=attention_mask,
                    image_pixel_values=pixel_values,
                )

                sim_img  = img_emb  @ concept_embeddings[:, :768].T
                sim_txt  = text_emb @ concept_embeddings[:, 768:].T
                features = torch.cat((sim_txt, sim_img), dim=-1)

                concept_embeddings, concept_frequency_total = update_concept_embeddings(
                    concept_embeddings, concept_frequency_total,
                    text_emb, img_emb, 0.0004, device=device,
                )

                recon     = autoencoder(features)
                recon_err = torch.mean((recon - features) ** 2, dim=-1)

                binary = np.where(labels > 0, 1, 0)
                ds_labels.extend(binary)
                ds_scores.extend(recon_err.cpu().numpy())
                all_labels.extend(binary)
                all_scores.extend(recon_err.cpu().numpy())

                total_time   += time.time() - t0
                total_inputs += len(labels)

            elapsed = time.time() - t_dl
            print(f"  [{name}] done in {elapsed:.1f}s  ({len(ds_labels)} samples)")
            per_dataset[name] = (np.array(ds_labels), np.array(ds_scores))

    def metrics(labels, scores):
        auroc = roc_auc_score(labels, scores)
        aupr  = average_precision_score(labels, scores)
        best_f1, best_th = 0.0, 0.0
        for th in np.linspace(scores.min(), scores.max(), 100):
            preds = (scores >= th).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=1)
            if f1 > best_f1:
                best_f1, best_th = f1, th
        final = (np.array(scores) >= best_th).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(labels, final, average="binary", zero_division=1)
        return {"AUROC": auroc, "AUPR": aupr, "F1": f1, "Precision": p, "Recall": r, "Threshold": best_th}

    results = {"overall": metrics(np.array(all_labels), np.array(all_scores))}
    results["overall"]["Avg ms/input"] = (total_time / total_inputs * 1000) if total_inputs else 0
    results["overall"]["N_safe"]   = int((np.array(all_labels) == 0).sum())
    results["overall"]["N_unsafe"] = int((np.array(all_labels) == 1).sum())

    for name, (lbl, scr) in per_dataset.items():
        if len(np.unique(lbl)) > 1:
            results[name] = metrics(lbl, scr)

    return results


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Seed: {SEED}\n")

    ds_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    ds_model     = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-large-patch14")

    print("=== Loading datasets ===")
    mmsafety_ds, _  = main_mmsafety(ds_model, ds_processor)
    figstep_ds,  _  = main_figstep(ds_model, ds_processor)
    nano_ds,     _  = main_nano(ds_model, ds_processor)
    safe_ds,     _  = main_safe_mmsafety(ds_model, ds_processor)

    unsafe_combined = ConcatDataset([mmsafety_ds, figstep_ds, nano_ds])
    print(f"\nMM-SafetyBench : {len(mmsafety_ds)}")
    print(f"FigStep        : {len(figstep_ds)}")
    print(f"JailbreakV-Nano: {len(nano_ds)}")
    print(f"Total unsafe   : {len(unsafe_combined)}")
    print(f"MMsafety safe  : {len(safe_ds)}  (hard safe — same domain as jailbreaks)")

    # Concept embeddings
    samples_per_category = 100
    num_categories       = 13
    print("\n=== Building concept embeddings (seed=42) ===")
    concept_embeddings_ori = build_concept_embeddings(samples_per_category * num_categories, device)

    # CLIP for memory network
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

    total_per_category   = 100
    categories_to_remove = [1, 2, 3, 4, 5]
    num_categories_train = 8

    rows_to_remove = []
    for cat_idx in categories_to_remove:
        rows_to_remove.extend(range(cat_idx * total_per_category, (cat_idx + 1) * total_per_category))
    keep_indices = sorted(set(range(concept_embeddings_ori.shape[0])) - set(rows_to_remove))
    reduced = concept_embeddings_ori[keep_indices]
    print(f"Reduced concept embeddings: {reduced.shape}")

    memory_network = MemoryNetwork(
        clip_model=clip_model,
        concept_embeddings=reduced,
        device=device,
    ).to(device)

    # Train/val split on safe data only
    train_size = int(0.8 * len(safe_ds))
    val_size   = len(safe_ds) - train_size
    train_safe, val_safe = random_split(safe_ds, [train_size, val_size])
    print(f"\nTrain safe: {train_size}  |  Val safe: {val_size}  |  Total OOD unsafe: {len(unsafe_combined)}")

    batch_size = 8
    train_dl     = DataLoader(train_safe,       batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_dl       = DataLoader(val_safe,         batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    mmsafety_dl  = DataLoader(mmsafety_ds,      batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    figstep_dl   = DataLoader(figstep_ds,       batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    nano_dl      = DataLoader(nano_ds,          batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Training
    num_epochs       = 5
    learning_rate    = 0.001
    update_threshold = 10
    top_k_update     = 50

    ae_input_dim = num_categories_train * samples_per_category * 2  # 1600
    autoencoder  = Autoencoder(ae_input_dim).to(device)
    ae_optim     = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    criterion    = nn.MSELoss()

    concept_embeddings = torch.nn.Parameter(
        torch.cat((reduced, reduced), dim=-1).to(device)
    )
    concept_optim  = optim.Adam([concept_embeddings], lr=learning_rate)
    sim_loss_fn    = nn.CosineEmbeddingLoss()
    concept_freq   = {i: 0 for i in range(len(concept_embeddings))}

    print("\n=== Training autoencoder (safe data only) ===")
    for epoch in range(num_epochs):
        autoencoder.train()
        total_loss, concept_loss_total = 0.0, 0.0
        freq_ep = {i: 0 for i in range(len(concept_embeddings))}
        accum   = {i: [] for i in range(len(concept_embeddings))}

        for batch in train_dl:
            ae_optim.zero_grad()
            concept_optim.zero_grad()

            pixel_values   = batch["pixel_values"].to(device)
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            _, _, text_emb, img_emb, _, _ = memory_network.forward(
                text_input_ids=input_ids,
                text_attention_mask=attention_mask,
                image_pixel_values=pixel_values,
            )

            sim_img  = img_emb  @ concept_embeddings[:, :768].T
            sim_txt  = text_emb @ concept_embeddings[:, 768:].T
            features = torch.cat((sim_txt, sim_img), dim=-1)
            recon    = autoencoder(features)
            ae_loss  = criterion(recon, features)

            top5_img = sim_img.topk(5, dim=-1).indices
            top5_txt = sim_txt.topk(5, dim=-1).indices
            for i in range(features.size(0)):
                matched = torch.cat((text_emb[i], img_emb[i]), dim=-1)
                for idx in top5_img[i].tolist() + top5_txt[i].tolist():
                    freq_ep[idx]      += 1
                    concept_freq[idx] += 1
                    accum[idx].append(matched.detach())

            ae_loss.backward(retain_graph=True)
            ae_optim.step()
            total_loss += ae_loss.item()

        sorted_c  = sorted(freq_ep.items(), key=lambda x: x[1], reverse=True)
        top_k_c   = [idx for idx, freq in sorted_c[:top_k_update] if freq > update_threshold]

        if top_k_c:
            concept_optim.zero_grad()
            closs = 0.0
            for idx in top_k_c:
                if accum[idx]:
                    matches  = torch.stack(accum[idx])
                    mean_vec = matches.mean(dim=0).detach()
                    var_vec  = (((matches - mean_vec) ** 2).mean(dim=0).detach()
                                if matches.size(0) > 1 else torch.zeros_like(mean_vec))
                    cvec   = concept_embeddings[idx].unsqueeze(0)
                    target = torch.ones(1, device=device)
                    closs += 5.0 * sim_loss_fn(cvec, mean_vec.unsqueeze(0), target) * (1 + var_vec.mean())
            (closs / len(top_k_c)).backward()
            concept_optim.step()
            concept_loss_total += closs.item()

        print(f"Epoch [{epoch+1}/{num_epochs}]  "
              f"AE Loss: {total_loss/len(train_dl):.4f}  "
              f"Concept Loss: {concept_loss_total/len(train_dl):.4f}")

    print("\nTraining complete.")

    # Save clean state before evaluation — adaptive memory update mutates concept_embeddings
    # across runs, so we reset between each evaluation pass
    concept_embeddings_clean = concept_embeddings.data.clone()
    concept_freq_clean = dict(concept_freq)

    def reset_state():
        concept_embeddings.data.copy_(concept_embeddings_clean)
        concept_freq.clear()
        concept_freq.update(concept_freq_clean)

    # Per-dataset evaluation: pair val_safe with each unsafe dataset so AUROC has both classes
    print("\n=== Evaluating (per-dataset: val_safe + each unsafe) ===")
    per_ds_configs = [
        ([val_dl, mmsafety_dl], ["val_safe", "MM-SafetyBench"]),
        ([val_dl, figstep_dl],  ["val_safe", "FigStep"]),
        ([val_dl, nano_dl],     ["val_safe", "JailbreakV-Nano"]),
    ]

    per_ds_results = {}
    for loaders, names in per_ds_configs:
        ds_name = names[1]
        print(f"\n  -- {ds_name} --")
        reset_state()
        r = evaluate(loaders, names, concept_embeddings, autoencoder, memory_network, device, concept_freq)
        per_ds_results[ds_name] = r["overall"]

    # Overall: val_safe + all unsafe combined
    print("\n=== Evaluating (overall: val_safe + all unsafe) ===")
    reset_state()
    all_loaders = [val_dl, mmsafety_dl, figstep_dl, nano_dl]
    all_names   = ["val_safe", "MM-SafetyBench", "FigStep", "JailbreakV-Nano"]
    overall_res = evaluate(all_loaders, all_names, concept_embeddings, autoencoder, memory_network, device, concept_freq)

    print("\n=== Results ===")
    print(f"{'Dataset':<20} {'AUROC':>7} {'AUPR':>7} {'F1':>7} {'Prec':>7} {'Recall':>7}")
    print("-" * 60)
    for name, r in per_ds_results.items():
        print(f"{name:<20} {r['AUROC']:>7.4f} {r['AUPR']:>7.4f} {r['F1']:>7.4f} {r['Precision']:>7.4f} {r['Recall']:>7.4f}")
    print("-" * 60)
    ov = overall_res["overall"]
    print(f"{'Overall':<20} {ov['AUROC']:>7.4f} {ov['AUPR']:>7.4f} {ov['F1']:>7.4f} {ov['Precision']:>7.4f} {ov['Recall']:>7.4f}")
    print(f"\nSafe evaluated : {ov['N_safe']}")
    print(f"Unsafe total   : {ov['N_unsafe']}")
    print(f"Avg latency    : {ov['Avg ms/input']:.2f} ms/input")
