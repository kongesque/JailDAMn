"""
Paper-faithful JailDAM evaluation.

Changes vs run_full.py to match the paper (arXiv:2504.03770):
  1. Concept embeddings: 30/category (paper optimal 20-40) instead of 100
  2. 3-way safe split: train (AE) / val (threshold) / test (report)
  3. Subsample unsafe to ~528 total (paper's test size)
  4. Subsample safe test to ~218 (paper's test size)
  5. Threshold selected on val set, metrics reported on held-out test set
"""

import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset, random_split

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
    sim_txt, sim_img,          # similarity scores [batch, num_concepts] — for topk index selection
    raw_txt, raw_img,          # raw CLIP embeddings [batch, 768] — for residual computation
    threshold, top_k=5, device="cuda",
):
    # sim_scores: bounded by num_concepts so topk indices are always valid
    # raw_emb:    768-dim, matches the col_slice width (768) for new_concept computation
    def _update(sim_scores, raw_emb, col_slice):
        softmax_probs = torch.nn.functional.softmax(sim_scores, dim=-1)
        max_probs, indices = torch.topk(softmax_probs, top_k, dim=-1)
        updates = []
        for i in range(sim_scores.shape[0]):
            if max_probs[i, 0] > threshold:
                freq_tensor = torch.tensor(list(concept_frequency_total.values()), device=device)
                min_idx = torch.argmin(freq_tensor).item()
                top_concepts = concept_embeddings[indices[i], col_slice.start:col_slice.stop]
                weighted = (max_probs[i].unsqueeze(-1) * top_concepts).sum(dim=0)
                new_concept = raw_emb[i] - weighted
                updates.append((min_idx, col_slice.start, col_slice.stop, new_concept.detach().clone()))
                concept_frequency_total[min_idx] = torch.max(freq_tensor).item() + 1
        for min_idx, start, stop, new_concept in updates:
            concept_embeddings[min_idx, start:stop] = new_concept

    _update(sim_txt, raw_txt, slice(0, 768))
    _update(sim_img, raw_img, slice(768, 1536))
    return concept_embeddings, concept_frequency_total


# -------------------------
# Score extraction (no threshold, no metrics — just labels + scores)
# -------------------------
def extract_scores(dataloaders, concept_embeddings, autoencoder, memory_network, device, concept_frequency_total):
    autoencoder.eval()
    concept_embeddings = concept_embeddings.to(device)

    all_labels, all_scores = [], []
    total_inputs, total_time = 0, 0.0

    with torch.no_grad():
        for dataloader in dataloaders:
            for batch in dataloader:
                t0 = time.time()
                pixel_values   = batch["pixel_values"].to(device)
                input_ids      = batch["input_ids"].to(device)
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

                # sim_scores for topk (bounded by num_concepts), raw embeddings for residual
                concept_embeddings, concept_frequency_total = update_concept_embeddings(
                    concept_embeddings, concept_frequency_total,
                    sim_txt, sim_img, text_emb, img_emb, 0.0004, device=device,
                )

                recon     = autoencoder(features)
                recon_err = torch.mean((recon - features) ** 2, dim=-1)

                binary = np.where(labels > 0, 1, 0)
                all_labels.extend(binary)
                all_scores.extend(recon_err.cpu().numpy())

                total_time   += time.time() - t0
                total_inputs += len(labels)

    return np.array(all_labels), np.array(all_scores), total_time, total_inputs


def find_best_threshold(labels, scores):
    best_f1, best_th = 0.0, 0.0
    for th in np.linspace(scores.min(), scores.max(), 200):
        preds = (scores >= th).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=1)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    return best_th, best_f1


def compute_metrics(labels, scores, threshold):
    auroc = roc_auc_score(labels, scores)
    aupr  = average_precision_score(labels, scores)
    preds = (scores >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=1)
    return {"AUROC": auroc, "AUPR": aupr, "F1": f1, "Precision": p, "Recall": r}


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Seed: {SEED}\n")

    ds_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    ds_model     = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-large-patch14")

    # --- Load all datasets ---
    print("=== Loading datasets ===")
    mmsafety_ds, _ = main_mmsafety(ds_model, ds_processor)
    figstep_ds,  _ = main_figstep(ds_model, ds_processor)
    nano_ds,     _ = main_nano(ds_model, ds_processor)
    safe_ds,     _ = main_safe_mmvet(ds_model, ds_processor)

    print(f"MM-SafetyBench : {len(mmsafety_ds)}")
    print(f"FigStep        : {len(figstep_ds)}")
    print(f"JailbreakV-Nano: {len(nano_ds)}")
    print(f"MM-Vet safe    : {len(safe_ds)}")

    # =================================================================
    # FIX 1: Use full MM-SafetyBench + FigStep (same as paper — these are
    #         complete datasets). JailbreakV-Nano is already our full subset.
    #         Do NOT subsample: subsampling hurts adaptive memory diversity.
    # =================================================================
    mmsafety_sub = mmsafety_ds
    figstep_sub  = figstep_ds
    nano_sub     = nano_ds

    print(f"\n=== Unsafe datasets (full, no subsampling) ===")
    print(f"MM-SafetyBench : {len(mmsafety_sub)}")
    print(f"FigStep        : {len(figstep_sub)}")
    print(f"JailbreakV-Nano: {len(nano_sub)}")
    print(f"Total unsafe   : {len(mmsafety_sub) + len(figstep_sub) + len(nano_sub)}")

    # =================================================================
    # FIX 2: 3-way safe split — train / val (threshold) / test (report)
    #         Paper uses 218 benign for testing
    # =================================================================
    n_safe = len(safe_ds)
    n_test_safe = 218                         # paper's test safe count
    n_val_safe  = 50                          # held-out for threshold selection
    n_train_safe = n_safe - n_test_safe - n_val_safe  # remainder for AE training

    assert n_train_safe > 0, f"Not enough safe samples: {n_safe} total"
    train_safe, val_safe, test_safe = random_split(safe_ds, [n_train_safe, n_val_safe, n_test_safe])

    print(f"\n=== Safe data split ===")
    print(f"Train (AE)     : {len(train_safe)}")
    print(f"Val (threshold): {len(val_safe)}")
    print(f"Test (report)  : {len(test_safe)}")

    # Split each unsafe dataset: small balanced val portion, rest goes to test
    def split_for_val(ds, n_val):
        n = len(ds)
        indices = list(range(n))
        random.shuffle(indices)
        val_n = min(n_val, n)
        return Subset(ds, indices[:val_n]), Subset(ds, indices[val_n:])

    # Allocate n_val_safe unsafe samples proportionally across datasets
    sub_sizes = [len(mmsafety_sub), len(figstep_sub), len(nano_sub)]
    sub_total = sum(sub_sizes)
    val_allocs = [max(1, int(n_val_safe * s / sub_total)) for s in sub_sizes]

    mmsafety_val, mmsafety_test = split_for_val(mmsafety_sub, val_allocs[0])
    figstep_val,  figstep_test  = split_for_val(figstep_sub,  val_allocs[1])
    nano_val,     nano_test     = split_for_val(nano_sub,     val_allocs[2])

    val_unsafe  = ConcatDataset([mmsafety_val,  figstep_val,  nano_val])
    test_unsafe = ConcatDataset([mmsafety_test, figstep_test, nano_test])

    print(f"\nVal unsafe  : {len(val_unsafe)}")
    print(f"Test unsafe : {len(test_unsafe)}")
    print(f"Test total  : {len(test_safe) + len(test_unsafe)}  "
          f"(safe {len(test_safe)} + unsafe {len(test_unsafe)})")

    # =================================================================
    # FIX 3: Concept embeddings — 30/category (paper's optimal range)
    # =================================================================
    samples_per_category = 40   # paper: 20-40 optimal, use upper end
    num_categories       = 13

    print(f"\n=== Building concept embeddings ({samples_per_category}/category) ===")
    concept_embeddings_ori = build_concept_embeddings(samples_per_category * num_categories, device)

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

    categories_to_remove = [1, 2, 3, 4, 5]
    num_categories_train = num_categories - len(categories_to_remove)  # 8

    rows_to_remove = []
    for cat_idx in categories_to_remove:
        rows_to_remove.extend(range(cat_idx * samples_per_category, (cat_idx + 1) * samples_per_category))
    keep_indices = sorted(set(range(concept_embeddings_ori.shape[0])) - set(rows_to_remove))
    reduced = concept_embeddings_ori[keep_indices]
    print(f"Reduced concept embeddings: {reduced.shape}")

    memory_network = MemoryNetwork(
        clip_model=clip_model,
        concept_embeddings=reduced,
        device=device,
    ).to(device)

    # --- Dataloaders ---
    batch_size = 8
    train_dl       = DataLoader(train_safe, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_safe_dl    = DataLoader(val_safe,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    val_unsafe_dl  = DataLoader(val_unsafe, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_safe_dl   = DataLoader(test_safe,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Per-dataset test loaders
    mmsafety_test_dl = DataLoader(mmsafety_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    figstep_test_dl  = DataLoader(figstep_test,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    nano_test_dl     = DataLoader(nano_test,     batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_unsafe_dl   = DataLoader(test_unsafe,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # --- Training ---
    num_epochs       = 5
    learning_rate    = 0.001
    update_threshold = 10
    top_k_update     = 50

    ae_input_dim = num_categories_train * samples_per_category * 2  # 8*30*2 = 480
    autoencoder  = Autoencoder(ae_input_dim).to(device)
    ae_optim     = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    criterion    = nn.MSELoss()

    concept_embeddings = torch.nn.Parameter(
        torch.cat((reduced, reduced), dim=-1).to(device)
    )
    concept_optim = optim.Adam([concept_embeddings], lr=learning_rate)
    sim_loss_fn   = nn.CosineEmbeddingLoss()
    concept_freq  = {i: 0 for i in range(len(concept_embeddings))}

    print(f"\n=== Training autoencoder (safe data only, {len(train_safe)} samples) ===")
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

        sorted_c = sorted(freq_ep.items(), key=lambda x: x[1], reverse=True)
        top_k_c  = [idx for idx, freq in sorted_c[:top_k_update] if freq > update_threshold]

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

    # Save clean state for resetting between eval passes
    concept_embeddings_clean = concept_embeddings.data.clone()
    concept_freq_clean = dict(concept_freq)

    def reset_state():
        concept_embeddings.data.copy_(concept_embeddings_clean)
        concept_freq.clear()
        concept_freq.update(concept_freq_clean)

    # =================================================================
    # FIX 4: Threshold on VAL set, metrics on TEST set
    # =================================================================
    print("\n=== Step 1: Select threshold on validation set ===")
    reset_state()
    val_labels, val_scores, _, _ = extract_scores(
        [val_safe_dl, val_unsafe_dl],
        concept_embeddings, autoencoder, memory_network, device, concept_freq,
    )
    threshold, val_f1 = find_best_threshold(val_labels, val_scores)
    val_m = compute_metrics(val_labels, val_scores, threshold)
    print(f"Val samples: {len(val_labels)} (safe {(val_labels==0).sum()}, unsafe {(val_labels==1).sum()})")
    print(f"Val AUROC: {val_m['AUROC']:.4f}  AUPR: {val_m['AUPR']:.4f}  F1: {val_m['F1']:.4f}")
    print(f"Selected threshold: {threshold:.4f}")

    # =================================================================
    # FIX 5: Report on held-out TEST set with the val-selected threshold
    # =================================================================
    print("\n=== Step 2: Evaluate on held-out test set ===")

    # Overall
    reset_state()
    test_labels, test_scores, t_time, t_inputs = extract_scores(
        [test_safe_dl, test_unsafe_dl],
        concept_embeddings, autoencoder, memory_network, device, concept_freq,
    )
    overall = compute_metrics(test_labels, test_scores, threshold)

    # Per-dataset (each paired with test_safe)
    per_ds = {}
    for name, ds_dl in [("MM-SafetyBench", mmsafety_test_dl),
                         ("FigStep", figstep_test_dl),
                         ("JailbreakV-Nano", nano_test_dl)]:
        reset_state()
        lbl, scr, _, _ = extract_scores(
            [test_safe_dl, ds_dl],
            concept_embeddings, autoencoder, memory_network, device, concept_freq,
        )
        per_ds[name] = compute_metrics(lbl, scr, threshold)

    # --- Print results ---
    print(f"\n{'='*70}")
    print(f"  PAPER-FAITHFUL EVALUATION  (threshold from val set)")
    print(f"{'='*70}")
    print(f"{'Dataset':<20} {'AUROC':>7} {'AUPR':>7} {'F1':>7} {'Prec':>7} {'Recall':>7}")
    print("-" * 62)
    for name, r in per_ds.items():
        print(f"{name:<20} {r['AUROC']:>7.4f} {r['AUPR']:>7.4f} {r['F1']:>7.4f} {r['Precision']:>7.4f} {r['Recall']:>7.4f}")
    print("-" * 62)
    print(f"{'Overall':<20} {overall['AUROC']:>7.4f} {overall['AUPR']:>7.4f} {overall['F1']:>7.4f} {overall['Precision']:>7.4f} {overall['Recall']:>7.4f}")

    print(f"\nTest safe   : {int((test_labels==0).sum())}")
    print(f"Test unsafe : {int((test_labels==1).sum())}")
    print(f"Threshold   : {threshold:.4f} (from val set)")
    print(f"Avg latency : {t_time/t_inputs*1000:.2f} ms/input")

    # Paper comparison
    print(f"\n{'='*70}")
    print(f"  COMPARISON WITH PAPER (Table 2/3)")
    print(f"{'='*70}")
    paper = {"MM-SafetyBench": (0.9472, 0.9155),
             "FigStep":        (0.9608, 0.9616),
             "Overall":        (0.9550, 0.9530)}
    print(f"{'Dataset':<20} {'Ours AUROC':>11} {'Paper AUROC':>12} {'Ours AUPR':>10} {'Paper AUPR':>11}")
    print("-" * 66)
    for name in ["MM-SafetyBench", "FigStep"]:
        ours = per_ds[name]
        pa, pp = paper[name]
        print(f"{name:<20} {ours['AUROC']:>11.4f} {pa:>12.4f} {ours['AUPR']:>10.4f} {pp:>11.4f}")
    pa, pp = paper["Overall"]
    print(f"{'Overall':<20} {overall['AUROC']:>11.4f} {pa:>12.4f} {overall['AUPR']:>10.4f} {pp:>11.4f}")
