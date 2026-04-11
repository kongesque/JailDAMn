"""
Text-only JailDAM evaluation on BeaverTails.

Keeps the paper's concept memory + soft attention architecture.
Replaces the autoencoder (anomaly detection) with a supervised MLP classifier,
because safe and unsafe text are not separable by reconstruction error in
CLIP embedding space.

Architecture (mirrors paper's MemoryNetwork: memory output + raw embedding → MLP):
  text prompt
    -> CLIP text encoder -> text_emb (768-dim)
    -> cosine similarity to concept memory -> sim_txt (num_concepts-dim)
    -> concat(text_emb, sim_txt) -> [batch, 768 + num_concepts]
    -> MLP classifier -> P(unsafe)
    -> test-time: adaptive concept update (same as paper)

Changes from run_paper_eval.py:
  - Image branch removed (text-only)
  - Autoencoder replaced with supervised binary classifier (BCE loss)
  - Classifier input: raw text embedding + concept similarity features
  - Training uses both safe AND unsafe data (not safe-only)
  - Concept memory + attention + adaptive update preserved

Dataset: PKU-Alignment/BeaverTails (330k_train for train/val, 330k_test for test)
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
from transformers import CLIPModel, CLIPTokenizer

from beavertails_loader import BeaverTailsDataset

# -------------------------
# Fixed seed
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

token = os.getenv('HF_TOKEN')
CLIP_NAME  = "openai/clip-vit-large-patch14"


# -------------------------
# Concept embedding generation  (text-only)
# -------------------------
def build_concept_embeddings(concept_numbers, device):
    clip_model = CLIPModel.from_pretrained(CLIP_NAME).to(device)
    tokenizer  = CLIPTokenizer.from_pretrained(CLIP_NAME)

    with open("concept.json", "r") as f:
        concept_dict = json.load(f)

    merged_list = [
        item.strip("[]")
        for key, value in concept_dict.items()
        for item in value.split(", ")
    ]
    sampled     = random.sample(merged_list, concept_numbers)
    merged_list = [random.choice(sampled) for _ in range(concept_numbers)]

    text_inputs = tokenizer(
        merged_list, return_tensors="pt", padding=True, truncation=True, max_length=77
    ).to(device)
    text_embeddings = clip_model.get_text_features(**text_inputs)
    if not isinstance(text_embeddings, torch.Tensor):
        text_embeddings = text_embeddings.pooler_output

    concept_embeddings = F.normalize(text_embeddings, p=2, dim=1).detach().cpu()
    del clip_model
    torch.cuda.empty_cache()
    return concept_embeddings  # [concept_numbers, 768]


# -------------------------
# Collate function  (text-only)
# -------------------------
def collate_fn(batch):
    texts, categories = zip(*batch)
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_NAME)
    device    = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = tokenizer(
        list(texts),
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=77,
    )
    inputs["category"] = torch.tensor(categories, dtype=torch.int64, device=device)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    return inputs


# -------------------------
# Classifier  (replaces Autoencoder — supervised on concept similarity features)
# -------------------------
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # [batch] logits


# -------------------------
# Concept embedding updater  (text-only: single branch, no col_slice)
# -------------------------
def update_concept_embeddings(
    concept_embeddings, concept_frequency_total,
    sim_txt,   # [batch, num_concepts]
    raw_txt,   # [batch, 768]
    threshold, top_k=5, device="cuda",
):
    softmax_probs = F.softmax(sim_txt, dim=-1)
    max_probs, indices = torch.topk(softmax_probs, top_k, dim=-1)
    updates = []
    for i in range(sim_txt.shape[0]):
        if max_probs[i, 0] > threshold:
            freq_tensor = torch.tensor(list(concept_frequency_total.values()), device=device)
            min_idx     = torch.argmin(freq_tensor).item()
            top_concepts = concept_embeddings[indices[i]]          # [top_k, 768]
            weighted     = (max_probs[i].unsqueeze(-1) * top_concepts).sum(dim=0)
            new_concept  = raw_txt[i] - weighted
            updates.append((min_idx, new_concept.detach().clone()))
            concept_frequency_total[min_idx] = torch.max(freq_tensor).item() + 1
    for min_idx, new_concept in updates:
        concept_embeddings[min_idx] = new_concept
    return concept_embeddings, concept_frequency_total


# -------------------------
# Encode text helper
# -------------------------
def encode_text(clip_model, input_ids, attention_mask):
    with torch.no_grad():
        text_emb = clip_model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )
    if not isinstance(text_emb, torch.Tensor):
        text_emb = text_emb.pooler_output
    return F.normalize(text_emb, p=2, dim=-1)


# -------------------------
# Score extraction  (classifier P(unsafe) as score instead of recon error)
# -------------------------
def extract_scores(dataloaders, concept_embeddings, classifier, clip_model, device, concept_frequency_total):
    classifier.eval()
    concept_embeddings = concept_embeddings.to(device)

    all_labels, all_scores = [], []
    total_inputs, total_time = 0, 0.0

    with torch.no_grad():
        for dataloader in dataloaders:
            for batch in dataloader:
                t0             = time.time()
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["category"].cpu().numpy()

                text_emb = encode_text(clip_model, input_ids, attention_mask)
                sim_txt  = text_emb @ concept_embeddings.T

                concept_embeddings, concept_frequency_total = update_concept_embeddings(
                    concept_embeddings, concept_frequency_total,
                    sim_txt, text_emb, 0.0004, device=device,
                )

                logits = classifier(sim_txt)
                scores = torch.sigmoid(logits)  # P(unsafe)

                binary = np.where(labels > 0, 1, 0)
                all_labels.extend(binary)
                all_scores.extend(scores.cpu().numpy())

                total_time   += time.time() - t0
                total_inputs += len(labels)

    return np.array(all_labels), np.array(all_scores), total_time, total_inputs


def extract_scores_tagged(tagged_dataloaders, concept_embeddings, classifier, clip_model, device, concept_frequency_total):
    """
    Joint forward pass over all dataloaders, tracking per-tag labels and scores.
    Adaptive memory updates continuously across all batches from all tags.
    """
    classifier.eval()
    concept_embeddings = concept_embeddings.to(device)

    tag_labels   = {}
    tag_scores   = {}
    total_inputs, total_time = 0, 0.0

    with torch.no_grad():
        for tag, dataloader in tagged_dataloaders:
            t_labels, t_scores = [], []
            for batch in dataloader:
                t0             = time.time()
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["category"].cpu().numpy()

                text_emb = encode_text(clip_model, input_ids, attention_mask)
                sim_txt  = text_emb @ concept_embeddings.T

                concept_embeddings, concept_frequency_total = update_concept_embeddings(
                    concept_embeddings, concept_frequency_total,
                    sim_txt, text_emb, 0.0004, device=device,
                )

                logits = classifier(sim_txt)
                scores = torch.sigmoid(logits)

                binary = np.where(labels > 0, 1, 0)
                t_labels.extend(binary)
                t_scores.extend(scores.cpu().numpy())

                total_time   += time.time() - t0
                total_inputs += len(labels)

            tag_labels[tag] = np.array(t_labels)
            tag_scores[tag] = np.array(t_scores)

    return tag_labels, tag_scores, total_time, total_inputs


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

    # --- Load datasets ---
    # Train/val from 330k_train, test from 330k_test (proper held-out)
    print("=== Loading datasets ===")
    train_safe_ds   = BeaverTailsDataset("330k_train", token=HF_TOKEN, safe_only=True,   max_samples=5000)
    train_unsafe_ds = BeaverTailsDataset("330k_train", token=HF_TOKEN, unsafe_only=True, max_samples=5000)
    test_safe_ds    = BeaverTailsDataset("330k_test",  token=HF_TOKEN, safe_only=True,   max_samples=2000)
    test_unsafe_ds  = BeaverTailsDataset("330k_test",  token=HF_TOKEN, unsafe_only=True, max_samples=2000)

    print(f"Train safe   : {len(train_safe_ds)}")
    print(f"Train unsafe : {len(train_unsafe_ds)}")
    print(f"Test safe    : {len(test_safe_ds)}")
    print(f"Test unsafe  : {len(test_unsafe_ds)}")

    # =================================================================
    # Split train data: classifier training / val (threshold selection)
    # =================================================================
    n_val = 500   # balanced val set for threshold

    def split_for_val(ds, n_val):
        n       = len(ds)
        indices = list(range(n))
        random.shuffle(indices)
        val_n   = min(n_val, n)
        return Subset(ds, indices[val_n:]), Subset(ds, indices[:val_n])

    train_safe, val_safe     = split_for_val(train_safe_ds,   n_val)
    train_unsafe, val_unsafe = split_for_val(train_unsafe_ds, n_val)

    # Combined training set (safe + unsafe, labeled)
    train_ds = ConcatDataset([train_safe, train_unsafe])

    print(f"\n=== Data split ===")
    print(f"Classifier train : {len(train_ds)} (safe {len(train_safe)} + unsafe {len(train_unsafe)})")
    print(f"Val (threshold)  : {len(val_safe) + len(val_unsafe)} (safe {len(val_safe)} + unsafe {len(val_unsafe)})")
    print(f"Test (report)    : {len(test_safe_ds) + len(test_unsafe_ds)} (safe {len(test_safe_ds)} + unsafe {len(test_unsafe_ds)})")

    # =================================================================
    # Concept embeddings
    # =================================================================
    samples_per_category = 40
    num_categories       = 13

    print(f"\n=== Building concept embeddings ({samples_per_category}/category) ===")
    concept_embeddings_ori = build_concept_embeddings(samples_per_category * num_categories, device)

    clip_model = CLIPModel.from_pretrained(CLIP_NAME).to(device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    categories_to_remove  = [1, 2, 3, 4, 5]
    num_categories_train  = num_categories - len(categories_to_remove)  # 8

    rows_to_remove = []
    for cat_idx in categories_to_remove:
        rows_to_remove.extend(range(cat_idx * samples_per_category, (cat_idx + 1) * samples_per_category))
    keep_indices = sorted(set(range(concept_embeddings_ori.shape[0])) - set(rows_to_remove))
    reduced = concept_embeddings_ori[keep_indices]
    num_concepts = len(reduced)
    print(f"Concept embeddings: {reduced.shape}")

    # --- Dataloaders ---
    batch_size = 32
    train_dl       = DataLoader(train_ds,       batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_safe_dl    = DataLoader(val_safe,        batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    val_unsafe_dl  = DataLoader(val_unsafe,      batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_safe_dl   = DataLoader(test_safe_ds,    batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_unsafe_dl = DataLoader(test_unsafe_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # --- Classifier + concept embeddings ---
    num_epochs       = 10
    learning_rate    = 0.001
    update_threshold = 10
    top_k_update     = 50

    classifier = Classifier(num_concepts).to(device)
    clf_optim  = optim.Adam(classifier.parameters(), lr=learning_rate)
    criterion  = nn.BCEWithLogitsLoss()

    concept_embeddings = nn.Parameter(reduced.to(device))
    concept_optim  = optim.Adam([concept_embeddings], lr=learning_rate)
    sim_loss_fn    = nn.CosineEmbeddingLoss()
    concept_freq   = {i: 0 for i in range(num_concepts)}

    print(f"\n=== Training classifier + concept memory ({len(train_ds)} samples, {num_epochs} epochs) ===")
    for epoch in range(num_epochs):
        classifier.train()
        total_loss, concept_loss_total = 0.0, 0.0
        correct, total = 0, 0
        freq_ep = {i: 0 for i in range(num_concepts)}
        accum   = {i: [] for i in range(num_concepts)}

        for batch in train_dl:
            clf_optim.zero_grad()
            concept_optim.zero_grad()

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["category"].to(device)
            binary_labels  = (labels > 0).float()  # 0=safe, 1=unsafe

            text_emb = encode_text(clip_model, input_ids, attention_mask)
            sim_txt  = text_emb @ concept_embeddings.T   # [batch, num_concepts]

            # Classifier loss
            logits   = classifier(sim_txt)
            clf_loss = criterion(logits, binary_labels)

            # Track top-5 concept hits for concept update
            top5_txt = sim_txt.topk(5, dim=-1).indices
            for i in range(sim_txt.size(0)):
                for idx in top5_txt[i].tolist():
                    freq_ep[idx]      += 1
                    concept_freq[idx] += 1
                    accum[idx].append(text_emb[i].detach())

            clf_loss.backward()
            clf_optim.step()
            total_loss += clf_loss.item()

            preds    = (logits > 0).long()
            correct += (preds == binary_labels.long()).sum().item()
            total   += len(binary_labels)

        # Concept embedding update (same as paper)
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

        acc = correct / total * 100
        print(f"Epoch [{epoch+1}/{num_epochs}]  "
              f"Loss: {total_loss/len(train_dl):.4f}  "
              f"Acc: {acc:.1f}%  "
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
    # Threshold on VAL set
    # =================================================================
    print("\n=== Step 1: Select threshold on validation set ===")
    reset_state()
    val_labels, val_scores, _, _ = extract_scores(
        [val_safe_dl, val_unsafe_dl],
        concept_embeddings, classifier, clip_model, device, concept_freq,
    )
    threshold, val_f1 = find_best_threshold(val_labels, val_scores)
    val_m = compute_metrics(val_labels, val_scores, threshold)
    print(f"Val samples: {len(val_labels)} (safe {(val_labels==0).sum()}, unsafe {(val_labels==1).sum()})")
    print(f"Val AUROC: {val_m['AUROC']:.4f}  AUPR: {val_m['AUPR']:.4f}  F1: {val_m['F1']:.4f}")
    print(f"Selected threshold: {threshold:.4f}")

    # =================================================================
    # Report on held-out TEST set
    # =================================================================
    print("\n=== Step 2: Evaluate on held-out test set (joint pass) ===")
    reset_state()
    tag_labels, tag_scores, t_time, t_inputs = extract_scores_tagged(
        [
            ("safe",   test_safe_dl),
            ("unsafe", test_unsafe_dl),
        ],
        concept_embeddings, classifier, clip_model, device, concept_freq,
    )

    safe_lbl = tag_labels["safe"]
    safe_scr = tag_scores["safe"]
    test_lbl = np.concatenate([safe_lbl, tag_labels["unsafe"]])
    test_scr = np.concatenate([safe_scr, tag_scores["unsafe"]])
    overall  = compute_metrics(test_lbl, test_scr, threshold)

    # --- Print results ---
    print(f"\n{'='*60}")
    print(f"  BEAVERTAILS TEXT-ONLY EVALUATION  (threshold from val set)")
    print(f"{'='*60}")
    print(f"{'Metric':<12} {'Value':>8}")
    print("-" * 22)
    for k, v in overall.items():
        print(f"{k:<12} {v:>8.4f}")

    print(f"\nTest safe   : {int((test_lbl==0).sum())}")
    print(f"Test unsafe : {int((test_lbl==1).sum())}")
    print(f"Threshold   : {threshold:.4f} (from val set)")
    print(f"Avg latency : {t_time/t_inputs*1000:.2f} ms/input")
