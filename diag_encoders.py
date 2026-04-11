"""
Deep diagnostic: which text encoder separates safe from unsafe BeaverTails text?
"""

# --- Bootstrap: import HF-dependent packages before local datasets/ shadows them ---
import os, sys
_proj = os.path.dirname(os.path.abspath(__file__))
_saved = sys.path[:]
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != _proj]
from datasets import load_dataset as _hf_load
from sentence_transformers import SentenceTransformer
sys.path[:] = _saved
# -----------------------------------------------------------------------------------

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, json, random, warnings
warnings.filterwarnings("ignore")
random.seed(42); np.random.seed(42); torch.manual_seed(42)

from transformers import CLIPModel, CLIPTokenizer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

device = "cuda" if torch.cuda.is_available() else "cpu"
token = os.getenv('HF_TOKEN')
N = 1000

# --- Load BeaverTails data directly ---
print("=== Loading BeaverTails ===")
raw = _hf_load("PKU-Alignment/BeaverTails", split="330k_train", token=HF_TOKEN)

safe_texts, unsafe_texts = [], []
for item in raw:
    if item["is_safe"] and len(safe_texts) < N:
        safe_texts.append(item["prompt"])
    elif not item["is_safe"] and len(unsafe_texts) < N:
        unsafe_texts.append(item["prompt"])
    if len(safe_texts) >= N and len(unsafe_texts) >= N:
        break

all_texts  = safe_texts + unsafe_texts
all_labels = np.array([0]*len(safe_texts) + [1]*len(unsafe_texts))
print(f"  {len(safe_texts)} safe, {len(unsafe_texts)} unsafe\n")

# --- Load concept strings ---
with open("concept.json") as f: cd = json.load(f)
concept_strs = [item.strip("[]") for v in cd.values() for item in v.split(", ")]
random.shuffle(concept_strs)
concept_strs = concept_strs[:320]


# --- Simple AE for recon-error scoring ---
class SimpleAE(nn.Module):
    def __init__(self, d, latent=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, latent), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(latent, 128), nn.ReLU(), nn.Linear(128, d))
    def forward(self, x): return self.dec(self.enc(x))

def ae_auroc(safe_feat, unsafe_feat, epochs=20):
    """Train AE on safe features, return AUROC of recon error on all."""
    d = safe_feat.shape[1]
    ae = SimpleAE(d).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=0.001)
    sf = torch.tensor(safe_feat, dtype=torch.float32, device=device)
    ae.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(ae(sf), sf)
        loss.backward(); opt.step()

    ae.eval()
    with torch.no_grad():
        all_feat = np.vstack([safe_feat, unsafe_feat])
        t = torch.tensor(all_feat, dtype=torch.float32, device=device)
        recon_err = torch.mean((ae(t) - t)**2, dim=-1).cpu().numpy()
    return roc_auc_score(all_labels, recon_err)


def eval_encoder(name, all_emb, safe_emb, unsafe_emb, concept_emb=None):
    """Compute separation metrics for an encoder."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # 1. Cosine similarity structure
    s_s = (safe_emb @ safe_emb.T).fill_diagonal_(0)
    u_u = (unsafe_emb @ unsafe_emb.T).fill_diagonal_(0)
    s_u = safe_emb @ unsafe_emb.T
    print(f"Intra-safe  cos sim : {s_s.mean():.4f} +/- {s_s.std():.4f}")
    print(f"Intra-unsafe cos sim: {u_u.mean():.4f} +/- {u_u.std():.4f}")
    print(f"Cross cos sim       : {s_u.mean():.4f} +/- {s_u.std():.4f}")

    # 2. Linear probe (80/20 split)
    X = all_emb.cpu().numpy()
    y = all_labels
    idx = np.random.permutation(len(y))
    split = int(0.8 * len(y))
    X_tr, X_te = X[idx[:split]], X[idx[split:]]
    y_tr, y_te = y[idx[:split]], y[idx[split:]]
    sc = StandardScaler().fit(X_tr)
    lr = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(X_tr), y_tr)
    prob = lr.predict_proba(sc.transform(X_te))[:, 1]
    auroc_lr = roc_auc_score(y_te, prob)
    print(f"\nLinear probe AUROC  : {auroc_lr:.4f}")

    # 3. One-class SVM (train on safe, score all)
    X_safe_tr = X[idx[:split]][y[idx[:split]] == 0]
    sc2 = StandardScaler().fit(X_safe_tr)
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1).fit(sc2.transform(X_safe_tr))
    ocsvm_scores = -ocsvm.decision_function(sc2.transform(X_te))
    auroc_ocsvm = roc_auc_score(y_te, ocsvm_scores)
    print(f"One-class SVM AUROC : {auroc_ocsvm:.4f}")

    # 4. Concept similarity + AE (JailDAM approach)
    if concept_emb is not None:
        safe_sim   = (safe_emb @ concept_emb.T).cpu().numpy()
        unsafe_sim = (unsafe_emb @ concept_emb.T).cpu().numpy()
        print(f"\nConcept sim (safe)  : mean {safe_sim.mean():.4f}  std {safe_sim.std():.4f}")
        print(f"Concept sim (unsafe): mean {unsafe_sim.mean():.4f}  std {unsafe_sim.std():.4f}")
        print(f"Concept sim gap     : {unsafe_sim.mean() - safe_sim.mean():.4f}")

        auroc_concept_ae = ae_auroc(safe_sim, unsafe_sim)
        print(f"AE concept-sim AUROC: {auroc_concept_ae:.4f}")

    # 5. Raw embedding AE (no concepts)
    raw_safe   = safe_emb.cpu().numpy()
    raw_unsafe = unsafe_emb.cpu().numpy()
    auroc_raw_ae = ae_auroc(raw_safe, raw_unsafe)
    print(f"AE raw-embed AUROC  : {auroc_raw_ae:.4f}")


# -----------------------------------------------------------------------
# Encoder 1: CLIP text
# -----------------------------------------------------------------------
print("\n--- Loading CLIP ---")
clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device); clip.eval()
tok  = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

def clip_embed(texts, bs=64):
    embs = []
    for i in range(0, len(texts), bs):
        inp = tok(texts[i:i+bs], return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        with torch.no_grad():
            e = clip.get_text_features(**inp)
        if not isinstance(e, torch.Tensor): e = e.pooler_output
        embs.append(F.normalize(e, p=2, dim=-1))
    return torch.cat(embs)

clip_all = clip_embed(all_texts)
ci = tok(concept_strs, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
with torch.no_grad():
    ce = clip.get_text_features(**ci)
if not isinstance(ce, torch.Tensor): ce = ce.pooler_output
ce = F.normalize(ce, p=2, dim=-1)

eval_encoder("CLIP text (current)", clip_all, clip_all[:N], clip_all[N:], ce)

# -----------------------------------------------------------------------
# Encoder 2: CLIP + mean-centering
# -----------------------------------------------------------------------
clip_mean = clip_all.mean(dim=0, keepdim=True)
clip_c = F.normalize(clip_all - clip_mean, p=2, dim=-1)
ce_c   = F.normalize(ce - clip_mean, p=2, dim=-1)
eval_encoder("CLIP text (mean-centered)", clip_c, clip_c[:N], clip_c[N:], ce_c)

del clip; torch.cuda.empty_cache()

# -----------------------------------------------------------------------
# Encoder 3: sentence-transformers/all-mpnet-base-v2
# -----------------------------------------------------------------------
print("\n\n--- Loading all-mpnet-base-v2 ---")
st = SentenceTransformer("all-mpnet-base-v2", device=device)
st_all = torch.tensor(st.encode(all_texts, batch_size=64, normalize_embeddings=True), device=device)
st_concepts = torch.tensor(st.encode(concept_strs, batch_size=64, normalize_embeddings=True), device=device)
eval_encoder("all-mpnet-base-v2", st_all, st_all[:N], st_all[N:], st_concepts)

del st; torch.cuda.empty_cache()

# -----------------------------------------------------------------------
# Encoder 4: sentence-transformers/all-MiniLM-L6-v2
# -----------------------------------------------------------------------
print("\n\n--- Loading all-MiniLM-L6-v2 ---")
mini = SentenceTransformer("all-MiniLM-L6-v2", device=device)
mini_all = torch.tensor(mini.encode(all_texts, batch_size=64, normalize_embeddings=True), device=device)
mini_concepts = torch.tensor(mini.encode(concept_strs, batch_size=64, normalize_embeddings=True), device=device)
eval_encoder("all-MiniLM-L6-v2", mini_all, mini_all[:N], mini_all[N:], mini_concepts)

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print(f"\n\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
print(f"Viable if:  Linear probe > 0.70,  OC-SVM > 0.60,  AE > 0.55")
