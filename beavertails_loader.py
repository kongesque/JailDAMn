"""
BeaverTails text-only dataset loader.

The local datasets/ directory shadows the HuggingFace `datasets` package on
sys.path. We fix this at module load time by temporarily removing the project
root from sys.path before importing HF datasets, then restoring it.
"""

# --- Bootstrap: must happen before any other import ---
import os as _os, sys as _sys
_proj = _os.path.dirname(_os.path.abspath(__file__))
_saved_path = _sys.path[:]
_sys.path[:] = [p for p in _sys.path if _os.path.abspath(p) != _proj]
from datasets import load_dataset as _hf_load  # HuggingFace datasets
_sys.path[:] = _saved_path
# ------------------------------------------------------

import torch
from torch.utils.data import Dataset

CATEGORY_NAMES = [
    "animal_abuse",
    "child_abuse",
    "controversial_topics,politics",
    "discrimination,stereotype,injustice",
    "drug_abuse,weapons,banned_substance",
    "financial_crime,property_crime,theft",
    "hate_speech,offensive_language",
    "misinformation_regarding_ethics,laws_and_safety",
    "non_violent_unethical_behavior",
    "privacy_violation",
    "self_harm",
    "sexually_explicit,adult_content",
    "terrorism,organized_crime",
    "violence,aiding_and_abetting,incitement",
]


class BeaverTailsDataset(Dataset):
    """
    Text-only wrapper around PKU-Alignment/BeaverTails.

    Args:
        hf_split:      HuggingFace split name ("330k_train" or "330k_test").
        token:         HuggingFace access token.
        safe_only:     If True, keep only safe samples (is_safe=True).
        unsafe_only:   If True, keep only unsafe samples (is_safe=False).
        max_samples:   Stop after collecting this many items (None = all).
        use_response:  If True, concatenate prompt + response as input text.
    """

    def __init__(
        self,
        hf_split,
        token=None,
        safe_only=False,
        unsafe_only=False,
        max_samples=None,
        use_response=False,
    ):
        raw = _hf_load("PKU-Alignment/BeaverTails", split=hf_split, token=token)

        self.data = []
        for item in raw:
            is_safe = item["is_safe"]
            if safe_only and not is_safe:
                continue
            if unsafe_only and is_safe:
                continue

            text = item["prompt"]
            if use_response:
                text = text + " " + item["response"]

            # First active harm category (1-indexed); 0 for safe samples
            category_idx = 0
            if not is_safe:
                for i, cat in enumerate(CATEGORY_NAMES):
                    if item["category"].get(cat, False):
                        category_idx = i + 1
                        break

            self.data.append((text, category_idx))
            if max_samples and len(self.data) >= max_samples:
                break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, category_idx = self.data[idx]
        return text, torch.tensor(category_idx, dtype=torch.int64)
