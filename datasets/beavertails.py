"""
BeaverTails text-only dataset loader.

Loads PKU-Alignment/BeaverTails from HuggingFace.
Each item returns (text, category_idx) where:
  category_idx = 0  → safe
  category_idx = 1..14 → first active harm category (1-indexed)

Import note: the local `datasets/` directory shadows the HF `datasets` package
when running from the project root. _load_hf() handles this by temporarily
removing the project root from sys.path before importing.
"""

import os
import sys
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


_hf_load_dataset_fn = None  # cached after first import


def _hf_load_dataset(name, split, token):
    """Import HuggingFace load_dataset once, bypassing the local datasets/ dir.

    The local `datasets/` package shadows the HF one in sys.modules. We fix
    this by stripping the project root from sys.path and evicting the cached
    local module before the first import, then caching load_dataset so we
    never re-import (PyArrow extension types can only be registered once).
    """
    global _hf_load_dataset_fn
    if _hf_load_dataset_fn is None:
        project_root  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        saved_path    = sys.path[:]
        saved_modules = {k: v for k, v in sys.modules.items()
                         if k == "datasets" or k.startswith("datasets.")}

        sys.path[:] = [p for p in sys.path if os.path.abspath(p) != project_root]
        for key in list(saved_modules):
            del sys.modules[key]

        try:
            from datasets import load_dataset
            _hf_load_dataset_fn = load_dataset
        finally:
            # Restore sys.path; keep HF modules so PyArrow types stay registered
            sys.path[:] = saved_path
            # Re-expose local datasets package under its own submodule keys
            for key in list(sys.modules.keys()):
                if key == "datasets" or key.startswith("datasets."):
                    if not key.startswith("datasets.beavertails"):
                        pass  # leave HF submodules in place
            sys.modules.update(saved_modules)

    return _hf_load_dataset_fn(name, split=split, token=token)


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
        raw = _hf_load_dataset("PKU-Alignment/BeaverTails", hf_split, token)

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
