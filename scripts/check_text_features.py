"""Check the text day-feature windows against a preprocessed bundle.

usage: python scripts/check_text_features.py <bundle_dir> <split> <finbert_sentences_<split>.pt>
The retrieved tweets and FinGPT structures are read from the bundle; FinBERT is not run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pacmoe.preprocess.text import day_features, load_sentence_table, sentence_embeddings, text_windows  # noqa: E402


def main() -> int:
    bundle_dir, split, table_path = sys.argv[1:4]
    d = torch.load(Path(bundle_dir) / f"samples_{split}.pt", map_location="cpu", weights_only=False)
    samples = list(zip(d["symbols"], d["dates"]))
    sentences = sentence_embeddings(d["x_text_docs_rag"], d["x_text_structures"], load_sentence_table(table_path))
    feats = day_features(samples, sentences, d["x_text_structures"])
    windows = text_windows(samples, feats)
    ref = d["x_text_day_feature_window"]
    shape_ok = all(a.shape == b.shape for a, b in zip(windows, ref))
    diff_all = max(float((a - b).abs().max()) for a, b in zip(windows, ref)) if shape_ok else float("nan")
    diff_bert = max(float((a[:, :768] - b[:, :768]).abs().max()) for a, b in zip(windows, ref)) if shape_ok else float("nan")
    exact = sum(bool(torch.equal(a, b)) for a, b in zip(windows, ref))
    out = {"split": split, "n": len(samples), "shapes_equal": shape_ok, "max_abs_diff": diff_all,
           "max_abs_diff_finbert_part": diff_bert, "bit_exact_windows": exact}
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
