"""Check the text CCA (256) against an original bundle's x_text_embed / x_text_embed_seq.

usage: python scripts/check_text_cca.py <original bundle dir> <text cache dir>
The text inputs and FinBERT sentence embeddings are read from the text cache (as in preprocess.py).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pacmoe.preprocess.text import FINBERT_DIM, aggregate_structures, load_sentence_table, pool_sentences, sample_inputs, text_key  # noqa: E402
from pacmoe.preprocess.text_cca import TextCCA  # noqa: E402


def views(text_dir: Path, split: str):
    """Pooled FinBERT (N, 768), structured view (N, 57), sentence embeddings [(L_i, 768)]."""
    rows = json.loads((text_dir / f"text_inputs_{split}.json").read_text(encoding="utf-8"))
    table = load_sentence_table(str(text_dir / f"finbert_sentences_{split}.pt"))
    pooled, structured, sentences = [], [], []
    for r in rows:
        keys = [text_key(t) for t in sample_inputs(r["tweets"], r["structures"])]
        emb = torch.stack([table[k] for k in keys]).float() if keys else torch.zeros(0, FINBERT_DIM)
        sentences.append(emb)
        pooled.append(pool_sentences(emb))
        structured.append(aggregate_structures(r["structures"], r["date"]))
    return [(r["symbol"], r["date"]) for r in rows], torch.stack(pooled), np.stack(structured), sentences


def main() -> int:
    bundle, text_dir = Path(sys.argv[1]), Path(sys.argv[2])
    keys_tr, pooled_tr, struct_tr, _ = views(text_dir, "train")
    cca = TextCCA(256).fit(pooled_tr, struct_tr)
    report = {"cca_components": int(cca.cca.n_components) if cca.cca is not None else 0,
              "pca_components": int(cca.pca.n_components_) if cca.pca is not None else 0}
    for split in ("train", "val", "test"):
        keys, pooled, _, sentences = views(text_dir, split)
        ref = torch.load(bundle / f"samples_{split}.pt", map_location="cpu", weights_only=False)
        assert keys == list(zip(ref["symbols"], ref["dates"]))
        emb = cca.transform(pooled)
        seq = cca.transform_sequences(sentences)
        ref_seq = ref["x_text_embed_seq"]
        shapes = all(a.shape == b.shape for a, b in zip(seq, ref_seq))
        report[split] = {
            "n": len(keys),
            "x_text_embed_max_abs_diff": float((emb - ref["x_text_embed"].float()).abs().max()),
            "x_text_embed_bit_exact": bool(torch.equal(emb, ref["x_text_embed"].float())),
            "seq_shapes_equal": shapes,
            "x_text_embed_seq_max_abs_diff": max((float((a - b.float()).abs().max()) for a, b in zip(seq, ref_seq) if a.numel()), default=0.0) if shapes else None,
            "seq_bit_exact": sum(bool(torch.equal(a, b.float())) for a, b in zip(seq, ref_seq)) if shapes else None,
        }
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
