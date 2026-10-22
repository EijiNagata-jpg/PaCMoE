"""Text features (Section 2.3): daily FinBERT + FinGPT-structure vectors and 20-day windows.

Inputs are cached model outputs; neither FinGPT nor FinBERT is run here:
  - FinGPT structured outputs per document (sentiment, confidence, relevance, event, horizon,
    entities, analysis), stored per sample in `fingpt_structures_<split>.json`;
  - FinBERT sentence embeddings (768) of "<tweet> [SEP] <FinGPT analysis>", stored in
    `finbert_sentences_<split>.pt` as a table keyed by the SHA-256 of the input text.

Day feature (825) = [pooled FinBERT (768), 9 scalars, event distribution (12), entity hash
embedding (32), horizon distribution (4)].
"""
from __future__ import annotations

import hashlib
import html
import math
import re
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

# Implementation details not stated in the paper.
TOP_K_DOCS = 3  # retrieved tweets per sample that are embedded
TEXT_WINDOW_DAYS = 20  # length of the text window
RECENCY_TAU_HOURS = 6.0  # recency weight exp(-age / tau)
ENTITY_DIM = 32
EVENT_VOCAB = ["earnings", "guidance", "merger", "acquisition", "partnership", "product", "regulation",
               "litigation", "macro", "analyst", "management", "other"]
HORIZON_VOCAB = ["intraday", "short", "mid", "long"]
SCALAR_KEYS = ("sentiment_mean", "sentiment_std", "confidence_mean", "doc_count", "avg_relevance",
               "avg_recency", "rag_score_mean", "rag_score_max", "sentence_count")
FINBERT_DIM = 768
DAY_FEATURE_DIM = FINBERT_DIM + len(SCALAR_KEYS) + len(EVENT_VOCAB) + ENTITY_DIM + len(HORIZON_VOCAB)

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+(?:\s+|$)|\n+")


# ------------------------------------------------------------------------------------------
# FinBERT input text
# ------------------------------------------------------------------------------------------
def finbert_input(tweet: str, analysis: str) -> str:
    """'<tweet> [SEP] <analysis>' (either part alone if the other is empty)."""
    raw, gen = str(tweet or "").strip(), str(analysis or "").strip()
    return f"{raw} [SEP] {gen}" if raw and gen else (raw or gen)


def clean_text(text: str) -> str:
    """HTML unescape, URLs removed, whitespace collapsed; '' for empty / 'nan'."""
    out = str(text if text is not None else "").strip()
    if not out or out.lower() == "nan":
        return ""
    out = _URL_RE.sub(" ", html.unescape(out))
    return _WS_RE.sub(" ", out).strip()


def text_key(cleaned: str) -> str:
    """Key of a FinBERT input in the sentence-embedding table."""
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()


def sample_inputs(tweets: Sequence[str], structures: Sequence[Dict]) -> List[str]:
    """Cleaned FinBERT inputs of one sample: the first TOP_K_DOCS retrieved tweets.

    The cleaning is applied twice, as in the original code; this matters only for doubly escaped
    HTML entities (e.g. '&amp;#x2019;' becomes a right single quote)."""
    texts = [finbert_input(doc, (structures[j] if j < len(structures) else {}).get("analysis", ""))
             for j, doc in enumerate(list(tweets)[:TOP_K_DOCS])]
    return [c for c in (clean_text(clean_text(t)) for t in texts) if c]


# ------------------------------------------------------------------------------------------
# Daily aggregation of FinGPT structures
# ------------------------------------------------------------------------------------------
def _clip(v, lo, hi) -> float:
    return float(max(lo, min(hi, v)))


def _timestamp(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _entity_embedding(entities: Sequence[str]) -> np.ndarray:
    """Sum of SHA-256 byte vectors of the upper-cased entity names, L2-normalized."""
    vec = np.zeros(ENTITY_DIM, dtype=np.float32)
    count = 0
    for entity in entities:
        token = str(entity).strip().upper()
        if not token:
            continue
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for i in range(ENTITY_DIM):
            vec[i] += np.float32(digest[i % len(digest)] / 255.0 * 2.0 - 1.0)
        count += 1
    norm = float(np.linalg.norm(vec))
    if count and norm > 1e-8:
        vec /= norm
    return vec


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return (e / float(e.sum())).astype(np.float32)


def aggregate_structures(structures: Sequence[Dict], date: str) -> np.ndarray:
    """57 structured dims of one sample-day. Documents are weighted by
    softmax(confidence * relevance * recency)."""
    out = np.zeros(DAY_FEATURE_DIM - FINBERT_DIM, dtype=np.float32)
    if not structures:
        return out
    now = _timestamp(date)
    tau = RECENCY_TAU_HOURS * 3600.0
    ev_idx = {n: i for i, n in enumerate(EVENT_VOCAB)}
    hz_idx = {n: i for i, n in enumerate(HORIZON_VOCAB)}
    sent, conf, rel, score, rec, n_sent, ev, hz, ents = [], [], [], [], [], [], [], [], []
    for s in structures:
        extracted = _clip(s.get("relevance", 0.0), 0.0, 1.0)
        retrieval = s.get("retrieval_relevance")
        if retrieval is None:
            eff = extracted
        else:
            retrieval = _clip(retrieval, 0.0, 1.0)
            eff = extracted * retrieval if extracted > 0.0 else retrieval
        ts = _timestamp(s.get("timestamp", s.get("date")))
        rec.append(float(math.exp(-max((now - ts).total_seconds(), 0.0) / tau)) if now is not None and ts is not None else 1.0)
        sent.append(_clip(s.get("sentiment", 0.0), -1.0, 1.0))
        conf.append(_clip(s.get("confidence", 0.0), 0.0, 1.0))
        rel.append(eff)
        score.append(_clip(retrieval if retrieval is not None else eff, 0.0, 1.0))
        text = str(s.get("text", s.get("analysis", "")) or "").strip()
        parts = [p for p in (c.strip() for c in _SENTENCE_SPLIT_RE.split(text)) if p] if text else []
        n_sent.append(len(parts) if parts else (1 if text else 0))
        ev.append(ev_idx.get(str(s.get("event", "other")).strip().lower() or "other", ev_idx["other"]))
        hz.append(hz_idx.get(str(s.get("horizon", "short")).strip().lower() or "short", hz_idx["short"]))
        ents.extend(str(e) for e in s.get("entities", []) if str(e).strip())
    f32 = lambda v: np.asarray(v, dtype=np.float32)  # noqa: E731
    w = _softmax(f32(conf) * f32(rel) * f32(rec))
    sent_a = f32(sent)
    mean = float(np.sum(w * sent_a))
    var = float(np.sum(w * np.square(sent_a - mean)))
    scores = f32(score)
    scalars = [mean, math.sqrt(max(var, 0.0)), float(np.sum(w * f32(conf))), len(structures),
               float(np.sum(w * f32(rel))), float(np.sum(w * f32(rec))), float(scores.mean()), float(scores.max()),
               int(sum(n_sent))]
    event = np.zeros(len(EVENT_VOCAB), dtype=np.float32)
    horizon = np.zeros(len(HORIZON_VOCAB), dtype=np.float32)
    for i, wi in zip(ev, w):
        event[i] += float(wi)
    for i, wi in zip(hz, w):
        horizon[i] += float(wi)
    return np.concatenate([np.asarray(scalars, dtype=np.float32), event, _entity_embedding(ents), horizon])


# ------------------------------------------------------------------------------------------
# Day features and windows
# ------------------------------------------------------------------------------------------
def pool_sentences(emb: torch.Tensor) -> torch.Tensor:
    """Mean of the sentence embeddings (zeros if none)."""
    if emb.shape[0] == 0:
        return torch.zeros(FINBERT_DIM, dtype=torch.float32)
    if emb.shape[0] == 1:
        return emb[0]
    weights = torch.softmax(torch.zeros(emb.shape[0], dtype=emb.dtype), dim=0)
    return torch.sum(weights.unsqueeze(-1) * emb, dim=0)


def sentence_embeddings(tweets: Sequence[Sequence[str]], structures: Sequence[Sequence[Dict]],
                        table: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
    """Per sample, the (L_i, 768) FinBERT embeddings of its inputs. `table` maps text_key -> (768,)."""
    out = []
    for docs, structs in zip(tweets, structures):
        keys = [text_key(t) for t in sample_inputs(docs, structs)]
        out.append(torch.stack([table[k] for k in keys]).float() if keys else torch.zeros(0, FINBERT_DIM))
    return out


def day_features(samples: Sequence[Tuple[str, str]], sentences: Sequence[torch.Tensor],
                 structures: Sequence[Sequence[Dict]]) -> torch.Tensor:
    """(N, 825) day features: [pooled FinBERT (768), structured (57)]."""
    rows = [torch.cat([pool_sentences(emb), torch.from_numpy(aggregate_structures(structs, date))])
            for (_, date), emb, structs in zip(samples, sentences, structures)]
    return torch.stack(rows) if rows else torch.zeros(0, DAY_FEATURE_DIM)


def text_windows(samples: Sequence[Tuple[str, str]], feats: torch.Tensor) -> List[torch.Tensor]:
    """Per sample, the day features of the same symbol's samples of the same split up to and
    including its date, last TEXT_WINDOW_DAYS rows."""
    by_symbol: Dict[str, List[Tuple[str, int]]] = {}
    for i, (sym, date) in enumerate(samples):
        by_symbol.setdefault(sym, []).append((date, i))
    out: List[torch.Tensor] = [None] * len(samples)  # type: ignore[list-item]
    for entries in by_symbol.values():
        order = [i for _, i in sorted(entries)]
        table = feats[order]
        for pos, i in enumerate(order):
            out[i] = table[max(0, pos - TEXT_WINDOW_DAYS + 1): pos + 1]
    return out


def load_sentence_table(path: str) -> Dict[str, torch.Tensor]:
    """{text_key: (768,)} from a `finbert_sentences_<split>.pt` file."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return {k: blob["embeddings"][i] for i, k in enumerate(blob["keys"])}
