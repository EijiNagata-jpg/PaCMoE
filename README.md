# PaCMoE: Pairwise Cross-Modal Attention Fusion with Mixture-of-Experts

Reference implementation of the experiments in

> Eiji Nagata, Riku Motomochi, Qiu Chen.
> PaCMoE: Pairwise Cross-Modal Attention Fusion with Mixture-of-Experts for Stock Movement Prediction.
> *Mathematics* **2026**, *14*(21), 3948. https://doi.org/10.3390/math14213948
> ([article page](https://www.mdpi.com/2227-7390/14/21/3948))


The repository contains only the computation that produced the reported results: preprocessing,
the model, the training objective, the training and evaluation loops, and the scripts that check
this code against the original research code.

## Quick start (Table 3)

```bash
pip install -r requirements.txt
# download bundle_main.tar and main_seed{1..10}.pt from the release of this repository (see "Data")
tar -xf bundle_main.tar                     # -> data/preprocessed/main/
mkdir -p checkpoints && mv main_seed*.pt checkpoints/
python main.py evaluate --config configs/main.toml --checkpoint checkpoints/main_seed8.pt --deterministic
# -> {"accuracy": 0.6132..., "mcc": 0.2312..., "n": 3666}   (all ten runs: see "Main results")
```

## Contents

| Path | Purpose |
|---|---|
| `preprocess.py`, `pacmoe/preprocess/` | Preprocessing (Sections 2.1-2.3): prices, technical indicators, labels, daily correlation graphs, text day features |
| `pacmoe/model.py` | PaCMoE (Sections 2.2-2.6): encoders -> pairwise experts -> MoE router -> heads |
| `pacmoe/nn/encoders.py` | PatchTST for OHLCV and for the technical indicators (channel-independent, three patch scales), text encoder, R-GCN |
| `pacmoe/nn/fusion.py` | Pairwise cross-modal attention experts and the router (Eqs. 2-15) |
| `pacmoe/nn/cca.py`, `pacmoe/nn/conditioning.py` | CCA residual adapters, router conditioning (quality and agreement scores) |
| `pacmoe/losses.py` | Multitask loss (Eqs. 16-26) and the additional training terms |
| `pacmoe/train.py`, `pacmoe/evaluate.py` | Training (AdamW, 15 epochs, model selection by validation MCC) and evaluation (accuracy, MCC at threshold 0.5) |
| `pacmoe/walk_forward.py` | Rolling walk-forward evaluation (Section 3.4) |
| `pacmoe/data.py` | Preprocessed feature bundles and date-grouped batching |
| `pacmoe/config.py`, `configs/*.toml` | Configuration. Only values stated in the paper are configurable |
| `pacmoe/constants.py` | Implementation details not stated in the paper (fixed values) |
| `scripts/check_*.py`, `scripts/compare_bundles.py` | Checks of the preprocessing against bundles written by the original research code |
| `scripts/dump_predictions.py` | Writes test predictions (direction, return and volatility heads) of checkpoints |
| `analysis/` | Section 3.5 (CAPM, Fama-French, statistical factors, GARCH, DCC-GARCH), DCC graphs, transaction costs |

## Requirements

Python 3.11, PyTorch 2.4.0 (CUDA 12.1), PyTorch Geometric, NumPy, pandas. The experiments were run on
one NVIDIA GeForce RTX 3090. See `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Data

The experiments use the ACL18/StockNet dataset (Xu and Cohen, 2018; https://github.com/yumoxu/stocknet-dataset).
The model reads *preprocessed bundles* (one directory per setting):

```
samples_{train,val,test}.pt   per sample: 60-day window of normalized prices and technical indicators
                              (122 columns), 20-day window of text day features (825), labels,
                              returns, symbol, date
graph_snapshots.pt            daily correlation graphs (computed from returns up to each day)
metadata.json                 node order of the graph, feature names, split dates
```

Files attached to the release of this repository (check them with `SHA256SUMS_checkpoints` and `SHA256SUMS_data`). Unpack the
archives in the repository root; they create the directories below:

| File | Unpacks to | Used for |
|---|---|---|
| `main_seed{1..10}.pt` | (move to `checkpoints/`) | checkpoints of the ten main runs (Table 3) |
| `bundle_main.tar` | `data/preprocessed/main/` | Tables 3 and 5, Section 3.5 |
| `bundle_main_volscaled.tar` | `data/preprocessed/main_volscaled/` | scaled volatility target |
| `bundle_ablation.tar`, `bundle_ablation_3cls.tar` | `data/preprocessed/ablation/`, `.../ablation_3cls/` | Table 4, Section 3.6, Figures 2-4 |
| `bundle_walk_forward_fold_00{0,1,2}.tar` | `data/preprocessed/walk_forward/fold_00{0,1,2}/` | Table 6 |
| `g3_dcc.pt` | (move to `data/graph_overrides/`) | DCC-GARCH daily graphs (Section 3.5) |

The bundles contain numerical features only (normalized price and indicator windows, FinBERT
embeddings and values derived from the FinGPT outputs, labels, returns, stock symbols and dates); they
do not contain the text of the tweets. The text caches from which the bundles are built (below)
contain the text of the retrieved tweets and are therefore not redistributed; they are available
from the corresponding author upon request. The StockNet dataset itself is public.

To rebuild the bundles, place the StockNet price files and the text caches as follows:

```
data/stocknet-dataset/price/raw/<SYMBOL>.csv
data/text_cache/<bundle name>/      text caches (one per bundle, upon request)
```

| Bundle | Used for |
|---|---|
| `preprocessed/main` | Tables 3 and 5, Section 3.5; with `graph_overrides/g3_dcc.pt`: DCC-GARCH graphs |
| `preprocessed/main_volscaled` | scaled volatility target |
| `preprocessed/ablation`, `preprocessed/ablation_3cls` | Table 4 (rows other than the full model), Section 3.6, Figures 2-4 |
| `preprocessed/walk_forward/fold_00{0,1,2}` | Table 6 (the same fold bundles were used for the three training seeds) |

### Building the bundles

This step is not needed to reproduce the results with the released bundles. `preprocess.py`
rebuilds a bundle from the StockNet price files and the text caches:

```bash
python preprocess.py --config configs/preprocess_main.toml           # -> data/preprocessed/main
python preprocess.py --config configs/preprocess_main_volscaled.toml # -> data/preprocessed/main_volscaled
python preprocess.py --config configs/preprocess_ablation.toml       # -> data/preprocessed/ablation
python preprocess.py --config configs/preprocess_ablation_3cls.toml  # -> data/preprocessed/ablation_3cls
python preprocess.py --config configs/preprocess_walk_forward.toml   # -> data/preprocessed/walk_forward/fold_00{0,1,2}
```

The rebuilt bundles are identical to the released ones (each from its own text cache: the FinGPT
outputs of a few samples differ between the caches because the bundles were built at different
times). The walk-forward bundles were built from indicator tables stored in single precision
(`FLOAT32_TABLES` in `pacmoe/preprocess/build.py`). The text CCA vectors depend on the number of
BLAS threads used for the fit (sklearn's iterative CCA and randomized PCA); the thread counts of the
original runs are set by `cca_blas_threads` in `configs/preprocess_*.toml` so that the vectors are
reproduced bit for bit (this is a numerical setting, not a model setting).

FinGPT and FinBERT are not run by this code. Their outputs are read from the text caches:

```
text_inputs_{split}.json        per sample: the three retrieved tweets and the FinGPT structured
                                outputs (sentiment, confidence, relevance, event, horizon, entities,
                                analysis text)
finbert_sentences_{split}.pt    FinBERT (ProsusAI/finbert) embeddings (768) of
                                "<tweet> [SEP] <FinGPT analysis>", keyed by the SHA-256 of the text
```

The tweets of a sample were retrieved from the 60 days before the sample date t (tweets of day t and
later are excluded; the three highest-ranked tweets are used): in all splits, every tweet used is
dated before t.


## Reproducing the results

### Main results (Table 3)

Evaluate the released checkpoints of the ten runs (with `bundle_main.tar`, see "Data"):

```bash
python main.py evaluate --config configs/main.toml --checkpoint checkpoints/main_seed8.pt --deterministic
```

Test accuracy / MCC of the released checkpoints with `--deterministic` (Table 3; seed 4 differs from
Table 3, see "Equivalence with the original code"):

| seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Acc | 0.6086 | 0.6113 | 0.6124 | 0.5881 | 0.6091 | 0.5925 | 0.6107 | 0.6132 | 0.6116 | 0.6113 |
| MCC | 0.2239 | 0.2293 | 0.2294 | 0.1829 | 0.2248 | 0.1914 | 0.2279 | 0.2312 | 0.2291 | 0.2290 |

Train a new model:

```bash
python main.py train --config configs/main.toml --seed <any natural number> --output-dir outputs/main_run1
```


Variants of the main setup: the scaled volatility target uses the bundle `main_volscaled`
(`--bundle-dir data/preprocessed/main_volscaled`); the DCC-GARCH graphs (Section 3.5) are used with
`--graph-snapshots data/graph_overrides/g3_dcc.pt` (train and evaluate).

### Ablation study (Table 4), three-class setting (Section 3.6), Figures 2-4

```bash
python main.py train --config configs/ablation.toml                       # full model of this setup
# set [ablation] variant = "router_uniform" | "only_pt" | "only_pg" | "only_tg" | "price_only" | "cls_only"
# variant = "three_class" with bundle_dir = "data/preprocessed/ablation_3cls"
```

Figures 2-4 use the predictions and router weights of the three runs of the full model
(`pacmoe.evaluate.predict(..., keep_router=True)`).

### Walk-forward evaluation (Table 6)

```bash
python main.py train --config configs/walk_forward.toml --seed 2025   # also 2026, 2027
```

For each fold, the epoch with the highest validation MCC among epochs 10-15 is reported.

### Econometric comparisons (Section 3.5) and transaction costs

```bash
python scripts/dump_predictions.py --setup main --bundle data/preprocessed/main \
    --checkpoints checkpoints/main_seed*.pt --out-dir outputs/predictions/main --deterministic
python -m analysis.econ_axes --pred-dir outputs/predictions/main \
    --ff-zip data/fama_french/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip   # IC and volatility AUC
python -m analysis.dcc_corr_eval --bundle data/preprocessed/main               # DCC-GARCH vs rolling correlations
python -m analysis.dcc_graph --bundle data/preprocessed/main --out data/graph_overrides/g3_dcc.pt
python -m analysis.direction_baselines factor --bundle data/preprocessed/main --all-symbols   # CAPM, sector, PCA
python -m analysis.direction_baselines factor --bundle data/preprocessed/main --all-symbols \
    --models FF3,FF5,FF5M --ff-dir data/fama_french                                       # Fama-French
python -m analysis.direction_baselines dcc --bundle data/preprocessed/main --all-symbols      # DCC-GARCH features
python -m analysis.transaction_costs --pred-dir outputs/predictions/main_all_days   # predictions for all stocks of each day
```

These scripts reproduce the values of Tables 7-9, the direction-prediction baselines of Section 3.5
and the transaction-cost analysis of the response letter exactly (checked against the outputs of the
original analysis code; the DCC graph file is identical).

The Fama-French daily factors are from the Kenneth R. French data library
(https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).


## Equivalence with the original code

The original research code is not part of this repository. To show that this code performs the same
computation, the checkpoints of the original runs were loaded into this code and both codes were
run on the same preprocessed bundles with deterministic CUDA kernels:

- predictions: identical predicted probabilities (maximum absolute difference 0.0) and identical
  accuracy and MCC for all validation and test samples of every run with a checkpoint: the ten
  main runs, the scaled volatility target (3 runs), the DCC graphs (3 runs), all rows of Table 4
  (3 runs of the full model and the variants), the three-class setting (3 runs) and the walk-forward
  evaluation (3 seeds x 3 folds). With deterministic kernels, accuracy and MCC equal Table 3 for nine
  of the ten main runs. Table 3 reports the training-time evaluation, which ran with the default
  (non-deterministic) kernels; with those kernels this code gives the Table 3 values for all ten
  runs (for seed 4, 0.5884 / 0.1835 in about half of the repetitions, 0.5881 / 0.1829 or 0.5887 / 0.1840 in the
  others);
- reported values: with deterministic kernels, the rows of Table 4 (accuracy, MCC, ROC-AUC), the
  three-class results of Section 3.6 and the scaled volatility target equal the reported values at
  the reported precision, except the mean three-class MCC (0.2085, reported as 0.209 from the
  per-seed values rounded to three digits) and seed 4 of Table 3 and run 3 of the DCC graphs, whose
  reported values come from the non-deterministic training-time evaluation (see above). The
  checkpoints of the epochs reported in Table 6 (epochs 10-15) were not saved; Table 6 equals the
  training logs of the original runs;
- training objective: identical values of the total loss and of every loss term on a training batch;
- training: started from the same initial state, the training trajectory matches;
- preprocessing: `preprocess.py` rebuilds every bundle (`main`, `main_volscaled`, `ablation`,
  `ablation_3cls` and the three walk-forward folds) bit for bit: price and indicator windows, labels,
  returns, text windows, the 256-dimensional text CCA vectors and all daily graphs, from the StockNet
  prices and the text caches.

```bash
python scripts/compare_bundles.py data/preprocessed/main <bundle written by the original code>
```

## Citation

If you use this code, please cite:

```bibtex
@article{nagata2026pacmoe,
  author  = {Nagata, Eiji and Motomochi, Riku and Chen, Qiu},
  title   = {PaCMoE: Pairwise Cross-Modal Attention Fusion with Mixture-of-Experts for Stock Movement Prediction},
  journal = {Mathematics},
  year    = {2026},
  volume  = {14},
  number  = {21},
  pages   = {3948},
  doi     = {10.3390/math14213948},
  url     = {https://www.mdpi.com/2227-7390/14/21/3948}
}
```

