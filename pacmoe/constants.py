"""
Implementation details.

These values are fixed because they were fixed in the implementation that produced the reported
results. Values stated in the paper (dimensions, heads, dropout, learning rate, loss weights,
label thresholds, ...) are read from the TOML config instead (see pacmoe/config.py).
"""

import torch

# Mixed precision of the training forward passes on CUDA (evaluation runs in float32).
AMP_DTYPE = torch.bfloat16

# ----------------------------------------------------------------------------------------
# Price encoders
# ----------------------------------------------------------------------------------------
OHLCV_CHANNELS = 5            # open, high, low, close, volume of the look-back window
OHLCV_PATCH_LEN = 16
OHLCV_STRIDE = 8
OHLCV_D_MODEL = 128
OHLCV_NHEAD = 4
OHLCV_LAYERS = 2
OHLCV_DROPOUT = 0.2
OHLCV_MAX_PATCHES = 256

TECH_D_MODEL = 128            # also the output size of the indicator PatchTST
TECH_LAYERS = 2
TECH_NHEAD = 4
TECH_DROPOUT = 0.0
TECH_SCALES = ((2, 1), (4, 2), (8, 4))   # (patch size, stride) per patch scale of the indicator PatchTST
TECH_MAX_SEQ_LEN = 512
PRICE_FEATURE_COLUMNS = 6     # x_ts[..., :6] = open, high, low, close, volume, amount; the rest are indicators

# ----------------------------------------------------------------------------------------
# Text encoder
# ----------------------------------------------------------------------------------------
TEXT_WINDOW_DAYS = 20         # day-level text features of the last 20 days (causal)
TEXT_LAYERS = 2

# ----------------------------------------------------------------------------------------
# Graph encoder
# ----------------------------------------------------------------------------------------
GRAPH_NUM_RELATIONS = 5       # RGCNConv weight slots; only relation 0 (return correlation) occurs
NUM_TICKERS = 88

# ----------------------------------------------------------------------------------------
# CCA residual adapters
# ----------------------------------------------------------------------------------------
CCA_RESIDUAL_WEIGHT = 0.2
CCA_RIDGE = 1e-3
CCA_MIN_BATCH = 8
CCA_STAT_MOMENTUM = 0.01
CCA_STAT_WARMUP_STEPS = 10

# ----------------------------------------------------------------------------------------
# Router conditioning
# ----------------------------------------------------------------------------------------
QUALITY_EMA_DECAY = 0.99
QUALITY_CONSISTENCY_WEIGHT = 0.3
GCCA_K = 16
GCCA_EMA_DECAY = 0.99
GCCA_REFRESH_EVERY = 50
GCCA_WARMUP_STEPS = 200
GCCA_EPS = 1e-4
ROUTER_REGIME_DIM = 1         # always zero in the reported configurations

# ----------------------------------------------------------------------------------------
# Router (tau of Eq. 14 = ROUTER_TEMPERATURE / ROUTER_LOGIT_SCALE = 1.35 / 0.75 = 1.8)
# ----------------------------------------------------------------------------------------
ROUTER_TEMPERATURE = 1.35
ROUTER_LOGIT_SCALE = 0.75
QUALITY_PRIOR_SCALE = 0.5
QUALITY_PRIOR_FLOOR = 0.15
EXPERT_RESIDUAL_SCALE = 0.3
ADAPTIVE_TOPK_MARGIN = 0.1
ADAPTIVE_TOPK_MARGIN_HIGH = 0.3

# ----------------------------------------------------------------------------------------
# Additional training objectives (weights)
# ----------------------------------------------------------------------------------------
ROUTER_ENTROPY_TARGET_RATIO = 0.9
ROUTER_MARGIN_TARGET = 0.25
QUALITY_PATTERN_THRESHOLD = 0.5
QUALITY_PATTERN_MARGIN = 0.2
ROUTER_PRIOR_WEIGHTS = {
    "entropy": 0.1,
    "margin": 0.01,
    "diversity": 0.003,
    "z": 0.03,
    "smoothing": 0.01,
    "pattern_consistency": 0.008,
    "pattern_separation": 0.008,
}
LOAD_BALANCE_WEIGHT = 0.1
RANKING_LOSS_WEIGHT = 0.1
BRANCH_AUX_WEIGHT = 0.5
BRANCH_AUX_MODALITY_WEIGHTS = {"price": 0.02, "text": 0.05, "graph": 0.05}
NEGATIVE_LEARNING_WEIGHT = 0.02
NEGATIVE_LEARNING_MARGIN = 0.1
NEGATIVE_LEARNING_CONFIDENCE = 0.55
ROUTING_AUX_EXPERT_WEIGHT = 0.075
ROUTING_AUX_TEMPERATURE = 0.5
ROUTING_AUX_START_EPOCH = 3
TEMPORAL_CONTRASTIVE_WEIGHT = 0.05
TEMPORAL_CONTRASTIVE_TEMPERATURE = 0.07
VOL_POS_WEIGHT = 0.695
CLS_LABEL_SMOOTHING = 0.05    # label smoothing of the direction cross-entropy

# ----------------------------------------------------------------------------------------
# Optimization (AdamW, the learning rate and the batch size are in the config)
# ----------------------------------------------------------------------------------------
WEIGHT_DECAY = 0.01
WARMUP_EPOCHS = 5            # linear warm-up of the learning rate over the first 5 epochs, then cosine
GRAD_CLIP = 5.0              # global gradient-norm clip
GRAPH_GRAD_CLIP = 0.2        # gradient-norm clip of the graph encoder and its pre-fusion block, applied first
CHECKPOINT_MIN_EPOCH = 1
