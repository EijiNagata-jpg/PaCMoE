"""Write the test predictions (direction probability, return head, volatility head) of checkpoints.

usage: python scripts/dump_predictions.py --setup main --bundle data/preprocessed/main \
           --checkpoints checkpoints/main_seed*.pt --out-dir outputs/predictions/main [--deterministic]
One <checkpoint name>.npz per checkpoint, read by the analysis/ scripts.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pacmoe.config import model_spec_for, setup_details  # noqa: E402
from pacmoe.data import Bundle  # noqa: E402
from pacmoe.evaluate import configure_backends, predict  # noqa: E402
from pacmoe.model import PaCMoE  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True)
    ap.add_argument("--variant", default="full")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--graph-snapshots", default=None)
    ap.add_argument("--deterministic", action="store_true")
    a = ap.parse_args()
    configure_backends(deterministic=a.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = Bundle(a.bundle, "test", setup_details(a.setup)["return_target"], graph_snapshots_path=a.graph_snapshots)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for path in a.checkpoints:
        model = PaCMoE(model_spec_for(a.setup, a.variant, text_feature_dim=int(bundle.text_features.shape[-1])))
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model_state"])
        model.to(device)
        r = predict(model, bundle, 128, device)
        name = Path(path).stem
        np.savez(out / f"{name}.npz", test_dates=r["dates"], test_symbols=r["symbols"], test_labels=r["labels"],
                 test_probs=r["probs"], test_pred_ret=r["pred_ret"], test_logit_vol=r["logit_vol"])
        print(name, len(r["dates"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
