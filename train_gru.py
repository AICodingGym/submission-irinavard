#!/usr/bin/env python3
"""Breath-level bidirectional GRU, masked MAE, GroupKFold validation."""
import argparse
import csv
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

FEATURES = ["R", "C", "time_step", "u_in", "u_out", "cum_u_in",
            "lag1", "lag2", "lag3", "lag5", "R_x_u_in", "cum_over_C"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare(df, with_target):
    required = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]
    if with_target:
        required.append("pressure")
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df[required].isna().any().any():
        raise ValueError("Missing input/target values")
    if df.id.duplicated().any():
        raise ValueError("Duplicate id values")
    if not df.u_out.isin([0, 1]).all():
        raise ValueError("u_out must be 0/1")
    d = df.sort_values(["breath_id", "time_step"], kind="stable").copy()
    sizes = d.groupby("breath_id", sort=False).size()
    if not (sizes == 80).all():
        raise ValueError("Every breath must contain exactly 80 steps")
    for col in ("R", "C"):
        if not (d.groupby("breath_id")[col].nunique() == 1).all():
            raise ValueError(f"{col} changes within a breath")
    if (d.C <= 0).any():
        raise ValueError("C must be positive")
    dt = d.groupby("breath_id", sort=False).time_step.diff().fillna(0)
    if (dt < 0).any():
        raise ValueError("time_step must increase within a breath")
    d["cum_u_in"] = (d.u_in * dt).groupby(d.breath_id).cumsum()
    for lag in (1, 2, 3, 5):
        d[f"lag{lag}"] = d.groupby("breath_id", sort=False).u_in.shift(lag).fillna(0)
    d["R_x_u_in"] = d.R * d.u_in
    d["cum_over_C"] = d.cum_u_in / d.C
    n = len(sizes)
    x = d[FEATURES].to_numpy(dtype=np.float32).reshape(n, 80, len(FEATURES))
    y = (d.pressure.to_numpy(dtype=np.float32).reshape(n, 80)
         if with_target else None)
    mask = (d.u_out.to_numpy().reshape(n, 80) == 0)
    ids = d.id.to_numpy().reshape(n, 80)
    breaths = d.breath_id.to_numpy().reshape(n, 80)[:, 0]
    rc = d[["R", "C"]].to_numpy().reshape(n, 80, 2)[:, 0, :].astype(int)
    if not np.isfinite(x).all() or (y is not None and not np.isfinite(y).all()):
        raise ValueError("Nonfinite features/targets")
    return x, y, mask, ids, breaths, rc


class BreathDataset(Dataset):
    def __init__(self, x, y, mask):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y) if y is not None else torch.zeros(x.shape[:2])
        self.mask = torch.from_numpy(mask)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.mask[i]


class PressureGRU(nn.Module):
    def __init__(self, input_size, hidden, layers, dropout, bidirectional):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0,
                          bidirectional=bidirectional)
        self.head = nn.Sequential(nn.Linear(hidden * (2 if bidirectional else 1), hidden),
                                  nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        h, _ = self.gru(x)
        return self.head(h).squeeze(-1)


def predict(model, loader, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for x, _, _ in loader:
            outputs.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(outputs)


def evaluate(pred, target, mask, rc):
    err = np.abs(pred - target)
    overall = float(err[mask].mean())
    groups = []
    for r, c in sorted(set(map(tuple, rc))):
        selected = (rc[:, 0] == r) & (rc[:, 1] == c)
        m = mask[selected]
        groups.append({"R": int(r), "C": int(c), "rows": int(m.sum()),
                       "mae": float(err[selected][m].mean()) if m.any() else None})
    return overall, groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="train.csv")
    p.add_argument("--test", default="test.csv")
    p.add_argument("--out", default="gru_runs")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold", type=int, default=0, help="0-based validation fold")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--causal", action="store_true", help="Unidirectional instead of bidirectional")
    p.add_argument("--predict-test", action="store_true", help="Predict test with best fold checkpoint")
    args = p.parse_args()
    if args.folds < 2 or not 0 <= args.fold < args.folds:
        p.error("Invalid folds/fold")
    out = Path(args.out) / f"fold_{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out / "train.log"), logging.StreamHandler()])
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("device=%s args=%s", device, vars(args))
    df = pd.read_csv(args.train)
    x, y, mask, ids, breaths, rc = prepare(df, True)
    splits = list(GroupKFold(n_splits=args.folds).split(np.zeros(len(breaths)), groups=breaths))
    tr, va = splits[args.fold]
    assert not set(breaths[tr]).intersection(breaths[va])
    # Fit feature scaling only on training breaths. Include all time steps as model inputs.
    mean = x[tr].mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = x[tr].std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    xtr = ((x[tr] - mean) / std).astype(np.float32)
    xva = ((x[va] - mean) / std).astype(np.float32)
    train_loader = DataLoader(BreathDataset(xtr, y[tr], mask[tr]), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(BreathDataset(xva, y[va], mask[va]), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    model = PressureGRU(len(FEATURES), args.hidden, args.layers, args.dropout,
                        not args.causal).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = float("inf")
    history_path = out / "metrics.csv"
    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_mae", "val_mae"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            abs_sum = 0.0
            count = 0
            for xb, yb, mb in train_loader:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)
                abs_err = (pred - yb).abs()
                loss = (abs_err * mb).sum() / mb.sum().clamp_min(1)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                abs_sum += (abs_err.detach() * mb).sum().item()
                count += mb.sum().item()
            train_mae = abs_sum / count
            val_pred = predict(model, val_loader, device)
            val_mae, groups = evaluate(val_pred, y[va], mask[va], rc[va])
            writer.writerow({"epoch": epoch, "train_mae": train_mae, "val_mae": val_mae})
            f.flush()
            checkpoint = {"epoch": epoch, "model": model.state_dict(),
                          "optimizer": optimizer.state_dict(), "mean": mean, "std": std,
                          "features": FEATURES, "args": vars(args), "val_mae": val_mae}
            torch.save(checkpoint, out / f"epoch_{epoch:03d}.pt")
            with (out / f"groups_epoch_{epoch:03d}.json").open("w") as gf:
                json.dump(groups, gf, indent=2)
            if val_mae < best:
                best = val_mae
                torch.save(checkpoint, out / "best.pt")
                pd.DataFrame({"id": ids[va].reshape(-1),
                              "pressure": val_pred.reshape(-1),
                              "target": y[va].reshape(-1),
                              "u_out": (~mask[va]).astype(int).reshape(-1)}).to_csv(
                                  out / "best_validation_predictions.csv", index=False)
                with (out / "best_groups.json").open("w") as gf:
                    json.dump(groups, gf, indent=2)
            logging.info("epoch=%d train_MAE=%.5f val_MAE=%.5f best=%.5f", epoch,
                         train_mae, val_mae, best)
            logging.info("per_R_C=%s", groups)
    if args.predict_test:
        test_df = pd.read_csv(args.test)
        xt, _, mt, test_ids, _, _ = prepare(test_df, False)
        checkpoint = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
        xt = ((xt - checkpoint["mean"]) / checkpoint["std"]).astype(np.float32)
        test_loader = DataLoader(BreathDataset(xt, None, mt), batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.workers)
        model.load_state_dict(checkpoint["model"])
        test_pred = predict(model, test_loader, device)
        pred_df = pd.DataFrame({"id": test_ids.reshape(-1), "pressure": test_pred.reshape(-1)})
        pred_df = test_df[["id"]].merge(pred_df, on="id", how="left", validate="one_to_one", sort=False)
        assert pred_df.pressure.notna().all() and len(pred_df) == len(test_df)
        pred_df.to_csv(out / "submission.csv", index=False)
        logging.info("Wrote %s", out / "submission.csv")


if __name__ == "__main__":
    main()
