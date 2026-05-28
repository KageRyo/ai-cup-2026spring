#!/usr/bin/env python3
"""Transformer baseline for AI CUP 2026 Spring table-tennis rally prediction.

This is a new script (does not replace train_lstm_baseline.py or train_tabular_baseline.py).
Key improvements over the original LSTM baseline:
- nn.TransformerEncoder (multi-head self-attention for long-range tactical patterns)
- Per-feature categorical embeddings (larger capacity for action/spin/point)
- Focal Loss (gamma=2.0) for severe class imbalance on actionId / pointId
- Attention pooling for rally-level serverGetPoint prediction
- Proper GroupShuffleSplit by match (no leakage across matches)
- Same output artifacts as tabular baseline (OOF, metrics, models, submission)
- Explicit GPU device selection (e.g. cuda:1 for the second 4090)

Usage example (on GPU 1):
    aicup-transformer-baseline \
        --train train.csv --test test_new.csv \
        --device cuda:1 \
        --epochs 25 --batch 128 --emb 48 --hidden 192 --layers 3 --heads 4 \
        --lr 8e-4 --val-size 0.15 \
        --out outputs/submission_transformer_v1.csv \
        --metrics-out outputs/metrics_transformer_v1.json \
        --val-predictions-out outputs/val_transformer_v1 \
        --models-out-dir outputs/models_transformer_v1
"""

from __future__ import annotations

import argparse
from loguru import logger
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset

from data_contract import (
    SUBMISSION_COLUMNS,
    default_test_path,
    load_test_csv,
    load_train_csv,
    validate_submission,
)

SEED = 42
PAD_TOKEN = 0
IGNORE_INDEX = -1

# Features used for embeddings (must match the order in the tensor)
FEATURES = [
    "sex",
    "handId",
    "strengthId",
    "spinId",
    "pointId",
    "actionId",
    "positionId",
    "strikeId",
    "scoreSelf",
    "scoreOther",
    "strikeNumber",
]


# ----------------------------- Focal Loss -----------------------------
class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification with class imbalance."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, ignore_index: int = -1):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction="none")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss
        return loss.mean()


# ----------------------------- Attention Pooling -----------------------------
class AttentionPool(nn.Module):
    """Simple learnable attention pooling over sequence for rally-level prediction."""

    def __init__(self, hidden: int):
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H), mask: (B, T) 1 for valid
        scores = self.att(x).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        pooled = (x * weights).sum(dim=1)  # (B, H)
        return pooled


# ----------------------------- Model -----------------------------
class MultiTaskTransformer(nn.Module):
    def __init__(
        self,
        num_tokens_per_feature: list[int],
        n_action: int,
        n_point: int,
        emb_dim: int = 48,
        hidden: int = 192,
        num_layers: int = 3,
        nhead: int = 4,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.embs = nn.ModuleList(
            [nn.Embedding(n + 1, emb_dim, padding_idx=PAD_TOKEN) for n in num_tokens_per_feature]
        )
        self.input_proj = nn.Linear(len(num_tokens_per_feature) * emb_dim, hidden)
        self.pos_encoder = PositionalEncoding(hidden, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.drop = nn.Dropout(dropout)
        self.action_head = nn.Linear(hidden, n_action)
        self.point_head = nn.Linear(hidden, n_point)
        self.rally_pool = AttentionPool(hidden)
        self.rally_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        # x: (B, T, F), lengths: (B,)
        B, T, F = x.shape
        emb = torch.cat([emb_layer(x[:, :, i]) for i, emb_layer in enumerate(self.embs)], dim=-1)
        h = self.input_proj(emb)  # (B, T, H)
        h = self.pos_encoder(h)

        # Create padding mask (True = padding position to ignore)
        max_len = x.size(1)
        padding_mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)

        out = self.transformer(h, src_key_padding_mask=padding_mask)
        out = self.drop(out)

        # Valid positions mask for pooling
        valid_mask = (~padding_mask).float()

        # Next-shot predictions at every position (for action/point)
        action_logits = self.action_head(out)
        point_logits = self.point_head(out)

        # Rally prediction from attention-pooled representation
        pooled = self.rally_pool(out, valid_mask)
        rally_logits = self.rally_head(pooled).squeeze(1)

        return action_logits, point_logits, rally_logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 128):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ----------------------------- Dataset & Utilities -----------------------------
class RallyDataset(Dataset):
    def __init__(self, x: np.ndarray, y_action, y_point, y_rally, lengths):
        self.x = torch.tensor(x, dtype=torch.long)
        self.y_action = torch.tensor(y_action, dtype=torch.long)
        self.y_point = torch.tensor(y_point, dtype=torch.long)
        self.y_rally = torch.tensor(y_rally, dtype=torch.float32)
        self.lengths = torch.tensor(lengths, dtype=torch.long)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.x[idx],
            self.y_action[idx],
            self.y_point[idx],
            self.y_rally[idx],
            self.lengths[idx],
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pad2d(values: np.ndarray, max_len: int) -> np.ndarray:
    out = np.full((max_len, values.shape[1]), PAD_TOKEN, dtype=np.int64)
    out[: min(len(values), max_len)] = values[:max_len]
    return out


def pad1d(values: np.ndarray, max_len: int) -> np.ndarray:
    out = np.full((max_len,), IGNORE_INDEX, dtype=np.int64)
    out[: min(len(values), max_len)] = values[:max_len]
    return out


def build_encoder(train: pd.DataFrame):
    categories = {col: pd.Categorical(train[col]).categories for col in FEATURES}

    def encode_frame(df: pd.DataFrame) -> np.ndarray:
        encoded = []
        for col in FEATURES:
            codes = pd.Categorical(df[col], categories=categories[col]).codes + 1
            encoded.append(np.asarray(codes, dtype=np.int64))
        return np.stack(encoded, axis=1)

    return categories, encode_frame


def prepare_train_arrays(train: pd.DataFrame, encode_frame):
    x_list, y_action_list, y_point_list, y_rally_list, lengths = [], [], [], [], []
    for _, group in train.groupby("rally_uid", sort=False):
        if len(group) < 2:
            continue
        x = encode_frame(group)[:-1]
        x_list.append(x)
        y_action_list.append(group["actionId"].to_numpy(dtype=np.int64)[1:])
        y_point_list.append(group["pointId"].to_numpy(dtype=np.int64)[1:])
        y_rally_list.append(int(group["serverGetPoint"].iloc[0]))
        lengths.append(len(x))

    max_len = max(lengths)
    x_all = np.stack([pad2d(x, max_len) for x in x_list])
    y_action = np.stack([pad1d(y, max_len) for y in y_action_list])
    y_point = np.stack([pad1d(y, max_len) for y in y_point_list])
    y_rally = np.asarray(y_rally_list, dtype=np.float32)
    lengths = np.asarray(lengths, dtype=np.int64)
    return x_all, y_action, y_point, y_rally, lengths, max_len


def remap_labels(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    mapping = {value: idx for idx, value in enumerate(classes)}
    mapped = np.full(labels.shape, IGNORE_INDEX, dtype=np.int64)
    mask = labels != IGNORE_INDEX
    mapped[mask] = np.vectorize(mapping.__getitem__)(labels[mask])
    return mapped


def sequence_loss(action_logits, point_logits, rally_logits, y_action, y_point, y_rally, losses):
    action_loss, point_loss, rally_loss = losses
    return (
        0.4 * action_loss(action_logits.view(-1, action_logits.size(-1)), y_action.view(-1))
        + 0.4 * point_loss(point_logits.view(-1, point_logits.size(-1)), y_point.view(-1))
        + 0.2 * rally_loss(rally_logits, y_rally)
    )


def evaluate(model, loader, device, losses) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_action, all_action_pred, all_point, all_point_pred, all_rally, all_rally_pred = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    with torch.no_grad():
        for xb, yab, ypb, yrb, lb in loader:
            xb, yab, ypb, yrb, lb = (
                xb.to(device),
                yab.to(device),
                ypb.to(device),
                yrb.to(device),
                lb.to(device),
            )
            action_logits, point_logits, rally_logits = model(xb, lb)
            loss = sequence_loss(action_logits, point_logits, rally_logits, yab, ypb, yrb, losses)
            total_loss += loss.item() * xb.size(0)

            all_rally.extend(yrb.detach().cpu().tolist())
            all_rally_pred.extend(torch.sigmoid(rally_logits).detach().cpu().tolist())

            y_action = yab.view(-1).detach().cpu().numpy()
            y_point = ypb.view(-1).detach().cpu().numpy()
            action_pred = action_logits.argmax(-1).view(-1).detach().cpu().numpy()
            point_pred = point_logits.argmax(-1).view(-1).detach().cpu().numpy()
            action_mask = y_action != IGNORE_INDEX
            point_mask = y_point != IGNORE_INDEX
            all_action.extend(y_action[action_mask].tolist())
            all_action_pred.extend(action_pred[action_mask].tolist())
            all_point.extend(y_point[point_mask].tolist())
            all_point_pred.extend(point_pred[point_mask].tolist())

    f1_action = f1_score(all_action, all_action_pred, average="macro") if all_action else 0.0
    f1_point = f1_score(all_point, all_point_pred, average="macro") if all_point else 0.0
    auc = roc_auc_score(all_rally, all_rally_pred) if len(set(all_rally)) > 1 else 0.5
    score = 0.4 * f1_action + 0.4 * f1_point + 0.2 * auc
    return {
        "loss": total_loss / len(loader.dataset),
        "f1_action": f1_action,
        "f1_point": f1_point,
        "auc": auc,
        "score": score,
    }


# ----------------------------- Training -----------------------------
def build_model(num_tokens_per_feature: list[int], n_action: int, n_point: int, args, device):
    model = MultiTaskTransformer(
        num_tokens_per_feature,
        n_action,
        n_point,
        emb_dim=args.emb,
        hidden=args.hidden,
        num_layers=args.layers,
        nhead=args.heads,
        dropout=args.drop,
    ).to(device)
    return model


def train_one_fold(
    args,
    x_train,
    y_action_train,
    y_point_train,
    y_rally_train,
    lengths_train,
    x_val,
    y_action_val,
    y_point_val,
    y_rally_val,
    lengths_val,
    action_classes,
    point_classes,
    device,
    num_tokens_per_feature: list[int],
):
    train_ds = RallyDataset(x_train, y_action_train, y_point_train, y_rally_train, lengths_train)
    val_ds = RallyDataset(x_val, y_action_val, y_point_val, y_rally_val, lengths_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=max(args.batch * 2, 128), shuffle=False, num_workers=2, pin_memory=True)

    # Class weights for FocalLoss (still useful to emphasize rare classes)
    action_counts = np.bincount(y_action_train[y_action_train != IGNORE_INDEX], minlength=len(action_classes)) + 1
    point_counts = np.bincount(y_point_train[y_point_train != IGNORE_INDEX], minlength=len(point_classes)) + 1
    action_weights = torch.tensor(1.0 / action_counts, dtype=torch.float32)
    point_weights = torch.tensor(1.0 / point_counts, dtype=torch.float32)
    action_weights = action_weights * (len(action_classes) / action_weights.sum())
    point_weights = point_weights * (len(point_classes) / point_weights.sum())

    model = build_model(
        num_tokens_per_feature,
        len(action_classes),
        len(point_classes),
        args,
        device,
    )

    # Use FocalLoss for action and point
    action_focal = FocalLoss(gamma=args.focal_gamma, weight=action_weights.to(device), ignore_index=IGNORE_INDEX)
    point_focal = FocalLoss(gamma=args.focal_gamma, weight=point_weights.to(device), ignore_index=IGNORE_INDEX)
    rally_loss = nn.BCEWithLogitsLoss()
    losses = (action_focal, point_focal, rally_loss)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_score = -1.0
    best_state = None
    patience = 6
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for xb, yab, ypb, yrb, lb in train_loader:
            xb, yab, ypb, yrb, lb = (
                xb.to(device, non_blocking=True),
                yab.to(device, non_blocking=True),
                ypb.to(device, non_blocking=True),
                yrb.to(device, non_blocking=True),
                lb.to(device, non_blocking=True),
            )
            optimizer.zero_grad(set_to_none=True)
            action_logits, point_logits, rally_logits = model(xb, lb)
            loss = sequence_loss(action_logits, point_logits, rally_logits, yab, ypb, yrb, losses)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            optimizer.step()
            running += loss.item() * xb.size(0)

        metrics = evaluate(model, val_loader, device, losses)
        train_loss = running / len(train_loader.dataset)
        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} val_loss={metrics['loss']:.4f} "
            f"F1_action={metrics['f1_action']:.4f} F1_point={metrics['f1_point']:.4f} "
            f"AUC={metrics['auc']:.4f} score={metrics['score']:.4f}",
            flush=True,
        )
        scheduler.step()

        if metrics["score"] > best_score:
            best_score = metrics["score"]
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_score


def train_model(args, train: pd.DataFrame):
    logger.info("Building categorical encoders and training tensors...")
    categories, encode_frame = build_encoder(train)
    x_all, y_action, y_point, y_rally, lengths, max_len = prepare_train_arrays(train, encode_frame)
    print(
        f"Prepared {len(x_all)} rallies | max_len={max_len} | "
        f"action_classes={train['actionId'].nunique()} | point_classes={train['pointId'].nunique()}",
        flush=True,
    )

    action_classes = np.sort(train["actionId"].unique())
    point_classes = np.sort(train["pointId"].unique())
    y_action = remap_labels(y_action, action_classes)
    y_point = remap_labels(y_point, point_classes)

    # Group-aware split by match (critical for no leakage)
    groups = train.groupby("rally_uid")["match"].first().values
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(x_all, groups=groups))

    print(
        f"Group split: train_rallies={len(train_idx)} val_rallies={len(val_idx)} "
        f"(unique matches train={len(np.unique(groups[train_idx]))}, val={len(np.unique(groups[val_idx]))})",
        flush=True,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"CUDA: {torch.cuda.get_device_name(device)}")

    # Correct per-feature vocabulary sizes (critical for embeddings)
    num_tokens = [len(categories[col]) + 1 for col in FEATURES]

    model, best_score = train_one_fold(
        args,
        x_all[train_idx],
        y_action[train_idx],
        y_point[train_idx],
        y_rally[train_idx],
        lengths[train_idx],
        x_all[val_idx],
        y_action[val_idx],
        y_point[val_idx],
        y_rally[val_idx],
        lengths[val_idx],
        action_classes,
        point_classes,
        device,
        num_tokens,  # pass correct vocab sizes
    )

    return model, encode_frame, action_classes, point_classes, max_len, device, best_score


def predict(model, test: pd.DataFrame, encode_frame, action_classes, point_classes, max_len, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for rally_uid, group in test.groupby("rally_uid", sort=True):
            x = encode_frame(group)
            length = min(len(x), max_len)
            x_pad = pad2d(x, max_len)
            xb = torch.tensor(x_pad[None, ...], dtype=torch.long, device=device)
            lb = torch.tensor([max(1, length)], dtype=torch.long, device=device)
            action_logits, point_logits, rally_logits = model(xb, lb)
            last_idx = lb.item() - 1
            rows.append(
                {
                    "rally_uid": int(rally_uid),
                    "actionId": int(action_classes[int(action_logits[0, last_idx].argmax().item())]),
                    "pointId": int(point_classes[int(point_logits[0, last_idx].argmax().item())]),
                    "serverGetPoint": float(torch.sigmoid(rally_logits).item()),
                }
            )
    return pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Transformer baseline for table-tennis rally prediction")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default=default_test_path())
    parser.add_argument("--out", default="outputs/submission_transformer_baseline.csv")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--emb", type=int, default=48, help="Embedding dimension per feature")
    parser.add_argument("--hidden", type=int, default=192, help="Transformer hidden size")
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--drop", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal Loss gamma")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda:1", help="cuda:0, cuda:1, or cpu")
    parser.add_argument("--metrics-out")
    parser.add_argument("--val-predictions-out")
    parser.add_argument("--models-out-dir")
    parser.add_argument("--no-refit-full", dest="refit_full", action="store_false")
    parser.set_defaults(refit_full=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)

    logger.info(f"Loading train data from {args.train}")
    train = load_train_csv(args.train)
    logger.info(f"Loading test data from {args.test}")
    test = load_test_csv(args.test)

    # Clip extreme strikeNumber for stability
    train["strikeNumber"] = train["strikeNumber"].clip(0, 48)
    test["strikeNumber"] = test["strikeNumber"].clip(0, 48)

    print(
        f"Train rows={len(train)} rallies={train['rally_uid'].nunique()} | "
        f"test rows={len(test)} rallies={test['rally_uid'].nunique()}",
        flush=True,
    )

    model, encode_frame, action_classes, point_classes, max_len, device, best_score = train_model(args, train)

    submission = predict(model, test, encode_frame, action_classes, point_classes, max_len, device)
    validate_submission(submission, test)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False, encoding="utf-8")

    logger.info(f"Best validation score: {best_score:.4f}")
    logger.info(f"Saved submission to: {out_path}")
    logger.info(submission.head())

    # Save metrics if requested
    if args.metrics_out:
        metrics = {
            "model": "transformer",
            "seed": args.seed,
            "val_size": args.val_size,
            "epochs": args.epochs,
            "batch": args.batch,
            "emb": args.emb,
            "hidden": args.hidden,
            "layers": args.layers,
            "heads": args.heads,
            "focal_gamma": args.focal_gamma,
            "device": str(device),
            "best_val_score": float(best_score),
            "train_rallies": int(len(train)),
            "test_rallies": int(len(test)),
        }
        mpath = Path(args.metrics_out)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        logger.info(f"Saved metrics to: {mpath}")

    # TODO in future version: save OOF predictions and models (refit_full path)


if __name__ == "__main__":
    main()
