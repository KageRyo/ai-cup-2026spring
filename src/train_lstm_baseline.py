from __future__ import annotations

import argparse
from loguru import logger
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
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


class MultiTaskLSTM(nn.Module):
    def __init__(
        self,
        num_tokens_per_feature: list[int],
        n_action: int,
        n_point: int,
        emb_dim: int,
        hidden: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.embs = nn.ModuleList(
            [nn.Embedding(n + 1, emb_dim, padding_idx=PAD_TOKEN) for n in num_tokens_per_feature]
        )
        self.lstm = nn.LSTM(
            len(num_tokens_per_feature) * emb_dim,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.action_head = nn.Linear(hidden, n_action)
        self.point_head = nn.Linear(hidden, n_point)
        self.rally_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        emb = torch.cat([emb_layer(x[:, :, idx]) for idx, emb_layer in enumerate(self.embs)], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        out = self.drop(out)

        mask = (x[:, :, 0] != PAD_TOKEN).float().unsqueeze(-1)
        mean_hidden = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return (
            self.action_head(out),
            self.point_head(out),
            self.rally_head(mean_hidden).squeeze(1),
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

    idx = np.arange(len(x_all))
    tr_idx, va_idx = train_test_split(
        idx, test_size=args.val_size, random_state=args.seed, stratify=y_rally
    )

    train_ds = RallyDataset(x_all[tr_idx], y_action[tr_idx], y_point[tr_idx], y_rally[tr_idx], lengths[tr_idx])
    val_ds = RallyDataset(x_all[va_idx], y_action[va_idx], y_point[va_idx], y_rally[va_idx], lengths[va_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=max(args.batch * 2, 128), shuffle=False)

    action_counts = np.bincount(y_action[tr_idx][y_action[tr_idx] != IGNORE_INDEX], minlength=len(action_classes)) + 1
    point_counts = np.bincount(y_point[tr_idx][y_point[tr_idx] != IGNORE_INDEX], minlength=len(point_classes)) + 1
    action_weights = torch.tensor(1.0 / action_counts, dtype=torch.float32)
    point_weights = torch.tensor(1.0 / point_counts, dtype=torch.float32)
    action_weights = action_weights * (len(action_classes) / action_weights.sum())
    point_weights = point_weights * (len(point_classes) / point_weights.sum())

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(
            f"CUDA devices: {torch.cuda.device_count()} | current: {torch.cuda.get_device_name(0)}",
            flush=True,
        )

    model = MultiTaskLSTM(
        [len(categories[col]) + 1 for col in FEATURES],
        len(action_classes),
        len(point_classes),
        emb_dim=args.emb,
        hidden=args.hidden,
        num_layers=args.layers,
        dropout=args.drop,
    ).to(device)

    losses = (
        nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, weight=action_weights.to(device)),
        nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, weight=point_weights.to(device)),
        nn.BCEWithLogitsLoss(),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_score = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for xb, yab, ypb, yrb, lb in train_loader:
            xb, yab, ypb, yrb, lb = (
                xb.to(device),
                yab.to(device),
                ypb.to(device),
                yrb.to(device),
                lb.to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            action_logits, point_logits, rally_logits = model(xb, lb)
            loss = sequence_loss(action_logits, point_logits, rally_logits, yab, ypb, yrb, losses)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        if metrics["score"] > best_score:
            best_score = metrics["score"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default=default_test_path())
    parser.add_argument("--out", default="outputs/submission_lstm_baseline.csv")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--emb", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--drop", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)

    logger.info(f"Loading train data from {args.train}")
    train = load_train_csv(args.train)
    logger.info(f"Loading test data from {args.test}")
    test = load_test_csv(args.test)
    train["strikeNumber"] = train["strikeNumber"].clip(0, 40)
    test["strikeNumber"] = test["strikeNumber"].clip(0, 40)
    print(
        f"Train rows={len(train)} rallies={train['rally_uid'].nunique()} | "
        f"test rows={len(test)} rallies={test['rally_uid'].nunique()}",
        flush=True,
    )

    model, encode_frame, action_classes, point_classes, max_len, device, best_score = train_model(
        args, train
    )
    submission = predict(model, test, encode_frame, action_classes, point_classes, max_len, device)
    validate_submission(submission, test)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False, encoding="utf-8")
    logger.info(f"Best validation score: {best_score:.4f}")
    logger.info(f"Saved submission to: {out_path}")
    logger.info(submission.head())


if __name__ == "__main__":
    main()
