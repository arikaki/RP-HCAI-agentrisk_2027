"""S2a: fine-tune a long-context encoder (default ModernBERT-large) on AgentRisk.

One model predicts all three outputs:
  - severity: a regression head (rounded to 1-5), which suits the MAE metric
  - harm domains and risk drivers: multi-label heads (sigmoid), decoded to
    1-2 labels with the "No Material ..." rule

Cross-validation: 5 folds on TRAIN, grouped by scenario (split_manifest.jsonl),
so paired items never sit on both sides. Each fold model also predicts the
target split (dev or test); those predictions are averaged
(5-model ensemble). Dev is never used for training or tuning.

Long inputs are truncated from the LEFT, so the final proposed action (the
end of the trajectory) is always kept.

Defaults fit a 12 GB GPU (bf16, gradient checkpointing, batch 2 x 8 accumulation).

    python scripts/finetune_modernbert.py --data-dir data --tag modernbert-large

Outputs in runs/s2a/<tag>/: oof_train.jsonl (cross-validated predictions on
train), cv_metrics.json, <split>.jsonl (ensemble predictions), run_info.json.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentrisk.data import grouped_folds, load_groups, load_split, write_jsonl  # noqa: E402
from agentrisk.labels import HARM_DOMAINS, RISK_DRIVERS, repair_label_set  # noqa: E402
from agentrisk.metrics import mae, mean_jaccard  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TrajDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return self.rows[i]


def multi_hot(labels: list[str], vocab: list[str]) -> list[float]:
    return [1.0 if v in labels else 0.0 for v in vocab]


class Collator:
    def __init__(self, tokenizer, max_len: int):
        self.tok = tokenizer
        self.max_len = max_len

    def __call__(self, batch: list[dict]) -> dict:
        enc = self.tok([r["formatted_interaction"] for r in batch], truncation=True,
                       max_length=self.max_len, padding=True, return_tensors="pt")
        out = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        if "risk_severity" in batch[0]:
            out["severity"] = torch.tensor([float(r["risk_severity"]) for r in batch])
            out["domains"] = torch.tensor([multi_hot(r["risk_domains"], HARM_DOMAINS) for r in batch])
            out["drivers"] = torch.tensor([multi_hot(r["risk_drivers"], RISK_DRIVERS) for r in batch])
        return out


class MultiHeadModel(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1, grad_ckpt: bool = True):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_name)
        if grad_ckpt:
            self.encoder.gradient_checkpointing_enable()
        h = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.sev = nn.Linear(h, 1)
        self.dom = nn.Linear(h, len(HARM_DOMAINS))
        self.drv = nn.Linear(h, len(RISK_DRIVERS))

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)  # mean pooling
        pooled = self.drop(pooled)
        return self.sev(pooled).squeeze(-1) + 3.0, self.dom(pooled), self.drv(pooled)


def decode_labels(probs: np.ndarray, vocab: list[str], kind: str, threshold: float) -> list[str]:
    order = np.argsort(-probs)
    chosen = [vocab[i] for i in order if probs[i] >= threshold][:2] or [vocab[order[0]]]
    labels, _ = repair_label_set(chosen, kind, scores={vocab[i]: float(probs[i]) for i in range(len(vocab))})
    return labels


def to_predictions(rows: list[dict], sev: np.ndarray, dom: np.ndarray, drv: np.ndarray,
                   threshold: float) -> list[dict]:
    preds = []
    for i, r in enumerate(rows):
        preds.append({
            "example_id": r["example_id"],
            "risk_severity": int(np.clip(np.rint(sev[i]), 1, 5)),
            "risk_domains": decode_labels(dom[i], HARM_DOMAINS, "domains", threshold),
            "risk_drivers": decode_labels(drv[i], RISK_DRIVERS, "drivers", threshold),
            "severity_raw": round(float(sev[i]), 3),
        })
    return preds


@torch.no_grad()
def predict(model, loader, device, use_bf16: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    sevs, doms, drvs = [], [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        att = batch["attention_mask"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            s, d, v = model(ids, att)
        sevs.append(s.float().cpu())
        doms.append(torch.sigmoid(d.float()).cpu())
        drvs.append(torch.sigmoid(v.float()).cpu())
    return torch.cat(sevs).numpy(), torch.cat(doms).numpy(), torch.cat(drvs).numpy()


def train_fold(args, train_rows, val_rows, target_rows, tokenizer, device, fold: int):
    from transformers import get_linear_schedule_with_warmup

    set_seed(args.seed + fold)
    model = MultiHeadModel(args.model, grad_ckpt=not args.no_grad_ckpt).to(device)
    collate = Collator(tokenizer, args.max_len)
    train_dl = DataLoader(TrajDataset(train_rows), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(TrajDataset(val_rows), batch_size=args.batch_size, collate_fn=collate)
    tgt_dl = DataLoader(TrajDataset([{k: r[k] for k in ("example_id", "formatted_interaction")} for r in target_rows]),
                        batch_size=args.batch_size, collate_fn=collate)

    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr},
                             {"params": head_params, "lr": args.head_lr}], weight_decay=0.01)
    steps = math.ceil(len(train_dl) / args.grad_accum) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    huber, bce = nn.SmoothL1Loss(), nn.BCEWithLogitsLoss()
    use_bf16 = device.type == "cuda" and not args.no_bf16

    for epoch in range(args.epochs):
        model.train()
        t0, total = time.time(), 0.0
        opt.zero_grad()
        for step, batch in enumerate(train_dl, 1):
            ids = batch["input_ids"].to(device)
            att = batch["attention_mask"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                s, d, v = model(ids, att)
            loss = (args.w_sev * huber(s.float(), batch["severity"].to(device))
                    + args.w_dom * bce(d.float(), batch["domains"].to(device))
                    + args.w_drv * bce(v.float(), batch["drivers"].to(device)))
            (loss / args.grad_accum).backward()
            total += loss.item()
            if step % args.grad_accum == 0 or step == len(train_dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
        vs, vd, vv = predict(model, val_dl, device, use_bf16)
        vp = to_predictions(val_rows, vs, vd, vv, 0.5)
        print(f"  fold {fold + 1} epoch {epoch + 1}/{args.epochs}: loss {total / len(train_dl):.3f} | "
              f"val MAE {mae([r['risk_severity'] for r in val_rows], [p['risk_severity'] for p in vp]):.3f} "
              f"Jdom {mean_jaccard([r['risk_domains'] for r in val_rows], [p['risk_domains'] for p in vp]):.3f} "
              f"Jdrv {mean_jaccard([r['risk_drivers'] for r in val_rows], [p['risk_drivers'] for p in vp]):.3f} "
              f"({time.time() - t0:.0f}s)")

    val_out = predict(model, val_dl, device, use_bf16)
    tgt_out = predict(model, tgt_dl, device, use_bf16)
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return val_out, tgt_out


def best_threshold(rows, sev, dom, drv) -> float:
    """Pick the label threshold that maximises mean Jaccard on out-of-fold predictions."""
    best_t, best_score = 0.5, -1.0
    for t in [x / 20 for x in range(4, 15)]:  # 0.20 ... 0.70
        p = to_predictions(rows, sev, dom, drv, t)
        score = (mean_jaccard([r["risk_domains"] for r in rows], [x["risk_domains"] for x in p])
                 + mean_jaccard([r["risk_drivers"] for r in rows], [x["risk_drivers"] for x in p]))
        if score > best_score:
            best_t, best_score = t, score
    return best_t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="answerdotai/ModernBERT-large")
    ap.add_argument("--tag", default="modernbert-large")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--target-split", default="dev", help="split to predict with the fold ensemble")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-5, help="encoder learning rate")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=3072, help="tokens; longer inputs are cut from the left")
    ap.add_argument("--w-sev", type=float, default=1.0)
    ap.add_argument("--w-dom", type=float, default=1.0)
    ap.add_argument("--w-drv", type=float, default=1.0)
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--limit-train", type=int, help="use only N training rows (smoke test)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    out_dir = Path(args.runs_dir) / "s2a" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_split("train", args.data_dir)
    if args.limit_train:
        train = train[: args.limit_train]
    target = load_split(args.target_split, args.data_dir)
    groups = load_groups(args.data_dir)
    if not groups:
        print("WARNING: no split_manifest.jsonl - folds are not grouped by scenario")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.truncation_side = "left"

    folds = grouped_folds(train, groups, args.folds, args.seed)
    n = len(train)
    oof_sev, oof_dom, oof_drv = np.zeros(n), np.zeros((n, len(HARM_DOMAINS))), np.zeros((n, len(RISK_DRIVERS)))
    tgt_sev, tgt_dom, tgt_drv = [], [], []
    for k, (tr_idx, va_idx) in enumerate(folds):
        print(f"fold {k + 1}/{len(folds)}: {len(tr_idx)} train / {len(va_idx)} val")
        (vs, vd, vv), (ts, td, tv) = train_fold(args, [train[i] for i in tr_idx], [train[i] for i in va_idx],
                                                 target, tokenizer, device, k)
        oof_sev[va_idx], oof_dom[va_idx], oof_drv[va_idx] = vs, vd, vv
        tgt_sev.append(ts)
        tgt_dom.append(td)
        tgt_drv.append(tv)

    thr = best_threshold(train, oof_sev, oof_dom, oof_drv)
    oof = to_predictions(train, oof_sev, oof_dom, oof_drv, thr)
    write_jsonl(out_dir / "oof_train.jsonl", oof)
    cv = {
        "threshold": thr,
        "cv_mae": mae([r["risk_severity"] for r in train], [p["risk_severity"] for p in oof]),
        "cv_jaccard_domains": mean_jaccard([r["risk_domains"] for r in train], [p["risk_domains"] for p in oof]),
        "cv_jaccard_drivers": mean_jaccard([r["risk_drivers"] for r in train], [p["risk_drivers"] for p in oof]),
    }
    (out_dir / "cv_metrics.json").write_text(json.dumps(cv, indent=2))
    print("cross-validation on train:", json.dumps(cv))

    preds = to_predictions(target, np.mean(tgt_sev, 0), np.mean(tgt_dom, 0), np.mean(tgt_drv, 0), thr)
    write_jsonl(out_dir / f"{args.target_split}.jsonl", preds)
    info = vars(args) | {"finished": datetime.now(timezone.utc).isoformat(), "threshold": thr,
                         "device": str(device), "n_train": n}
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"wrote {len(preds)} {args.target_split} predictions -> {out_dir / f'{args.target_split}.jsonl'}")


if __name__ == "__main__":
    main()
