"""Train a CSLR model with CTC loss.

Example
-------
    python -m src.train --group full --head bilstm --epochs 40 --run full_bilstm
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import (ClipStore, FixedPhraseDataset, PhraseDataset, build_fixed_items,
                      collate, compute_stats, split_clip_pools)
from .decode import decode_batch
from .metrics import wer
from .models import build_model, count_params


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    # CTC on MPS is unreliable; these models are small enough that CPU is fine.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, loader, device, method="greedy", beam_width=10):
    model.eval()
    refs, hyps = [], []
    with torch.no_grad():
        for X, y, in_len, tgt_len in loader:
            lp = model(X.to(device), in_len.to(device)).cpu().numpy()
            out_len = torch.div(in_len, 2, rounding_mode="floor").clamp(min=1).numpy()
            hyps.extend(decode_batch(lp, out_len, method, beam_width))
            off = 0
            for L in tgt_len.tolist():
                refs.append(y[off:off + L].tolist())
                off += L
    return wer(refs, hyps), refs, hyps


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/raw")
    p.add_argument("--group", default="full", choices=["hands", "hands_pose", "full"])
    p.add_argument("--head", default="bilstm", choices=["bilstm", "transformer"])
    p.add_argument("--run", default=None, help="run name (default: <group>_<head>)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-phrases", type=int, default=1200)
    p.add_argument("--eval-phrases", type=int, default=300)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--holdout-session", default=None,
                   help="session id held out as the test set (recommended)")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    run = args.run or f"{args.group}_{args.head}"
    outdir = Path(args.out) / run
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)

    print(f"[{run}] loading clips from {args.data} (group={args.group})")
    store = ClipStore.load(args.data, group=args.group)
    print(f"  vocab={len(store.vocab)}  clips/gloss={store.counts()}")

    tr_pool, va_pool, te_pool = split_clip_pools(
        store, seed=args.seed, holdout_session=args.holdout_session)
    stats = compute_stats(store, tr_pool)

    train_ds = PhraseDataset(store, tr_pool, args.train_phrases, seed=args.seed, stats=stats)
    val_ds = FixedPhraseDataset(store, build_fixed_items(store, va_pool, args.eval_phrases,
                                                         seed=1234), stats=stats)
    test_ds = FixedPhraseDataset(store, build_fixed_items(store, te_pool, args.eval_phrases,
                                                          seed=99), stats=stats)
    dl = dict(batch_size=args.batch_size, collate_fn=collate, num_workers=0)
    train_dl = DataLoader(train_ds, shuffle=True, **dl)
    val_dl = DataLoader(val_ds, shuffle=False, **dl)
    test_dl = DataLoader(test_ds, shuffle=False, **dl)

    model = build_model(vars(args) | {"head": args.head, "d_model": args.d_model,
                                      "hidden": args.hidden, "layers": args.layers},
                        store.feature_dim, len(store.vocab)).to(device)
    print(f"  {args.head} head, {count_params(model):,} params, device={device}")

    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history, best, bad = [], float("inf"), 0
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, nb, t0 = 0.0, 0, time.time()
        for X, y, in_len, tgt_len in train_dl:
            X, in_len = X.to(device), in_len.to(device)
            lp = model(X, in_len).transpose(0, 1)          # (T, B, C) for CTCLoss
            out_len = model.out_lengths(in_len).cpu()
            loss = ctc(lp, y, out_len, tgt_len)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        vm, _, _ = evaluate(model, val_dl, device, "greedy")
        history.append({"epoch": ep, "loss": tot / max(nb, 1), "val_wer": vm["wer"],
                        "secs": time.time() - t0})
        print(f"  ep {ep:3d}  loss {tot / max(nb, 1):.4f}  val WER {vm['wer']:.4f}"
              f"  ({time.time() - t0:.1f}s)")

        if vm["wer"] < best - 1e-4:
            best, bad = vm["wer"], 0
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "vocab": store.vocab, "feature_dim": store.feature_dim,
                        "stats": (stats[0], stats[1])}, outdir / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop at epoch {ep} (best val WER {best:.4f})")
                break

    ck = torch.load(outdir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    greedy, refs, hyps = evaluate(model, test_dl, device, "greedy")
    beam, refs_b, hyps_b = evaluate(model, test_dl, device, "beam", 10)
    summary = {"run": run, "group": args.group, "head": args.head,
               "params": count_params(model), "best_val_wer": best,
               "test_greedy": greedy, "test_beam": beam, "vocab": store.vocab,
               "clips_per_gloss": store.counts(), "history": history,
               "holdout_session": args.holdout_session}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    np.save(outdir / "test_refs.npy", np.array(refs_b, dtype=object), allow_pickle=True)
    np.save(outdir / "test_hyps.npy", np.array(hyps_b, dtype=object), allow_pickle=True)
    print(f"[{run}] TEST  greedy WER {greedy['wer']:.4f} | beam WER {beam['wer']:.4f} "
          f"| sentence acc {beam['sentence_acc']:.4f}")


if __name__ == "__main__":
    main()
