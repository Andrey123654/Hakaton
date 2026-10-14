# -*- coding: utf-8 -*-
"""
blend_generations.py — объединение нескольких поколений моделей по рангам.

Разные поколения решения (разный набор признаков, разные схемы ансамблирования)
ошибаются на разных объектах. Ранговое усреднение таких предсказаний устойчиво
даёт прирост: веса подбираются жадным покоординатным поиском по OOF F1, а сама
шкала рангов переносится на тест без переобучения (ранги нормированы в [0,1],
поэтому подобранный порог применим напрямую).

Запуск:
    python tools/blend_generations.py --out submission_blend.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from submission import build_submission  # noqa: E402


def rank01(x: np.ndarray) -> np.ndarray:
    return rankdata(x) / len(x)


def best_f1(y: np.ndarray, score: np.ndarray, grid: int = 400):
    best, bt = 0.0, 0.5
    for t in np.quantile(score, np.linspace(0.02, 0.98, grid)):
        f1 = f1_score(y, (score >= t).astype(int))
        if f1 > best:
            best, bt = f1, float(t)
    return best, bt


def main() -> None:
    ap = argparse.ArgumentParser(description="Ранговое объединение поколений моделей")
    ap.add_argument("--sources", default="",
                    help="список пар oof:test через запятую; по умолчанию ищутся автоматически")
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--test", default="data/test.csv")
    ap.add_argument("--sample", default="data/sample.csv")
    ap.add_argument("--out", default="submission_blend.csv")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--weights", default="",
                    help="явные веса источников: meta_nn=0.3,meta_nn2=0.3,nn2=0.2,final=0.2; "
                         "если не заданы — жадный подбор по OOF F1")
    args = ap.parse_args()

    tr = pd.read_csv(args.train)
    grp = tr.groupby("url")["result"].nunique()
    tr = tr[~tr["url"].isin(set(grp[grp > 1].index))]
    tr = tr.drop_duplicates(subset=["url"]).reset_index(drop=True)
    y = tr["result"].values

    if args.sources:
        pairs = [tuple(p.split(":")) for p in args.sources.split(",")]
    else:
        pairs = []
        for oof_path in sorted(glob.glob(os.path.join(ROOT, "models", "oof_final_*.npy"))):
            tag = os.path.basename(oof_path)[len("oof_final_"):-len(".npy")]
            t = os.path.join(ROOT, "models", f"test_proba_{tag}.npy")
            if os.path.exists(t):
                pairs.append((oof_path, t))
        for oof_path in sorted(glob.glob(os.path.join(ROOT, "models", "meta_oof_*.npy"))):
            tag = os.path.basename(oof_path)[len("meta_oof_"):-len(".npy")]
            t = os.path.join(ROOT, "models", f"meta_proba_test_{tag}.npy")
            if os.path.exists(t):
                pairs.append((oof_path, t))

    names, R_oof, R_te, single = [], [], [], {}
    for oof_path, te_path in pairs:
        o = np.load(oof_path)
        t = np.load(te_path)
        if len(o) != len(y):
            print(f"[warn] {oof_path}: {len(o)} != {len(y)} — пропускаю")
            continue
        label = os.path.basename(oof_path).replace("oof_final_", "").replace("meta_oof_", "meta_").replace(".npy", "")
        names.append(label)
        R_oof.append(rank01(o))
        R_te.append(rank01(t))
        single[label] = best_f1(y, o)[0]
    R_oof = np.array(R_oof)
    R_te = np.array(R_te)
    print(f"[info] источников: {len(names)}")
    for n_, s_ in single.items():
        print(f"   {n_:<22} OOF F1 = {s_:.5f}")

    m = len(names)
    if args.weights:
        wmap = {}
        for part in args.weights.split(","):
            k, _, v = part.partition("=")
            wmap[k.strip()] = float(v)
        w = np.array([wmap.get(n_, 0.0) for n_ in names], dtype=float)
        if w.sum() <= 0:
            raise SystemExit("нулевые веса")
        w = w / w.sum()
        score = (R_oof * w[:, None]).sum(axis=0)
        best_score, best_thr = best_f1(y, score)
        print("\n[info] заданные веса:")
        for n_, w_ in zip(names, w):
            if w_ > 0:
                print(f"   {n_:<22} {w_:.3f}")
        print(f"[info] OOF F1 объединения: {best_score:.5f} (порог {best_thr:.4f})")
        score_te = (R_te * w[:, None]).sum(axis=0)
        test = pd.read_csv(args.test)
        sub = build_submission(test, score_te, best_thr, sample_path=args.sample)
        out_path = os.path.join(ROOT, args.out)
        sub.to_csv(out_path, index=False)
        print(f"[ok] {out_path}: {len(sub)} строк, доля фишинга {sub[sub.columns[-1]].mean():.4f}")
        with open(os.path.join(ROOT, "reports", "blend_report.json"), "w", encoding="utf-8") as fh:
            json.dump({"sources": names, "weights": w.tolist(), "oof_f1": best_score,
                       "threshold": best_thr, "single": single}, fh, ensure_ascii=False, indent=2)
        return

    counts = np.zeros(m)
    best_score, best_thr, best = 0.0, 0.5, None
    for it in range(args.iters):
        cand_best = None
        for j in range(m):
            w = counts.copy()
            w[j] += 1
            score = (R_oof * (w / w.sum())[:, None]).sum(axis=0)
            f1, thr = best_f1(y, score)
            if cand_best is None or f1 > cand_best[0]:
                cand_best = (f1, thr, j, score)
        if cand_best is None or (best is not None and cand_best[0] <= best_score + 1e-7):
            break
        best_score, best_thr, j, score = cand_best
        counts[j] += 1
        best = score
        print(f"  шаг {it + 1}: +{names[j]:<22} F1={best_score:.5f}")
    weights = counts / counts.sum()
    print("\n[info] итоговые веса:")
    for n_, w_ in zip(names, weights):
        if w_ > 0:
            print(f"   {n_:<22} {w_:.3f}")
    print(f"[info] OOF F1 объединения: {best_score:.5f} (порог {best_thr:.4f})")

    score_te = (R_te * weights[:, None]).sum(axis=0)
    test = pd.read_csv(args.test)
    sub = build_submission(test, score_te, best_thr, sample_path=args.sample)
    out_path = os.path.join(ROOT, args.out)
    sub.to_csv(out_path, index=False)
    print(f"[ok] {out_path}: {len(sub)} строк, доля фишинга {sub[sub.columns[-1]].mean():.4f}")
    with open(os.path.join(ROOT, "reports", "blend_report.json"), "w", encoding="utf-8") as fh:
        json.dump({"sources": names, "weights": weights.tolist(), "oof_f1": best_score,
                   "threshold": best_thr, "single": single}, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
