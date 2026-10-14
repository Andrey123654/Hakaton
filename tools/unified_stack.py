# -*- coding: utf-8 -*-
"""
unified_stack.py — единый стекинг по всем поколениям моделей.

Вместо рангового усреднения готовых решений обучается одна мета-модель, на вход
которой подаются:

  * логиты предсказаний всех базовых моделей всех поколений
    (`models/oof_<tag>.npz` + `models/test_base_<tag>.npz`);
  * для поколений без сохранённых базовых предсказаний — их итоговое
    решение как отдельный столбец (`oof_final_<tag>.npy` + `test_proba_<tag>.npy`);
  * соседские признаки по четырём представлениям контента
    (слова, разметка, шаблоны URL-путей, заголовки) с исключением соседей
    из валидационного фолда.

Мета-модель — логистическая регрессия; порог подбирается по OOF F1.

Запуск:
    python tools/unified_stack.py --tags nn,nn2,nn3 --blend-tags final \
        --views "cache:word,cache:markup,cache:urlpath,cache_k100:title" \
        --out submission_final.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from submission import build_submission  # noqa: E402
from train import load_nn, nn_feature_block  # noqa: E402

EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def main() -> None:
    ap = argparse.ArgumentParser(description="Единый стекинг по поколениям моделей")
    ap.add_argument("--tags", default="nn,nn2,nn3",
                    help="поколения с сохранёнными базовыми предсказаниями")
    ap.add_argument("--blend-tags", default="final",
                    help="поколения, у которых берётся только итоговое решение")
    ap.add_argument("--views", default="cache:word,cache:markup,cache:urlpath,cache_k100:title")
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--test", default="data/test.csv")
    ap.add_argument("--sample", default="data/sample.csv")
    ap.add_argument("--out", default="submission_final.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    tr = pd.read_csv(args.train)
    grp = tr.groupby("url")["result"].nunique()
    tr = tr[~tr["url"].isin(set(grp[grp > 1].index))]
    tr = tr.drop_duplicates(subset=["url"]).reset_index(drop=True)
    y = tr["result"].values
    n = len(y)

    skf = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    fold_ids = np.zeros(n, dtype=np.int16)
    for f, (_, iva) in enumerate(skf.split(np.zeros(n), y), 1):
        fold_ids[iva] = f

    # ---------- логиты базовых моделей ----------
    cols_tr, cols_te, col_names = [], [], []
    for tag in [t.strip() for t in args.tags.split(",") if t.strip()]:
        o = np.load(os.path.join(ROOT, "models", f"oof_{tag}.npz"))
        tb = np.load(os.path.join(ROOT, "models", f"test_base_{tag}.npz"))
        for k in [k for k in o.files if k != "y"]:
            if k not in tb.files:
                raise SystemExit(f"нет test_base для {tag}/{k}")
            cols_tr.append(logit(o[k]))
            cols_te.append(logit(tb[k]))
            col_names.append(f"{tag}:{k}")
    for tag in [t.strip() for t in (args.blend_tags or "").split(",") if t.strip()]:
        o = os.path.join(ROOT, "models", f"oof_final_{tag}.npy")
        t = os.path.join(ROOT, "models", f"test_proba_{tag}.npy")
        if os.path.exists(o) and os.path.exists(t):
            cols_tr.append(logit(np.load(o)))
            cols_te.append(logit(np.load(t)))
            col_names.append(f"{tag}:blend")
    print(f"[info] базовых столбцов: {len(col_names)}")

    # ---------- соседские признаки ----------
    for spec in [s.strip() for s in args.views.split(",") if s.strip()]:
        d_, _, v = spec.rpartition(":")
        d_ = d_ or "cache"
        a = load_nn(os.path.join(ROOT, d_, f"nn_{v}_train.npz"))
        b = load_nn(os.path.join(ROOT, d_, f"nn_{v}_test.npz"))
        if a is None or b is None:
            print(f"[warn] {spec} недоступно — пропускаю")
            continue
        blk_tr = np.zeros((n, 16), dtype=np.float32)
        for fold in range(1, args.folds + 1):
            iva = fold_ids == fold
            f_blk, _ = nn_feature_block(a[0], a[1], y, fold_of=fold_ids,
                                        current_fold=fold, prefix=v)
            blk_tr[iva] = f_blk[iva]
        f_te, _ = nn_feature_block(b[0], b[1], y, prefix=v)
        cols_tr.append(blk_tr)
        cols_te.append(f_te)
        col_names.append(f"nn:{v}")
    print(f"[info] всего столбцов мета-модели: {len(col_names)}")

    Z = np.column_stack(cols_tr)
    Z_te = np.column_stack(cols_te)

    # ---------- OOF-оценка мета-модели ----------
    meta_oof = np.zeros(n)
    for fold in range(1, args.folds + 1):
        itr, iva = fold_ids != fold, fold_ids == fold
        sc = StandardScaler().fit(Z[itr])
        m = LogisticRegression(C=args.C, max_iter=4000, class_weight="balanced",
                               random_state=args.seed)
        m.fit(sc.transform(Z[itr]), y[itr])
        meta_oof[iva] = m.predict_proba(sc.transform(Z[iva]))[:, 1]
    best_f1, best_thr = 0.0, 0.5
    for t in np.quantile(meta_oof, np.linspace(0.005, 0.995, 600)):
        f1 = f1_score(y, (meta_oof >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_thr = f1, float(t)
    print(f"[info] OOF F1 единого стекинга: {best_f1:.5f} (порог {best_thr:.4f})")

    # ---------- финальная мета-модель ----------
    sc_full = StandardScaler().fit(Z)
    meta = LogisticRegression(C=args.C, max_iter=4000, class_weight="balanced",
                              random_state=args.seed)
    meta.fit(sc_full.transform(Z), y)
    proba = meta.predict_proba(sc_full.transform(Z_te))[:, 1]
    np.save(os.path.join(ROOT, "models", "unified_meta_proba_test.npy"), proba)
    np.save(os.path.join(ROOT, "models", "unified_meta_oof.npy"), meta_oof)

    test = pd.read_csv(args.test)
    sub = build_submission(test, proba, best_thr, sample_path=args.sample)
    out_path = os.path.join(ROOT, args.out)
    sub.to_csv(out_path, index=False)
    print(f"[ok] {out_path}: {len(sub)} строк, доля фишинга "
          f"{sub[sub.columns[-1]].mean():.4f}")
    with open(os.path.join(ROOT, "reports", "unified_stack_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"columns": col_names, "oof_f1": best_f1, "threshold": best_thr,
                   "C": args.C}, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
