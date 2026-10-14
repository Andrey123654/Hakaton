# -*- coding: utf-8 -*-
"""
meta_submission.py — пересборка стекинга поверх сохранённых предсказаний
базовых моделей с добавлением соседских признаков на мета-уровне.

Зачем: эксперименты показали, что признаки «похожесть на известные страницы»
эффективнее работают не только внутри базовых моделей, но и как вход
мета-модели. Скрипт берёт сохранённые OOF-предсказания базовых моделей
(`models/oof_<tag>.npz`) и их усреднённые по фолдам предсказания для теста
(`models/test_base_<tag>.npz`), добавляет соседские признаки (с исключением
соседей из валидационного фолда для train) и обучает мета-логистическую
регрессию с подбором порога по OOF.

Запуск:
    python tools/meta_submission.py --tag nn --out submission_nn_meta.csv
"""

from __future__ import annotations

import argparse
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

from train import load_nn, nn_feature_block  # noqa: E402
from submission import build_submission  # noqa: E402


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def main() -> None:
    ap = argparse.ArgumentParser(description="Стекинг с соседскими признаками")
    ap.add_argument("--tag", default="nn")
    ap.add_argument("--views", default="cache:word,cache:markup,cache:urlpath,cache_k100:title")
    ap.add_argument("--nn-dir", default="cache", help="каталог со списками соседей")
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--test", default="data/test.csv")
    ap.add_argument("--sample", default="data/sample.csv")
    ap.add_argument("--out", default="submission_nn_meta.csv")
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

    d = np.load(os.path.join(ROOT, "models", f"oof_{args.tag}.npz"))
    names = [k for k in d.files if k != "y"]
    oof = {k: d[k] for k in names}
    tb = np.load(os.path.join(ROOT, "models", f"test_base_{args.tag}.npz"))
    test_urls = pd.read_csv(args.test)["url"].astype(str).tolist()

    skf = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    fold_ids = np.zeros(n, dtype=np.int16)
    for f, (_, iva) in enumerate(skf.split(np.zeros(n), y), 1):
        fold_ids[iva] = f

    # спецификация вида "cache:word,cache:markup,cache_k100:title"
    views = []
    for spec in args.views.split(","):
        spec = spec.strip()
        if not spec:
            continue
        d_, _, v_ = spec.rpartition(":")
        views.append((d_ or args.nn_dir, v_))
    nn_tr = {}
    nn_te = {}
    for d_, v in views:
        a = load_nn(os.path.join(ROOT, d_, f"nn_{v}_train.npz"))
        b = load_nn(os.path.join(ROOT, d_, f"nn_{v}_test.npz"))
        if a is None or b is None:
            print(f"[warn] представление {v} недоступно — пропускаю")
            continue
        nn_tr[v] = a
        nn_te[v] = b
    print(f"[info] базовые модели: {names}; соседские представления: {list(nn_tr)}")

    # --- OOF-признаки мета-модели: логиты базовых моделей + соседи (с исключением фолда)
    Z_oof = np.column_stack([logit(oof[k]) for k in names])
    for v, (idx, sim) in nn_tr.items():   # ключи — имена представлений
        blk = np.zeros((n, 16), dtype=np.float32)
        for fold in range(1, args.folds + 1):
            iva = fold_ids == fold
            f_blk, _ = nn_feature_block(idx, sim, y, fold_of=fold_ids,
                                        current_fold=fold, prefix=v)
            blk[iva] = f_blk[iva]
        Z_oof = np.column_stack([Z_oof, blk])
    print(f"[info] мета-признаки (train): {Z_oof.shape}")

    # --- признаки для test: логиты моделей + соседи по полному train
    Z_te = np.column_stack([logit(tb[k]) for k in names])
    for v, (idx, sim) in nn_te.items():
        f_blk, _ = nn_feature_block(idx, sim, y, prefix=v)
        Z_te = np.column_stack([Z_te, f_blk])
    print(f"[info] мета-признаки (test):  {Z_te.shape}")

    # --- кросс-валидированная оценка мета-модели
    meta_oof = np.zeros(n)
    for fold in range(1, args.folds + 1):
        itr, iva = fold_ids != fold, fold_ids == fold
        sc = StandardScaler().fit(Z_oof[itr])
        m = LogisticRegression(C=args.C, max_iter=3000, class_weight="balanced",
                               random_state=args.seed)
        m.fit(sc.transform(Z_oof[itr]), y[itr])
        meta_oof[iva] = m.predict_proba(sc.transform(Z_oof[iva]))[:, 1]
    best_f1, best_thr = 0.0, 0.5
    for t in np.quantile(meta_oof, np.linspace(0.01, 0.99, 400)):
        f1 = f1_score(y, (meta_oof >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_thr = f1, float(t)
    base_best = max(f1_score(y, (np.mean([oof[k] for k in names], axis=0) >= t).astype(int))
                    for t in np.quantile(np.mean([oof[k] for k in names], axis=0),
                                         np.linspace(0.01, 0.99, 200)))
    print(f"[info] OOF F1 мета-модели: {best_f1:.5f} (порог {best_thr:.4f}); "
          f"простое усреднение базовых: {base_best:.5f}")

    # --- финальная мета-модель на всех данных
    sc_full = StandardScaler().fit(Z_oof)
    meta = LogisticRegression(C=args.C, max_iter=3000, class_weight="balanced",
                              random_state=args.seed)
    meta.fit(sc_full.transform(Z_oof), y)
    proba = meta.predict_proba(sc_full.transform(Z_te))[:, 1]
    np.save(os.path.join(ROOT, "models", f"meta_proba_test_{args.tag}.npy"), proba)
    np.save(os.path.join(ROOT, "models", f"meta_oof_{args.tag}.npy"), meta_oof)
    print(f"[info] сохранены meta_oof_{args.tag}.npy и meta_proba_test_{args.tag}.npy")

    test = pd.read_csv(args.test)
    sub = build_submission(test, proba, best_thr, sample_path=args.sample)
    out_path = os.path.join(ROOT, args.out)
    sub.to_csv(out_path, index=False)
    print(f"[ok] {out_path}: {len(sub)} строк, доля фишинга "
          f"{sub[sub.columns[-1]].mean():.4f}, порог {best_thr:.4f}")
    with open(os.path.join(ROOT, "reports", f"meta_report_{args.tag}.md"), "w",
              encoding="utf-8") as fh:
        fh.write(f"""# Стекинг с соседскими признаками (tag={args.tag})

* базовые модели: {", ".join(names)}
* соседские представления: {", ".join(nn_tr)}
* мета-признаков: {Z_oof.shape[1]}
* OOF F1 мета-модели: **{best_f1:.5f}** (порог {best_thr:.4f})
* OOF F1 простого усреднения базовых моделей: {base_best:.5f}
""")


if __name__ == "__main__":
    main()
