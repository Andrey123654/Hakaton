# -*- coding: utf-8 -*-
"""
eval_dns_gain.py — проверка прироста от DNS-признаков на кросс-валидации
и сборка итогового решения.

Мета-модель получает на вход:
  * логиты предсказаний обеих линий решения (V2/V3/V4 линии A и единый
    стекинг линии Б) — OOF для обучения и усреднённые по фолдам для теста;
  * соседские признаки по четырём представлениям контента (с исключением
    соседей из валидационного фолда);
  * DNS-признаки домена (резолвинг, MX/NS/TXT/DMARC/CNAME, парковка).

Запуск:
    python tools/eval_dns_gain.py --out submission_dns.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from features import normalize_url, split_host  # noqa: E402
from submission import build_submission  # noqa: E402
from train import load_nn, nn_feature_block  # noqa: E402

EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def host_of(url: str) -> str:
    from urllib.parse import urlsplit
    h = urlsplit(normalize_url(url)).netloc.lower()
    if "@" in h:
        h = h.rsplit("@", 1)[1]
    return h.partition(":")[0].strip(".")


def net_matrix(urls: pd.Series, tables) -> np.ndarray:
    """
    Сетевые признаки (DNS, TLS, HTTP) по хосту или по полному URL.
    tables — список (DataFrame, key), где key = 'host' или 'url'.
    """
    from rdap_features import registrable_domain
    keys = {"host": urls.map(host_of).values,
            "url": urls.astype(str).values,
            "domain": np.array([registrable_domain(u)[0] for u in urls])}
    blocks = []
    for table, key in tables:
        cols = [c for c in table.columns
                if c.startswith(("dns_", "tls_", "http_", "rdap_", "ip_"))
                and c != "dns_host_len"]
        blocks.append(table.reindex(keys[key])[cols].astype(np.float32).values)
    X = np.column_stack(blocks) if blocks else np.zeros((len(urls), 0), dtype=np.float32)
    return np.nan_to_num(X, nan=0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="submission_dns.csv")
    ap.add_argument("--views", default="cache:word,cache:markup,cache:urlpath,cache_k100:title")
    ap.add_argument("--dns", default="cache/dns_a.pkl",
                    help="pkl с сетевыми признаками; можно список через запятую")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", choices=["lr", "gbdt"], default="gbdt",
                    help="мета-модель: логистическая регрессия или градиентный бустинг")
    ap.add_argument("--all-sources", action="store_true",
                    help="добавить в мета-модель все доступные источники линии A")
    ap.add_argument("--no-neighbors", action="store_true",
                    help="не добавлять соседские признаки (они ухудшают мета-модель)")
    args = ap.parse_args()

    tr = pd.read_csv(os.path.join(ROOT, "data/train.csv"))
    g = tr.groupby("url")["result"].nunique()
    tr = tr[~tr["url"].isin(set(g[g > 1].index))].drop_duplicates("url").reset_index(drop=True)
    y = tr["result"].values
    n = len(y)
    test = pd.read_csv(os.path.join(ROOT, "data/test.csv"))
    nt = len(test)

    skf = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    fold_ids = np.zeros(n, dtype=np.int16)
    for f, (_, iva) in enumerate(skf.split(np.zeros(n), y), 1):
        fold_ids[iva] = f

    # ---------- источники предсказаний ----------
    oof_cols, te_cols, names = [], [], []
    sources = [("A_v2", "improved_v2/blend_predictions.npz"),
               ("A_v3", "improved_v3/blend_predictions.npz"),
               ("A_v4", "improved_v4/blend_predictions.npz")]
    if args.all_sources:
        sources += [("A_v8", "improved_v8/predictions.npz"),
                    ("A_v10_lgb", "improved_v10/lightgbm_predictions.npz"),
                    ("A_v10_xgb", "improved_v10/xgboost_predictions.npz")]
    for tag, path in sources:
        d = np.load(os.path.join(ROOT, path))
        oof_cols.append(logit(d["oof"]))
        te_cols.append(logit(d["test"]))
        names.append(tag)
    oof_cols.append(logit(np.load(os.path.join(ROOT, "models/unified_meta_oof.npy"))))
    te_cols.append(logit(np.load(os.path.join(ROOT, "models/unified_meta_proba_test.npy"))))
    names.append("B_unified")
    print(f"[info] источников предсказаний: {len(names)} ({', '.join(names)})")

    # ---------- соседские признаки ----------
    nn_tr, nn_te = [], []
    for spec in [s.strip() for s in args.views.split(",") if s.strip()]:
        d_, _, v = spec.rpartition(":")
        d_ = d_ or "cache"
        a = load_nn(os.path.join(ROOT, d_, f"nn_{v}_train.npz"))
        b = load_nn(os.path.join(ROOT, d_, f"nn_{v}_test.npz"))
        if a is None or b is None:
            continue
        blk = np.zeros((n, 16), dtype=np.float32)
        for fold in range(1, args.folds + 1):
            iva = fold_ids == fold
            f_blk, _ = nn_feature_block(a[0], a[1], y, fold_of=fold_ids,
                                        current_fold=fold, prefix=v)
            blk[iva] = f_blk[iva]
        nn_tr.append(blk)
        f_te, _ = nn_feature_block(b[0], b[1], y, prefix=v)
        nn_te.append(f_te)
    print(f"[info] соседских представлений: {len(nn_tr)}")

    # ---------- DNS-признаки ----------
    tables = []
    for part in args.dns.split(","):
        spec = part.strip()
        path_part, _, key = spec.rpartition(":")
        if not path_part:
            path_part, key = spec, "host"
        path = os.path.join(ROOT, path_part)
        if os.path.exists(path):
            tbl = pd.read_pickle(path)
            tables.append((tbl, key))
            print(f"[info] сетевые признаки: {os.path.basename(path)}[{key}] "
                  f"({tbl.shape[1]} колонок)")
        else:
            print(f"[warn] нет файла {path}")
    dns_tr = net_matrix(tr["url"], tables)
    dns_te = net_matrix(test["url"], tables)
    print(f"[info] сетевых признаков всего: {dns_tr.shape[1]}")

    Z = np.column_stack(oof_cols + nn_tr + [dns_tr])
    Z_te = np.column_stack(te_cols + nn_te + [dns_te])
    print(f"[info] мета-признаков: {Z.shape[1]}")

    def cv_f1(feature_slice=None):
        X = Z if feature_slice is None else Z[:, feature_slice]
        pred = np.zeros(n)
        for fold in range(1, args.folds + 1):
            itr, iva = fold_ids != fold, fold_ids == fold
            sc = StandardScaler().fit(X[itr])
            m = (LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced",
                                    random_state=args.seed) if args.model == "lr" else
                 HistGradientBoostingClassifier(max_iter=200, learning_rate=0.10,
                                                max_leaf_nodes=4, l2_regularization=1.0,
                                                random_state=args.seed))
            m.fit(sc.transform(X[itr]), y[itr])
            pred[iva] = m.predict_proba(sc.transform(X[iva]))[:, 1]
        f1, thr = max((f1_score(y, (pred >= t).astype(int)), t)
                      for t in np.quantile(pred, np.linspace(0.005, 0.995, 600)))
        return f1, thr, pred

    # базовые ориентиры
    base_scores = {}
    for i, nm in enumerate(names):
        p = 1 / (1 + np.exp(-oof_cols[i]))
        base_scores[nm] = max(f1_score(y, (p >= t).astype(int))
                              for t in np.quantile(p, np.linspace(0.005, 0.995, 400)))
    print("\n=== ориентиры (OOF F1) ===")
    for k, v in base_scores.items():
        print(f"  {k:<12} {v:.5f}")

    n_logits = len(oof_cols)
    n_nn = 0 if args.no_neighbors else sum(b.shape[1] for b in nn_tr)
    i_logits = list(range(n_logits))
    i_nn = list(range(n_logits, n_logits + n_nn))
    if args.no_neighbors:
        nn_tr, nn_te = [], []
    i_dns = list(range(n_logits + n_nn, Z.shape[1]))

    print("\n=== мета-модели ===")
    res = {}
    for label, sl in [("логиты обеих линий", i_logits),
                      ("логиты + соседи", i_logits + i_nn),
                      ("логиты + соседи + DNS", i_logits + i_nn + i_dns),
                      ("логиты + DNS", i_logits + i_dns)]:
        f1, thr, _ = cv_f1(sl)
        res[label] = {"f1": f1, "threshold": thr, "n_features": len(sl)}
        print(f"  {label:<26} F1={f1:.5f} (признаков {len(sl)}, порог {thr:.4f})")

    best_label = max(res, key=lambda k: res[k]["f1"])
    print(f"\n[info] лучшая конфигурация: {best_label}")

    # ---------- финальная модель и файл ответов ----------
    sl = {"логиты обеих линий": i_logits, "логиты + соседи": i_logits + i_nn,
          "логиты + соседи + DNS": i_logits + i_nn + i_dns,
          "логиты + DNS": i_logits + i_dns}[best_label]
    X, X_te = Z[:, sl], Z_te[:, sl]
    sc = StandardScaler().fit(X)
    meta = (LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced",
                               random_state=args.seed) if args.model == "lr" else
            HistGradientBoostingClassifier(max_iter=200, learning_rate=0.10,
                                           max_leaf_nodes=4, l2_regularization=1.0,
                                           random_state=args.seed))
    meta.fit(sc.transform(X), y)
    proba = meta.predict_proba(sc.transform(X_te))[:, 1]
    np.save(os.path.join(ROOT, "models", "dns_meta_proba_test.npy"), proba)
    sub = build_submission(test, proba, res[best_label]["threshold"],
                           sample_path=os.path.join(ROOT, "data/sample.csv"))
    sub.to_csv(os.path.join(ROOT, args.out), index=False)
    print(f"[ok] {args.out}: {len(sub)} строк, доля фишинга "
          f"{sub[sub.columns[-1]].mean():.4f}")
    with open(os.path.join(ROOT, "reports", "dns_gain_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"baselines": base_scores, "meta": res, "best": best_label,
                   "dns_resolves_train": float(dns_tr[:, 0].mean()),
                   "dns_resolves_test": float(dns_te[:, 0].mean())}, fh,
                  ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
