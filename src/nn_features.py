# -*- coding: utf-8 -*-
"""
nn_features.py — признаки «похожесть на известные страницы» (nearest neighbours).

Идея. Фишинговые страницы собираются из готовых «китов», поэтому HTML и текст
страницы-клона почти совпадают у разных доменов одной кампании. Для адресов на
доменах, которых не было в обучении (а это 55.9 % выборки, и именно на них
приходится 87 % ошибок модели), близость к известным фишинговым и безопасным
страницам — сильнейший сигнал обобщения.

Модуль:
  1. строит TF-IDF двух видов: слова (1–2 грамма) и символьные n-граммы (3–5)
     видимого текста страницы;
  2. считает top-K ближайших соседей (косинус) для каждой строки train и test
     блоками, чтобы не держать в памяти матрицу 64k x 64k;
  3. сохраняет списки соседей (индексы + близости) — из них на этапе
     кросс-валидации строятся признаки с честным исключением соседей из
     валидационного фолда (см. train.py: nn_block()).

Запуск:
    python src/nn_features.py --out-dir cache --topk 50 --workers 1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Tuple

import re
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import align_text, detect_columns, read_table, to_binary  # noqa: E402

VIEWS = {
    "word": dict(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=40000,
                 sublinear_tf=True, strip_accents="unicode"),
    "markup": dict(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=20000,
                   sublinear_tf=True, token_pattern=r"\S+"),
    "title": dict(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=30000,
                  sublinear_tf=True, strip_accents="unicode"),
    "fingerprint": dict(analyzer="word", ngram_range=(1, 1), min_df=2,
                        max_features=8000, sublinear_tf=True, token_pattern=r"\S+"),
    # шаблоны URL-путей: одна кампания часто переиспользует структуру пути на
    # разных доменах, поэтому путь рассматривается отдельно от хоста
    "urlpath": dict(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                    max_features=60000, sublinear_tf=True),
    "char": dict(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=120000,
                 sublinear_tf=True, lowercase=True),
}


def load_train(path: str) -> pd.DataFrame:
    tr = read_table(path)
    url_col, label_col = detect_columns(tr, need_label=True)
    tr = tr.rename(columns={url_col: "url", label_col: "result"})
    tr["url"] = tr["url"].astype(str)
    tr["result"] = to_binary(tr["result"])
    grp = tr.groupby("url")["result"].nunique()
    tr = tr[~tr["url"].isin(set(grp[grp > 1].index))]
    return tr.drop_duplicates(subset=["url"]).reset_index(drop=True)


def url_path_text(u: str) -> str:
    """Путь и параметры URL без домена; цифры нормализуются (шаблон кампании)."""
    from features import normalize_url
    p = urlsplit(normalize_url(u))
    path = (p.path or "/") + (("?" + p.query) if p.query else "")
    path = path.lower()
    path = re.sub(r"\d+", "0", path)
    return path


def l2_normalize(X: sp.csr_matrix) -> sp.csr_matrix:
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return sp.diags(1.0 / norms).dot(X).tocsr()


def topk_similarity(Xq: sp.csr_matrix, Xr: sp.csr_matrix, k: int,
                    exclude_self_offset: int | None = None,
                    chunk: int = 512, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Top-K косинусных соседей каждой строки Xq среди Xr.
    exclude_self_offset — если задано, для строки i исключается столбец
    (exclude_self_offset + i) (совпадение train с самим собой).
    """
    n = Xq.shape[0]
    idx = np.zeros((n, k), dtype=np.int32)
    sim = np.zeros((n, k), dtype=np.float32)
    t0 = time.time()
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        S = (Xq[s:e] @ Xr.T).toarray()
        if exclude_self_offset is not None:
            rows = np.arange(e - s)
            cols = exclude_self_offset + s + rows
            ok = cols < Xr.shape[0]
            S[rows[ok], cols[ok]] = -1.0
        kk = min(k, S.shape[1])
        part = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        part_sim = np.take_along_axis(S, part, axis=1)
        order = np.argsort(-part_sim, axis=1)
        idx[s:e, :kk] = np.take_along_axis(part, order, axis=1)
        sim[s:e, :kk] = np.take_along_axis(part_sim, order, axis=1)
        if kk < k:
            idx[s:e, kk:] = idx[s:e, kk - 1:kk]
            sim[s:e, kk:] = -1.0
        if verbose and (s // chunk) % 20 == 0:
            print(f"    {s}/{n}  {time.time() - t0:.0f} c", flush=True)
    return idx, sim


def main() -> None:
    ap = argparse.ArgumentParser(description="Top-K соседей по контенту страниц")
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--test", default="data/test.csv")
    ap.add_argument("--text-train", default="cache/content_text_train.pkl")
    ap.add_argument("--text-test", default="cache/content_text_test.pkl")
    ap.add_argument("--markup-train", default="cache/content_markup_train.pkl")
    ap.add_argument("--markup-test", default="cache/content_markup_test.pkl")
    ap.add_argument("--fp-train", default="cache/content_fingerprint_train.pkl")
    ap.add_argument("--fp-test", default="cache/content_fingerprint_test.pkl")
    ap.add_argument("--title-train", default="cache/content_title_train.pkl")
    ap.add_argument("--title-test", default="cache/content_title_test.pkl")
    ap.add_argument("--out-dir", default="cache")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--views", default="word,markup")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tr = load_train(args.train)
    te = read_table(args.test)
    url_col, _ = detect_columns(te, need_label=False)
    te = te.rename(columns={url_col: "url"})
    te["url"] = te["url"].astype(str)
    te = te.reset_index(drop=True)
    y = tr["result"].values
    print(f"[info] train {tr.shape}, test {te.shape}, фишинг {y.mean():.4f}")

    sources = {
        "word": (args.text_train, args.text_test, "ctext"),
        "markup": (args.markup_train, args.markup_test, "markup"),
        "fingerprint": (args.fp_train, args.fp_test, "fingerprint"),
        "title": (args.title_train, args.title_test, "title"),
    }
    texts = {"urlpath": ([url_path_text(u) for u in tr["url"]],
                         [url_path_text(u) for u in te["url"]])}
    for view, (p_tr, p_te, field) in sources.items():
        texts[view] = (align_text(tr["url"], p_tr, f"{view} train", field=field),
                       align_text(te["url"], p_te, f"{view} test", field=field))

    for view in args.views.split(","):
        view = view.strip()
        if view not in VIEWS:
            continue
        if view not in texts:
            print(f"[warn] нет источника текста для представления {view} — пропускаю")
            continue
        txt_tr, txt_te = texts[view]
        t0 = time.time()
        vec = TfidfVectorizer(**VIEWS[view])
        vec.fit(txt_tr + txt_te)                       # трандуктивно: словарь по train+test
        Xtr = l2_normalize(vec.transform(txt_tr))
        Xte = l2_normalize(vec.transform(txt_te))
        print(f"[{view}] TF-IDF train {Xtr.shape}, test {Xte.shape}, "
              f"{time.time() - t0:.0f} c")

        idx_tr, sim_tr = topk_similarity(Xtr, Xtr, args.topk, exclude_self_offset=0,
                                         chunk=args.chunk)
        np.savez_compressed(os.path.join(args.out_dir, f"nn_{view}_train.npz"),
                            idx=idx_tr, sim=sim_tr)
        print(f"[{view}] train-соседи готовы за {time.time() - t0:.0f} c")

        idx_te, sim_te = topk_similarity(Xte, Xtr, args.topk, exclude_self_offset=None,
                                         chunk=args.chunk)
        np.savez_compressed(os.path.join(args.out_dir, f"nn_{view}_test.npz"),
                            idx=idx_te, sim=sim_te)
        print(f"[{view}] test-соседи готовы за {time.time() - t0:.0f} c")

    print("[ok] соседи сохранены в", args.out_dir)


if __name__ == "__main__":
    main()
