# -*- coding: utf-8 -*-
"""
predict.py — применение обученной модели: формирование submission.csv
и проверка отдельных URL.

Примеры:
    # получить файл ответов для платформы Cyberhackai
    python src/predict.py --test data/test.csv --out submission.csv

    # проверить один адрес (демонстрация)
    python src/predict.py --url "http://sberbank-online-verify.tk/login?account=1"
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import build_feature_matrix, char_ngram_text, split_host, url_word_text  # noqa: E402
from train import read_table  # noqa: E402
from submission import build_submission  # noqa: E402
from scipy.sparse import csr_matrix, hstack  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")

COUNT_KEYS = ["host_count", "sld_count", "suffix_count", "path_count"]


def load_bundle(tag: str = "run"):
    import joblib
    path = os.path.join(MODELS_DIR, f"bundle_{tag}.joblib")
    if not os.path.exists(path):
        raise SystemExit(f"Не найден бандл модели: {path}. Сначала запустите src/train.py")
    return joblib.load(path)


def apply_count_features(F: pd.DataFrame, urls: pd.Series, count_maps: dict) -> pd.DataFrame:
    """Применяет сохранённые частотные словари корпуса (train+test)."""
    norm = urls.astype(str).str.replace(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", "", regex=True)
    host = norm.str.split("/").str[0].str.lower().str.strip(".")
    sld = host.map(lambda h: split_host(h)[1])
    suffix = host.map(lambda h: split_host(h)[2])
    path = norm.str.split("/", n=1).str[1].fillna("").str.lower()
    keys = {"host_count": host, "sld_count": sld, "suffix_count": suffix, "path_count": path}
    default = {"host_count": 1.0, "sld_count": 1.0, "suffix_count": 1.0, "path_count": 1.0}
    for name in COUNT_KEYS:
        mp = count_maps.get(name, {})
        F[name] = keys[name].map(mp).fillna(default[name]).astype(float).values
    return F


def attach_content(F: pd.DataFrame, content: pd.DataFrame) -> pd.DataFrame:
    """Добавляет к признакам URL признаки HTML-контента (по порядку строк)."""
    content = content.drop(columns=[c for c in ("content_url",) if c in content.columns])
    content = content.select_dtypes(include=[np.number])
    if len(content) > len(F):
        content = content.iloc[: len(F)]
    for c in content.columns:
        F[c] = np.resize(content[c].values, len(F))
    return F


def predict_urls(urls, bundle, count_maps=None, content=None, html_texts=None,
                 scheme: str = "auto", nn_matrix=None) -> np.ndarray:
    """
    Возвращает вероятность фишинга для списка URL.

    html_texts — видимый текст страниц (из cache/content_text_test.pkl).
    Если модель обучена с текстом HTML, а текст не передан, соответствующий
    блок признаков заполняется нулями (модель деградирует мягко, без ошибки).
    """
    F = build_feature_matrix(list(urls), verbose=False)
    if count_maps:
        F = apply_count_features(F, pd.Series(list(urls)), count_maps)
    if content is not None:
        F = attach_content(F, content)
    cols = bundle["numeric_cols"]
    nn_names = bundle.get("nn_names") or []
    if nn_names and nn_matrix is not None and len(nn_matrix) == len(urls):
        for j, c in enumerate(nn_names):
            F[c] = nn_matrix[:, j]
    for c in cols:
        if c not in F.columns:
            F[c] = 0.0
    X_num = np.nan_to_num(F[cols].astype(np.float32).values, nan=0.0, posinf=0.0, neginf=0.0)

    Xc = bundle["char_vec"].transform([char_ngram_text(u) for u in urls])
    Xw = bundle["word_vec"].transform([url_word_text(u) for u in urls])
    Xs = bundle["scaler"].transform(X_num)

    blocks = [Xc, Xw]
    html_vec = bundle.get("html_vec")
    if html_vec is not None:
        if html_texts is not None:
            Xh = html_vec.transform(list(html_texts))
        else:
            # нулевой блок той же ширины, что и словарь HTML-текста
            width = len(html_vec.get_feature_names_out())
            Xh = csr_matrix((len(urls), width), dtype=np.float32)
        blocks.append(Xh)
    X_lin = hstack([csr_matrix(Xs)] + blocks).tocsr()
    X_lgb = np.hstack([X_num, bundle["svd"].transform(hstack(blocks).tocsr())]).astype(np.float32)

    base = {}
    for name, model in bundle["models"].items():
        if name in ("logreg", "sgd"):
            base[name] = model.predict_proba(X_lin)[:, 1]
        elif name == "lightgbm":
            base[name] = model.predict_proba(X_lgb)[:, 1]
        else:
            base[name] = model.predict_proba(X_num)[:, 1]

    # Если итоговая схема — стекинг, воспроизводим её мета-моделью
    meta = bundle.get("meta_model")
    if scheme == "greedy":
        meta = None          # принудительно используем жадный отбор (вариант решения № 2)
    if meta is not None:
        names = bundle.get("base_names") or list(bundle["models"].keys())
        eps = 1e-6
        blocks = [np.log(np.clip(base[n], eps, 1 - eps)
                         / (1 - np.clip(base[n], eps, 1 - eps))) for n in names]
        if nn_names and nn_matrix is not None and len(nn_matrix) == len(urls):
            blocks.append(np.asarray(nn_matrix, dtype=float))
        elif getattr(meta, "n_features_in_", len(blocks)) > np.column_stack(blocks).shape[1]:
            # модель ждала соседские признаки, но их нет — добираем нулями
            missing = int(meta.n_features_in_ - np.column_stack(blocks).shape[1])
            blocks.append(np.zeros((len(urls), missing)))
        Z = np.column_stack(blocks)
        sc = bundle.get("meta_scaler")
        if sc is not None:
            Z = sc.transform(Z)
        return meta.predict_proba(Z)[:, 1]

    proba = np.zeros(len(urls))
    weights = bundle["weights"]
    for name, p in base.items():
        proba += p * weights.get(name, 0.0)
    total = sum(weights.values()) or 1.0
    return proba / total


def main() -> None:
    ap = argparse.ArgumentParser(description="Предсказание фишинга и формирование submission.csv")
    ap.add_argument("--test", default="data/test.csv")
    ap.add_argument("--sample", default=None,
                    help="образец формата ответов (sample.csv / sample_submission.csv)")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--url", default=None, help="проверить один URL и выйти")
    ap.add_argument("--threshold", type=float, default=None, help="переопределить порог")
    ap.add_argument("--recompute", action="store_true",
                    help="игнорировать сохранённые вероятности и пересчитать моделью")
    ap.add_argument("--scheme", choices=["auto", "greedy"], default="auto",
                    help="auto — сохранённая схема; greedy — жадный отбор базовых моделей")
    ap.add_argument("--with-url", action="store_true",
                    help="добавить колонку url (по регламенту не требуется)")
    ap.add_argument("--content-test", default=None,
                    help="pkl с признаками контента для test (cache/content_test.pkl)")
    ap.add_argument("--content-text-test", default=None,
                    help="pkl с текстом страниц для test (cache/content_text_test.pkl)")
    args = ap.parse_args()

    if args.sample is None:
        for cand in ("data/sample.csv", "data/sample_submission.csv", "sample.csv"):
            if os.path.exists(cand):
                args.sample = cand
                break

    bundle = load_bundle(args.tag)
    thr = args.threshold if args.threshold is not None else float(bundle["threshold"])

    if args.url:
        p = predict_urls([args.url], bundle, count_maps=bundle.get("count_maps"))[0]
        verdict = "ФИШИНГ" if p >= thr else "безопасный"
        print(f"URL: {args.url}")
        print(f"Вероятность фишинга: {p:.4f} (порог {thr:.4f}) -> {verdict}")
        return

    if not os.path.exists(args.test):
        raise SystemExit(f"Не найден файл {args.test}")
    test = read_table(args.test)
    url_col = next((c for c in test.columns if str(c).lower() == "url"), test.columns[-1])
    if url_col != "url":
        test = test.rename(columns={url_col: "url"})
    urls = test["url"].astype(str).tolist()

    html_texts = None
    if args.content_text_test and os.path.exists(args.content_text_test):
        from train import align_text
        html_texts = align_text(test["url"], args.content_text_test, "test")

    nn_matrix = None
    nn_path = os.path.join(MODELS_DIR, f"nn_test_{args.tag}.npy")
    if os.path.exists(nn_path):
        nn_matrix = np.load(nn_path)
        print(f"[info] соседские признаки загружены: {nn_matrix.shape}")

    content = None
    if args.content_test and os.path.exists(args.content_test):
        content = pd.read_pickle(args.content_test)
        if len(content) != len(test):
            print(f"[warn] контентных строк {len(content)}, test {len(test)} — обрезаю/дополняю")

    # Быстрый путь: если train.py уже сохранил вероятности для test, берём их —
    # тогда submission.csv гарантированно совпадает с оценкой модели (OOF-порог).
    saved = os.path.join(MODELS_DIR, f"test_proba_{args.tag}.npy")
    if os.path.exists(saved) and not args.content_test and html_texts is None \
            and not args.recompute and args.scheme == "auto":
        proba = np.load(saved)
        if len(proba) == len(test):
            print(f"[info] использованы сохранённые вероятности {os.path.basename(saved)}")
        else:
            print("[warn] длина сохранённых вероятностей не совпала — пересчитываю моделью")
            proba = predict_urls(urls, bundle, count_maps=bundle.get("count_maps"),
                                 content=content, html_texts=html_texts,
                                 scheme=args.scheme, nn_matrix=nn_matrix)
    else:
        proba = predict_urls(urls, bundle, count_maps=bundle.get("count_maps"),
                             content=content, html_texts=html_texts,
                             scheme=args.scheme, nn_matrix=nn_matrix)
    sub = build_submission(test, proba, thr, sample_path=args.sample, with_url=args.with_url)
    sub.to_csv(args.out, index=False)
    pred_col = "Predicted" if "Predicted" in sub.columns else sub.columns[-1]
    rate = float(sub[pred_col].mean())
    print(f"[ok] {args.out}: {len(sub)} строк, колонки {list(sub.columns)}, "
          f"доля предсказанного фишинга {rate:.4f} (порог {thr:.4f})")
    print(sub.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
