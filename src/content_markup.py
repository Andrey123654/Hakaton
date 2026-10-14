# -*- coding: utf-8 -*-
"""
content_markup.py — структурная подпись страницы (последовательность тегов).

Идея: фишинговые «киты» воспроизводят не только текст, но и разметку страницы
(порядок тегов, наличие формы, тип полей, редиректы). Последовательность тегов
с короткими атрибутными маркерами — компактное представление структуры, по
которому страницы одной кампании совпадают даже при другой вёрстке и тексте.

Результат используется как второе пространство для поиска ближайших соседей
(src/nn_features.py) наряду со словами текста.

Запуск:
    python src/content_markup.py --csv train_with_content.csv \
        --out cache/content_markup_train.pkl --workers 6
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from multiprocessing import Pool
from typing import List, Tuple

import pandas as pd

TAG_RE = re.compile(r"<([a-zA-Z][a-zA-Z0-9]*)((?:\s+[^<>]*)?)/?>")
SKIP = {"br", "b", "i", "u", "em", "strong", "font", "span", "small", "nobr", "wbr"}
MAX_INPUT = 120_000
MAX_TAGS = 500


def markup_signature(html: str) -> str:
    """Компактная структурная подпись: последовательность тегов + ключевые атрибуты."""
    html = (html or "")[:MAX_INPUT]
    out: List[str] = []
    for m in TAG_RE.finditer(html):
        tag = m.group(1).lower()
        if tag in SKIP:
            continue
        attrs = (m.group(2) or "").lower()
        marker = tag
        if tag == "input":
            if "password" in attrs:
                marker = "input:password"
            elif "hidden" in attrs:
                marker = "input:hidden"
            elif "email" in attrs or "user" in attrs or "login" in attrs:
                marker = "input:user"
            elif "text" in attrs:
                marker = "input:text"
            elif "submit" in attrs or "button" in attrs:
                marker = "input:submit"
            else:
                marker = "input:other"
        elif tag == "form":
            marker = "form:action" if "action" in attrs else "form:noaction"
        elif tag == "script":
            marker = "script:src" if "src" in attrs else "script:inline"
        elif tag == "meta":
            if "refresh" in attrs:
                marker = "meta:refresh"
            elif "content" in attrs:
                marker = "meta"
            else:
                marker = "meta"
        elif tag == "iframe":
            marker = "iframe"
        elif tag == "link":
            marker = "link:css" if "stylesheet" in attrs else "link"
        out.append(marker)
        if len(out) >= MAX_TAGS:
            break
    return " ".join(out)


def _worker(args: Tuple[List[str], List[str]]) -> List[Tuple[str, str]]:
    urls, htmls = args
    return [(u, markup_signature(h)) for u, h in zip(urls, htmls)]


def build(csv_path: str, out_path: str, workers: int = 6, chunksize: int = 4000,
          limit: int | None = None) -> pd.DataFrame:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rows: List[Tuple[str, str]] = []
    processed = 0
    reader = pd.read_csv(csv_path, chunksize=chunksize, on_bad_lines="skip",
                         engine="c", dtype=str, keep_default_na=False,
                         usecols=lambda c: c in ("url", "content"))
    with Pool(workers) as pool:
        for chunk in reader:
            ucol = "url" if "url" in chunk.columns else chunk.columns[0]
            ccol = "content" if "content" in chunk.columns else chunk.columns[-1]
            urls = chunk[ucol].astype(str).tolist()
            htmls = chunk[ccol].astype(str).tolist()
            n = len(urls)
            step = max(n // workers + 1, 1)
            parts = [(urls[i:i + step], htmls[i:i + step]) for i in range(0, n, step)]
            for res in pool.imap(_worker, parts):
                rows.extend(res)
            processed += n
            print(f"  разметка: обработано {processed} строк", flush=True)
            if limit and processed >= limit:
                break
    df = pd.DataFrame(rows, columns=["content_url", "markup"])
    df.to_pickle(out_path)
    print(f"[ok] {out_path}: {df.shape}, средняя длина подписи "
          f"{df['markup'].str.len().mean():.0f} символов")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Структурная подпись страниц")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunksize", type=int, default=4000)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    build(args.csv, args.out, workers=args.workers, chunksize=args.chunksize, limit=args.limit)


if __name__ == "__main__":
    main()
