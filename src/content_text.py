# -*- coding: utf-8 -*-
"""
content_text.py — извлечение видимого текста страницы для TF-IDF-модели.

Фишинговые страницы часто собираются из готовых «китов», поэтому их HTML и
текст повторяются шаблонами. Символьные/словесные n-граммы видимого текста
(заголовок + тело страницы) дают линейной модели сильный дополнительный сигнал,
которого нет в признаках URL.

Модуль потоково читает `*_with_content.csv`, вырезает <script>/<style>, теги,
сжимает пробелы и сохраняет короткий текст (первые N символов) вместе с URL.

Запуск:
    python src/content_text.py --csv train_with_content.csv \
        --out cache/content_text_train.pkl --workers 4
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from multiprocessing import Pool
from typing import List, Tuple

import numpy as np
import pandas as pd

SCRIPT_STYLE_RE = re.compile(r"<(script|style|noscript)\b[^>]*>.*?</\1>", re.I | re.S)
COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
TEXT_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
META_DESC_RE = re.compile(
    r"""<meta[^>]+name\s*=\s*["']?description["']?[^>]+content\s*=\s*["']([^"']*)["']""", re.I)
H1_RE = re.compile(r"<h[12][^>]*>(.*?)</h[12]>", re.I | re.S)

MAX_TEXT = 3000
MAX_INPUT = 300_000


def page_text(url: str, html: str) -> str:
    """Видимый текст страницы: заголовок + h1/h2 + meta description + тело."""
    html = (html or "")[:MAX_INPUT]
    head_parts: List[str] = []
    m = TITLE_RE.search(html)
    if m:
        head_parts.append(TEXT_RE.sub(" ", m.group(1)))
    m = META_DESC_RE.search(html)
    if m:
        head_parts.append(m.group(1))
    head_parts.extend(TEXT_RE.sub(" ", h) for h in H1_RE.findall(html)[:3])

    body = COMMENT_RE.sub(" ", SCRIPT_STYLE_RE.sub(" ", html))
    body = TEXT_RE.sub(" ", body)
    body = WS_RE.sub(" ", body).strip()
    head = WS_RE.sub(" ", " ".join(head_parts)).strip()
    return (head + " || " + body)[:MAX_TEXT].lower()


def _worker(args: Tuple[List[str], List[str]]) -> List[Tuple[str, str]]:
    urls, htmls = args
    out = []
    for u, h in zip(urls, htmls):
        try:
            out.append((u, page_text(u, h)))
        except Exception:  # noqa: BLE001
            out.append((u, ""))
    return out


def build(csv_path: str, out_path: str, workers: int = 4, chunksize: int = 2000,
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
            print(f"  текст: обработано {processed} строк", flush=True)
            if limit and processed >= limit:
                break
    df = pd.DataFrame(rows, columns=["content_url", "ctext"])
    df.to_pickle(out_path)
    print(f"[ok] {out_path}: {df.shape}, средняя длина текста "
          f"{df['ctext'].str.len().mean():.0f} символов")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Видимый текст страниц для TF-IDF")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunksize", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    build(args.csv, args.out, workers=args.workers, chunksize=args.chunksize, limit=args.limit)


if __name__ == "__main__":
    main()
