# -*- coding: utf-8 -*-
"""
content_fingerprint.py — текстовая «подпись» конфигурации страницы.

Числовые признаки контента дискретизируются и превращаются в токены вида
`pass=1 form=2 refresh=0 ext=3 ...`. Такая подпись описывает не текст, а
конфигурацию страницы (есть ли форма пароля, редирект, обфускация, сколько
внешних доменов) — то есть «почерк» фишингового набора. Используется как ещё
одно пространство для поиска ближайших соседей.

Запуск:
    python src/content_fingerprint.py --in cache/content_train.pkl \
        --out cache/content_fingerprint_train.pkl
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

# ключевые признаки контента и способ их огрубления
BINNED = {
    "c_n_form": [0, 1, 2, 5],
    "c_n_input": [0, 1, 3, 8],
    "c_n_password": [0, 1, 3],
    "c_n_iframe": [0, 1, 4],
    "c_n_script": [0, 3, 10, 30],
    "c_n_external_domains": [0, 2, 6, 15],
    "c_n_external_scripts": [0, 1, 4],
    "c_text_words": [0, 50, 200, 800],
    "c_n_phish_phrases": [0, 1, 3],
    "c_n_login_hints": [0, 1, 4],
}
FLAGS = [
    "c_has_password", "c_form_action_external", "c_form_action_php", "c_meta_refresh",
    "c_js_redirect", "c_eval", "c_base64_blobs", "c_hidden_style", "c_parked",
    "c_brand_domain_mismatch", "c_title_brand_mismatch", "c_minimal_page",
    "c_has_external_iframe", "c_favicon_external", "c_clone_signature", "c_steal_form",
]


def fingerprint_row(row: pd.Series) -> str:
    tokens = []
    for col, edges in BINNED.items():
        v = float(row.get(col, 0.0))
        bucket = int(np.searchsorted(edges, v, side="right"))
        tokens.append(f"{col}={bucket}")
    for col in FLAGS:
        tokens.append(f"{col}={int(bool(row.get(col, 0.0)))}")
    return " ".join(tokens)


def main() -> None:
    ap = argparse.ArgumentParser(description="Подпись конфигурации страницы")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_pickle(args.inp)
    url_col = "content_url" if "content_url" in df.columns else df.columns[0]
    out = pd.DataFrame({
        "content_url": df[url_col].astype(str),
        "fingerprint": [fingerprint_row(r) for _, r in df.iterrows()],
    })
    out.to_pickle(args.out)
    print(f"[ok] {args.out}: {out.shape}, пример: {out['fingerprint'].iloc[0][:120]}")


if __name__ == "__main__":
    main()
