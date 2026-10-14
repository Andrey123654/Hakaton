# -*- coding: utf-8 -*-
"""
content_title.py — отдельное представление «заголовок страницы».

Заголовок (<title>) — компактный и очень информативный признак: фишинговые
страницы часто дословно копируют заголовок бренда («Вход в Сбербанк Онлайн»),
поэтому близость заголовков хорошо группирует кампании. Берётся из уже
сохранённого текста страниц (первая часть до разделителя ' || ').

Запуск:
    python src/content_title.py --in cache/content_text_train.pkl \
        --out cache/content_title_train.pkl
"""

from __future__ import annotations

import argparse

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df = pd.read_pickle(args.inp)
    out = pd.DataFrame({
        "content_url": df["content_url"].astype(str),
        "title": df["ctext"].astype(str).str.split("||", regex=False).str[0].str.strip(),
    })
    out.to_pickle(args.out)
    print(f"[ok] {args.out}: {out.shape}, пример: {out['title'].iloc[0][:90]!r}")


if __name__ == "__main__":
    main()
