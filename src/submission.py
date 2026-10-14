# -*- coding: utf-8 -*-
"""
submission.py — формирование файла ответов в формате соревнования.

Регламент Cyberhackai:
    файл ответов содержит заголовок и колонки `Id,Predicted`;
    порядок строк совпадает с порядком строк test.csv.

Если рядом есть образец (sample.csv / sample_submission.csv / sample_submit.csv),
структура колонок берётся из него — это исключает расхождение с проверяющей
системой даже при смене названий колонок.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

PRED_ALIASES = ("predicted", "prediction", "predict", "target", "label", "result")


def build_submission(test: pd.DataFrame, proba: np.ndarray, threshold: float,
                     sample_path: str | None = None, url_col: str = "url",
                     with_url: bool = False) -> pd.DataFrame:
    """Таблица ответов: Id,Predicted (опционально с колонкой url)."""
    pred = (proba >= threshold).astype(int)
    ids = test["Id"].values if "Id" in test.columns else np.arange(len(test))

    if sample_path and os.path.exists(sample_path):
        sample = pd.read_csv(sample_path, nrows=5)
        cols = [str(c) for c in sample.columns]
        target = next((c for c in cols if c.lower() in PRED_ALIASES), None)
        id_col = next((c for c in cols if c.lower() == "id"), "Id")
        url_like = next((c for c in cols if c.lower() == "url"), None)
        if target and len(sample) > 0:
            out = pd.DataFrame({id_col: ids, target: pred})
            if url_like:
                out[url_like] = test[url_col].values
            ordered = [c for c in cols if c in out.columns]
            return out[ordered] if ordered else out[[id_col, target]]

    out = pd.DataFrame({"Id": ids, "Predicted": pred})
    if with_url:
        out["url"] = test[url_col].values
        out = out[["Id", "url", "Predicted"]]
    return out
