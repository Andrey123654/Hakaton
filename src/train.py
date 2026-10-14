# -*- coding: utf-8 -*-
"""
train.py — обучение и валидация моделей обнаружения фишинговых URL.

Пайплайн:
  1. Чтение train.csv (колонки определяются автоматически).
  2. Извлечение признаков (features.py) + частотные признаки по корпусу train+test.
  3. Кросс-валидация (StratifiedKFold) четырёх разнородных моделей:
       - CatBoost      (градиентный бустинг, плотные признаки)
       - LightGBM      (бустинг + SVD от символьных n-грамм)
       - RandomForest  (бэггинг, устойчив к шуму)
       - LogisticRegression (линейная модель на TF-IDF + масштабированные признаки)
  4. Ансамблирование OOF-предсказаний жадным отбором (Caruana ensemble selection).
  5. Подбор порога бинаризации, максимизирующего F1 на OOF.
  6. Сохранение моделей, весов, порога и отчёта в models/ и reports/.

Запуск:
    python src/train.py --train data/train.csv --test data/test.csv
    python src/train.py --quick            # быстрый режим для проверки
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import (  # noqa: E402
    build_feature_matrix,
    char_ngram_text,
    split_host,
    url_word_text,
)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 confusion_matrix, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import TruncatedSVD
    from scipy.sparse import hstack, csr_matrix
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Не установлены зависимости scikit-learn/scipy: {exc}")

SEED = 42
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")

URL_ALIASES = ["url", "URL", "Url", "link", "uri", "address", "domain", "url_address"]
LABEL_ALIASES = ["label", "Label", "LABEL", "target", "Target", "Predicted", "predicted",
                 "phishing", "Phishing", "is_phishing", "class", "Class", "status",
                 "Status", "result", "y", "malicious", "type"]


# --------------------------------------------------------------------------
# Загрузка данных
# --------------------------------------------------------------------------

def read_table(path: str) -> pd.DataFrame:
    """
    Читает CSV/TSV/XLSX с автоопределением разделителя и кодировки.
    Поддерживает файлы с одной колонкой (test.csv содержит только url).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
        for sep in (",", ";", "\t"):
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, engine="python",
                                 on_bad_lines="skip")
                if df.shape[1] >= 1 and len(df) > 0:
                    return df
            except Exception as exc:  # noqa: BLE001
                last_err = exc
    raise SystemExit(f"Не удалось прочитать {path}: {last_err}")


def detect_columns(df: pd.DataFrame, need_label: bool) -> Tuple[str, str | None]:
    """Определяет колонку с URL и (опционально) колонку с меткой."""
    url_col = next((c for c in df.columns if c in URL_ALIASES), None)
    if url_col is None:
        url_col = next((c for c in df.columns if "url" in str(c).lower()), df.columns[0])
    label_col = None
    if need_label:
        label_col = next((c for c in df.columns if c in LABEL_ALIASES and c != url_col), None)
        if label_col is None:
            for c in df.columns:
                if c == url_col:
                    continue
                vals = pd.unique(df[c].dropna())
                if len(vals) <= 2 and set(map(str, vals)) <= {"0", "1", "0.0", "1.0",
                                                              "True", "False", "true", "false"}:
                    label_col = c
                    break
        if label_col is None:
            raise SystemExit(f"Не найдена колонка с меткой среди: {list(df.columns)}")
    return url_col, label_col


def to_binary(series: pd.Series) -> pd.Series:
    """Приводит метку к 0/1 (поддерживает строковые значения)."""
    if series.dtype == bool:
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return (series.astype(float) > 0.5).astype(int)
    mapping = {"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0,
               "phishing": 1, "malicious": 1, "bad": 1, "legit": 0, "benign": 0,
               "safe": 0, "good": 0, "ham": 0, "spam": 1}
    return series.astype(str).str.strip().str.lower().map(mapping).fillna(0).astype(int)


# --------------------------------------------------------------------------
# Признаковое пространство
# --------------------------------------------------------------------------

def add_corpus_features(f_tr: pd.DataFrame, f_te: pd.DataFrame,
                        urls_tr: pd.Series, urls_te: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Частотные признаки, вычисляемые по объединённому корпусу train+test.
    Это unsupervised-статистика (сколько раз встречается домен/хост/путь),
    она не использует метки, поэтому утечки целевой переменной не возникает.
    """
    all_urls = pd.concat([urls_tr, urls_te], ignore_index=True).astype(str)
    norm = all_urls.str.replace(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", "", regex=True)
    host = norm.str.split("/").str[0].str.lower().str.strip(".")
    sld = host.map(lambda h: split_host(h)[1])
    suffix = host.map(lambda h: split_host(h)[2])
    path = norm.str.split("/", n=1).str[1].fillna("").str.lower()

    counts = {
        "host_count": host.value_counts(),
        "sld_count": sld.value_counts(),
        "suffix_count": suffix.value_counts(),
        "path_count": path.value_counts(),
    }
    for name, vc in counts.items():
        key = {"host_count": host, "sld_count": sld,
               "suffix_count": suffix, "path_count": path}[name]
        values = key.map(vc).astype(float)
        f_tr[name] = values.iloc[: len(f_tr)].values
        f_te[name] = values.iloc[len(f_tr):].values
    count_maps = {name: vc.to_dict() for name, vc in counts.items()}
    return f_tr, f_te, count_maps


def make_text_matrices(train_texts: List[str], test_texts: List[str],
                       max_char: int = 40000, max_word: int = 12000):
    """TF-IDF: символьные n-граммы (2-5) + слова в гомоглиф-нормализованном виде."""
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                               max_features=max_char, sublinear_tf=True)
    word_vec = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2),
                               min_df=3, max_features=max_word, sublinear_tf=True)
    Xc_tr = char_vec.fit_transform(train_texts)
    Xc_te = char_vec.transform(test_texts)
    Xw_tr = word_vec.fit_transform(train_texts)
    Xw_te = word_vec.transform(test_texts)
    return (Xc_tr, Xc_te), (Xw_tr, Xw_te), char_vec, word_vec


def align_text(urls: pd.Series, text_path: str, label: str = "train",
               field: str = "ctext") -> List[str]:
    """
    Выравнивает видимый текст страниц (src/content_text.py) по списку URL.
    Для отсутствующих страниц возвращает пустую строку.
    """
    if not text_path or not os.path.exists(text_path):
        print(f"[warn] файл текста страниц не найден: {text_path}")
        return [""] * len(urls)
    T = pd.read_pickle(text_path)
    T = T.drop_duplicates(subset=["content_url"], keep="first").set_index("content_url")
    if field not in T.columns:
        print(f"[warn] в {text_path} нет колонки {field}")
        return [""] * len(urls)
    aligned = T[field].reindex(urls.astype(str).values)
    found = float(aligned.notna().mean())
    print(f"[info] текст страниц ({label}): сопоставлено {found * 100:.2f} % строк")
    return aligned.fillna("").astype(str).tolist()


def attach_content_features(F: pd.DataFrame, urls: pd.Series, content_path: str,
                            label: str = "train") -> pd.DataFrame:
    """
    Присоединяет признаки HTML-контента к признакам URL по значению URL.

    Важно: train.csv после удаления дубликатов и противоречивых записей
    не совпадает построчно с train_with_content.csv, поэтому соединение
    выполняется именно по URL, а не по номеру строки.
    """
    if not content_path or not os.path.exists(content_path):
        print(f"[warn] файл контентных признаков не найден: {content_path}")
        return F
    C = pd.read_pickle(content_path)
    if "content_url" not in C.columns:
        print(f"[warn] в {content_path} нет колонки content_url — пропускаю")
        return F
    C = C.drop_duplicates(subset=["content_url"], keep="first")
    C = C.set_index("content_url").select_dtypes(include=[np.number])
    urls_str = urls.astype(str).values
    aligned = C.reindex(urls_str)
    found = aligned.notna().any(axis=1).values
    aligned = aligned.fillna(0.0).reset_index(drop=True)
    for c in aligned.columns:
        F[c] = aligned[c].values.astype(np.float32)
    F["content_row_missing"] = np.where(found, 0.0, 1.0).astype(np.float32)
    print(f"[info] контентные признаки ({label}): {aligned.shape[1]} колонок, "
          f"сопоставлено по URL {found.mean() * 100:.2f} % строк")
    return F


# --------------------------------------------------------------------------
# Модели
# --------------------------------------------------------------------------


NN_FEATURE_NAMES = [
    "nn_valid", "nn_max_sim", "nn_top1_label", "nn_sim_pos", "nn_sim_neg",
    "nn_sim_diff", "nn_wlabel", "nn_wlabel_top10", "nn_mean_top10",
    "nn_n_pos_50", "nn_n_neg_50", "nn_n_pos_80", "nn_n_neg_80",
    "nn_n_pos_95", "nn_n_neg_95", "nn_frac_pos_60",
]


def nn_feature_block(idx: np.ndarray, sim: np.ndarray, y_source: np.ndarray,
                     fold_of=None, current_fold=None, prefix: str = "nn",
                     top: int = 10) -> Tuple[np.ndarray, List[str]]:
    """
    Признаки по списку ближайших соседей (контентная похожесть).

    idx, sim      — (n, K) индексы и близости соседей в массиве-источнике;
    y_source      — метки источника (train);
    fold_of       — массив «номер фолда» для источника; соседи, попавшие в
                    current_fold, исключаются, чтобы признаки для валидационной
                    части не подглядывали в неё (честная OOF-оценка).
    """
    sims = sim.astype(np.float32, copy=True)
    valid = sims > 0
    if fold_of is not None and current_fold is not None:
        valid &= (fold_of[idx] != current_fold)
    sims = np.where(valid, sims, -1.0)
    labels = y_source[idx]
    pos = valid & (labels == 1)
    neg = valid & (labels == 0)

    def masked_max(mask):
        v = np.where(mask, sims, -1.0).max(axis=1)
        return np.where(v < 0, 0.0, v).astype(np.float32)

    sim_pos = masked_max(pos)
    sim_neg = masked_max(neg)
    n_valid = valid.sum(axis=1).astype(np.float32)
    max_sim = masked_max(valid)
    top1 = np.where(n_valid > 0, labels[np.arange(len(idx)), np.argmax(sims, axis=1)], -1).astype(np.float32)

    wsum = np.where(valid, sims, 0.0).sum(axis=1)
    wlab = np.where(valid, sims * labels, 0.0).sum(axis=1) / np.maximum(wsum, 1e-6)

    k_top = min(top, sims.shape[1])
    order = np.argsort(-sims, axis=1)[:, :k_top]
    sims_top = np.take_along_axis(sims, order, axis=1)
    lab_top = np.take_along_axis(labels, order, axis=1)
    w = np.where(sims_top > 0, sims_top, 0.0)
    wlab_top = (w * lab_top).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-6)
    mean_top = np.where(w > 0, w, np.nan)
    with np.errstate(invalid="ignore"):
        mean_top = np.nanmean(mean_top, axis=1)
    mean_top = np.nan_to_num(mean_top, nan=0.0)

    feats = np.column_stack([
        n_valid, max_sim, top1, sim_pos, sim_neg, sim_pos - sim_neg, wlab, wlab_top,
        mean_top,
        (valid & (sims >= 0.5) & (labels == 1)).sum(axis=1),
        (valid & (sims >= 0.5) & (labels == 0)).sum(axis=1),
        (valid & (sims >= 0.8) & (labels == 1)).sum(axis=1),
        (valid & (sims >= 0.8) & (labels == 0)).sum(axis=1),
        (valid & (sims >= 0.95) & (labels == 1)).sum(axis=1),
        (valid & (sims >= 0.95) & (labels == 0)).sum(axis=1),
    ]).astype(np.float32)
    npos60 = (valid & (sims >= 0.6) & (labels == 1)).sum(axis=1).astype(np.float32)
    nneg60 = (valid & (sims >= 0.6) & (labels == 0)).sum(axis=1).astype(np.float32)
    feats = np.column_stack([feats, npos60 / np.maximum(npos60 + nneg60, 1.0)]).astype(np.float32)
    names = [f"{prefix}_{n[3:]}" for n in NN_FEATURE_NAMES]   # nn_ + valid -> nnword_valid
    return feats, names


def load_nn(path: str):
    """Загружает списки соседей (idx, sim) или None."""
    if not path or not os.path.exists(path):
        return None
    d = np.load(path)
    return d["idx"], d["sim"]


def get_models(quick: bool = False) -> Dict[str, object]:
    """Набор базовых моделей. Разнородность важнее единичного качества."""
    models: Dict[str, object] = {}
    try:
        from catboost import CatBoostClassifier
        models["catboost"] = CatBoostClassifier(
            iterations=400 if quick else 2000,
            learning_rate=0.08 if quick else 0.05,
            depth=6,
            l2_leaf_reg=3.0,
            loss_function="Logloss",
            eval_metric="F1",
            random_seed=SEED,
            verbose=0,
            thread_count=-1,
            allow_writing_files=False,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] CatBoost недоступен: {exc}")
    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = LGBMClassifier(
            n_estimators=400 if quick else 1500,
            learning_rate=0.05,
            num_leaves=64,
            min_child_samples=30,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] LightGBM недоступен: {exc}")

    from sklearn.ensemble import HistGradientBoostingClassifier
    # Резервный бустинг из scikit-learn: работает без внешних бинарных зависимостей
    # (например, если LightGBM не может загрузить libomp).
    models["hist_gbm"] = HistGradientBoostingClassifier(
        max_iter=300 if quick else 900,
        learning_rate=0.06,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=SEED,
    )
    from sklearn.ensemble import ExtraTreesClassifier
    models["extra_trees"] = ExtraTreesClassifier(
        n_estimators=300 if quick else 800,
        max_features="sqrt",
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=SEED,
        class_weight="balanced_subsample",
    )
    models["random_forest"] = RandomForestClassifier(
        n_estimators=200 if quick else 600,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=SEED,
        class_weight="balanced_subsample",
    )
    # dual=True заметно быстрее при числе признаков > числа объектов (разреженный TF-IDF)
    if os.environ.get("PHISH_WITH_SGD") == "1":
        # эксперимент: линейная модель с другой функцией потерь. На разреженной
        # матрице (150 тыс. признаков) обучается кратно дольше logreg, поэтому
        # по умолчанию выключена.
        from sklearn.linear_model import SGDClassifier
        models["sgd"] = SGDClassifier(
            loss="modified_huber", alpha=1e-6, max_iter=20, tol=1e-5,
            class_weight="balanced", random_state=SEED,
        )
    models["logreg"] = LogisticRegression(
        C=4.0, max_iter=3000, solver="liblinear", dual=True,
        class_weight="balanced", random_state=SEED,
    )
    return models


def fit_predict_dense(model, Xtr, ytr, Xva_list) -> List[np.ndarray]:
    """Обучает модель на плотных признаках и возвращает вероятности для Xva_list."""
    if model.__class__.__name__ == "CatBoostClassifier":
        model.fit(Xtr, ytr)
    else:
        model.fit(Xtr, ytr)
    return [model.predict_proba(X)[:, 1] for X in Xva_list]


# --------------------------------------------------------------------------
# Ансамблирование и порог
# --------------------------------------------------------------------------

def best_threshold(y_true: np.ndarray, proba: np.ndarray, n_grid: int = 400) -> Tuple[float, float]:
    """Порог, максимизирующий F1 (перебор по квантилям распределения вероятностей)."""
    grid = np.unique(np.quantile(proba, np.linspace(0.01, 0.99, n_grid)))
    best_f1, best_t = 0.0, 0.5
    for t in grid:
        pred = (proba >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def ensemble_selection(oof: Dict[str, np.ndarray], y: np.ndarray,
                       n_iter: int = 25) -> Tuple[Dict[str, float], float, float]:
    """
    Жадный отбор ансамбля (Caruana): на каждом шаге добавляем модель,
    максимально повышающую F1 усреднённого предсказания.
    """
    names = list(oof.keys())
    counts = {n: 0 for n in names}
    current = np.zeros_like(y, dtype=float)
    best_f1, best_t, total = 0.0, 0.5, 0
    for _ in range(n_iter):
        step_best = None
        for n in names:
            cand = (current * total + oof[n]) / (total + 1)
            t, f1 = best_threshold(y, cand)
            if step_best is None or f1 > step_best[1]:
                step_best = (n, f1, t, cand)
        if step_best is None or (total > 0 and step_best[1] <= best_f1 + 1e-6):
            break
        name, f1, t, cand = step_best
        counts[name] += 1
        total += 1
        current = cand
        best_f1, best_t = f1, t
    weights = {n: c / max(total, 1) for n, c in counts.items() if c > 0}
    return weights, best_t, best_f1


def classification_report_dict(y: np.ndarray, proba: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "threshold": float(thr),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# --------------------------------------------------------------------------
# Основной сценарий
# --------------------------------------------------------------------------

def main() -> None:
    global SEED
    ap = argparse.ArgumentParser(description="Обучение модели обнаружения фишинговых URL")
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--test", default="data/test.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="быстрый режим (мало итераций)")
    ap.add_argument("--cache", default="cache", help="каталог кэша признаков")
    ap.add_argument("--tag", default="run", help="метка запуска для имён файлов")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="seed разбиения и моделей (для декорреляции поколений)")
    ap.add_argument("--content-train", default=None,
                    help="pkl с признаками HTML-контента для train (cache/content_train.pkl)")
    ap.add_argument("--content-test", default=None,
                    help="pkl с признаками HTML-контента для test (cache/content_test.pkl)")
    ap.add_argument("--content-text-train", default=None,
                    help="pkl с видимым текстом страниц для train (cache/content_text_train.pkl)")
    ap.add_argument("--content-text-test", default=None,
                    help="pkl с видимым текстом страниц для test (cache/content_text_test.pkl)")
    ap.add_argument("--nn-word-train", default="cache/nn_word_train.npz",
                    help="top-K соседей по словам текста (train)")
    ap.add_argument("--nn-word-test", default="cache/nn_word_test.npz",
                    help="top-K соседей по словам текста (test)")
    ap.add_argument("--nn-char-train", default="cache/nn_char_train.npz",
                    help="top-K соседей по символьным n-граммам (train)")
    ap.add_argument("--nn-char-test", default="cache/nn_char_test.npz",
                    help="top-K соседей по символьным n-граммам (test)")
    ap.add_argument("--nn-markup-train", default="cache/nn_markup_train.npz",
                    help="top-K соседей по структуре страницы (train)")
    ap.add_argument("--nn-markup-test", default="cache/nn_markup_test.npz",
                    help="top-K соседей по структуре страницы (test)")
    ap.add_argument("--nn-urlpath-train", default="cache/nn_urlpath_train.npz",
                    help="top-K соседей по шаблонам URL-путей (train)")
    ap.add_argument("--nn-urlpath-test", default="cache/nn_urlpath_test.npz",
                    help="top-K соседей по шаблонам URL-путей (test)")
    ap.add_argument("--nn-title-train", default="cache/nn_title_train.npz",
                    help="top-K соседей по заголовкам страниц (train)")
    ap.add_argument("--nn-title-test", default="cache/nn_title_test.npz",
                    help="top-K соседей по заголовкам страниц (test)")
    ap.add_argument("--sample", default=None, help="образец формата ответов")
    ap.add_argument("--out-submission", default=None,
                    help="куда сохранить submission.csv (по умолчанию не пишется)")
    args = ap.parse_args()
    SEED = args.seed          # все модели и разбиения используют этот seed
    print(f"[info] seed = {SEED}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)
    t0 = time.time()

    # ---------- 1. Данные ----------
    tr = read_table(args.train)
    url_col, label_col = detect_columns(tr, need_label=True)
    tr = tr.rename(columns={url_col: "url", label_col: "label"})
    tr["url"] = tr["url"].astype(str)
    tr["label"] = to_binary(tr["label"])
    tr = tr.dropna(subset=["url"])
    tr = tr[tr["url"].str.len() > 0]

    # Противоречивые дубликаты (один URL с разными метками) удаляем
    grp = tr.groupby("url")["label"].nunique()
    bad_urls = set(grp[grp > 1].index)
    if bad_urls:
        print(f"[info] удалено противоречивых URL: {len(bad_urls)}")
        tr = tr[~tr["url"].isin(bad_urls)]
    tr = tr.drop_duplicates(subset=["url"]).reset_index(drop=True)
    print(f"[info] train: {tr.shape}, доля фишинга = {tr['label'].mean():.4f}")

    test = None
    if args.test and os.path.exists(args.test):
        te = read_table(args.test)
        te_url_col, _ = detect_columns(te, need_label=False)
        te = te.rename(columns={te_url_col: "url"})
        te["url"] = te["url"].astype(str)
        test = te.reset_index(drop=True)
        print(f"[info] test: {test.shape}")

    # ---------- 2. Признаки ----------
    cache_f = os.path.join(args.cache, f"feat_{args.tag}_train.pkl")
    if os.path.exists(cache_f) and not args.quick:
        F_tr = pd.read_pickle(cache_f)
        print(f"[info] признаки train взяты из кэша {cache_f}")
    else:
        F_tr = build_feature_matrix(tr["url"].tolist())
        F_tr.to_pickle(cache_f)
    F_te = None
    if test is not None:
        cache_te = os.path.join(args.cache, f"feat_{args.tag}_test.pkl")
        if os.path.exists(cache_te) and not args.quick:
            F_te = pd.read_pickle(cache_te)
        else:
            F_te = build_feature_matrix(test["url"].tolist())
            F_te.to_pickle(cache_te)

    count_maps: Dict[str, dict] = {}
    if F_te is not None:
        F_tr, F_te, count_maps = add_corpus_features(F_tr, F_te, tr["url"], test["url"])

    # Признаки HTML-контента (если подготовлены src/content_features.py)
    if args.content_train:
        F_tr = attach_content_features(F_tr, tr["url"], args.content_train, "train")
    if args.content_test and F_te is not None:
        F_te = attach_content_features(F_te, test["url"], args.content_test, "test")
    if F_te is not None:
        # одинаковая колонка-порядок для train и test
        for c in F_tr.columns:
            if c not in F_te.columns:
                F_te[c] = 0.0
        F_te = F_te[[c for c in F_tr.columns]]

    numeric_cols = [c for c in F_tr.columns if c != "tld"]
    X_num = F_tr[numeric_cols].astype(np.float32).values
    X_num_te = F_te[numeric_cols].astype(np.float32).values if F_te is not None else None

    # Текстовые представления
    txt_char_tr = [char_ngram_text(u) for u in tr["url"]]
    txt_word_tr = [url_word_text(u) for u in tr["url"]]
    if test is not None:
        txt_char_te = [char_ngram_text(u) for u in test["url"]]
        txt_word_te = [url_word_text(u) for u in test["url"]]

    (Xc_tr, Xc_te), (Xw_tr, Xw_te), char_vec, word_vec = make_text_matrices(
        txt_char_tr, txt_char_te if test is not None else [""]
    )

    # Текст страницы (видимый HTML-текст) -> TF-IDF: ловит шаблоны фишинговых «китов»
    Xh_tr = Xh_te = None
    html_vec = None
    if args.content_text_train:
        txt_html_tr = align_text(tr["url"], args.content_text_train, "train")
        txt_html_te = (align_text(test["url"], args.content_text_test, "test")
                       if (args.content_text_test and test is not None) else [""] * (len(test) if test is not None else 0))
        html_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3,
                                   max_features=60000, sublinear_tf=True,
                                   strip_accents="unicode")
        Xh_tr = html_vec.fit_transform(txt_html_tr)
        Xh_te = html_vec.transform(txt_html_te) if test is not None else None
        print(f"[info] HTML-текст: {Xh_tr.shape[1]} n-грамм, "
              f"непустых страниц {np.mean([bool(t.strip()) for t in txt_html_tr]) * 100:.1f} %")

    # Разреженный текстовый блок (URL char/word + HTML word) — одинаков для всех фолдов
    text_blocks_tr = [Xc_tr, Xw_tr] + ([Xh_tr] if Xh_tr is not None else [])
    X_text_tr = hstack(text_blocks_tr).tocsr()
    X_text_te = None
    if F_te is not None:
        text_blocks_te = [Xc_te, Xw_te] + ([Xh_te] if Xh_te is not None else [])
        X_text_te = hstack(text_blocks_te).tocsr()

    svd_dim = 64 if args.quick else 128
    svd = TruncatedSVD(n_components=svd_dim, random_state=SEED)
    S_tr = svd.fit_transform(X_text_tr)
    S_te = svd.transform(X_text_te) if X_text_te is not None else None

    y = tr["label"].values
    X_num_base = np.nan_to_num(X_num, nan=0.0, posinf=0.0, neginf=0.0)
    X_num_te_base = (np.nan_to_num(X_num_te, nan=0.0, posinf=0.0, neginf=0.0)
                     if F_te is not None else None)

    # ---- Признаки «похожесть на известные страницы» (top-K соседей по контенту) ----
    nn_sources = []
    for key, tr_path, te_path in (("word", args.nn_word_train, args.nn_word_test),
                                  ("markup", args.nn_markup_train, args.nn_markup_test),
                                  ("urlpath", args.nn_urlpath_train, args.nn_urlpath_test),
                                  ("title", args.nn_title_train, args.nn_title_test),
                                  ("char", args.nn_char_train, args.nn_char_test)):
        a = load_nn(tr_path)
        b = load_nn(te_path)
        if a is None or b is None:
            continue
        # защита от несовпадения размеров (например, списки соседей собраны
        # для другого набора данных)
        if a[0].shape[0] != len(y) or (F_te is not None and b[0].shape[0] != len(test)):
            print(f"[warn] {key}: размер списка соседей не совпадает с данными — пропускаю")
            continue
        nn_sources.append((key, a[0], a[1], b[0], b[1]))
    if nn_sources:
        print(f"[info] подключены соседские представления: "
              f"{', '.join(k for k, *_ in nn_sources)} (top-K={nn_sources[0][1].shape[1]})")
    else:
        print("[info] соседские признаки не найдены — обучение без них")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    fold_ids = np.zeros(len(y), dtype=np.int16)
    for f, (_, iva) in enumerate(skf.split(X_num_base, y), 1):
        fold_ids[iva] = f

    def nn_matrices(current_fold):
        """Матрицы NN-признаков для train и test при данном разбиении."""
        blocks_tr, blocks_te, names = [], [], []
        for key, idx_tr, sim_tr, idx_te, sim_te in nn_sources:
            f_tr, nm = nn_feature_block(idx_tr, sim_tr, y, fold_of=fold_ids,
                                        current_fold=current_fold, prefix=f"nn{key}")
            blocks_tr.append(f_tr)
            names += nm
            if F_te is not None:
                f_te, _ = nn_feature_block(idx_te, sim_te, y, prefix=f"nn{key}")
                blocks_te.append(f_te)
        out_tr = np.hstack(blocks_tr) if blocks_tr else np.zeros((len(y), 0), dtype=np.float32)
        out_te = (np.hstack(blocks_te) if blocks_te and F_te is not None
                  else np.zeros((len(test) if F_te is not None else 0, 0), dtype=np.float32))
        return out_tr, out_te, names

    # ---------- 3. Кросс-валидация ----------
    models = get_models(quick=args.quick)
    oof: Dict[str, np.ndarray] = {n: np.zeros(len(y)) for n in models}
    test_pred: Dict[str, np.ndarray] = {}
    fold_metrics: Dict[str, List[Dict[str, float]]] = {n: [] for n in models}
    nn_names: List[str] = []

    for fold, (itr, iva) in enumerate(skf.split(X_num_base, y), 1):
        print(f"\n===== fold {fold}/{args.folds} =====")
        ytr, yva = y[itr], y[iva]

        nn_tr, nn_te, nn_names = nn_matrices(fold)
        X_num_f = np.hstack([X_num_base, nn_tr]).astype(np.float32)
        scaler = StandardScaler()
        Xs_f = scaler.fit_transform(X_num_f[itr])
        Xs_f_all = scaler.transform(X_num_f)
        X_lin_tr = hstack([csr_matrix(Xs_f_all), X_text_tr]).tocsr()
        X_lgb_tr = np.hstack([X_num_f, S_tr]).astype(np.float32)

        if F_te is not None:
            X_num_te_f = np.hstack([X_num_te_base, nn_te]).astype(np.float32)
            X_lin_te = hstack([csr_matrix(scaler.transform(X_num_te_f)), X_text_te]).tocsr()
            X_lgb_te = np.hstack([X_num_te_f, S_te]).astype(np.float32)

        for name, proto in models.items():
            import copy
            model = copy.deepcopy(proto)
            if name in ("logreg", "sgd"):
                model.fit(X_lin_tr[itr], ytr)
                p_va = model.predict_proba(X_lin_tr[iva])[:, 1]
                if F_te is not None:
                    p_te = model.predict_proba(X_lin_te)[:, 1]
            elif name == "lightgbm":
                model.fit(X_lgb_tr[itr], ytr)
                p_va = model.predict_proba(X_lgb_tr[iva])[:, 1]
                if F_te is not None:
                    p_te = model.predict_proba(X_lgb_te)[:, 1]
            else:
                model.fit(X_num_f[itr], ytr)
                p_va = model.predict_proba(X_num_f[iva])[:, 1]
                if F_te is not None:
                    p_te = model.predict_proba(X_num_te_f)[:, 1]
            oof[name][iva] = p_va
            thr, f1 = best_threshold(yva, p_va)
            m = classification_report_dict(yva, p_va, thr)
            fold_metrics[name].append(m)
            print(f"  {name:<14} F1={m['f1']:.4f} (thr={thr:.3f})  "
                  f"P={m['precision']:.4f} R={m['recall']:.4f} AUC={m['roc_auc']:.4f}")
            if F_te is not None:
                test_pred.setdefault(name, np.zeros(len(test)))
                test_pred[name] += p_te / args.folds
        # чекпоинт после каждого фолда: защита от потери длинного прогона
        try:
            np.savez(os.path.join(MODELS_DIR, f"ckpt_{args.tag}.npz"),
                     fold=fold, y=y, oof=np.column_stack([oof[n] for n in models]),
                     test_pred=np.column_stack([test_pred[n] for n in models]) if test_pred else np.zeros((0, 0)),
                     model_names=np.array(list(models.keys())))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] чекпоинт не сохранён: {exc}")

    # ---------- 4. Ансамбль ----------
    print("\n===== OOF-качество базовых моделей =====")
    summary = {}
    for name in models:
        thr, f1 = best_threshold(y, oof[name])
        m = classification_report_dict(y, oof[name], thr)
        summary[name] = m
        print(f"  {name:<14} F1={m['f1']:.4f} (thr={thr:.3f}) P={m['precision']:.4f} "
              f"R={m['recall']:.4f} AUC={m['roc_auc']:.4f}")

    weights, thr_ens, f1_ens = ensemble_selection(oof, y, n_iter=25)
    blended_oof = sum(oof[n] * w for n, w in weights.items())
    thr_ens, f1_ens = best_threshold(y, blended_oof)
    m_ens = classification_report_dict(y, blended_oof, thr_ens)
    print("\n===== Ансамбль (жадный отбор) =====")
    print(f"  веса: { {k: round(v, 3) for k, v in weights.items()} }")
    print(f"  F1={m_ens['f1']:.4f} thr={thr_ens:.3f} P={m_ens['precision']:.4f} "
          f"R={m_ens['recall']:.4f} AUC={m_ens['roc_auc']:.4f} "
          f"TN={m_ens['tn']} FP={m_ens['fp']} FN={m_ens['fn']} TP={m_ens['tp']}")

    # ---------- 4б. Стекинг (мета-модель на OOF-предсказаниях) ----------
    m_stack = None
    stacked_oof = None
    stacked_test = None
    meta_full = None
    meta_scaler = None
    base_names = list(models.keys())
    thr_stack = 0.5
    if len(models) > 1 and F_te is not None:
        names = list(models.keys())
        eps = 1e-6

        def to_logit(p: np.ndarray) -> np.ndarray:
            p = np.clip(p, eps, 1 - eps)
            return np.log(p / (1 - p))

        # В мета-модель подаём логиты базовых моделей и те же соседские признаки
        # (именно они дают основной прирост на незнакомых доменах). Для train
        # соседи, попавшие в свой фолд, исключаются — оценка остаётся честной.
        _, nn_te_meta, _ = nn_matrices(current_fold=None)
        extra_meta = [to_logit(oof[n]) for n in names]
        extra_te = [to_logit(test_pred[n]) for n in names]
        if nn_te_meta.shape[1]:
            nn_meta_tr = np.zeros((len(y), nn_te_meta.shape[1]), dtype=np.float32)
            for f_ in range(1, args.folds + 1):
                iva_f = fold_ids == f_
                blk_f, _, _ = nn_matrices(current_fold=f_)
                nn_meta_tr[iva_f] = blk_f[iva_f]
            extra_meta.append(nn_meta_tr)
            extra_te.append(nn_te_meta)
        Z = np.column_stack(extra_meta)
        Z_te = np.column_stack(extra_te)
        meta_oof = np.zeros(len(y))
        meta_te = np.zeros(len(test))
        for f_ in range(1, args.folds + 1):
            itr, iva = (fold_ids != f_), (fold_ids == f_)
            sc = StandardScaler().fit(Z[itr])        # масштаб внутри фолда — без утечки
            meta = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                                      random_state=SEED)
            meta.fit(sc.transform(Z[itr]), y[itr])
            meta_oof[iva] = meta.predict_proba(sc.transform(Z[iva]))[:, 1]
            meta_te += meta.predict_proba(sc.transform(Z_te))[:, 1] / args.folds
        meta_scaler = StandardScaler().fit(Z)
        meta_full = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                                       random_state=SEED)
        meta_full.fit(meta_scaler.transform(Z), y)   # финальная мета-модель для бандла
        thr_stack, f1_stack = best_threshold(y, meta_oof)
        m_stack = classification_report_dict(y, meta_oof, thr_stack)
        stacked_oof, stacked_test = meta_oof, meta_te
        print("\n===== Стекинг (мета-логистическая регрессия) =====")
        print(f"  F1={m_stack['f1']:.4f} thr={thr_stack:.3f} P={m_stack['precision']:.4f} "
              f"R={m_stack['recall']:.4f} AUC={m_stack['roc_auc']:.4f}")

    # выбор лучшей схемы ансамблирования по OOF
    use_stack = bool(m_stack and m_stack["f1"] > m_ens["f1"])
    if use_stack:
        print(f"[info] выбрана схема: СТЕКИНГ (F1 {m_stack['f1']:.4f} > {m_ens['f1']:.4f})")
        final_oof, final_thr, final_metrics = stacked_oof, thr_stack, m_stack
        scheme = "stacking"
    else:
        print(f"[info] выбрана схема: ЖАДНЫЙ ОТБОР (F1 {m_ens['f1']:.4f} >= "
              f"{m_stack['f1'] if m_stack else float('nan'):.4f})")
        final_oof, final_thr, final_metrics = blended_oof, thr_ens, m_ens
        scheme = "greedy_blend"

    # ---------- 5. Финальное обучение на всех данных ----------
    # NN-признаки считаются по полному train (для train — без самого объекта,
    # для test — по всем известным страницам).
    nn_tr_full, nn_te_full, nn_names = nn_matrices(current_fold=None)
    X_num_full = np.hstack([X_num_base, nn_tr_full]).astype(np.float32)
    scaler = StandardScaler()
    Xs_full = scaler.fit_transform(X_num_full)
    X_lin_full = hstack([csr_matrix(Xs_full), X_text_tr]).tocsr()
    X_lgb_full = np.hstack([X_num_full, S_tr]).astype(np.float32)
    if F_te is not None:
        X_num_te_full = np.hstack([X_num_te_base, nn_te_full]).astype(np.float32)
        X_lin_te_full = hstack([csr_matrix(scaler.transform(X_num_te_full)), X_text_te]).tocsr()
        X_lgb_te_full = np.hstack([X_num_te_full, S_te]).astype(np.float32)
        np.save(os.path.join(MODELS_DIR, f"nn_test_{args.tag}.npy"), nn_te_full)
    numeric_cols = numeric_cols + list(nn_names)

    final_models = {}
    for name, proto in models.items():
        import copy
        model = copy.deepcopy(proto)
        if name in ("logreg", "sgd"):
            model.fit(X_lin_full, y)
        elif name == "lightgbm":
            model.fit(X_lgb_full, y)
        else:
            model.fit(X_num_full, y)
        final_models[name] = model
    print(f"[info] финальные модели обучены на полном train "
          f"({len(numeric_cols)} числовых признаков, включая {len(nn_names)} соседских)")

    # ---------- 6. Предсказание для test ----------
    if F_te is not None:
        # Для финального ответа используются усреднённые по фолдам предсказания
        # (test_pred): их распределение совпадает с тем, на котором обучалась
        # мета-модель стекинга и подбирался порог.
        proba_te = np.zeros(len(test))
        for name, w in weights.items():
            proba_te += test_pred[name] * w
        proba_te /= max(sum(weights.values()), 1e-9)
        if use_stack:
            proba_te = stacked_test
        np.save(os.path.join(MODELS_DIR, f"test_proba_{args.tag}.npy"), proba_te)
        np.savez(os.path.join(MODELS_DIR, f"test_base_{args.tag}.npz"),
                 **{k: v for k, v in test_pred.items()})
        np.save(os.path.join(MODELS_DIR, f"oof_final_{args.tag}.npy"), final_oof)

        # ---------- 6б. Файл ответов для платформы ----------
        if args.out_submission:
            from submission import build_submission
            sample = args.sample
            if sample is None:
                for cand in ("data/sample.csv", "data/sample_submission.csv",
                             "data/sample_submit.csv", "sample.csv"):
                    if os.path.exists(cand):
                        sample = cand
                        break
            sub = build_submission(test, proba_te, final_thr, sample_path=sample)
            sub.to_csv(args.out_submission, index=False)
            print(f"[ok] файл ответов: {args.out_submission} ({len(sub)} строк, "
                  f"колонки {list(sub.columns)}, доля фишинга "
                  f"{sub[sub.columns[-1]].mean():.4f})")

    # ---------- 7. Сохранение ----------
    import joblib
    joblib.dump({"models": final_models, "weights": weights, "threshold": final_thr,
                 "scheme": scheme, "numeric_cols": numeric_cols, "scaler": scaler,
                 "svd": svd, "char_vec": char_vec, "word_vec": word_vec,
                 "html_vec": html_vec, "count_maps": count_maps,
                 "meta_model": meta_full if use_stack else None,
                 "base_names": base_names, "nn_names": list(nn_names),
                 "meta_scaler": meta_scaler if use_stack else None},
                os.path.join(MODELS_DIR, f"bundle_{args.tag}.joblib"))
    np.save(os.path.join(MODELS_DIR, f"oof_blend_{args.tag}.npy"), blended_oof)
    with open(os.path.join(MODELS_DIR, f"oof_{args.tag}.npz"), "wb") as fh:
        np.savez(fh, y=y, **{k: v for k, v in oof.items()})

    report = {
        "tag": args.tag,
        "n_train": int(len(tr)), "n_features": len(numeric_cols),
        "phishing_rate": float(tr["label"].mean()),
        "folds": args.folds,
        "base_models": summary,
        "ensemble_weights": weights,
        "ensemble": m_ens,
        "stacking": m_stack,
        "scheme": scheme,
        "final_metrics": final_metrics,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(REPORTS_DIR, f"cv_report_{args.tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    lines = ["# Отчёт кросс-валидации", "",
             f"- Дата запуска: {time.strftime('%Y-%m-%d %H:%M')}",
             f"- Обучающая выборка: {len(tr)} URL, доля фишинга {tr['label'].mean():.4f}",
             f"- Признаков: {len(numeric_cols)} (+ TF-IDF char/word)",
             f"- Схема валидации: StratifiedKFold, {args.folds} фолдов", "",
             "## Базовые модели (OOF)", "",
             "| Модель | F1 | Precision | Recall | ROC-AUC | PR-AUC | Точность (accuracy) |",
             "|---|---|---|---|---|---|---|"]
    for name, m in summary.items():
        lines.append(f"| {name} | {m['f1']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | "
                     f"{m['roc_auc']:.4f} | {m['pr_auc']:.4f} | {m['accuracy']:.4f} |")
    lines += ["", "## Ансамбль", "",
              f"Веса: {weights}", "",
              f"| F1 | Precision | Recall | ROC-AUC | Порог |",
              "|---|---|---|---|---|",
              f"| {m_ens['f1']:.4f} | {m_ens['precision']:.4f} | {m_ens['recall']:.4f} | "
              f"{m_ens['roc_auc']:.4f} | {thr_ens:.4f} |", "",
              f"Матрица ошибок: TN={m_ens['tn']}, FP={m_ens['fp']}, FN={m_ens['fn']}, TP={m_ens['tp']}"]

    if m_stack:
        lines += ["", "## Стекинг (мета-модель на OOF-предсказаниях)", "",
                  "| F1 | Precision | Recall | ROC-AUC | Порог |",
                  "|---|---|---|---|---|",
                  f"| {m_stack['f1']:.4f} | {m_stack['precision']:.4f} | "
                  f"{m_stack['recall']:.4f} | {m_stack['roc_auc']:.4f} | {thr_stack:.4f} |"]
    lines += ["", "## Итоговая схема", "",
              f"Выбрана схема **{scheme}** (сравнение по OOF F1):", "",
              "| F1 | Precision | Recall | ROC-AUC | Accuracy | Порог |",
              "|---|---|---|---|---|---|",
              f"| {final_metrics['f1']:.4f} | {final_metrics['precision']:.4f} | "
              f"{final_metrics['recall']:.4f} | {final_metrics['roc_auc']:.4f} | "
              f"{final_metrics['accuracy']:.4f} | {final_thr:.4f} |", "",
              f"Матрица ошибок: TN={final_metrics['tn']}, FP={final_metrics['fp']}, "
              f"FN={final_metrics['fn']}, TP={final_metrics['tp']}", "",
              f"Время обучения: {time.time() - t0:.0f} с."]
    with open(os.path.join(REPORTS_DIR, f"cv_report_{args.tag}.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\n[info] готово за {time.time() - t0:.1f} с. Отчёт: reports/cv_report_{args.tag}.md")


if __name__ == "__main__":
    main()
