# -*- coding: utf-8 -*-
"""
explain.py — анализ техник фишинга и объяснение работы модели.

Формирует:
  * reports/technique_analysis.md — какие приёмы злоумышленников реально
    встречаются в данных и насколько они «сигнальны» (lift относительно
    доброкачественных URL);
  * reports/figures/*.png — графики распределений, важности признаков,
    матрица ошибок, распределение предсказанных вероятностей;
  * примеры тайпсквоттинга, мимикрии и «ключевых слов-приманок» с указанием
    бренда, под который маскируется адрес;
  * разбор ложноположительных и ложноотрицательных срабатываний модели.

Запуск:
    python src/explain.py --train data/train.csv --tag run
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import (  # noqa: E402
    build_feature_matrix,
    extract_features,
    find_mimicked_brand,
    find_typosquat_brand,
    split_host,
)
from train import URL_ALIASES, read_table, to_binary  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(ROOT, "reports")
FIGDIR = os.path.join(REPORTS, "figures")
MODELS = os.path.join(ROOT, "models")


# --------------------------------------------------------------------------
# Техники атак
# --------------------------------------------------------------------------

TECHNIQUES = {
    "is_typosquatting": "Тайпсквоттинг (опечатка в домене)",
    "is_homoglyph_attack": "Гомоглифы (paypa1, rnicrosoft)",
    "is_hyphen_brand_attack": "Бренд с добавкой/дефисом (sberbank-online)",
    "brand_mimicry": "Мимикрия: бренд вне домена",
    "brand_in_subdomain": "Бренд в поддомене",
    "brand_in_path": "Бренд в пути",
    "attack_keywords": "Ключевые слова-приманки (>=2)",
    "has_login_word": "Слова login/signin",
    "has_verify_word": "Слова verify/confirm/update",
    "has_money_word": "Финансовые слова (pay/bank/wallet)",
    "has_prize_word": "Призы и бонусы (bonus/prize/free)",
    "has_delivery_word": "Доставка (parcel/tracking/order)",
    "is_ip_host": "IP-адрес вместо домена",
    "has_userinfo": "Конструкция user@host",
    "suspicious_tld": "«Плохая» зона (.tk/.xyz/.top...)",
    "is_free_host": "Бесплатный хостинг",
    "is_shortener": "Сокращатель ссылок",
    "many_subdomains": "Много поддоменов (>=3)",
    "has_hex_encoding": "Процентное кодирование",
    "is_punycode": "Punycode (xn--)",
    "has_port": "Нестандартный порт",
    "url_is_long": "Длинный URL (>75 символов)",
    "double_extension": "Двойное расширение (.jpg.php)",
    "has_many_params": "Много параметров запроса",
}


def technique_table(df: pd.DataFrame) -> pd.DataFrame:
    """Считает распространённость техник в классах и lift."""
    rows = []
    y = df["result"].values
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    for col, title in TECHNIQUES.items():
        if col not in df.columns:
            continue
        if col == "has_many_params":
            values = (df["n_query_params"] >= 2).astype(int).values
        else:
            values = df[col].values
        p_pos = float(values[y == 1].mean()) if n_pos else 0.0
        p_neg = float(values[y == 0].mean()) if n_neg else 0.0
        rows.append({
            "technique": title,
            "feature": col,
            "phishing_share": p_pos,
            "legit_share": p_neg,
            "lift": (p_pos + 1e-6) / (p_neg + 1e-6),
            "n_phishing": int(values[y == 1].sum()),
        })
    return pd.DataFrame(rows).sort_values("lift", ascending=False)


# --------------------------------------------------------------------------
# Графики
# --------------------------------------------------------------------------

def plot_distributions(df: pd.DataFrame) -> List[str]:
    files = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    y = df["result"].values

    for ax, col, title in [
        (axes[0, 0], "url_len", "Длина URL"),
        (axes[0, 1], "entropy_host", "Энтропия домена"),
        (axes[1, 0], "n_phish_words", "Число слов-приманок"),
        (axes[1, 1], "n_subdomains", "Число поддоменов"),
    ]:
        bins = np.linspace(0, np.percentile(df[col], 99) or 1, 40)
        ax.hist(df[col][y == 0], bins=bins, alpha=0.6, label="безопасные", density=True)
        ax.hist(df[col][y == 1], bins=bins, alpha=0.6, label="фишинг", density=True)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIGDIR, "distributions.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    files.append(path)

    tt = technique_table(df)
    tt = tt[tt["phishing_share"] >= 0.005]          # отбрасываем сверхредкие признаки
    top = tt.sort_values("phishing_share", ascending=False).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["technique"], top["phishing_share"] * 100, color="#c0392b", alpha=0.85,
            label="фишинг")
    ax.barh(top["technique"], top["legit_share"] * 100, color="#2980b9", alpha=0.55,
            label="безопасные")
    ax.set_xlabel("Доля URL с признаком, %")
    ax.set_title("Распространённость техник в классах")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIGDIR, "techniques.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    files.append(path)
    return files


def plot_importances(df: pd.DataFrame, top_n: int = 25) -> List[str]:
    """Важность признаков усредняется по доступным моделям из бандла."""
    import joblib
    files = []
    importances: Dict[str, np.ndarray] = {}
    for tag in ("final", "run"):
        path = os.path.join(MODELS, f"bundle_{tag}.joblib")
        if not os.path.exists(path):
            continue
        bundle = joblib.load(path)
        cols = bundle["numeric_cols"]
        acc = np.zeros(len(cols))
        used = 0
        for name, model in bundle["models"].items():
            imp = None
            if hasattr(model, "feature_importances_"):
                imp = np.asarray(model.feature_importances_, dtype=float)
            elif hasattr(model, "get_feature_importance"):
                try:
                    imp = np.asarray(model.get_feature_importance(), dtype=float)
                except Exception:  # noqa: BLE001
                    imp = None
            if imp is not None and len(imp) == len(cols) and imp.sum() > 0:
                acc += imp / imp.sum()
                used += 1
        if used:
            importances = {"model": acc / used, "cols": cols}
            break
    if not importances:
        return files

    order = np.argsort(importances["model"])[::-1][:top_n]
    labels = [importances["cols"][i] for i in order]
    values = importances["model"][order]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(labels[::-1], values[::-1], color="#16a085")
    ax.set_title(f"Топ-{top_n} признаков по важности (среднее по моделям)")
    ax.set_xlabel("Нормированная важность")
    fig.tight_layout()
    path = os.path.join(FIGDIR, "importances.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    files.append(path)
    return files


def plot_oof(tag: str = "run") -> List[str]:
    """Распределение OOF-вероятностей + матрица ошибок при рабочем пороге."""
    import joblib
    files = []
    oof_path = os.path.join(MODELS, f"oof_{tag}.npz")
    bundle_path = os.path.join(MODELS, f"bundle_{tag}.joblib")
    if not (os.path.exists(oof_path) and os.path.exists(bundle_path)):
        return files
    data = np.load(oof_path)
    y = data["y"]
    blend_path = os.path.join(MODELS, f"oof_final_{tag}.npy")
    if not os.path.exists(blend_path):
        blend_path = os.path.join(MODELS, f"oof_blend_{tag}.npy")
    proba = np.load(blend_path)
    thr = float(joblib.load(bundle_path)["threshold"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(proba[y == 0], bins=60, alpha=0.7, label="безопасные", color="#2980b9")
    axes[0].hist(proba[y == 1], bins=60, alpha=0.7, label="фишинг", color="#c0392b")
    axes[0].axvline(thr, color="k", ls="--", label=f"порог {thr:.3f}")
    axes[0].set_yscale("log")
    axes[0].set_title("OOF-вероятности по классам")
    axes[0].legend()

    pred = (proba >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    cm = np.array([[tn, fp], [fn, tp]])
    im = axes[1].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f"{cm[i, j]:,}".replace(",", " "), ha="center", va="center",
                         color="black", fontsize=12)
    axes[1].set_xticks([0, 1], ["pred 0", "pred 1"])
    axes[1].set_yticks([0, 1], ["true 0", "true 1"])
    axes[1].set_title(f"Матрица ошибок (порог {thr:.3f})")
    fig.colorbar(im, ax=axes[1], shrink=0.8)
    fig.tight_layout()
    path = os.path.join(FIGDIR, "oof_errors.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    files.append(path)
    return files


# --------------------------------------------------------------------------
# Примеры и разбор ошибок
# --------------------------------------------------------------------------

def technique_examples(df: pd.DataFrame, limit: int = 12) -> Dict[str, pd.DataFrame]:
    """Собирает наглядные примеры по каждой технике."""
    out: Dict[str, pd.DataFrame] = {}
    pos = df[df["result"] == 1]

    typo = pos[pos["is_typosquatting"] == 1].head(limit).copy()
    if len(typo):
        typo["imitates"] = typo["url"].map(lambda u: find_typosquat_brand(split_host_netloc(u)))
        typo["distance"] = [find_typosquat_brand(split_host_netloc(u))[1] for u in typo["url"]]
    out["typosquatting"] = typo

    homo = pos[pos["is_homoglyph_attack"] == 1].head(limit).copy()
    if len(homo):
        homo["imitates"] = homo["url"].map(lambda u: find_typosquat_brand(split_host_netloc(u))[0])
    out["homoglyph"] = homo

    mimic = pos[pos["brand_mimicry"] == 1].head(limit).copy()
    if len(mimic):
        mimic["imitates"] = mimic["url"].map(find_mimicked_brand)
    out["mimicry"] = mimic

    kw = pos[pos["n_phish_words"] >= 3].head(limit)
    out["keywords"] = kw[["url", "n_phish_words"]]
    ip = pos[pos["is_ip_host"] == 1].head(limit)
    out["ip"] = ip[["url"]]
    return out


def split_host_netloc(url: str) -> str:
    from urllib.parse import urlsplit
    from features import normalize_url
    host = urlsplit(normalize_url(url)).netloc.lower()
    if "@" in host:
        host = host.rsplit("@", 1)[1]
    return host.partition(":")[0].strip(".")


def error_examples(df: pd.DataFrame, tag: str = "run", limit: int = 15):
    """Топ ложноположительных и ложноотрицательных срабатываний."""
    path = os.path.join(MODELS, f"oof_final_{tag}.npy")
    if not os.path.exists(path):
        path = os.path.join(MODELS, f"oof_blend_{tag}.npy")
    if not os.path.exists(path) or len(df) != len(np.load(path)):
        return None, None
    import joblib
    proba = np.load(path)
    thr = float(joblib.load(os.path.join(MODELS, f"bundle_{tag}.joblib"))["threshold"])
    d = df.copy()
    d["proba"] = proba
    pred = (proba >= thr).astype(int)
    d["pred"] = pred
    fp = d[(d["result"] == 0) & (d["pred"] == 1)].sort_values("proba", ascending=False).head(limit)
    fn = d[(d["result"] == 1) & (d["pred"] == 0)].sort_values("proba").head(limit)
    return fp[["url", "proba"]], fn[["url", "proba"]]


# --------------------------------------------------------------------------
# Отчёт
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Анализ техник фишинга и объяснение модели")
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args()

    os.makedirs(FIGDIR, exist_ok=True)
    tr = read_table(args.train)
    url_col = next((c for c in tr.columns if c in URL_ALIASES), tr.columns[0])
    label_col = next((c for c in tr.columns if c != url_col), tr.columns[-1])
    tr = tr.rename(columns={url_col: "url", label_col: "result"})
    tr["result"] = to_binary(tr["result"])
    # та же предобработка, что в train.py — иначе OOF-предсказания не совпадут по строкам
    grp = tr.groupby("url")["result"].nunique()
    bad = set(grp[grp > 1].index)
    if bad:
        tr = tr[~tr["url"].isin(bad)]
    tr = tr.drop_duplicates(subset=["url"]).reset_index(drop=True)

    cache = os.path.join(ROOT, "cache", "feat_explain_train.pkl")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    if os.path.exists(cache) and len(pd.read_pickle(cache)) == len(tr):
        F = pd.read_pickle(cache)
    else:
        F = build_feature_matrix(tr["url"].tolist())
        F.to_pickle(cache)
    df = pd.concat([tr[["url", "result"]], F], axis=1)

    # ---------------- графики ----------------
    figs = plot_distributions(df) + plot_importances(df) + plot_oof(args.tag)
    print(f"[info] графиков сохранено: {len(figs)}")

    # ---------------- таблицы ----------------
    tt = technique_table(df)
    ex = technique_examples(df, args.limit)
    fp, fn = error_examples(df, args.tag)

    lines: List[str] = []
    lines.append("# Анализ техник фишинга и объяснение модели\n")
    lines.append(f"Обучающая выборка: **{len(df)}** URL "
                 f"(фишинг: {int((df['result'] == 1).sum())}, "
                 f"безопасные: {int((df['result'] == 0).sum())}).\n")
    lines.append("## 1. Распространённость техник атак\n")
    lines.append("`lift` = доля признака среди фишинга / доля среди безопасных URL. "
                 "Значение > 1 означает, что признак чаще встречается у фишинга.\n")
    lines.append("| Техника | Доля у фишинга | Доля у безопасных | lift | Найдено у фишинга |")
    lines.append("|---|---|---|---|---|")
    for _, r in tt.iterrows():
        lines.append(f"| {r['technique']} | {r['phishing_share'] * 100:.2f}% | "
                     f"{r['legit_share'] * 100:.2f}% | {r['lift']:.1f} | {r['n_phishing']} |")

    def add_examples(title: str, frame: pd.DataFrame, cols: List[str]) -> None:
        lines.append(f"\n## {title}\n")
        if frame is None or not len(frame):
            lines.append("_Примеры не найдены._")
            return
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for _, r in frame.iterrows():
            vals = []
            for c in cols:
                v = r.get(c, "")
                v = "" if pd.isna(v) else str(v)
                vals.append(v.replace("|", "\\|")[:110])
            lines.append("| " + " | ".join(vals) + " |")

    add_examples("2. Тайпсквоттинг: домены-двойники",
                 ex.get("typosquatting"), ["url", "imitates", "distance"])
    add_examples("3. Гомоглифные подмены",
                 ex.get("homoglyph"), ["url", "imitates"])
    add_examples("4. Мимикрия под бренд (бренд вне домена)",
                 ex.get("mimicry"), ["url", "imitates"])
    add_examples("5. Ключевые слова-приманки",
                 ex.get("keywords"), ["url", "n_phish_words"])
    add_examples("6. IP-адрес вместо домена",
                 ex.get("ip"), ["url"])

    lines.append("\n## 7. Ошибки модели на OOF-предсказаниях\n")
    if fp is None:
        lines.append("_Файл OOF-предсказаний не найден: сначала запустите src/train.py._")
    else:
        add_examples("7.1. Ложноположительные (безопасный сайт признан фишингом)",
                     fp, ["url", "proba"])
        add_examples("7.2. Ложноотрицательные (фишинг пропущен)",
                     fn, ["url", "proba"])

    lines.append("\n## 8. Графики\n")
    for f in figs:
        lines.append(f"![{os.path.basename(f)}](figures/{os.path.basename(f)})\n")

    path = os.path.join(REPORTS, "technique_analysis.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"[ok] отчёт: {path}")
    print(tt.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
