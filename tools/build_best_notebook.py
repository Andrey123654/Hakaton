# -*- coding: utf-8 -*-
"""
build_best_notebook.py — сборка тетрадки по лучшему решению.

Тетрадка воспроизводит финальное решение хакатона (публичный F1 = 0.996062):
мета-модель на логитах моделей обеих линий и сетевых признаках домена.
Использует только артефакты из каталога artifacts/, поэтому воспроизводится
детерминированно и не зависит от сетевых запросов.

Запуск:
    python tools/build_best_notebook.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/best_solution_0.996062.ipynb
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notebooks", "best_solution_0.996062.ipynb")

CELLS = []


def md(text: str) -> None:
    CELLS.append(new_markdown_cell(text))


def code(text: str) -> None:
    CELLS.append(new_code_cell(text))


md("""# Обнаружение фишинговых URL: решение с публичным F1 = 0.996062

**Практическая работа 4 — хакатон на платформе Cyberhackai**
Дисциплина: «Применение ML в кибербезопасности». Итог: 3-е место.

Тетрадка воспроизводит финальное решение: **публичный F1 = 0.996062**,
кросс-валидация (OOF) **F1 ≈ 0.9919** на 63 900 размеченных URL.

## Идея решения

1. **Две независимо разработанные линии моделей.**
   Линия A — иерархические счётчики меток по домену и пути (с исключением
   проверочных фолдов), ресурсные и словарные признаки, бустинги
   LightGBM/XGBoost/CatBoost. Линия B — лексика URL, HTML-контент страницы
   (формы, редиректы, мимирия под бренд) и признаки «похожести на известные
   страницы» (top-K соседей по тексту, разметке, шаблонам путей, заголовкам).
2. **Сетевые признаки — ключевое дополнение.** Набор собран в 2024 году, а
   фишинговые домены живут недолго: их блокируют и удаляют. Поэтому текущее
   состояние домена (резолвится ли он, отвечает ли по HTTPS, что отдаёт по HTTP,
   возраст регистрации по RDAP, ASN и возраст выделения IP-префикса) сильно
   различает классы. Одиночные AUC: «домен не резолвится» 0.79, «нет TLS» 0.82,
   «HTTP-ошибка» 0.75, «возраст домена» 0.88.
3. **Мета-модель.** Градиентный бустинг (HistGradientBoosting: 200 итераций,
   4 листа, L2 = 1) на логитах всех базовых моделей обеих линий и 67 сетевых
   признаках. Мелкие деревья на 138 признаках устойчивее глубоких.

## Состав каталога artifacts/

* `artifacts/data/` — `train.csv` (64 000 URL с метками), `test.csv` (16 000 URL),
  `sample.csv` (формат ответа);
* `artifacts/base_preds/` — OOF-предсказания (для обучения мета-модели) и
  усреднённые по фолдам предсказания для теста: шесть моделей линии A
  и единый стекинг линии B;
* `artifacts/network/` — снимки сетевого состояния доменов (DNS, TLS, HTTP,
  RDAP, IP/ASN).

**Воспроизводимость.** Сетевое состояние меняется во времени, поэтому запросы
не повторяются: в репозитории лежат снимки, и результат воспроизводится
детерминированно из них.""")

code("""import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)

# корень проекта определяется независимо от того, откуда запущена тетрадка
ROOT = Path.cwd()
while not (ROOT / "artifacts").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
ART = ROOT / "artifacts"
print("данные и артефакты найдены:", ART.relative_to(ROOT))""")

md("""## 1. Данные

Обучающую выборку приводим к тому же виду, что использовался при обучении:
удаляем противоречивые дубликаты (один URL с разными метками) и повторы URL.""")

code("""train = pd.read_csv(ART / "data" / "train.csv")
test = pd.read_csv(ART / "data" / "test.csv")

grp = train.groupby("url")["result"].nunique()
train = train[~train["url"].isin(set(grp[grp > 1].index))]
train = train.drop_duplicates("url").reset_index(drop=True)
y = train["result"].values
n, nt = len(train), len(test)
print(f"обучающих URL: {n}, тестовых: {nt}, доля фишинга: {y.mean():.4f}")""")

md("""## 2. Сетевые признаки

Для каждого URL признаки соединяются по трём ключам: **хост** (DNS, TLS, ASN),
**полный URL** (HTTP-состояние) и **регистрируемый домен** (RDAP). Используются
те же функции разбора адреса, что и в основном пайплайне (`src/features.py`).""")

code("""import sys
from urllib.parse import urlsplit

sys.path.insert(0, str(ROOT / "src"))
from features import normalize_url, split_host


def host_of(url: str) -> str:
    h = urlsplit(normalize_url(url)).netloc.lower()
    if "@" in h:
        h = h.rsplit("@", 1)[1]
    return h.partition(":")[0].strip(".")


def registrable_domain(url: str):
    \"\"\"(регистрируемый домен, зона) — как в src/rdap_features.py\"\"\"
    _, sld, suf = split_host(host_of(url))
    if suf and sld:
        return f"{sld}.{suf}", suf.split(".")[-1]
    return host_of(url), ""


NET_FILES = ["dns_a.pkl", "dns_rcode.pkl", "dns_all.pkl", "tls_all.pkl",
             "http_all.pkl", "rdap_all.pkl", "ip_asn.pkl"]
NET_KEY = {"dns_a.pkl": "host", "dns_rcode.pkl": "host", "dns_all.pkl": "host",
           "tls_all.pkl": "host", "http_all.pkl": "url", "rdap_all.pkl": "domain",
           "ip_asn.pkl": "host"}


def network_matrix(urls):
    keys = {"host": urls.map(host_of).values,
            "url": urls.astype(str).values,
            "domain": np.array([registrable_domain(u)[0] for u in urls])}
    blocks = []
    for name in NET_FILES:
        path = ART / "network" / name
        if not path.exists():
            continue
        tbl = pd.read_pickle(path)
        cols = [c for c in tbl.columns
                if c != "dns_host_len" and tbl[c].dtype != object]
        blocks.append(np.nan_to_num(
            tbl.reindex(keys[NET_KEY[name]])[cols].astype(np.float32).values,
            nan=0.0))
    return np.column_stack(blocks)


NET = network_matrix(train["url"])
NET_te = network_matrix(test["url"])
print("сетевых признаков:", NET.shape[1])

dns = pd.read_pickle(ART / "network" / "dns_a.pkl")["dns_resolves"]
print("резолвится доменов: train %.3f, test %.3f" % (
    dns.reindex(train["url"].map(host_of)).fillna(0).mean(),
    dns.reindex(test["url"].map(host_of)).fillna(0).mean()))""")

md("""## 3. Предсказания базовых моделей

Для обучения мета-модели используются **честные OOF-предсказания**: каждая
строка предсказана моделью, которая её не видела. Для теста берутся
предсказания, усреднённые по фолдам.""")

code("""BASE = ["A_v2", "A_v3", "A_v4", "A_v8", "A_v10_lgb", "A_v10_xgb", "B_unified"]
eps = 1e-6


def logit(p):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


LO, LO_te = [], []
for name in BASE:
    d = np.load(ART / "base_preds" / f"{name}.npz")
    LO.append(logit(d["oof"]))
    LO_te.append(logit(d["test"]))
LO = np.column_stack(LO)
LO_te = np.column_stack(LO_te)
print(f"загружено базовых моделей: {len(BASE)}, строк OOF: {LO.shape[0]}")""")

md("""## 4. Мета-модель и кросс-валидация

Схема валидации та же, что при обучении базовых моделей:
`StratifiedKFold(5, shuffle=True, random_state=42)`.""")

code("""X = np.hstack([LO, NET])
X_te = np.hstack([LO_te, NET_te])

skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
oof_pred = np.zeros(n)
for fold, (itr, iva) in enumerate(skf.split(X, y), 1):
    sc = StandardScaler().fit(X[itr])
    model = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.10, max_leaf_nodes=4,
        l2_regularization=1.0, random_state=SEED)
    model.fit(sc.transform(X[itr]), y[itr])
    oof_pred[iva] = model.predict_proba(sc.transform(X[iva]))[:, 1]
    print(f"  фолд {fold}: F1 = "
          f"{f1_score(y[iva], (oof_pred[iva] >= 0.5).astype(int)):.5f}")

grid = np.quantile(oof_pred, np.linspace(0.005, 0.995, 800))
best_f1, best_thr = max((f1_score(y, (oof_pred >= t).astype(int)), t) for t in grid)
pred = (oof_pred >= best_thr).astype(int)
print(f"\\nOOF F1 = {best_f1:.5f}  (порог {best_thr:.4f})")
print(f"precision = {precision_score(y, pred):.5f}  "
      f"recall = {recall_score(y, pred):.5f}  "
      f"ROC-AUC = {roc_auc_score(y, oof_pred):.5f}")
print(f"ошибок: {(pred != y).sum()} из {n} "
      f"(FP {(pred[y == 0] == 1).sum()}, FN {(pred[y == 1] == 0).sum()})")""")

md("""## 5. Финальная модель и файл ответов

Мета-модель обучается на всей обучающей выборке; ответ формируется в формате
соревнования (`Id,Predicted`).""")

code("""sc_full = StandardScaler().fit(X)
final = HistGradientBoostingClassifier(
    max_iter=200, learning_rate=0.10, max_leaf_nodes=4,
    l2_regularization=1.0, random_state=SEED)
final.fit(sc_full.transform(X), y)
proba_te = final.predict_proba(sc_full.transform(X_te))[:, 1]
pred_te = (proba_te >= best_thr).astype(int)

submission = pd.DataFrame({"Id": np.arange(nt), "Predicted": pred_te})
out_path = ROOT / "submission_from_notebook.csv"
submission.to_csv(out_path, index=False)
print(submission.head().to_string(index=False))
print("доля предсказанного фишинга: %.4f (в train %.4f)" % (pred_te.mean(), y.mean()))

ref_path = ROOT / "submission_net_full.csv"
if ref_path.exists():
    ref = pd.read_csv(ref_path)
    diff = int((ref["Predicted"].values != pred_te).sum())
    print(f"\\nотличий от загруженного submission_net_full.csv: {diff} из {nt}")
    print("sha256 загруженного файла:",
          hashlib.sha256(ref_path.read_bytes()).hexdigest()[:16])""")

md("""## 6. Итог и ограничения

| Метрика | Значение |
|---|---|
| OOF F1 (63 900 объектов, 5 фолдов) | **0.99195** |
| Публичный F1 на Cyberhackai | **0.996062** |
| Итоговое место | **3-е** |

**Ограничение метода.** Сетевые признаки описывают состояние домена на момент
запроса, а не его содержимое. Внутри соревнования это корректно (обучающая и
тестовая выборки собраны в один период, публичная и приватная части — случайные
подмножества одного теста), и правила разрешают сторонние данные. Для потоковой
детекции «здесь и сейчас» признак не переносится: домен атаки там ещё жив.

**Про остаточные ошибки.** Они упираются в шум разметки: в обучающем наборе
встречаются `login.yahoo.com` с меткой «фишинг» и спам-домены в зонах
`.dance`/`.webcam` с меткой «безопасный». Публичный рейтинг считается по
3 200 адресам (стандартная ошибка ≈ 0.00125), поэтому решения в группе
0.9954–0.9961 статистически неразличимы.
""")

nb = new_notebook(cells=CELLS, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9"},
})
nbf.validate(nb)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as fh:
    nbf.write(nb, fh)
print(f"[ok] {OUT}: ячеек {len(CELLS)}")
