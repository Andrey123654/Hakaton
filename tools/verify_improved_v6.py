from pathlib import Path
import json,hashlib,shutil
import numpy as np,pandas as pd
from scipy.stats import rankdata
from catboost import CatBoostClassifier
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'improved_v6';a=np.load(ROOT/'improved_v5/full_features.npz');saved=np.load(out/'full_predictions.npz')['test'];P=[]
for seed in [42,43,44]:
 m=CatBoostClassifier();m.load_model(str(ROOT/'improved_v5/full_model.cbm') if seed==42 else str(out/f'full_seed_{seed}.cbm'));P.append(m.predict_proba(a['test'])[:,1])
P=np.column_stack(P);assert np.allclose(P,saved,rtol=0,atol=1e-12);q=np.mean(np.column_stack([rankdata(P[:,j])/len(P) for j in range(3)]),axis=1);b=np.load(ROOT/'improved_v2/blend_predictions.npz')['test'];sample=pd.read_csv(ROOT/'data/sample.csv');old=pd.read_csv(ROOT/'improved_v5/submission_v5_refit.csv');report={'max_score_difference':float(np.max(abs(P-saved))),'files':[]};sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,w in [('main',.25),('alternative',.1)]:
 path=out/f'submission_v6_{name}.csv';s=pd.read_csv(path);pred=((1-w)*b+w*q>=.6257334898278559).astype(int);assert s.columns.tolist()==['Id','Predicted'];assert len(s)==16000;assert s.Id.equals(sample.Id);assert s.Predicted.isin([0,1]).all();assert np.array_equal(pred,s.Predicted)
 target=Path('/Users/andrey/Downloads')/f'submission_v6_{name}.csv';shutil.copy2(path,target);assert sha(path)==sha(target);report['files'].append({'path':str(target),'rows':len(s),'changes_vs_v5':int(np.sum(pred!=old.Predicted.to_numpy())),'sha256':sha(target)})
assert sha(ROOT/'improved_v5/submission_v5_refit.csv')=='110bec074aff3462323c6acaa8713bd8103818965b52a4c0a7730eade5f50c0a';assert sha(ROOT/'improved_v2/submission_v2.csv')=='f28adee1f5dd1ba5982726f1d36e3dd9558df4476cd130ea927455ae2aedad94';report['prior_winners_unchanged']=True;(out/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
(out/'README.md').write_text('''# V6: ансамбль итогового обучения

Лучший подтверждённый публичный результат V5: **0.995076**. Результат V6 неизвестен. Для V6 не заявляется независимая локальная оценка F1.

Вместо одной модели полного обучения используются три с фиксированными seeds 42, 43, 44. Рецепт и признаки V5 сохранены: 63 900 очищенных обучающих строк, 288 признаков, CatBoost depth=7, iterations=1400, learning_rate=0.045, l2_leaf_reg=5. Признаки обучения, зависящие от меток, сформированы с исключением меток своего проверочного разбиения. Тестовые признаки используют всю обучающую выборку.

Ранги вероятностей каждой модели усредняются. Основной файл submission_v6_main.csv использует 25% этого ансамбля и 75% V2; альтернативный — 10% и 90%. Порог сохранён из V2: 0.6257334898278559. Изменение весов мотивировано сообщённым публичным приростом V5; прирост на приватной части неизвестен. Метки теста не использованы.

Воспроизведение из корня проекта:

    .venv/bin/python tools/refit_improved_v5.py
    .venv/bin/python tools/improve_v6.py
    .venv/bin/python tools/verify_improved_v6.py

Первый шаг нужен, если отсутствуют full_features.npz и full_model.cbm в improved_v5. Для него нужны исходные data CSV, cache, cache_k100. Для V6 также нужны вероятности improved_v2/blend_predictions.npz. Скрипт обучения повторно использует существующие сохранённые модели; для полного переобучения удалите только improved_v6/full_seed_43.cbm и full_seed_44.cbm. Второй скрипт независимо загружает все модели, проверяет совпадение вероятностей и CSV, порядок 16 000 строк, сохранность V2 и V5 и создаёт копии в Downloads. Изменённые строки сохранены в changes_main.csv и changes_alternative.csv.

Окружение: Python 3.9, numpy 2.0.2, pandas 2.3.3, scipy 1.13.1, scikit-learn 1.6.1, catboost 1.2.10, joblib 1.5.3. Версии текущего окружения также записаны в environment.txt.
''',encoding='utf8')
