from pathlib import Path
import json,hashlib,shutil
import numpy as np,pandas as pd
from scipy.stats import rankdata
from catboost import CatBoostClassifier
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'improved_v5'
a=np.load(out/'full_features.npz');m=CatBoostClassifier();m.load_model(str(out/'full_model.cbm'));p=m.predict_proba(a['test'])[:,1];saved=np.load(out/'full_predictions.npz')['test'];assert np.allclose(p,saved,rtol=0,atol=1e-12)
b=np.load(ROOT/'improved_v2/blend_predictions.npz');pred=(.9*b['test']+.1*rankdata(p)/len(p)>=.6257334898278559).astype(int)
sample=pd.read_csv(ROOT/'data/sample.csv');s=pd.read_csv(out/'submission_v5_refit.csv');assert s.columns.tolist()==['Id','Predicted'];assert len(s)==16000;assert s.Id.equals(sample.Id);assert s.Predicted.isin([0,1]).all();assert np.array_equal(pred,s.Predicted)
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
assert sha(ROOT/'improved_v2/submission_v2.csv')=='f28adee1f5dd1ba5982726f1d36e3dd9558df4476cd130ea927455ae2aedad94'
old=pd.read_csv(ROOT/'improved_v2/submission_v2.csv');test=pd.read_csv(ROOT/'data/test.csv');ix=np.flatnonzero(s.Predicted.to_numpy()!=old.Predicted.to_numpy());changes=pd.DataFrame({'Id':ix,'url':test.url.iloc[ix].to_numpy(),'V2':old.Predicted.iloc[ix].to_numpy(),'V5':s.Predicted.iloc[ix].to_numpy()});changes.to_csv(out/'changes_refit.csv',index=False)
target=Path.home()/'Downloads'/'submission_v5_refit.csv';shutil.copy2(out/'submission_v5_refit.csv',target);assert sha(target)==sha(out/'submission_v5_refit.csv')
r={'rows':len(s),'changed_answers':len(ix),'model_max_score_difference':float(np.max(abs(p-saved))),'submission_prediction_parity':True,'baseline_unchanged':True,'sha256':sha(target),'delivery':str(target),'public_score':None};(out/'verification_refit.json').write_text(json.dumps(r,indent=2))
(out/'README.md').write_text('''# V5: итоговый эксперимент

Лучший подтверждённый публичный результат: V2, **0.994747**. Публичный результат V5 ещё неизвестен; улучшение и достижение 1.0 не заявляются.

Независимая NB-SVM и новые признаки сочетаний домена и пути проверены на пяти отложенных частях. Их смеси не превзошли V2; они исключены из итогового файла. Результаты сохранены в blend_report.json.

Файл submission_v5_refit.csv: 90% ответов V2 и 10% рангов модели V1, заново обученной на всех 63 900 очищенных обучающих примерах. Вес 0.1 фиксирован без оценки на публичном тесте; порог взят из V2. Это эксперимент с итоговым обучением, независимая локальная оценка его F1 отсутствует. Изменены 3 ответа относительно V2.

Признаки с использованием меток обучающих URL для обучения сформированы по пяти разбиениям с исключением проверочных меток; признаки теста используют всю обучающую выборку. Метки теста не использованы.

Воспроизведение из корня проекта:

    .venv/bin/python tools/refit_improved_v5.py
    .venv/bin/python tools/deliver_improved_v5.py

Нужны исходные data CSV, cache, cache_k100 и подтверждённые вероятности improved_v2/blend_predictions.npz. Полное исходное построение признаков описано в tools/improve_v1.py, обучение V2 — в соответствующих скриптах tools. Сохранены full_features.npz, full_model.cbm, full_predictions.npz. Для воспроизведения на сохранённых признаках достаточно модели, признаков и вероятностей V2; второй скрипт проверяет точное совпадение результатов.

Окружение: Python 3.9, numpy 2.0.2, pandas 2.3.3, scipy 1.13.1, scikit-learn 1.6.1, catboost 1.2.10, joblib 1.5.3. Исходные и прежние ответы сохранены. Формат, порядок всех 16 000 строк и предсказания проверены.
''',encoding='utf8');print(json.dumps(r,indent=2));print(changes.to_string(index=False))
