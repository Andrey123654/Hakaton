from pathlib import Path
import ctypes,json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[1];ctypes.CDLL(str(ROOT/'.venv/lib/python3.9/site-packages/sklearn/.dylibs/libomp.dylib'),mode=ctypes.RTLD_GLOBAL)
import numpy as np,pandas as pd,joblib
from scipy.stats import rankdata
out=ROOT/'improved_v10';r=json.loads((out/'report.json').read_text());verification={'models':{}};T=[]
for name in ['lightgbm','xgboost']:
 saved=np.load(out/f'{name}_predictions.npz');test=np.zeros(len(saved['test']));oof=np.zeros(len(saved['y']))
 for fold in range(5):
  a=np.load(ROOT/f'improved_v1/features_fold_{fold}.npz');m=joblib.load(out/f'{name}_fold_{fold}.joblib');test+=m.predict_proba(a['test'])[:,1]/5;oof[a['iva']]=m.predict_proba(a['valid'])[:,1]
 assert np.allclose(test,saved['test'],rtol=0,atol=1e-12);assert np.allclose(oof,saved['oof'],rtol=0,atol=1e-12);verification['models'][name]={'test_max_difference':float(np.max(abs(test-saved['test']))),'oof_max_difference':float(np.max(abs(oof-saved['oof'])))};T.append(rankdata(test)/len(test))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert sha(ROOT/'improved_v5/submission_v5_refit.csv')=='110bec074aff3462323c6acaa8713bd8103818965b52a4c0a7730eade5f50c0a';verification['V5_unchanged']=True
if r['selected']:
 b=np.load(ROOT/'improved_v2/blend_predictions.npz')['test'];tree=np.load(ROOT/'improved_v5/full_predictions.npz')['test'];v5=.9*b+.1*rankdata(tree)/len(b);w=r['selected']['weights'];pred=(w[0]*v5+w[1]*T[0]+w[2]*T[1]>=r['selected']['threshold']).astype(int);s=pd.read_csv(out/'submission_v10.csv');sample=pd.read_csv(ROOT/'data/sample.csv');assert s.columns.tolist()==['Id','Predicted'];assert len(s)==16000;assert s.Id.equals(sample.Id);assert s.Predicted.isin([0,1]).all();assert np.array_equal(pred,s.Predicted)
 if r['changes_vs_v5']:
  target=Path.home()/'Downloads'/'submission_v10.csv';shutil.copy2(out/'submission_v10.csv',target);assert sha(target)==sha(out/'submission_v10.csv');verification['submission']={'rows':len(s),'sha256':sha(target),'delivery':str(target),'changed_answers_vs_v5':r['changes_vs_v5']}
(out/'verification.json').write_text(json.dumps(verification,indent=2));print(json.dumps(verification,indent=2))
(out/'README.md').write_text('''# V10: независимые алгоритмы бустинга

Лучший подтверждённый публичный результат V5: **0.995076**. Публичный результат V10 неизвестен. Локальные оценки новой модели и смеси сохранены в report.json; они не являются оценкой публичного теста.

Используются 288 признаков из пяти исходных матриц V1. Каждый набор train/valid/test имеет признаки, построенные без меток своей внешней проверочной части; признаки обучения, зависящие от меток, дополнительно сформированы по внутренним разбиениям. Новые модели обучаются с фиксированным числом итераций, без подбора по внешним проверочным ответам.

LightGBM: 1200 деревьев, learning_rate=0.035, num_leaves=31, min_child_samples=50, reg_lambda=8, reg_alpha=0.1, colsample_bytree=0.85, subsample=0.85, subsample_freq=1. XGBoost: 900 деревьев, learning_rate=0.04, max_depth=5, min_child_weight=5, reg_lambda=10, reg_alpha=0.1, colsample_bytree=subsample=0.9, tree_method=hist. Потоки: 6; seeds соответственно 3042..3046 и 4042..4046.

Проверяются веса 0.05, 0.1, 0.2, 0.35 для каждого алгоритма отдельно и пополам. Подбор веса и порога производится на OOF; report.json дополнительно содержит оценку с выбором веса и порога без проверочных меток каждого разбиения. Ранее сохранённые upstream OOF переиспользуются, поэтому это не полностью вложенная оценка всего ансамбля. Итоговый тестовый файл сохраняет компонент полного обучения V5, для которого независимого OOF нет: итоговая локальная оценка V10 не заявляется.

OpenMP уже входит в установленный scikit-learn. Установка системных библиотек не требуется. Воспроизведение на этом Mac из корня проекта:

    DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.9/site-packages/sklearn/.dylibs" .venv/bin/python tools/improve_v10_boosting.py
    DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.9/site-packages/sklearn/.dylibs" .venv/bin/python tools/verify_improved_v10.py

Первый скрипт повторно использует сохранённые fold-модели, если они есть. Для полного переобучения удалите только improved_v10/lightgbm_fold_*.joblib и xgboost_fold_*.joblib. Нужны improved_v1/features_fold_0..4.npz, improved_v2/blend_predictions.npz, improved_v5/full_predictions.npz и data CSV. Построение исходных признаков сохранено в tools/improve_v1.py. Метки теста не используются.

Проверка независимо загружает все 10 моделей, сравнивает OOF и тестовые вероятности, пересчитывает итоговый CSV, проверяет заголовок, порядок 16 000 строк и сохранность V5. verification.json содержит хеши и результаты; changes.csv — изменённые строки. CSV копируется в Downloads только при наличии новых ответов и выбранной смеси.

Версии: Python 3.9, LightGBM 4.6.0, XGBoost 2.1.4, numpy 2.0.2, pandas 2.3.3, scipy 1.13.1, scikit-learn 1.6.1, joblib 1.5.3.
''',encoding='utf8')
