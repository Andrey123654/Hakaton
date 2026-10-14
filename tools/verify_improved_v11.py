from pathlib import Path
import ctypes,json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[1];ctypes.CDLL(str(ROOT/'.venv/lib/python3.9/site-packages/sklearn/.dylibs/libomp.dylib'),mode=ctypes.RTLD_GLOBAL)
import numpy as np,pandas as pd,joblib
from scipy.stats import rankdata
out=ROOT/'improved_v11';a=np.load(ROOT/'improved_v5/full_features.npz');saved=np.load(out/'full_predictions.npz')['test'];P=[]
for seed in [3042,3043,3044]:
 m=joblib.load(out/f'full_seed_{seed}.joblib');P.append(m.predict_proba(a['test'])[:,1])
P=np.column_stack(P);assert np.allclose(P,saved,rtol=0,atol=1e-12);nt=len(P);full=np.mean(np.column_stack([rankdata(P[:,j])/nt for j in range(3)]),axis=1);b=np.load(ROOT/'improved_v2/blend_predictions.npz')['test'];tree=np.load(ROOT/'improved_v5/full_predictions.npz')['test'];lgb=np.load(ROOT/'improved_v10/lightgbm_predictions.npz')['test'];fold=rankdata(lgb)/nt;v10=.8*(.9*b+.1*rankdata(tree)/nt)+.2*fold;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert sha(ROOT/'improved_v10/submission_v10.csv')=='ef5e3f07e04f25acea23c81971c7d55fae7098448d31a3c88f11a035a5c3233e';assert sha(ROOT/'improved_v5/submission_v5_refit.csv')=='110bec074aff3462323c6acaa8713bd8103818965b52a4c0a7730eade5f50c0a';sample=pd.read_csv(ROOT/'data/sample.csv');old=pd.read_csv(ROOT/'improved_v10/submission_v10.csv');report={'model_prediction_max_difference':float(np.max(abs(P-saved))),'prior_winners_unchanged':True,'files':[]}
for name,w in [('half',.1),('full',.2)]:
 path=out/f'submission_v11_{name}.csv';s=pd.read_csv(path);pred=(v10+w*(full-fold)>=.6250374960876368).astype(int);assert s.columns.tolist()==['Id','Predicted'];assert len(s)==16000;assert s.Id.equals(sample.Id);assert s.Predicted.isin([0,1]).all();assert np.array_equal(pred,s.Predicted);target=Path('/Users/andrey/Downloads')/f'submission_v11_{name}.csv';shutil.copy2(path,target);assert sha(target)==sha(path);report['files'].append({'file':str(target),'rows':len(s),'changes_vs_v10':int(np.sum(pred!=old.Predicted.to_numpy())),'sha256':sha(target)})
(out/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
(out/'README.md').write_text('''# V11: полное обучение LightGBM

Лучший подтверждённый публичный F1: V10 **0.995404**. Публичные результаты V11 неизвестны. Независимой локальной оценки F1 итогового полного обучения нет.

LightGBM с тем же рецептом V10 обучен на всех 63 900 очищенных обучающих строках для seeds 3042, 3043, 3044. Ранги ответов трёх моделей усредняются. Признаки полного обучения и теста взяты из improved_v5/full_features.npz: 288 столбцов, признаки обучения, зависящие от меток, сформированы по пяти разбиениям с исключением собственных проверочных меток; тестовые признаки используют всю обучающую выборку. Настройки и число итераций фиксированы до публичной проверки V11.

Общий вклад LightGBM (20%), остальные веса и порог V10 (0.6250374960876368) сохранены. Основной файл submission_v11_half.csv заменяет половину прежнего вклада LightGBM на полное обучение: 10% прежних моделей по разбиениям + 10% нового ансамбля. Альтернативный submission_v11_full.csv заменяет весь вклад: 20% нового ансамбля. Соответственно изменены 3 и 4 ответа относительно V10. Это кандидаты для турнирной проверки; прирост не заявляется.

Параметры: n_estimators=1200, learning_rate=0.035, num_leaves=31, min_child_samples=50, reg_lambda=8, reg_alpha=0.1, colsample_bytree=0.85, subsample=0.85, subsample_freq=1, n_jobs=6. Метки теста не используются.

Воспроизведение на этом Mac из корня проекта:

    DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.9/site-packages/sklearn/.dylibs" .venv/bin/python tools/improve_v11_refit.py
    DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.9/site-packages/sklearn/.dylibs" .venv/bin/python tools/verify_improved_v11.py

OpenMP берётся из уже установленного scikit-learn. Первый скрипт повторно использует сохранённые модели; для полного переобучения удалите только improved_v11/full_seed_3042.joblib, full_seed_3043.joblib и full_seed_3044.joblib. Нужны data CSV, improved_v5/full_features.npz и full_predictions.npz, improved_v2/blend_predictions.npz, improved_v10/lightgbm_predictions.npz. Построение исходных признаков сохранено в tools/improve_v1.py и refit_improved_v5.py; обучение LightGBM по разбиениям — improve_v10_boosting.py.

Сохранены модели, full_predictions.npz, report.json, changes_half.csv и changes_full.csv. Проверка независимо загружает модели, сравнивает вероятности и ответы, проверяет формат и порядок 16 000 строк, хеши и сохранность V5/V10. Копии проверенных CSV находятся в Downloads.

Окружение: Python 3.9, LightGBM 4.6.0, numpy 2.0.2, pandas 2.3.3, scipy 1.13.1, scikit-learn 1.6.1, joblib 1.5.3.
''',encoding='utf8')
