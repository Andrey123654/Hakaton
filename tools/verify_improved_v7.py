from pathlib import Path
import json,hashlib,shutil
import numpy as np,pandas as pd,joblib
from scipy.sparse import load_npz
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'improved_v7';m=joblib.load(out/'linear_full.joblib');p=m.decision_function(load_npz(out/'linear_test_features.npz'));saved=np.load(out/'linear_predictions.npz')['test'];assert np.allclose(p,saved,rtol=0,atol=1e-12);nt=len(p)
b=np.load(ROOT/'improved_v2/blend_predictions.npz')['test'];oldlinear=np.load(ROOT/'improved_v2/linear_predictions.npz')['test'];tree=np.load(ROOT/'improved_v5/full_predictions.npz')['test'];v5=.9*b+.1*rankdata(tree)/nt;score=v5+.045*(rankdata(p)-rankdata(oldlinear))/nt;pred=(score>=.6257334898278559).astype(int)
sample=pd.read_csv(ROOT/'data/sample.csv');s=pd.read_csv(out/'submission_v7.csv');assert s.columns.tolist()==['Id','Predicted'];assert len(s)==16000;assert s.Id.equals(sample.Id);assert s.Predicted.isin([0,1]).all();assert np.array_equal(pred,s.Predicted)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert sha(ROOT/'improved_v5/submission_v5_refit.csv')=='110bec074aff3462323c6acaa8713bd8103818965b52a4c0a7730eade5f50c0a';target=Path.home()/'Downloads'/'submission_v7.csv';shutil.copy2(out/'submission_v7.csv',target);assert sha(target)==sha(out/'submission_v7.csv');report={'model_prediction_max_difference':float(np.max(abs(p-saved))),'csv_prediction_parity':True,'rows':len(s),'sha256':sha(target),'best_v5_unchanged':True,'delivery':str(target)};(out/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
(out/'README.md').write_text('''# V7: дообучение текстовой модели

Лучший подтверждённый публичный F1: V5 **0.995076**. V6: 0.994420, ухудшение. Результат V7 неизвестен; независимый локальный F1 итогового дообучения отсутствует.

Сохраняются порог V5 и вклад его модели полного обучения. В текстовом компоненте, ранее проверенном в V2, половина вклада модели из пяти разбиений заменена моделью с теми же фиксированными параметрами, обученной на всех 63 900 очищенных примерах. Формула: score_V5 + 0.045 * (rank_full_SVM - rank_fold_SVM); ранги нормированы на размер теста, порог 0.6257334898278559.

LinearSVC C=1, max_iter=2500, tol=1e-4, seed=42. Используются сохранённые словари V2: символы и слова текста страницы, символы URL и заголовка. Словари не переобучаются; данные берутся из офлайн-снимков. Метки теста не использованы. Величина замены фиксирована до публичной оценки V7. Пять разбиений V2 проверяли рецепт модели, но не являются независимой оценкой итогового полного дообучения.

Проверка устойчивого порога на 60 bootstrap-выборках V2 не дала существенного выигрыша: F1 с crossfit-порогом 0.9901159753, с медианным bootstrap-порогом 0.9901163884. Поэтому порог не изменён. Эти значения относятся к V2 и не являются оценкой V7.

Воспроизведение из корня проекта:

    .venv/bin/python tools/improve_v7.py
    .venv/bin/python tools/verify_improved_v7.py

Необходимы data/train.csv, data/test.csv, data/sample.csv, cache/content_text_{train,test}.pkl, cache/content_title_{train,test}.pkl, improved_v2/linear_vectorizers.joblib, linear_predictions.npz, blend_predictions.npz и improved_v5/full_predictions.npz. Предыдущие этапы построения сохранены в tools/improve_v2_linear.py, blend_improved_v2.py и refit_improved_v5.py. Модель linear_full.joblib и её тестовые признаки сохранены. Скрипт проверки независимо загружает модель, проверяет совпадение ответов, формат, порядок 16 000 строк и сохранность лучшего V5. Изменения — changes.csv.

Окружение: Python 3.9; numpy 2.0.2; pandas 2.3.3; scipy 1.13.1; scikit-learn 1.6.1; catboost 1.2.10; joblib 1.5.3.
''',encoding='utf8')
