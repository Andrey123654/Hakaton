#!/usr/bin/env bash
# run_all.sh — полное воспроизведение итогового решения с нуля.
#
# Требуется: Python 3.9+, ~16 ГБ ОЗУ, ~10 ГБ свободного места на диске,
# исходные файлы организаторов в каталоге data/ и Архив 3 с контентом.
#
# Время выполнения на 12-ядерном CPU: ~5-6 часов (из них ~40 минут —
# извлечение признаков контента из файлов с HTML).
#
# Контрольные значения (для проверки воспроизведения):
#   поколение final : OOF F1 = 0.98606
#   поколение nn    : OOF F1 = 0.98656 (стекинг 0.98661)
#   поколение nn2   : OOF F1 = 0.98807 (стекинг 0.98807)
#   поколение nn3   : OOF F1 = 0.98853 (стекинг 0.98840)
#   единый стекинг  : OOF F1 = 0.98982 -> submission_unified.csv
#                     (публичный F1 = 0.993440; у submission_blend4 — 0.994416)
#
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
CONTENT_DIR="${CONTENT_DIR:-./Архив 3}"   # переопределяется переменной окружения
TRAIN_CONTENT="$CONTENT_DIR/train_with_content.csv"
TEST_CONTENT="$CONTENT_DIR/test_with_content.csv"

echo "=== 0. Окружение ==="
[ -x "$PY" ] || { python3 -m venv .venv && .venv/bin/pip install -r requirements.txt; }
$PY -c "import pandas, sklearn, catboost; print('зависимости в порядке')"
mkdir -p cache cache_k100 models reports

echo "=== 1. Признаки HTML-контента (78 признаков) ==="
for split in test train; do
  src_file="$TEST_CONTENT"; [ "$split" = "train" ] && src_file="$TRAIN_CONTENT"
  [ -f "cache/content_${split}.pkl" ] || \
    $PY src/content_features.py --csv "$src_file" --out "cache/content_${split}.pkl" --workers 8
done

echo "=== 2. Видимый текст и структурная подпись страниц ==="
[ -f cache/content_text_test.pkl ] || $PY src/content_text.py --csv "$TEST_CONTENT" \
    --out cache/content_text_test.pkl --workers 4
[ -f cache/content_text_train.pkl ] || $PY src/content_text.py --csv "$TRAIN_CONTENT" \
    --out cache/content_text_train.pkl --workers 4
[ -f cache/content_markup_test.pkl ] || $PY src/content_markup.py --csv "$TEST_CONTENT" \
    --out cache/content_markup_test.pkl --workers 5
[ -f cache/content_markup_train.pkl ] || $PY src/content_markup.py --csv "$TRAIN_CONTENT" \
    --out cache/content_markup_train.pkl --workers 5
[ -f cache/content_title_test.pkl ] || $PY src/content_title.py \
    --in cache/content_text_test.pkl --out cache/content_title_test.pkl
[ -f cache/content_title_train.pkl ] || $PY src/content_title.py \
    --in cache/content_text_train.pkl --out cache/content_title_train.pkl

echo "=== 3. Пространства поиска ближайших соседей ==="
# top-50 по словам, разметке и шаблонам URL-путей; top-100 по заголовкам страниц —
# именно так собраны поколения, давшие итоговое решение
[ -f cache/nn_word_train.npz ] || $PY src/nn_features.py --out-dir cache \
    --topk 50 --views word,markup,urlpath
[ -f cache_k100/nn_title_train.npz ] || $PY src/nn_features.py --out-dir cache_k100 \
    --topk 100 --views title

echo "=== 4. Поколение 1: URL + контент + текст (без соседей) ==="
$PY src/train.py --seed 42 --tag final --train data/train.csv --test data/test.csv --folds 5 \
    --content-train cache/content_train.pkl --content-test cache/content_test.pkl \
    --content-text-train cache/content_text_train.pkl --content-text-test cache/content_text_test.pkl \
    --nn-word-train "" --nn-markup-train "" --nn-urlpath-train "" --nn-title-train "" \
    --out-submission submission.csv --sample data/sample.csv

echo "=== 5. Поколения 2-4: соседские признаки (2, 3 и 4 представления) ==="
$PY src/train.py --seed 42 --tag nn --train data/train.csv --test data/test.csv --folds 5 \
    --content-train cache/content_train.pkl --content-test cache/content_test.pkl \
    --content-text-train cache/content_text_train.pkl --content-text-test cache/content_text_test.pkl \
    --nn-word-train cache/nn_word_train.npz --nn-word-test cache/nn_word_test.npz \
    --nn-markup-train cache/nn_markup_train.npz --nn-markup-test cache/nn_markup_test.npz \
    --nn-urlpath-train "" --nn-title-train "" --out-submission submission_nn.csv --sample data/sample.csv

$PY src/train.py --seed 42 --tag nn2 --train data/train.csv --test data/test.csv --folds 5 \
    --content-train cache/content_train.pkl --content-test cache/content_test.pkl \
    --content-text-train cache/content_text_train.pkl --content-text-test cache/content_text_test.pkl \
    --nn-word-train cache/nn_word_train.npz --nn-word-test cache/nn_word_test.npz \
    --nn-markup-train cache/nn_markup_train.npz --nn-markup-test cache/nn_markup_test.npz \
    --nn-urlpath-train cache/nn_urlpath_train.npz --nn-urlpath-test cache/nn_urlpath_test.npz \
    --nn-title-train "" --out-submission submission_nn2.csv --sample data/sample.csv

$PY src/train.py --seed 42 --tag nn3 --train data/train.csv --test data/test.csv --folds 5 \
    --content-train cache/content_train.pkl --content-test cache/content_test.pkl \
    --content-text-train cache/content_text_train.pkl --content-text-test cache/content_text_test.pkl \
    --nn-word-train cache/nn_word_train.npz --nn-word-test cache/nn_word_test.npz \
    --nn-markup-train cache/nn_markup_train.npz --nn-markup-test cache/nn_markup_test.npz \
    --nn-urlpath-train cache/nn_urlpath_train.npz --nn-urlpath-test cache/nn_urlpath_test.npz \
    --nn-title-train cache_k100/nn_title_train.npz --nn-title-test cache_k100/nn_title_test.npz \
    --out-submission submission_nn3.csv --sample data/sample.csv

echo "=== 6. Итоговое решение: единый стекинг по всем поколениям ==="
$PY tools/unified_stack.py --tags nn,nn2,nn3 --blend-tags final \
    --views "cache:word,cache:markup,cache:urlpath,cache_k100:title" \
    --out submission_unified.csv

echo "=== 7. Резервные решения и отчёты ==="
$PY tools/meta_submission.py --tag nn  --views "cache:word,cache:markup,cache:urlpath,cache_k100:title" \
    --out submission_nn_meta.csv
$PY tools/meta_submission.py --tag nn2 --views "cache:word,cache:markup,cache:urlpath,cache_k100:title" \
    --out submission_nn2_meta.csv
$PY tools/blend_generations.py --weights "meta_nn=0.3,meta_nn2=0.3,nn2=0.2,final=0.2" --out submission_blend4.csv
$PY src/explain.py --train data/train.csv --tag final
$PY tools/build_report.py

echo
echo "Готово. Файл для загрузки: submission_unified.csv"
echo "Резервные варианты: submission_blend4.csv (лучшая публичная оценка), submission_nn3.csv"
