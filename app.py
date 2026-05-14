"""
=============================================================================
ВЕБ-ПРИЛОЖЕНИЕ КРЕДИТНОГО СКОРИНГА (Flask)

Запуск:
    python app.py

Откройте в браузере: http://localhost:5000
=============================================================================
"""

import os
import json
import logging
import numpy as np
import joblib
from datetime import datetime

from flask import Flask, render_template, request, jsonify
from tensorflow import keras

# ─── Логирование ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s — %(message)s',
)
log = logging.getLogger(__name__)

# ─── Пути к артефактам ─────────────────────────────────────────────────────
MODEL_DIR        = '/home/tatyana/Documents/Baumana/ВКР_МГТУ/files/credit_scoring_app/models'
MODEL_PATH       = os.path.join(MODEL_DIR, 'credit_scoring_model.keras')
SCALER_PATH      = os.path.join(MODEL_DIR, 'scaler.joblib')
FEATURES_PATH    = os.path.join(MODEL_DIR, 'features.json')
METADATA_PATH    = os.path.join(MODEL_DIR, 'model_metadata.json')
COMPARISON_PATH  = os.path.join(MODEL_DIR, 'comparison_results.json')
FEATURE_STATS_PATH = os.path.join(MODEL_DIR, 'feature_stats.json')

# ─── Инициализация Flask ───────────────────────────────────────────────────
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# ─── Загрузка модели и артефактов (однократно при старте) ──────────────────
log.info("Загрузка модели и артефактов...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Модель не найдена: {MODEL_PATH}\n"
        f"Сначала обучите модель: python train_model.py"
    )

model     = keras.models.load_model(MODEL_PATH)
scaler    = joblib.load(SCALER_PATH)
with open(FEATURES_PATH, 'r', encoding='utf-8') as f:
    FEATURES = json.load(f)
with open(METADATA_PATH, 'r', encoding='utf-8') as f:
    METADATA = json.load(f)

# Сравнительный анализ (опциональный)
COMPARISON = None
if os.path.exists(COMPARISON_PATH):
    with open(COMPARISON_PATH, 'r', encoding='utf-8') as f:
        COMPARISON = json.load(f)
    log.info(f"  Сравнительный анализ загружен: {len(COMPARISON['models'])} моделей")

# Статистика признаков (опциональная)
FEATURE_STATS = None
if os.path.exists(FEATURE_STATS_PATH):
    with open(FEATURE_STATS_PATH, 'r', encoding='utf-8') as f:
        FEATURE_STATS = json.load(f)
    log.info(f"  Статистика признаков загружена: {FEATURE_STATS['n_features']} признаков")

THRESHOLD = METADATA['threshold']

log.info(f"  Модель загружена: {len(FEATURES)} признаков")
log.info(f"  Порог классификации: {THRESHOLD:.3f}")
log.info(f"  ROC-AUC модели: {METADATA['metrics']['roc_auc']:.4f}")

SCORING_HISTORY = []


# ═══════════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════════════════
def build_features_from_form(form_data: dict) -> dict:
    """Преобразует данные веб-формы в полный набор инженерных признаков."""
    f = {x: float(form_data.get(x, 0) or 0) for x in [
        'ежемесячный_платеж', 'пенсия', 'дополнительный_доход',
        'директорские_выплаты', 'региональные_выплаты',
        'перепродажа_товаров', 'неофициальный_доход',
        'оценка_автомобиля', 'лимит_карты', 'свободный_лимит',
        'платёжеспособность',
    ]}
    code_blacklist  = int(form_data.get('blacklist',    900))
    code_age        = int(form_data.get('age_check',    900))
    code_criminal   = int(form_data.get('criminal',     900))
    code_clientsc   = int(form_data.get('client_score', 900))
    code_card       = int(form_data.get('card',         704))
    code_pension    = int(form_data.get('pensioner',    802))
    has_contract    = int(form_data.get('has_contract', 0))

    total_income = (f['ежемесячный_платеж'] + f['пенсия'] +
                    f['дополнительный_доход'] + f['директорские_выплаты'] +
                    f['региональные_выплаты'] + f['перепродажа_товаров'] +
                    f['неофициальный_доход'])

    features = {
        'log_суммарный_доход':  np.log1p(max(0, total_income)),
        'log_ежемесячный':      np.log1p(max(0, f['ежемесячный_платеж'])),
        'log_пенсия':           np.log1p(max(0, f['пенсия'])),
        'log_неофиц_доход':     np.log1p(max(0, f['неофициальный_доход'])),
        'log_директорские':     np.log1p(max(0, f['директорские_выплаты'])),
        'log_оценка_авто':      np.log1p(max(0, f['оценка_автомобиля'])),
        'имеет_пенсию':         int(f['пенсия'] > 0),
        'имеет_неоф_доход':     int(f['неофициальный_доход'] > 0),
        'имеет_доп_доход':      int(f['дополнительный_доход'] > 0),
        'имеет_директорские':   int(f['директорские_выплаты'] > 0),
        'имеет_авто':           int(f['оценка_автомобиля'] > 0),
        'есть_лимит':           int(f['лимит_карты'] > 0),
        'log_лимит':            np.log1p(max(0, f['лимит_карты'])),
        'использование_лимита': 0.0,
        'платёжеспособность':   f['платёжеспособность'] / 100.0,
        'blacklist_пройден':    int(code_blacklist == 900),
        'blacklist_отказ':      int(code_blacklist == 901),
        'blacklist_нет_данных': int(code_blacklist == -1),
        'age_check_пройден':    int(code_age == 900),
        'age_check_отказ':      int(code_age == 901),
        'criminal_пройден':     int(code_criminal == 900),
        'criminal_отказ':       int(code_criminal == 901),
        'client_score_пройден': int(code_clientsc == 900),
        'client_score_отказ':   int(code_clientsc == 901),
        'карта_хорошо':         int(code_card == 702),
        'карта_плохо':          int(code_card == 701),
        'карта_нет':            int(code_card == 700),
        'карта_не_провер':      int(code_card == 704),
        'является_пенсионером':   int(code_pension == 801),
        'пенсионер_по_инвал':     int(code_pension == 803),
        'пенсионер_не_проверено': int(code_pension == 804),
        'есть_договор':           has_contract,
    }
    if f['лимит_карты'] > 0:
        used = (f['лимит_карты'] - f['свободный_лимит']) / max(1, f['лимит_карты'])
        features['использование_лимита'] = float(np.clip(used, 0, 2))
    return features


def predict_application(features_dict: dict) -> dict:
    """Применяет модель к словарю инженерных признаков."""
    x = np.array([[features_dict.get(name, 0) for name in FEATURES]],
                 dtype=np.float32)
    x_scaled = scaler.transform(x)
    prob = float(model.predict(x_scaled, verbose=0).ravel()[0])

    if prob >= 0.75:
        decision_label = 'Автоматическое одобрение'
        decision_class = 'success'
    elif prob >= THRESHOLD:
        decision_label = 'Одобрено'
        decision_class = 'success'
    elif prob >= 0.20:
        decision_label = 'На рассмотрение эксперту'
        decision_class = 'warning'
    else:
        decision_label = 'Автоматический отказ'
        decision_class = 'danger'

    return {
        'probability':    round(prob, 4),
        'probability_pct': round(prob * 100, 2),
        'threshold':      THRESHOLD,
        'decision_label': decision_label,
        'decision_class': decision_class,
        'timestamp':      datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


# ═══════════════════════════════════════════════════════════════════════════
# МАРШРУТЫ
# ═══════════════════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template('index.html', metadata=METADATA, threshold=THRESHOLD)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form.to_dict()
        features  = build_features_from_form(form_data)
        result    = predict_application(features)

        SCORING_HISTORY.append({**result, 'features_raw': form_data})
        if len(SCORING_HISTORY) > 100:
            SCORING_HISTORY.pop(0)

        log.info(f"Скоринг: prob={result['probability']:.4f}  "
                 f"→ {result['decision_label']}")
        return jsonify({'success': True, 'result': result})

    except Exception as exc:
        log.exception("Ошибка при скоринге")
        return jsonify({'success': False, 'error': str(exc)}), 400


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API эндпоинт для интеграции."""
    try:
        data = request.get_json(force=True)
        if 'features' in data:
            features = data['features']
        else:
            features = build_features_from_form(data)
        result = predict_application(features)
        return jsonify({'success': True, 'result': result})
    except Exception as exc:
        log.exception("API ошибка")
        return jsonify({'success': False, 'error': str(exc)}), 400


@app.route('/dashboard')
def dashboard():
    """Дашборд с метриками и историей."""
    stats = {'total': len(SCORING_HISTORY)}
    if SCORING_HISTORY:
        decisions = [h['decision_label'] for h in SCORING_HISTORY]
        stats['approve']     = sum(1 for d in decisions
                                   if d in ['Автоматическое одобрение', 'Одобрено'])
        stats['review']      = decisions.count('На рассмотрение эксперту')
        stats['auto_reject'] = decisions.count('Автоматический отказ')
        probs = [h['probability'] for h in SCORING_HISTORY]
        stats['avg_probability'] = round(sum(probs) / len(probs) * 100, 2)
    else:
        stats.update(approve=0, review=0, auto_reject=0, avg_probability=0)

    return render_template(
        'dashboard.html',
        metadata=METADATA,
        comparison=COMPARISON,
        stats=stats,
        history=list(reversed(SCORING_HISTORY[-20:])),
    )


@app.route('/comparison')
def comparison_page():
    """Отдельная страница сравнительного анализа."""
    if not COMPARISON:
        return render_template('no_comparison.html', metadata=METADATA)
    return render_template(
        'comparison.html',
        metadata=METADATA,
        comparison=COMPARISON,
    )


@app.route('/features')
def features_page():
    """Страница анализа признаков: распределения, статистика."""
    if not FEATURE_STATS:
        return render_template('no_features.html', metadata=METADATA)
    # Параметр ?detail=N показывает детальный график конкретного признака
    selected = request.args.get('detail', type=int)
    return render_template(
        'features.html',
        metadata=METADATA,
        stats=FEATURE_STATS,
        selected=selected,
    )


@app.route('/api/features')
def api_features():
    """REST API: статистика всех признаков."""
    if not FEATURE_STATS:
        return jsonify({'error': 'feature stats not available'}), 404
    return jsonify(FEATURE_STATS)


@app.route('/api/health')
def health():
    return jsonify({
        'status':   'healthy',
        'model':    'loaded',
        'version':  METADATA.get('version', '1.0.0'),
        'features': len(FEATURES),
        'comparison_available': COMPARISON is not None,
        'feature_stats_available': FEATURE_STATS is not None,
    })


@app.route('/api/metadata')
def api_metadata():
    return jsonify(METADATA)


@app.route('/api/comparison')
def api_comparison():
    """REST API: данные сравнительного анализа."""
    if not COMPARISON:
        return jsonify({'error': 'comparison data not available'}), 404
    return jsonify(COMPARISON)


# ═══════════════════════════════════════════════════════════════════════════
# ЗАПУСК
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print()
    print("=" * 65)
    print("ВЕБ-ПРИЛОЖЕНИЕ КРЕДИТНОГО СКОРИНГА")
    print("=" * 65)
    print(f"  Модель:           {MODEL_PATH}")
    print(f"  ROC-AUC:          {METADATA['metrics']['roc_auc']:.4f}")
    print(f"  Порог:            {THRESHOLD:.3f}")
    print(f"  Признаков:        {len(FEATURES)}")
    print(f"  Сравнит. анализ:  "
          f"{'доступен' if COMPARISON else 'не загружен'}")
    print(f"  Анализ признаков: "
          f"{'доступен' if FEATURE_STATS else 'не загружен'}")
    print("─" * 65)
    print("  Веб-интерфейс:    http://localhost:5000")
    print("  Дашборд:          http://localhost:5000/dashboard")
    print("  Сравнение:        http://localhost:5000/comparison")
    print("  Признаки:         http://localhost:5000/features")
    print("  API:              POST /api/predict (JSON)")
    print("=" * 65)
    print()
    app.run(host='0.0.0.0', port=5000, debug=False)
