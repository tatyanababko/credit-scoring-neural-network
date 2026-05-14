"""
=============================================================================
ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ ДЛЯ КРЕДИТНОГО СКОРИНГА
с сохранением артефактов и СРАВНИТЕЛЬНЫМ АНАЛИЗОМ альтернативных моделей

Запуск:
    python train_model.py

После запуска в папке models/ появятся:
    credit_scoring_model.keras    — обученная нейросеть
    scaler.joblib                 — нормализатор признаков
    features.json                 — список признаков
    model_metadata.json           — метаданные (порог, метрики, гиперпараметры)
    training_history.json         — история обучения
    comparison_results.json       — результаты сравнительного анализа
=============================================================================
"""

import os
import json
import time
import warnings
import joblib

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks, optimizers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, roc_curve
)

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
np.random.seed(42)
tf.random.set_seed(42)

# ─── Пути ───────────────────────────────────────────────────────────────────
DATA_PATH    = '/home/tatyana/Documents/Baumana/ВКР_МГТУ/files/Scoring.csv'    # путь к датасету
MODEL_DIR    = '/home/tatyana/Documents/Baumana/ВКР_МГТУ/files/credit_scoring_app/models'
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH    = os.path.join(MODEL_DIR, 'credit_scoring_model.keras')
SCALER_PATH   = os.path.join(MODEL_DIR, 'scaler.joblib')
FEATURES_PATH = os.path.join(MODEL_DIR, 'features.json')
METADATA_PATH = os.path.join(MODEL_DIR, 'model_metadata.json')
HISTORY_PATH  = os.path.join(MODEL_DIR, 'training_history.json')
COMPARISON_PATH  = os.path.join(MODEL_DIR, 'comparison_results.json')

print("=" * 65)
print("ОБУЧЕНИЕ МОДЕЛИ КРЕДИТНОГО СКОРИНГА И СРАВНИТЕЛЬНЫЙ АНАЛИЗ")
print("=" * 65)
print(f"TensorFlow: {tf.__version__}")
print(f"GPU доступен: {bool(tf.config.list_physical_devices('GPU'))}")


# ═══════════════════════════════════════════════════════════════════════════
# 1. ЗАГРУЗКА И ПРЕДОБРАБОТКА
# ═══════════════════════════════════════════════════════════════════════════
print("\n[1/7] Загрузка и предобработка данных...")

df = pd.read_csv(
    DATA_PATH,
    sep=';', encoding='utf-8-sig',
    on_bad_lines='skip', decimal=',', low_memory=False
)
df = df[df['Целевое значение'].isin(['confirmed', 'reject'])].copy()
df['target'] = (df['Целевое значение'] == 'confirmed').astype(int)

n_total = len(df)
n_pos   = int(df['target'].sum())
n_neg   = n_total - n_pos
print(f"  Записей: {n_total:,}  (confirmed: {n_pos:,}, reject: {n_neg:,})")

# В данных типа object имеются строки
obj_cols = df.select_dtypes(include='object').columns
for col in obj_cols:
    df[col] = (df[col].astype(str).str.replace(',','.',regex=False).str.strip())
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ── Инженерия признаков ───────────────────────────────────────────────────
income_parts = ['Ежемесячный платеж', 'Пенсия', 'Дополнительный доход',
                'Директорские выплаты', 'Региональные выплаты',
                'Перепродажа товаров', 'Неофициальный доход']
df['суммарный_доход']     = df[income_parts].clip(lower=0).sum(axis=1)
df['log_суммарный_доход'] = np.log1p(df['суммарный_доход'])
df['log_ежемесячный']     = np.log1p(df['Ежемесячный платеж'].clip(lower=0))
df['log_пенсия']          = np.log1p(df['Пенсия'].clip(lower=0))
df['log_неофиц_доход']    = np.log1p(df['Неофициальный доход'].clip(lower=0))
df['log_директорские']    = np.log1p(df['Директорские выплаты'].clip(lower=0))
df['log_оценка_авто']     = np.log1p(df['Оценка автомобиля'].clip(lower=0))

df['имеет_пенсию']        = (df['Пенсия'] > 0).astype(int)
df['имеет_неоф_доход']    = (df['Неофициальный доход'] > 0).astype(int)
df['имеет_доп_доход']     = (df['Дополнительный доход'] > 0).astype(int)
df['имеет_директорские']  = (df['Директорские выплаты'] > 0).astype(int)
df['имеет_авто']          = (df['Оценка автомобиля'] > 0).astype(int)

lim  = df['Лимит по карте'].fillna(0).clip(lower=0)
free = df['Свободный лимит'].fillna(0)
df['есть_лимит'] = df['Лимит по карте'].notna().astype(int)
df['log_лимит']  = np.log1p(lim)
df['использование_лимита'] = np.where(
    lim > 0, ((lim - free) / lim.clip(lower=1)).clip(0, 2), 0.0)
df['платёжеспособность'] = df['Платежеспособность'].fillna(0) / 100.0

for col, alias in [
    ('Клиент в черном списке', 'blacklist'),
    ('Возраст',                'age_check'),
    ('Криминальное прошлое',   'criminal'),
    ('Оценка клиента',         'client_score'),
]:
    df[f'{alias}_пройден']   = (df[col] == 900).astype(int)
    df[f'{alias}_отказ']     = (df[col] == 901).astype(int)
    df[f'{alias}_нет_данных'] = (df[col] == -1).astype(int)

card = df['Код погашения кредита по карте']
df['карта_хорошо']    = (card == 702).astype(int)
df['карта_плохо']     = (card == 701).astype(int)
df['карта_нет']       = (card == 700).astype(int)
df['карта_не_провер'] = (card == 704).astype(int)
df['является_пенсионером']   = (df['Код пенсионера'] == 801).astype(int)
df['пенсионер_по_инвал']     = (df['Код пенсионера'] == 803).astype(int)
df['пенсионер_не_проверено'] = (df['Код пенсионера'] == 804).astype(int)
df['есть_договор'] = df['Код договора'].notna().astype(int)

FEATURES = [
    'log_суммарный_доход', 'log_ежемесячный', 'log_пенсия',
    'log_неофиц_доход', 'log_директорские', 'log_оценка_авто',
    'имеет_пенсию', 'имеет_неоф_доход', 'имеет_доп_доход',
    'имеет_директорские', 'имеет_авто',
    'есть_лимит', 'log_лимит', 'использование_лимита',
    'платёжеспособность',
    'blacklist_пройден',    'blacklist_отказ',    'blacklist_нет_данных',
    'age_check_пройден',    'age_check_отказ',
    'criminal_пройден',     'criminal_отказ',
    'client_score_пройден', 'client_score_отказ',
    'карта_хорошо', 'карта_плохо', 'карта_нет', 'карта_не_провер',
    'является_пенсионером', 'пенсионер_по_инвал', 'пенсионер_не_проверено',
    'есть_договор',
]
X = df[FEATURES].fillna(0).values.astype(np.float32)
y = df['target'].values.astype(np.float32)

# ── Train/Val/Test ────────────────────────────────────────────────────────
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=0.125, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.125 / 0.875, stratify=y_tmp, random_state=42)

# Все признаки имеют большую разбежность. После StandardScaler [-3;3] одинаково 
# эффективно используем их
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

w_pos    = n_neg / n_pos
sw_train = np.where(y_train == 1, w_pos, 1.0)

print(f"  Train: {len(y_train):,}  Val: {len(y_val):,}  Test: {len(y_test):,}")
print(f"  Признаков: {len(FEATURES)}  Вес+: {w_pos:.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. АРХИТЕКТУРА МОДЕЛИ
# ═══════════════════════════════════════════════════════════════════════════
print("\n[2/7] Сборка нейронной сети...")

HYPERPARAMS = {
    'units_1':   256, 'units_2':   128, 'units_3':   64,
    'dropout_1': 0.30,'dropout_2': 0.30,'dropout_3': 0.20,
    'l2_reg':    4e-4,'lr':        1e-3,
}

def build_model(n_in, hp):
    reg = regularizers.l2(hp['l2_reg'])
    inp = keras.Input(shape=(n_in,), name='input')
    x = layers.Dense(hp['units_1'], kernel_regularizer=reg, name='dense_1')(inp)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Activation('relu', name='relu_1')(x)
    x = layers.Dropout(hp['dropout_1'], name='dropout_1')(x)
    x = layers.Dense(hp['units_2'], kernel_regularizer=reg, name='dense_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Activation('relu', name='relu_2')(x)
    x = layers.Dropout(hp['dropout_2'], name='dropout_2')(x)
    x = layers.Dense(hp['units_3'], kernel_regularizer=reg, name='dense_3')(x)
    x = layers.Activation('relu', name='relu_3')(x)
    x = layers.Dropout(hp['dropout_3'], name='dropout_3')(x)
    out = layers.Dense(1, activation='sigmoid', name='output')(x)
    m = keras.Model(inp, out)
    m.compile(
        optimizer=optimizers.Adam(learning_rate=hp['lr']),
        loss='binary_crossentropy',
        metrics=[keras.metrics.AUC(name='roc_auc'),
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    return m

model = build_model(len(FEATURES), HYPERPARAMS)
print(f"  Параметров: {model.count_params():,}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ
# ═══════════════════════════════════════════════════════════════════════════
print("\n[3/7] Обучение нейронной сети...")

cb_list = [
    callbacks.EarlyStopping(monitor='val_roc_auc', patience=15, mode='max',
                            restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_roc_auc', factor=0.5, patience=7,
                                mode='max', min_lr=1e-6, verbose=1),
    callbacks.ModelCheckpoint(filepath=MODEL_PATH, monitor='val_roc_auc',
                              save_best_only=True, mode='max'),
]
t0 = time.time()
history = model.fit(
    X_train_s, y_train,
    validation_data=(X_val_s, y_val),
    sample_weight=sw_train,
    epochs=200, batch_size=1024,
    callbacks=cb_list, verbose=2,
)
train_time = time.time() - t0
n_epochs   = len(history.history['loss'])
print(f"\n  Эпох: {n_epochs}   Время: {train_time:.0f} сек")


# ═══════════════════════════════════════════════════════════════════════════
# 4. ПОДБОР ПОРОГА И ОЦЕНКА НЕЙРОННОЙ СЕТИ
# ═══════════════════════════════════════════════════════════════════════════
print("\n[4/7] Подбор порога классификации и оценка нейронной сети...")

def find_best_threshold(y_true, y_prob):
    """Поиск оптимального порога по F1-score."""
    thrs = np.linspace(0.05, 0.95, 181)
    best_t, best_f1 = 0.5, 0
    for t in thrs:
        f1_t = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1_t > best_f1:
            best_f1, best_t = f1_t, t
    return float(best_t), float(best_f1)


def compute_all_metrics(y_true, y_prob, threshold):
    """Полный расчёт всех метрик для бинарной классификации."""
    y_pred = (y_prob >= threshold).astype(int)
    auc     = roc_auc_score(y_true, y_prob)
    gini    = 2 * auc - 1
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks      = (tpr - fpr).max()
    acc     = accuracy_score(y_true, y_pred)
    prec1   = precision_score(y_true, y_pred, zero_division=0)
    rec1    = recall_score(y_true, y_pred, zero_division=0)
    f1_1    = f1_score(y_true, y_pred, zero_division=0)
    prec0   = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    rec0    = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_0    = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    cm      = confusion_matrix(y_true, y_pred)
    return {
        'roc_auc':   float(auc),
        'gini':      float(gini),
        'ks':        float(ks),
        'accuracy':  float(acc),
        'precision_class_1': float(prec1),
        'recall_class_1':    float(rec1),
        'f1_class_1':        float(f1_1),
        'precision_class_0': float(prec0),
        'recall_class_0':    float(rec0),
        'f1_class_0':        float(f1_0),
        'threshold':  float(threshold),
        'confusion_matrix': {
            'TN': int(cm[0, 0]), 'FP': int(cm[0, 1]),
            'FN': int(cm[1, 0]), 'TP': int(cm[1, 1]),
        },
    }


y_prob_val_nn  = model.predict(X_val_s,  batch_size=4096, verbose=0).ravel()
best_thr_nn, best_f1_val_nn = find_best_threshold(y_val, y_prob_val_nn)
print(f"  Порог нейронной сети: {best_thr_nn:.3f}  (Val F1: {best_f1_val_nn:.4f})")

y_prob_test_nn = model.predict(X_test_s, batch_size=4096, verbose=0).ravel()
nn_metrics     = compute_all_metrics(y_test, y_prob_test_nn, best_thr_nn)

print(f"  Тест: ROC-AUC={nn_metrics['roc_auc']:.4f}  F1={nn_metrics['f1_class_1']:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# 5. СРАВНИТЕЛЬНЫЙ АНАЛИЗ С АЛЬТЕРНАТИВНЫМИ МОДЕЛЯМИ
# ═══════════════════════════════════════════════════════════════════════════
print("\n[5/7] Обучение альтернативных моделей для сравнительного анализа...")

competitors = {
    'Логистическая регрессия': {
        'model': LogisticRegression(
            C=0.5, max_iter=1000, class_weight='balanced',
            solver='saga', n_jobs=-1, random_state=42),
        'description': 'Линейная модель, базовый бенчмарк',
    },
    'Дерево решений': {
        'model': DecisionTreeClassifier(
            max_depth=10, min_samples_leaf=30,
            class_weight='balanced', random_state=42),
        'description': 'Одно дерево решений (CART)',
    },
    'Случайный лес': {
        'model': RandomForestClassifier(
            n_estimators=200, max_depth=14, min_samples_leaf=20,
            class_weight='balanced', n_jobs=-1, random_state=42),
        'description': 'Ансамбль из 200 деревьев (бэггинг)',
    },
    'Градиентный бустинг': {
        'model': GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42),
        'description': 'Последовательный ансамбль с градиентным бустингом',
    },
}

comparison_results = []

# Сначала записываем результаты нейронной сети
comparison_results.append({
    'name':         'Нейронная сеть (TensorFlow/Keras)',
    'short_name':   'Нейронная сеть',
    'description':  'Многослойный персептрон 3 скрытых слоя (256-128-64) + Dropout + BatchNorm',
    'training_time_sec': round(train_time, 1),
    'metrics':      nn_metrics,
    'is_main':      True,
})

print(f"\n  {'Модель':<28} {'AUC':>8} {'Gini':>7} {'KS':>7} "
      f"{'F1':>7} {'Prec':>7} {'Rec':>7} {'Время':>7}")
print("  " + "─" * 86)
print(f"  {'Нейронная сеть':<28} {nn_metrics['roc_auc']:>8.4f} "
      f"{nn_metrics['gini']:>7.4f} {nn_metrics['ks']:>7.4f} "
      f"{nn_metrics['f1_class_1']:>7.4f} {nn_metrics['precision_class_1']:>7.4f} "
      f"{nn_metrics['recall_class_1']:>7.4f} {train_time:>6.0f}с")

# Обучаем альтернативные модели
for name, cfg in competitors.items():
    clf = cfg['model']
    t_start = time.time()
    try:
        clf.fit(X_train_s, y_train, sample_weight=sw_train)
    except TypeError:
        clf.fit(X_train_s, y_train)
    elapsed = time.time() - t_start

    prob_v = clf.predict_proba(X_val_s)[:, 1]
    thr_c, f1_val_c = find_best_threshold(y_val, prob_v)

    prob_t  = clf.predict_proba(X_test_s)[:, 1]
    metrics = compute_all_metrics(y_test, prob_t, thr_c)

    comparison_results.append({
        'name':              name,
        'short_name':        name,
        'description':       cfg['description'],
        'training_time_sec': round(elapsed, 1),
        'metrics':           metrics,
        'is_main':           False,
    })

    print(f"  {name:<28} {metrics['roc_auc']:>8.4f} "
          f"{metrics['gini']:>7.4f} {metrics['ks']:>7.4f} "
          f"{metrics['f1_class_1']:>7.4f} {metrics['precision_class_1']:>7.4f} "
          f"{metrics['recall_class_1']:>7.4f} {elapsed:>6.0f}с")

# ── Анализ преимущества ──────────────────────────────────────────────────
print("\n  Анализ преимуществ:")
nn_auc       = nn_metrics['roc_auc']
best_alt     = max((r for r in comparison_results if not r['is_main']),
                   key=lambda r: r['metrics']['roc_auc'])
best_alt_auc = best_alt['metrics']['roc_auc']
delta        = nn_auc - best_alt_auc

print(f"    Лучшая альтернатива: {best_alt['name']}  (AUC = {best_alt_auc:.4f})")
print(f"    Преимущество нейронной сети: {delta:+.4f} по ROC-AUC")

# Сортировка по AUC для дашборда
comparison_results.sort(key=lambda r: r['metrics']['roc_auc'], reverse=True)


# ═══════════════════════════════════════════════════════════════════════════
# 6. РАНЖИРОВАНИЕ
# ═══════════════════════════════════════════════════════════════════════════
print("\n[6/7] Финальное ранжирование моделей...")

for i, r in enumerate(comparison_results, 1):
    r['rank'] = i

print(f"\n  {'#':>2}  {'Модель':<32} {'ROC-AUC':>9} {'Gini':>7}")
print("  " + "─" * 56)
for r in comparison_results:
    marker = ' ← нейросеть' if r['is_main'] else ''
    print(f"  {r['rank']:>2}  {r['short_name']:<32} "
          f"{r['metrics']['roc_auc']:>9.4f} {r['metrics']['gini']:>7.4f}"
          f"{marker}")


# ═══════════════════════════════════════════════════════════════════════════
# 7. СОХРАНЕНИЕ АРТЕФАКТОВ
# ═══════════════════════════════════════════════════════════════════════════
print("\n[7/7] Сохранение артефактов...")

model.save(MODEL_PATH)
print(f"  ✓ Модель:        {MODEL_PATH}")

joblib.dump(scaler, SCALER_PATH)
print(f"  ✓ Scaler:        {SCALER_PATH}")

with open(FEATURES_PATH, 'w', encoding='utf-8') as f:
    json.dump(FEATURES, f, ensure_ascii=False, indent=2)
print(f"  ✓ Features:      {FEATURES_PATH}")

metadata = {
    'version': '1.1.0',
    'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    'tensorflow_version': tf.__version__,
    'threshold': float(best_thr_nn),
    'n_features': len(FEATURES),
    'class_weights': {'reject_0': 1.0, 'confirmed_1': float(w_pos)},
    'hyperparameters': HYPERPARAMS,
    'training': {
        'epochs_run':       int(n_epochs),
        'training_time_sec': round(train_time, 1),
        'samples_train':    int(len(y_train)),
        'samples_val':      int(len(y_val)),
        'samples_test':     int(len(y_test)),
    },
    'metrics': nn_metrics,
}
with open(METADATA_PATH, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)
print(f"  ✓ Metadata:      {METADATA_PATH}")

hist_clean = {k: [float(v) for v in vals] for k, vals in history.history.items()}
with open(HISTORY_PATH, 'w', encoding='utf-8') as f:
    json.dump(hist_clean, f, ensure_ascii=False, indent=2)
print(f"  ✓ History:       {HISTORY_PATH}")

comparison_data = {
    'created_at':    time.strftime('%Y-%m-%d %H:%M:%S'),
    'test_samples':  int(len(y_test)),
    'models':        comparison_results,
    'summary': {
        'best_model':                 comparison_results[0]['name'],
        'best_roc_auc':               float(comparison_results[0]['metrics']['roc_auc']),
        'nn_advantage_over_best_alt': float(delta),
        'best_alternative':           best_alt['name'],
    },
}
with open(COMPARISON_PATH, 'w', encoding='utf-8') as f:
    json.dump(comparison_data, f, ensure_ascii=False, indent=2)
print(f"  ✓ Comparison:    {COMPARISON_PATH}")

print("\n" + "=" * 70)
print("✓ ОБУЧЕНИЕ И СРАВНИТЕЛЬНЫЙ АНАЛИЗ ЗАВЕРШЕНЫ")
print("=" * 70)
print(f"\nДля запуска веб-приложения:  python app.py")