"""
=============================================================================
АНАЛИЗ ПРИЗНАКОВ: распределения, статистика, нормализация
=============================================================================

Скрипт строит:
  • Графики распределения каждого признака ДО нормализации
  • Графики того же признака ПОСЛЕ нормализации (StandardScaler)
  • JSON-файл со статистикой (min/max/mean/std/медиана/квантили)
  • Сводную таблицу всех признаков

Запуск:
    python analyze_features.py

Требования: должна быть обучена модель (запущен train_model.py).
Результаты сохраняются в:
    static/feature_plots/   — PNG-графики
    models/feature_stats.json — статистика признаков
=============================================================================
"""

import os
import json
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─── Пути ───────────────────────────────────────────────────────────────────
DATA_PATH    = '/home/tatyana/Documents/Baumana/ВКР_МГТУ/files/Scoring.csv'    # путь к датасету
MODELS_DIR     = '/home/tatyana/Documents/Baumana/ВКР_МГТУ/files/credit_scoring_app/models'
PLOTS_DIR      = '/home/tatyana/Documents/Baumana/ВКР_МГТУ/files/credit_scoring_app/static/feature_plots'
SCALER_PATH    = os.path.join(MODELS_DIR, 'scaler.joblib')
FEATURES_PATH  = os.path.join(MODELS_DIR, 'features.json')
STATS_PATH     = os.path.join(MODELS_DIR, 'feature_stats.json')

os.makedirs(PLOTS_DIR, exist_ok=True)

print("=" * 70)
print("АНАЛИЗ ПРИЗНАКОВ — РАСПРЕДЕЛЕНИЯ ДО И ПОСЛЕ НОРМАЛИЗАЦИИ")
print("=" * 70)

# ─── Проверка наличия артефактов ────────────────────────────────────────────
if not os.path.exists(SCALER_PATH) or not os.path.exists(FEATURES_PATH):
    print(f"\n✗ Не найдены артефакты модели.")
    print(f"  Сначала запустите: python train_model.py\n")
    exit(1)

scaler = joblib.load(SCALER_PATH)
with open(FEATURES_PATH, 'r', encoding='utf-8') as f:
    FEATURES = json.load(f)

print(f"  Загружен scaler с {scaler.n_features_in_} признаков")
print(f"  Список признаков: {len(FEATURES)}")


# ─── Загрузка и предобработка данных (как в train_model.py) ─────────────────
print("\n[1/4] Загрузка и подготовка данных...")

df = pd.read_csv(
    DATA_PATH, sep=';', encoding='utf-8-sig',
    on_bad_lines='skip', decimal=',', low_memory=False
)
df = df[df['Целевое значение'].isin(['confirmed', 'reject'])].copy()
df['target'] = (df['Целевое значение'] == 'confirmed').astype(int)

# Та же инженерия признаков, что и в train_model.py
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

df['имеет_пенсию']       = (df['Пенсия'] > 0).astype(int)
df['имеет_неоф_доход']   = (df['Неофициальный доход'] > 0).astype(int)
df['имеет_доп_доход']    = (df['Дополнительный доход'] > 0).astype(int)
df['имеет_директорские'] = (df['Директорские выплаты'] > 0).astype(int)
df['имеет_авто']         = (df['Оценка автомобиля'] > 0).astype(int)

lim  = df['Лимит по карте'].fillna(0).clip(lower=0)
free = df['Свободный лимит'].fillna(0)
df['есть_лимит']           = df['Лимит по карте'].notna().astype(int)
df['log_лимит']            = np.log1p(lim)
df['использование_лимита'] = np.where(
    lim > 0, ((lim - free) / lim.clip(lower=1)).clip(0, 2), 0.0)
df['платёжеспособность']   = df['Платежеспособность'].fillna(0) / 100.0

for col, alias in [
    ('Клиент в черном списке', 'blacklist'),
    ('Возраст',                'age_check'),
    ('Криминальное прошлое',   'criminal'),
    ('Оценка клиента',         'client_score'),
]:
    df[f'{alias}_пройден']    = (df[col] == 900).astype(int)
    df[f'{alias}_отказ']      = (df[col] == 901).astype(int)
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

X_raw = df[FEATURES].fillna(0).values.astype(np.float64)
X_norm = scaler.transform(X_raw)
y = df['target'].values

print(f"  Подготовлено {len(df):,} наблюдений × {len(FEATURES)} признаков")


# ─── Расчёт статистики по каждому признаку ─────────────────────────────────
print("\n[2/4] Расчёт статистики...")

def get_stats(values):
    """Возвращает словарь статистики по массиву чисел."""
    arr = np.array(values, dtype=np.float64)
    return {
        'min':      float(np.min(arr)),
        'max':      float(np.max(arr)),
        'mean':     float(np.mean(arr)),
        'std':      float(np.std(arr)),
        'median':   float(np.median(arr)),
        'q25':      float(np.percentile(arr, 25)),
        'q75':      float(np.percentile(arr, 75)),
        'unique':   int(len(np.unique(arr))),
        'zeros_pct': float((arr == 0).sum() / len(arr) * 100),
    }


# Тип признака: бинарный или непрерывный
def is_binary(values):
    unique_vals = np.unique(values)
    return len(unique_vals) <= 2 and set(unique_vals.astype(int)).issubset({0, 1})


feature_stats = []
for i, fname in enumerate(FEATURES):
    raw_vals  = X_raw[:, i]
    norm_vals = X_norm[:, i]
    binary    = is_binary(raw_vals)

    feature_stats.append({
        'name':          fname,
        'index':         i,
        'is_binary':     binary,
        'type':          'binary' if binary else 'continuous',
        'raw':           get_stats(raw_vals),
        'normalized':    get_stats(norm_vals),
        'scaler_mean':   float(scaler.mean_[i]),
        'scaler_scale':  float(scaler.scale_[i]),
        'plot_filename': f'feature_{i:02d}_{fname.replace(" ", "_")}.png',
    })

# Сохранение статистики
stats_data = {
    'created_at':    pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'n_samples':     int(len(df)),
    'n_features':    len(FEATURES),
    'features':      feature_stats,
}
with open(STATS_PATH, 'w', encoding='utf-8') as f:
    json.dump(stats_data, f, ensure_ascii=False, indent=2)
print(f"  ✓ Статистика сохранена: {STATS_PATH}")


# ─── Построение графиков для каждого признака ──────────────────────────────
print("\n[3/4] Построение графиков распределения...")

plt.rcParams.update({
    'figure.dpi': 110,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.grid': True,
    'grid.alpha': 0.25,
})
COLOR_RAW  = '#2E75B6'
COLOR_NORM = '#E91E63'
COLOR_POS  = '#10b981'
COLOR_NEG  = '#ef4444'

idx_pos = y == 1
idx_neg = y == 0

for i, fname in enumerate(FEATURES):
    raw_vals  = X_raw[:, i]
    norm_vals = X_norm[:, i]
    info      = feature_stats[i]
    binary    = info['is_binary']

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    fig.suptitle(
        f'Признак №{i+1}: «{fname}»  ({info["type"]})',
        fontsize=13, fontweight='bold'
    )

    # ── Слева: ДО нормализации ────────────────────────────────────────────
    ax = axes[0]
    if binary:
        # Бар-чарт для бинарных
        counts = pd.Series(raw_vals.astype(int)).value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values, color=COLOR_RAW,
               alpha=0.85, edgecolor='white', width=0.5)
        for x, v in zip(counts.index, counts.values):
            ax.text(str(x), v * 1.01, f'{v:,}\n({v/len(raw_vals)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_ylim(0, counts.max() * 1.18)
        ax.set_xlabel('Значение')
        ax.set_ylabel('Количество')
    else:
        # Гистограмма для непрерывных, разделение по классам
        ax.hist(raw_vals[idx_pos], bins=50, alpha=0.6, color=COLOR_POS,
                label='confirmed (1)', density=True)
        ax.hist(raw_vals[idx_neg], bins=50, alpha=0.6, color=COLOR_NEG,
                label='reject (0)', density=True)
        ax.set_xlabel('Значение признака (сырые данные)')
        ax.set_ylabel('Плотность')
        ax.legend(fontsize=9)
    ax.set_title('ДО нормализации')

    # Аннотация со статистикой
    r = info['raw']
    txt_raw = (f"min: {r['min']:.4f}\nmax: {r['max']:.4f}\n"
               f"mean: {r['mean']:.4f}\nstd: {r['std']:.4f}\n"
               f"median: {r['median']:.4f}\nуникальных: {r['unique']}")
    ax.text(0.98, 0.97, txt_raw, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8fafc',
                      edgecolor='#cbd5e1', alpha=0.95),
            family='monospace')

    # ── Справа: ПОСЛЕ нормализации ────────────────────────────────────────
    ax = axes[1]
    if binary:
        counts_n = pd.Series(np.round(norm_vals, 4)).value_counts().sort_index()
        labels = [f'{x:.3f}' for x in counts_n.index]
        ax.bar(labels, counts_n.values, color=COLOR_NORM,
               alpha=0.85, edgecolor='white', width=0.5)
        for x_lbl, v in zip(labels, counts_n.values):
            ax.text(x_lbl, v * 1.01, f'{v:,}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_ylim(0, counts_n.max() * 1.15)
        ax.set_xlabel('Нормализованное значение')
        ax.set_ylabel('Количество')
    else:
        ax.hist(norm_vals[idx_pos], bins=50, alpha=0.6, color=COLOR_POS,
                label='confirmed (1)', density=True)
        ax.hist(norm_vals[idx_neg], bins=50, alpha=0.6, color=COLOR_NEG,
                label='reject (0)', density=True)
        # Линии mean = 0 и ±1 std
        ax.axvline(x=0, color='black', lw=1.2, ls='--', alpha=0.5)
        ax.axvline(x=1, color='black', lw=0.8, ls=':',  alpha=0.4)
        ax.axvline(x=-1, color='black', lw=0.8, ls=':', alpha=0.4)
        ax.set_xlabel('Значение признака (после StandardScaler)')
        ax.set_ylabel('Плотность')
        ax.legend(fontsize=9)
    ax.set_title('ПОСЛЕ нормализации (StandardScaler)')

    n = info['normalized']
    txt_norm = (f"min: {n['min']:.4f}\nmax: {n['max']:.4f}\n"
                f"mean: {n['mean']:.4f}\nstd: {n['std']:.4f}\n"
                f"median: {n['median']:.4f}\n"
                f"μ={info['scaler_mean']:.4g}\nσ={info['scaler_scale']:.4g}")
    ax.text(0.98, 0.97, txt_norm, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fef3c7',
                      edgecolor='#fbbf24', alpha=0.95),
            family='monospace')

    plt.tight_layout()
    filepath = os.path.join(PLOTS_DIR, info['plot_filename'])
    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)

    if (i + 1) % 5 == 0 or (i + 1) == len(FEATURES):
        print(f"  Построено графиков: {i + 1}/{len(FEATURES)}")


# ─── Сводный график: распределение всех признаков ──────────────────────────
print("\n[4/4] Сводные графики...")

# Boxplot до нормализации (для непрерывных признаков)
continuous_idx = [i for i, fs in enumerate(feature_stats) if not fs['is_binary']]
if continuous_idx:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle('Сравнение распределений всех непрерывных признаков',
                 fontsize=14, fontweight='bold')

    cont_names = [FEATURES[i] for i in continuous_idx]
    cont_raw   = X_raw[:, continuous_idx]
    cont_norm  = X_norm[:, continuous_idx]

    ax = axes[0]
    bp = ax.boxplot(cont_raw, labels=cont_names, patch_artist=True,
                    showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor(COLOR_RAW)
        patch.set_alpha(0.7)
    ax.set_title('ДО нормализации — разные масштабы признаков')
    ax.set_ylabel('Значение')
    ax.tick_params(axis='x', rotation=45)

    ax = axes[1]
    bp = ax.boxplot(cont_norm, labels=cont_names, patch_artist=True,
                    showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor(COLOR_NORM)
        patch.set_alpha(0.7)
    ax.axhline(y=0, color='black', lw=1, ls='--', alpha=0.4)
    ax.set_title('ПОСЛЕ нормализации — все на одной шкале')
    ax.set_ylabel('Нормализованное значение')
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'summary_continuous.png'),
                bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Сводный график непрерывных: summary_continuous.png")


# Тепловая карта корреляций
fig, ax = plt.subplots(figsize=(13, 11))
corr = pd.DataFrame(X_raw, columns=FEATURES).corr()
sns.heatmap(corr, cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax,
            cbar_kws={'label': 'Коэффициент корреляции Пирсона'},
            square=True, linewidths=0.3, linecolor='white',
            annot=False)
ax.set_title('Корреляционная матрица всех признаков',
             fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, 'summary_correlation.png'),
            bbox_inches='tight')
plt.close(fig)
print(f"  ✓ Корреляционная матрица: summary_correlation.png")


# ─── Итоги ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"✓ АНАЛИЗ ЗАВЕРШЁН")
print("=" * 70)

n_files = len([f for f in os.listdir(PLOTS_DIR) if f.endswith('.png')])
print(f"\n  Файлов в {PLOTS_DIR}/: {n_files}")
print(f"  Статистика: {STATS_PATH}")

# Краткая таблица в консоль
print(f"\n  Топ-5 признаков с наибольшим диапазоном (max-min) ДО нормализации:")
sorted_features = sorted(feature_stats,
                         key=lambda f: f['raw']['max'] - f['raw']['min'],
                         reverse=True)[:5]
print(f"  {'Признак':<32}{'min':>12}{'max':>14}{'диапазон':>14}")
print("  " + "─" * 72)
for f in sorted_features:
    rng = f['raw']['max'] - f['raw']['min']
    print(f"  {f['name']:<32}{f['raw']['min']:>12.4f}"
          f"{f['raw']['max']:>14.4f}{rng:>14.4f}")

print(f"\nДля просмотра в браузере запустите: python3 app.py")
print(f"Затем откройте: http://localhost:5000/features\n")
