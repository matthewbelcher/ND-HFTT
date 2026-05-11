#!/usr/bin/env python3
"""
Ensemble + BTC Momentum — Augmented Model
==========================================
Takes the teammate's GBM setup (ensemble_profitability.py) and adds
btc_mom_5s / btc_mom_10s / btc_accel as new features.

Compares:
  - Baseline: original 16-feature GBM (teammate's model)
  - Augmented: 16 + 3 BTC momentum features
  - Filter:    baseline GBM predictions, only kept when BTC momentum agrees

Train/test split is identical to the teammate's:
  Train: 2026-03-16, 03-17, 03-22, 03-24
  Test:  2026-03-25, 04-08
"""

import os, warnings, glob
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')

ROOT   = Path(__file__).resolve().parent.parent   # Kalshi-Orderbook-Predictor/
OUTPUT = ROOT / 'output'
CACHE  = OUTPUT / 'btc_mom_cache'

# ── Teammate's feature set ─────────────────────────────────────────────────────
BASE_FEATS = [
    'obi_1','obi_3','obi_5','obi_10','spread','mid_price','depth_skew',
    'best_level_imbalance','toxicity_10s','depth_hhi','mom_5s','mom_10s',
    'mr_30s','obi_vel_1s','obi_vel_5s','btc_vol_imbalance'
]
NEW_FEATS  = ['btc_mom_5s', 'btc_mom_10s', 'btc_accel']
ALL_FEATS  = BASE_FEATS + NEW_FEATS

TRAIN_DATES = {'2026-03-16', '2026-03-17', '2026-03-22', '2026-03-24'}
TEST_DATES  = {'2026-03-25', '2026-04-08'}


# ── 1. Load expanded_features and merge in BTC momentum ───────────────────────

print('Loading expanded_features.parquet…')
kalshi = pd.read_parquet(OUTPUT / 'expanded_features.parquet')

# Normalise timestamps
if kalshi['timestamp'].dtype == object:
    kalshi['timestamp'] = pd.to_datetime(kalshi['timestamp'], utc=True)
elif kalshi['timestamp'].dt.tz is None:
    kalshi['timestamp'] = kalshi['timestamp'].dt.tz_localize('UTC')

kalshi['ts_sec'] = kalshi['timestamp'].dt.floor('s')
kalshi['date']   = kalshi['contract_dt'].dt.strftime('%Y-%m-%d')
print(f'  {len(kalshi):,} rows  |  contracts: {kalshi["contract"].nunique()}')

# Load btc_mom_cache parquets for matching sessions
# cache contract col: 'KXBTC15M-26MAR161945-45'
# kalshi contract col: '26MAR161945-45'  → prepend prefix to match
cache_files = sorted(CACHE.glob('KXBTC15M-*.parquet'))
print(f'\nLoading {len(cache_files)} btc_mom_cache sessions…')

btc_frames = []
for cf in cache_files:
    df = pd.read_parquet(cf)
    # index is the 1s timestamp
    df = df.reset_index().rename(columns={'index': 'ts_sec'})
    if 'ts_sec' not in df.columns:
        # handle if index had a different name
        df.columns = ['ts_sec'] + list(df.columns[1:])
    df['ts_sec'] = pd.to_datetime(df['ts_sec'], utc=True).dt.floor('s')
    # strip KXBTC15M- prefix to match kalshi contract col
    short = cf.stem.replace('KXBTC15M-', '')
    df['contract_short'] = short
    btc_frames.append(df[['ts_sec', 'contract_short', 'btc_mom_5s', 'btc_mom_10s', 'btc_accel']])

btc_all = pd.concat(btc_frames, ignore_index=True)
print(f'  {len(btc_all):,} BTC momentum rows')

# Merge
kalshi['contract_short'] = kalshi['contract']   # already in short form
merged = kalshi.merge(btc_all, on=['contract_short', 'ts_sec'], how='left')
nan_rate = merged['btc_mom_10s'].isna().mean()
print(f'  Merge NaN rate on btc_mom_10s: {nan_rate:.1%}')
print(f'  Final rows: {len(merged):,}')


# ── 2. Train/test split ────────────────────────────────────────────────────────

train = merged[merged['date'].isin(TRAIN_DATES)]
test  = merged[merged['date'].isin(TEST_DATES)]
print(f'\nTrain: {len(train):,} rows  |  Test: {len(test):,} rows')


def get_X(df, feats):
    return df[feats].to_numpy(dtype=np.float64)

def fee(price, qty, ftype='taker'):
    rate = 0.07 if ftype == 'taker' else 0.0175
    return np.ceil(rate * qty * price * (1 - price) * 100) / 100


# ── 3. Train both models for each horizon ─────────────────────────────────────

results = {}

for horizon in [5, 10, 30]:
    tgt = f'future_dir_{horizon}s'
    print(f'\n{"─"*60}')
    print(f'Horizon: {horizon}s  |  target: {tgt}')

    tr_full = train[BASE_FEATS + NEW_FEATS + [tgt]].dropna()
    te_full = test [BASE_FEATS + NEW_FEATS + [tgt]].dropna()

    if len(tr_full) < 100 or len(te_full) < 10:
        print(f'  [skip] insufficient data')
        continue

    tr_base = tr_full[BASE_FEATS + [tgt]].dropna()
    te_base = te_full[BASE_FEATS + [tgt]].dropna()

    # ── Baseline (teammate's model) ──
    gb_base = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    gb_base.fit(get_X(tr_base, BASE_FEATS), tr_base[tgt].values)
    prob_base = gb_base.predict_proba(get_X(te_base, BASE_FEATS))[:, 1]
    auc_base  = roc_auc_score(te_base[tgt].values, prob_base)

    # ── Augmented (+ BTC momentum) ──
    gb_aug = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    gb_aug.fit(get_X(tr_full, ALL_FEATS), tr_full[tgt].values)
    prob_aug = gb_aug.predict_proba(get_X(te_full, ALL_FEATS))[:, 1]
    auc_aug  = roc_auc_score(te_full[tgt].values, prob_aug)

    print(f'  Baseline AUC  (16 feats):     {auc_base:.4f}')
    print(f'  Augmented AUC (16+3 feats):   {auc_aug:.4f}   Δ={auc_aug-auc_base:+.4f}')

    # ── Feature importances — how much do new features contribute? ──
    feat_imp = dict(zip(ALL_FEATS, gb_aug.feature_importances_))
    new_total = sum(feat_imp[f] for f in NEW_FEATS)
    print(f'\n  BTC momentum feature importances:')
    for f in NEW_FEATS:
        print(f'    {f:<18} {feat_imp[f]:.4f}')
    print(f'    {"TOTAL":<18} {new_total:.4f}  '
          f'(vs btc_vol_imbalance: {feat_imp.get("btc_vol_imbalance", 0):.4f})')

    # ── BTC momentum filter on baseline predictions ──
    # Only trade when |btc_mom_10s| > threshold AND agrees with model direction
    te_aug_df = te_full.copy()
    te_aug_df['prob']     = prob_aug
    te_aug_df['pred_dir'] = (prob_aug >= 0.5).astype(int)   # 0 or 1
    te_aug_df['btc_dir']  = (te_aug_df['btc_mom_10s'].fillna(0) > 0).astype(int)

    for thr in [0.0, 0.0001, 0.0002, 0.0005]:
        if thr == 0.0:
            mask = pd.Series([True] * len(te_aug_df), index=te_aug_df.index)
        else:
            mask = (te_aug_df['btc_mom_10s'].abs() >= thr) & \
                   (te_aug_df['btc_dir'] == te_aug_df['pred_dir'])
        sub = te_aug_df[mask]
        if len(sub) < 10:
            continue
        keep_pct = mask.mean()
        auc_filt = roc_auc_score(sub[tgt].values,
                                  sub['prob'].values)
        acc_filt = (sub['pred_dir'] == sub[tgt]).mean()
        print(f'\n  BTC filter thr={thr:.4f}  '
              f'kept={keep_pct:.0%}  '
              f'AUC={auc_filt:.4f}  '
              f'acc={acc_filt:.3f}')

    results[horizon] = {
        'auc_base': auc_base,
        'auc_aug':  auc_aug,
        'delta':    auc_aug - auc_base,
    }

    # Save augmented model for 10s horizon
    if horizon == 10:
        path = OUTPUT / 'ensemble_gbm_10s_btcmom.joblib'
        # Save as dict with feature_names so model_runner.py can validate on load.
        joblib.dump({"model": gb_aug, "feature_names": ALL_FEATS}, path)
        print(f'\n  Saved augmented model → {path}')


# ── 4. Summary ─────────────────────────────────────────────────────────────────

print(f'\n{"═"*60}')
print('SUMMARY')
print(f'{"═"*60}')
print(f'{"Horizon":<10} {"Baseline AUC":<16} {"Augmented AUC":<16} {"Δ AUC":<10}')
print('─' * 52)
for h, r in results.items():
    flag = '▲' if r['delta'] > 0 else '▼'
    print(f'{h}s{"":<8} {r["auc_base"]:.4f}{"":<10} {r["auc_aug"]:.4f}{"":<10} '
          f'{flag} {r["delta"]:+.4f}')

print('\nNew features added: btc_mom_5s, btc_mom_10s, btc_accel')
print('(BTC price return over 5s/10s windows + acceleration)')
