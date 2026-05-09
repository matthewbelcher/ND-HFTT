#!/usr/bin/env python3
"""Ensemble strategy profitability with Kalshi fees."""
import os, warnings, numpy as np, pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')

ROOT   = Path(__file__).resolve().parent.parent   # Kalshi-Orderbook-Predictor/
OUTPUT = ROOT / 'output'
# Trained only on `train` dates below; consumed by live_simulator (no training at runtime).
LIVE_GBM_10S_PATH = OUTPUT / 'ensemble_gbm_10s_live.joblib'
df = pd.read_parquet(OUTPUT / 'expanded_features.parquet')
df['date'] = df['contract_dt'].dt.strftime('%Y-%m-%d')
train = df[df['date'].isin(['2026-03-16','2026-03-17','2026-03-22','2026-03-24'])]
test  = df[df['date'].isin(['2026-03-25','2026-04-08'])]

FEATS = [
    'obi_1','obi_3','obi_5','obi_10','spread','mid_price','depth_skew',
    'best_level_imbalance','toxicity_10s','depth_hhi','mom_5s','mom_10s',
    'mr_30s','obi_vel_1s','obi_vel_5s','btc_vol_imbalance'
]

def get_X(data):
    return data[FEATS].to_numpy(dtype=np.float64)

def fee(price, qty, ftype='taker'):
    rate = 0.07 if ftype == 'taker' else 0.0175
    return np.ceil(rate * qty * price * (1 - price) * 100) / 100

# Train models
print('Training ensemble models...')
models = {}
for h in [5, 10, 30]:
    tgt = f'future_dir_{h}s'
    tr = train[FEATS + [tgt]].dropna()
    X_tr, y_tr = get_X(tr), tr[tgt].values
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    gb.fit(X_tr, y_tr)
    te = test[FEATS + [tgt]].dropna()
    X_te, y_te = get_X(te), te[tgt].values
    ta = roc_auc_score(y_tr, gb.predict_proba(X_tr)[:, 1])
    tea = roc_auc_score(y_te, gb.predict_proba(X_te)[:, 1])
    models[h] = gb
    print(f'  {h}s: Train AUC={ta:.4f}, Test AUC={tea:.4f}')

if 10 in models:
    joblib.dump(
        {
            'model': models[10],
            'feature_names': FEATS,
            'target': 'future_dir_10s',
            'train_dates': ['2026-03-16', '2026-03-17', '2026-03-22', '2026-03-24'],
        },
        LIVE_GBM_10S_PATH,
    )
    print(f"\nSaved 10s GBM for live simulator -> {LIVE_GBM_10S_PATH}")

# Evaluate as strategy
def run_strategy(model, data, horizon, thresh, qty, ftype):
    fm = f'future_mid_{horizon}s'
    needed = list(set(FEATS + [fm, 'mid_price']))
    v = data[needed].dropna()
    if len(v) < 50:
        return None
    X = v[FEATS].to_numpy()
    # Remove duplicate columns that may arise from mid_price appearing twice
    if X.shape[1] != len(FEATS):
        X = X[:, :len(FEATS)]
    probs = model.predict_proba(X)[:, 1]
    
    gross_list, fee_list, net_list = [], [], []
    for i in range(len(probs)):
        p = probs[i]
        ep = v['mid_price'].values[i]
        xp = v[fm].values[i]
        if np.isnan(xp):
            continue
        if p > thresh:
            d = 1
        elif p < (1 - thresh):
            d = -1
        else:
            continue
        g = d * (xp - ep) * qty
        f_tot = fee(ep, qty, ftype) + fee(xp, qty, ftype)
        gross_list.append(g)
        fee_list.append(f_tot)
        net_list.append(g - f_tot)
    
    if not gross_list:
        return None
    return {
        'n': len(gross_list),
        'gross': sum(gross_list),
        'fees': sum(fee_list),
        'net': sum(net_list),
        'wr': sum(1 for n in net_list if n > 0) / len(net_list),
    }

print('\n' + '='*95)
print('ENSEMBLE GBM STRATEGY RESULTS (OUT-OF-SAMPLE)')
print('='*95)

for h in [5, 10, 30]:
    m = models[h]
    print(f'\n--- Horizon: {h}s ---')
    print(f'  {"Thresh":>6} {"Fee":>5} {"Qty":>4} | {"N":>7} | {"Gross":>10} | {"Fees":>9} | {"NET":>10} | {"WR_net":>7}')
    print('  ' + '-'*78)
    for t in [0.52, 0.55, 0.58, 0.60, 0.65]:
        for ft in ['taker', 'maker']:
            for q in [1, 100]:
                r = run_strategy(m, test, h, t, q, ft)
                if r is None or r['n'] < 20:
                    continue
                marker = ' <<' if r['net'] > 0 else ''
                print(f'  {t:6.2f} {ft:>5} {q:4d} | {r["n"]:7d} | ${r["gross"]:9.2f} | ${r["fees"]:8.2f} | ${r["net"]:9.2f} | {r["wr"]:.3f}{marker}')

# Head-to-head
print('\n' + '='*95)
print('HEAD-TO-HEAD: OBI-ONLY vs ENSEMBLE (10s, maker, 100 contracts)')
print('='*95)
m10 = models[10]

# OBI only
for t in [0.20, 0.30]:
    v = test[['obi_1','future_mid_10s','mid_price']].dropna()
    sig = np.where(v['obi_1']>t, 1, np.where(v['obi_1']<-t, -1, 0))
    mask = sig != 0
    ep = v['mid_price'].values[mask]
    xp = np.where(np.isnan(v['future_mid_10s'].values[mask]), ep, v['future_mid_10s'].values[mask])
    g = np.where(sig[mask]==1, xp-ep, ep-xp) * 100
    f = np.array([fee(e,100,'maker')+fee(x,100,'maker') for e,x in zip(ep,xp)])
    n = g - f
    print(f'  OBI L1 t={t:.2f}:       N={mask.sum():7d} Gross=${g.sum():10.2f} Fees=${f.sum():9.2f} NET=${n.sum():10.2f}')

for t in [0.55, 0.58, 0.60]:
    r = run_strategy(m10, test, 10, t, 100, 'maker')
    if r:
        print(f'  Ensemble t={t:.2f}:     N={r["n"]:7d} Gross=${r["gross"]:10.2f} Fees=${r["fees"]:9.2f} NET=${r["net"]:10.2f}')

# Calibration
print('\n' + '='*95)
print('MODEL CALIBRATION (10s)')
print('='*95)
v = test[list(set(FEATS + ['future_dir_10s']))].dropna()
X = v[FEATS].to_numpy()[:, :len(FEATS)]
probs = m10.predict_proba(X)[:, 1]
y = v['future_dir_10s'].values

bins = [(0,0.40),(0.40,0.45),(0.45,0.50),(0.50,0.55),(0.55,0.60),(0.60,1.0)]
print(f'  {"P(up) range":>12} | {"N":>7} | {"Actual P(up)":>12} | {"% of data":>10}')
print('  ' + '-'*50)
for lo,hi in bins:
    mask = (probs >= lo) & (probs < hi)
    n = mask.sum()
    actual = y[mask].mean() if n > 0 else 0
    pct = n/len(probs)*100
    print(f'  [{lo:.2f}, {hi:.2f})  | {n:7d} | {actual:12.3f} | {pct:9.1f}%')

# Gross profitability by confidence bucket
print('\n' + '='*95)
print('GROSS P&L BY MODEL CONFIDENCE (10s, 100ct, maker)')
print('='*95)
v = test[list(set(FEATS + ['future_mid_10s','mid_price']))].dropna()
X = v[FEATS].to_numpy()
if X.shape[1] != len(FEATS):
    X = X[:, :len(FEATS)]
probs = m10.predict_proba(X)[:, 1]
ep = v['mid_price'].values
xp = v['future_mid_10s'].values

for lo, hi in [(0.50,0.55),(0.55,0.60),(0.60,0.65),(0.65,0.70),(0.70,1.0)]:
    mask = (probs >= lo) & (probs < hi) & ~np.isnan(xp)
    if mask.sum() < 10:
        continue
    g = np.where(probs[mask]>0.5, xp[mask]-ep[mask], ep[mask]-xp[mask]) * 100
    f = np.array([fee(e,100,'maker')+fee(x,100,'maker') for e,x in zip(ep[mask],xp[mask])])
    n = g - f
    print(f'  Confidence [{lo:.2f}-{hi:.2f}): N={mask.sum():6d} Gross=${g.sum():9.2f} Fees=${f.sum():8.2f} NET=${n.sum():9.2f}')
