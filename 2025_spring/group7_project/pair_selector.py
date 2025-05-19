import dask.dataframe as dd
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
import logging

'''
This file selects cointegrated pairs (e.g., HD and Lowes) by computing correlations and performing cointegration tests.
It identifies lead-lag relationships and feeds results to stat_arb_strat.py.
'''

class PairSelector:
    def __init__(self, ddf, symbols=None):
        self.ddf = ddf
        if symbols is None:
            counts = ddf.SYM_ROOT.value_counts().nlargest(50).compute()
            symbols = counts.index.tolist()
        self.symbols = symbols
        logging.info(f"Initialized PairSelector with {len(self.symbols)} symbols: {self.symbols}")

    def _corr_pair(self, sym1, sym2):
        logging.debug(f"Computing correlation for {sym1}/{sym2}")
        d1 = (self.ddf[self.ddf.SYM_ROOT == sym1]
              [['datetime', 'mid']]
              .rename(columns={'mid': 'm1'}))
        d2 = (self.ddf[self.ddf.SYM_ROOT == sym2]
              [['datetime', 'mid']]
              .rename(columns={'mid': 'm2'}))
        merged = dd.merge(d1, d2, on='datetime', how='inner').compute()
        if len(merged) < 1000:
            logging.warning(f"Insufficient data for {sym1}/{sym2}: {len(merged)} rows")
            return 0.0, len(merged)
        
        m1 = merged['m1']
        m2 = merged['m2']
        n = len(merged)
        sum_x = m1.sum()
        sum_y = m2.sum()
        sum_xx = (m1 * m1).sum()
        sum_yy = (m2 * m2).sum()
        sum_xy = (m1 * m2).sum()
        num = n * sum_xy - sum_x * sum_y
        den = np.sqrt((n * sum_xx - sum_x**2) * (n * sum_yy - sum_y**2))
        corr = float(num / den) if den != 0 else 0.0
        logging.debug(f"Correlation {sym1}/{sym2}: {corr:.3f}, n={n}")
        return corr, n

    def _coint_pair(self, sym1, sym2):
        logging.debug(f"Computing cointegration for {sym1}/{sym2}")
        d1 = (self.ddf[self.ddf.SYM_ROOT == sym1]
              [['datetime', 'mid']]
              .rename(columns={'mid': 'm1'}))
        d2 = (self.ddf[self.ddf.SYM_ROOT == sym2]
              [['datetime', 'mid']]
              .rename(columns={'mid': 'm2'}))
        merged = dd.merge(d1, d2, on='datetime', how='inner').compute()
        if len(merged) < 1000:
            logging.warning(f"Insufficient data for cointegration {sym1}/{sym2}: {len(merged)} rows")
            return None, 1.0, None
        
        m1 = merged['m1']
        m2 = merged['m2']
        try:
            stat, pval, crit = coint(m1, m2)
            logging.info(f"Cointegration {sym1}/{sym2}: stat={stat:.3f}, pval={pval:.4f}, crit={crit}")
            return stat, pval, crit
        except Exception as e:
            logging.error(f"Cointegration test failed for {sym1}/{sym2}: {str(e)}")
            return None, 1.0, None

    def _lead_lag(self, sym1, sym2, max_lag=10):
        logging.debug(f"Computing lead-lag for {sym1}/{sym2}")
        d1 = (self.ddf[self.ddf.SYM_ROOT == sym1]
              [['datetime', 'mid']]
              .rename(columns={'mid': 'm1'}))
        d2 = (self.ddf[self.ddf.SYM_ROOT == sym2]
              [['datetime', 'mid']]
              .rename(columns={'mid': 'm2'}))
        merged = dd.merge(d1, d2, on='datetime', how='inner').compute()
        if len(merged) < 1000:
            return 0, 0.0
        
        m1 = merged['m1']
        m2 = merged['m2']
        correlations = []
        for lag in range(-max_lag, max_lag + 1):
            corr = m1.corr(m2.shift(lag))
            correlations.append((lag, corr))
        max_corr = max(correlations, key=lambda x: abs(x[1]), default=(0, 0.0))
        logging.debug(f"Lead-lag {sym1}/{sym2}: lag={max_corr[0]}, corr={max_corr[1]:.3f}")
        return max_corr[0], max_corr[1]

    def select_pairs(self, corr_threshold=0.70, coint_pval_threshold=0.05, max_lag=10):
        pairs = []
        for i, s1 in enumerate(self.symbols):
            for s2 in self.symbols[i + 1:]:
                corr, n = self._corr_pair(s1, s2)
                if n > 1000 and abs(corr) >= corr_threshold:
                    stat, pval, crit = self._coint_pair(s1, s2)
                    if pval < coint_pval_threshold:
                        lag, lag_corr = self._lead_lag(s1, s2, max_lag)
                        pairs.append((s1, s2, corr, pval, lag, lag_corr))
        logging.info(f"Found {len(pairs)} cointegrated pairs with |ρ| ≥ {corr_threshold:.2f}, pval < {coint_pval_threshold:.2f}")
        return sorted(pairs, key=lambda x: x[3])  # Sort by p-value