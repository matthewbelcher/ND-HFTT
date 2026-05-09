"""
dashboard.py
============
Interactive signal comparison dashboard for KXBTC15M order book signals.

Usage
-----
    # First time — loads from ../bigdata/, recalibrates G*, caches results
    python dashboard.py --data-dir ../bigdata/ --recalibrate

    # Subsequent runs — loads cache instantly, sessions stay in memory
    python dashboard.py --data-dir ../bigdata/

    # Multiple directories
    python dashboard.py --data-dir ../bigdata/ completed-data/

Then open http://127.0.0.1:8050 in your browser.

How interactive re-run works
-----------------------------
All sessions are loaded into memory at startup and kept in a module-level
list. When you click Re-run, a background thread re-evaluates all signals
with the new params. A polling interval checks every 800ms and updates the
charts when the thread finishes. No JSON round-trips — results stay
server-side so 200+ sessions don't freeze the browser.
"""

import argparse
import sys
import warnings
import threading
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, no_update

warnings.filterwarnings('ignore', category=FutureWarning)

# ── Colors ────────────────────────────────────────────────────────────────────
BG    = '#0f1117'
PANEL = '#151821'
GRID  = '#1e2130'
SPINE = '#2a2f45'
TEXT  = '#c8d0e0'
MUTED = '#6b7280'
BLUE  = '#00d4ff'
GREEN = '#00e676'
RED   = '#ff1744'
PURPLE= '#d400ff'
ORANGE= '#ff6b35'
YELLOW= '#ffd600'

SIGNAL_COLORS = {
    'obi_delta'          : BLUE,
    'microprice'         : PURPLE,
    'spread_filtered_obi': GREEN,
    'time_window_obi'    : ORANGE,
}
SIGNAL_LABELS = {
    'obi_delta'          : 'OBI Delta (obi3)',
    'microprice'         : 'Micro-price',
    'spread_filtered_obi': 'Spread-Filtered OBI',
    'time_window_obi'    : 'Time-Window OBI',
}

BASE_LAYOUT = dict(
    paper_bgcolor=BG, plot_bgcolor=PANEL,
    font=dict(color=TEXT, family='Arial', size=11),
)

# ── Server-side state — keeps DataFrames in Python, not the browser ───────────
_STATE: dict = {
    'results' : pd.DataFrame(),
    'stats'   : pd.DataFrame(),
    'sessions': [],
    'running' : False,
    'last_msg': 'Ready.',
    'progress': '',   # live progress string shown in status bar during re-run
}


# ── Signal builder ────────────────────────────────────────────────────────────

def _build_signals(threshold, horizon, signal_win, cooldown,
                   mp_threshold, gstar_path, start_sec, end_sec, obi_col):
    from signals import (OBIDeltaSignal, MicropriceSignal,
                         SpreadFilteredOBI, TimeWindowOBI)
    sigs = [
        OBIDeltaSignal(obi_col=obi_col, horizon=horizon, signal_win=signal_win,
                       threshold=threshold, min_tick=0.01, cooldown=cooldown),
        MicropriceSignal(gstar_path=gstar_path, threshold=mp_threshold,
                         horizon=horizon, min_tick=0.01, cooldown=cooldown),
        SpreadFilteredOBI(max_spread_ticks=1, obi_col=obi_col, horizon=horizon,
                          signal_win=signal_win, threshold=threshold,
                          min_tick=0.01, cooldown=cooldown),
        TimeWindowOBI(start_sec=int(start_sec), end_sec=int(end_sec),
                      obi_col=obi_col, horizon=horizon, signal_win=signal_win,
                      threshold=threshold, min_tick=0.01, cooldown=cooldown),
    ]
    for sig in sigs:
        if hasattr(sig, 'load'):
            sig.load()
    return sigs


def _run_signals(sigs, sessions):
    all_res = []
    n_total = len(sessions)
    for i, df in enumerate(sessions):
        ticker = df['ticker'].iloc[0]
        _STATE['progress'] = f'Session {i+1}/{n_total} — {ticker}'
        for sig in sigs:
            try:
                res = sig.evaluate(df)
                if not res.empty:
                    res = res.copy()
                    res['ticker'] = ticker
                    all_res.append(res)
            except Exception:
                pass

    if not all_res:
        return pd.DataFrame(), pd.DataFrame()

    results = pd.concat(all_res).sort_index()

    def _agg(g):
        hr = g['hit'].mean()
        ar = g['adverse'].mean()
        return pd.Series({
            'n_signals'   : len(g),
            'hit_rate'    : round(hr, 4),
            'adverse_rate': round(ar, 4),
            'no_move_rate': round(1 - hr - ar, 4),
            'avg_fwd_move': round(g['fwd_move'].mean(), 5),
        })

    stats = (results.groupby(['signal_name', 'ticker'])
             .apply(_agg, include_groups=False).reset_index())
    return results, stats


# ── Figure builders ───────────────────────────────────────────────────────────

def _empty(msg=''):
    fig = go.Figure()
    if msg:
        fig.add_annotation(text=msg, x=0.5, y=0.5, xref='paper', yref='paper',
                           showarrow=False, font=dict(color=MUTED, size=12))
    fig.update_layout(**BASE_LAYOUT, margin=dict(l=20,r=20,t=20,b=20), height=200)
    return fig


def fig_table(results):
    if results.empty:
        return _empty('No signals — adjust params and click Re-run.')
    rows = []
    for sig, grp in results.groupby('signal_name'):
        hr  = grp['hit'].mean()
        ar  = grp['adverse'].mean()
        rows.append([SIGNAL_LABELS.get(sig, sig),
                     grp['ticker'].nunique(), len(grp),
                     f'{hr:.1%}', f'{ar:.1%}', f'{1-hr-ar:.1%}',
                     f'{grp["fwd_move"].mean():+.5f}'])
    hr_vals = [float(r[3].rstrip('%')) / 100 for r in rows]
    hr_fill = [('rgba(0,230,118,.18)' if v >= 0.46 else
                'rgba(255,23,68,.15)' if v < 0.35 else PANEL)
               for v in hr_vals]
    n = len(rows)
    cols = ['Signal', 'Sess.', 'N', 'Hit Rate', 'Adverse', 'No-Move', 'Avg Fwd $']
    fig = go.Figure(go.Table(
        columnwidth=[220, 55, 65, 75, 75, 75, 85],
        header=dict(values=[f'<b>{c}</b>' for c in cols],
                    fill_color=GRID, font=dict(color=TEXT, size=11),
                    align='left', line_color=SPINE, height=26),
        cells=dict(values=list(zip(*rows)),
                   fill_color=[[PANEL]*n,[PANEL]*n,[PANEL]*n,
                                hr_fill,[PANEL]*n,[PANEL]*n,[PANEL]*n],
                   font=dict(color=TEXT, size=11),
                   align=['left','right','right','right','right','right','right'],
                   line_color=SPINE, height=24),
    ))
    fig.update_layout(**BASE_LAYOUT, margin=dict(l=0,r=0,t=4,b=4),
                      height=max(130, 52 + 24 * n))
    return fig


def fig_bars(results):
    if results.empty:
        return _empty()
    fig = go.Figure()
    for sig, grp in results.groupby('signal_name'):
        hr = grp['hit'].mean()
        fig.add_trace(go.Bar(
            x=[SIGNAL_LABELS.get(sig, sig)], y=[hr],
            marker_color=SIGNAL_COLORS.get(sig, YELLOW), opacity=0.85,
            text=[f'{hr:.1%}'], textposition='outside',
            textfont=dict(color=TEXT, size=12),
            name=SIGNAL_LABELS.get(sig, sig),
        ))
    fig.add_hline(y=0.50, line_dash='dash', line_color=TEXT, opacity=0.3,
                  annotation_text='50%', annotation_font_color=MUTED,
                  annotation_position='right')
    fig.add_hline(y=0.33, line_dash='dot', line_color=RED, opacity=0.2,
                  annotation_text='random', annotation_font_color=MUTED,
                  annotation_position='right')
    fig.update_layout(**BASE_LAYOUT, showlegend=False, bargap=0.4,
                      margin=dict(l=50,r=60,t=16,b=20),
                      yaxis=dict(tickformat='.0%', range=[0,.80], title='Hit Rate',
                                 gridcolor=GRID),
                      xaxis=dict(showticklabels=False),
                      height=290)
    return fig


def fig_per_session(stats, selected):
    if stats.empty:
        return _empty()
    fig = go.Figure()
    df = stats[stats['signal_name'].isin(selected)]
    for sig, grp in df.groupby('signal_name'):
        grp = grp.sort_values('ticker')
        fig.add_trace(go.Scatter(
            x=[t.split('-',1)[-1] for t in grp['ticker']],
            y=grp['hit_rate'],
            mode='lines+markers',
            name=SIGNAL_LABELS.get(sig, sig),
            line=dict(color=SIGNAL_COLORS.get(sig, YELLOW), width=1.8),
            marker=dict(size=5),
            customdata=np.stack([grp['n_signals'].values,
                                 grp['adverse_rate'].values,
                                 grp['avg_fwd_move'].values], axis=-1),
            hovertemplate=('<b>%{x}</b><br>Hit: <b>%{y:.1%}</b><br>'
                           'N: %{customdata[0]}<br>Adverse: %{customdata[1]:.1%}'
                           '<extra>%{fullData.name}</extra>'),
        ))
    fig.add_hline(y=0.5, line_dash='dash', line_color=TEXT, opacity=0.2)
    fig.update_layout(**BASE_LAYOUT, yaxis=dict(tickformat='.0%', title='Hit Rate',
                                                gridcolor=GRID),
                      xaxis=dict(tickangle=-40, title='Session'),
                      height=310, margin=dict(l=50,r=20,t=40,b=80),
                      legend=dict(orientation='h', yanchor='bottom', y=1.01,
                                  x=0, font=dict(color=TEXT, size=10)))
    return fig


def fig_fwd_dist(results, selected):
    if results.empty:
        return _empty()
    fig = go.Figure()
    for sig, grp in results[results['signal_name'].isin(selected)].groupby('signal_name'):
        fig.add_trace(go.Histogram(
            x=grp['fwd_move'], name=SIGNAL_LABELS.get(sig, sig),
            nbinsx=50, opacity=0.55, histnorm='probability',
            marker_color=SIGNAL_COLORS.get(sig, YELLOW),
        ))
    fig.add_vline(x=0, line_color=TEXT, opacity=0.3)
    fig.update_layout(**BASE_LAYOUT, barmode='overlay',
                      xaxis=dict(title='Forward move ($)', tickformat='+.3f',
                                 gridcolor=GRID),
                      yaxis=dict(title='Probability', gridcolor=GRID),
                      height=270, margin=dict(l=50,r=20,t=40,b=50),
                      legend=dict(orientation='h', yanchor='bottom', y=1.01,
                                  x=0, font=dict(color=TEXT, size=10)))
    return fig


def fig_time_bucket(results):
    obi = results[results['signal_name'] == 'obi_delta'].copy()
    if obi.empty:
        return _empty('Run OBI delta signal to see time-in-window analysis.')

    # Compute elapsed seconds from market open per ticker.
    # Use the minimum timestamp per ticker as the market open reference.
    # This works whether the data came from memory or a parquet cache.
    if 'elapsed_sec' not in obi.columns:
        market_opens = obi.groupby('ticker').apply(
            lambda g: g.index.min(), include_groups=False)
        obi = obi.copy()
        obi['elapsed_sec'] = obi.apply(
            lambda row: (row.name - market_opens[row['ticker']]).total_seconds(),
            axis=1)

    # Drop any NaN elapsed values before bucketing
    obi = obi.dropna(subset=['elapsed_sec'])
    if obi.empty:
        return _empty('Could not compute elapsed time from signal data.')

    obi['minute'] = (obi['elapsed_sec'] // 60).clip(0, 14).astype(int)
    b = (obi.groupby('minute')
            .agg(hit_rate=('hit', 'mean'), n=('hit', 'count'))
            .reset_index())
    fig = go.Figure(go.Bar(
        x=b['minute'], y=b['hit_rate'], opacity=0.85,
        marker_color=[GREEN if v >= 0.46 else RED if v < 0.35 else YELLOW
                      for v in b['hit_rate']],
        text=[f'{v:.0%}<br>n={n}' for v, n in zip(b['hit_rate'], b['n'])],
        textposition='outside', textfont=dict(color=TEXT, size=9),
        hovertemplate='Minute %{x}<br>Hit: %{y:.1%}<extra></extra>',
    ))
    fig.add_hline(y=0.5, line_dash='dash', line_color=TEXT, opacity=0.3)
    fig.update_layout(**BASE_LAYOUT,
                      xaxis=dict(title='Minute in 15-min window', dtick=1,
                                 tickmode='linear', gridcolor=GRID),
                      yaxis=dict(tickformat='.0%', title='Hit Rate',
                                 range=[0, max(.75, b['hit_rate'].max() + .1)],
                                 gridcolor=GRID),
                      height=260, margin=dict(l=50, r=20, t=16, b=50))
    return fig


# ── Layout helpers ────────────────────────────────────────────────────────────

def _card(children, extra=None):
    s = {'background': PANEL, 'padding': '12px', 'borderRadius': '6px',
         'border': f'1px solid {SPINE}'}
    if extra:
        s.update(extra)
    return html.Div(children, style=s)


def _label(text):
    return html.Label(text, style={'color': MUTED, 'fontSize': '11px',
                                   'marginBottom': '2px', 'display': 'block'})


def _title(text):
    return html.H4(text, style={'color': TEXT, 'fontSize': '13px',
                                'fontWeight': 'bold', 'margin': '0 0 8px 0'})


def build_layout(results, stats):
    all_sigs   = sorted(results['signal_name'].unique()) if not results.empty else []
    n_sessions = results['ticker'].nunique() if not results.empty else 0
    n_total    = len(results)

    return html.Div(
        style={'backgroundColor': BG, 'minHeight': '100vh',
               'fontFamily': 'Arial, sans-serif', 'padding': '16px 20px'},
        children=[

            # Header
            html.Div([
                html.Div([
                    html.H2('KXBTC15M Signal Dashboard',
                            style={'color': BLUE, 'margin': 0,
                                   'fontWeight': 'bold', 'fontSize': '22px'}),
                    html.Span(f'{n_sessions} sessions · {n_total:,} events loaded',
                              style={'color': MUTED, 'fontSize': '12px'}),
                ]),
                html.Div(id='status-bar', children='Ready.',
                         style={'color': GREEN, 'fontSize': '12px',
                                'alignSelf': 'center', 'fontStyle': 'italic'}),
            ], style={'display': 'flex', 'justifyContent': 'space-between',
                      'alignItems': 'flex-start', 'marginBottom': '14px',
                      'borderBottom': f'2px solid {SPINE}', 'paddingBottom': '10px'}),

            # Controls
            html.Div([

                # Signal checkboxes
                _card([
                    _title('Signals'),
                    dcc.Checklist(
                        id='signal-selector',
                        options=[{'label': f'  {SIGNAL_LABELS.get(s,s)}', 'value': s}
                                 for s in all_sigs],
                        value=all_sigs,
                        inputStyle={'marginRight': '6px', 'accentColor': BLUE},
                        labelStyle={'display': 'block', 'marginBottom': '5px',
                                    'color': TEXT, 'fontSize': '12px'},
                    ),
                ], {'minWidth': '200px'}),

                # OBI params
                _card([
                    _title('OBI Parameters'),
                    _label('Depth column'),
                    dcc.Dropdown(id='obi-col-dropdown',
                                 options=[{'label': c, 'value': c}
                                          for c in ['obi1','obi3','obi5','obi10','obi']],
                                 value='obi3',
                                 style={'backgroundColor': GRID, 'color': TEXT,
                                        'border': f'1px solid {SPINE}',
                                        'marginBottom': '10px', 'fontSize': '12px'}),
                    _label('Signal threshold  |d_obi| ≥'),
                    dcc.Slider(id='threshold-slider', min=0.10, max=0.80, step=0.05,
                               value=0.40,
                               marks={v: {'label': f'{v:.2f}',
                                          'style': {'color': MUTED, 'fontSize': '9px'}}
                                      for v in [0.10, 0.25, 0.40, 0.60, 0.80]},
                               tooltip={'placement': 'bottom'}),
                    html.Div(style={'height': '10px'}),
                    _label('Signal window (s)'),
                    dcc.Slider(id='sigwin-slider', min=0.10, max=1.0, step=0.05,
                               value=0.25,
                               marks={v: {'label': f'{v:.2f}s',
                                          'style': {'color': MUTED, 'fontSize': '9px'}}
                                      for v in [0.10, 0.25, 0.50, 1.0]},
                               tooltip={'placement': 'bottom'}),
                ], {'flex': '1', 'marginLeft': '10px'}),

                # Evaluation params
                _card([
                    _title('Evaluation Parameters'),
                    _label('Forward horizon (s)'),
                    dcc.Slider(id='horizon-slider', min=0.5, max=5.0, step=0.5,
                               value=1.0,
                               marks={v: {'label': f'{v:.1f}s',
                                          'style': {'color': MUTED, 'fontSize': '9px'}}
                                      for v in [0.5, 1.0, 2.0, 3.0, 5.0]},
                               tooltip={'placement': 'bottom'}),
                    html.Div(style={'height': '10px'}),
                    _label('Cooldown (s)'),
                    dcc.Slider(id='cooldown-slider', min=0.25, max=2.0, step=0.25,
                               value=0.5,
                               marks={v: {'label': f'{v:.2f}s',
                                          'style': {'color': MUTED, 'fontSize': '9px'}}
                                      for v in [0.25, 0.5, 1.0, 2.0]},
                               tooltip={'placement': 'bottom'}),
                    html.Div(style={'height': '10px'}),
                    _label('Micro-price threshold ($)'),
                    dcc.Slider(id='mp-threshold-slider', min=0.0005, max=0.003,
                               step=0.0005, value=0.001,
                               marks={v: {'label': f'${v:.4f}',
                                          'style': {'color': MUTED, 'fontSize': '9px'}}
                                      for v in [0.0005, 0.001, 0.002, 0.003]},
                               tooltip={'placement': 'bottom'}),
                    html.Div(style={'height': '10px'}),
                    _label('Time window start/end (s)'),
                    dcc.RangeSlider(id='timewin-slider', min=0, max=900, step=30,
                                    value=[60, 840],
                                    marks={v: {'label': f'{v}s',
                                               'style': {'color': MUTED, 'fontSize': '9px'}}
                                           for v in [0, 60, 300, 600, 840, 900]},
                                    tooltip={'placement': 'bottom'}),
                    html.Div(style={'height': '14px'}),
                    html.Button('↺  Re-run signals', id='rerun-btn', n_clicks=0,
                                style={'backgroundColor': BLUE, 'color': BG,
                                       'border': 'none', 'borderRadius': '4px',
                                       'padding': '9px 0', 'fontWeight': 'bold',
                                       'cursor': 'pointer', 'fontSize': '13px',
                                       'width': '100%'}),
                ], {'flex': '1', 'marginLeft': '10px'}),

            ], style={'display': 'flex', 'alignItems': 'flex-start',
                      'marginBottom': '12px'}),

            # Row 1: table + bars
            html.Div([
                _card([_title('Signal Summary'),
                       dcc.Graph(id='summary-table',
                                 config={'displayModeBar': False},
                                 figure=fig_table(results))],
                      {'flex': '1.7'}),
                _card([_title('Hit Rate'),
                       dcc.Graph(id='hitrate-bars',
                                 config={'displayModeBar': False},
                                 figure=fig_bars(results))],
                      {'flex': '1', 'marginLeft': '10px'}),
            ], style={'display': 'flex', 'marginBottom': '10px'}),

            # Row 2: per-session
            _card([_title('Hit Rate by Session'),
                   dcc.Graph(id='per-session-chart',
                             config={'displayModeBar': False},
                             figure=fig_per_session(stats, all_sigs))],
                  {'marginBottom': '10px'}),

            # Row 3: fwd dist + time bucket
            html.Div([
                _card([_title('Forward Move Distribution'),
                       dcc.Graph(id='fwd-dist', config={'displayModeBar': False},
                                 figure=fig_fwd_dist(results, all_sigs))],
                      {'flex': '1'}),
                _card([_title('Hit Rate by Minute in Window  (OBI delta)'),
                       dcc.Graph(id='time-bucket', config={'displayModeBar': False},
                                 figure=fig_time_bucket(results))],
                      {'flex': '1', 'marginLeft': '10px'}),
            ], style={'display': 'flex', 'marginBottom': '10px'}),

            # Polling interval — disabled until a re-run starts
            dcc.Interval(id='poll-interval', interval=2000,
                         n_intervals=0, disabled=True),
        ]
    )


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app, gstar_path):

    # Kick off background re-run
    @app.callback(
        Output('poll-interval', 'disabled'),
        Output('status-bar', 'children'),
        Output('status-bar', 'style'),
        Input('rerun-btn', 'n_clicks'),
        State('threshold-slider', 'value'),
        State('sigwin-slider', 'value'),
        State('horizon-slider', 'value'),
        State('cooldown-slider', 'value'),
        State('mp-threshold-slider', 'value'),
        State('timewin-slider', 'value'),
        State('obi-col-dropdown', 'value'),
        prevent_initial_call=True,
    )
    def kick_rerun(_, threshold, signal_win, horizon, cooldown,
                   mp_threshold, timewin, obi_col):
        if _STATE['running']:
            return True, 'Already running…', {'color': YELLOW, 'fontSize': '12px'}
        if not _STATE['sessions']:
            return True, 'No sessions in memory — restart with --data-dir', \
                   {'color': RED, 'fontSize': '12px'}

        start_sec = timewin[0] if timewin else 60
        end_sec   = timewin[1] if timewin else 840

        def _worker():
            _STATE['running'] = True
            try:
                sigs = _build_signals(threshold, horizon, signal_win, cooldown,
                                      mp_threshold, gstar_path,
                                      start_sec, end_sec, obi_col)
                res, stats = _run_signals(sigs, _STATE['sessions'])
                _STATE['results'] = res
                _STATE['stats']   = stats
                _STATE['progress'] = ''
                n = len(res)
                s = res['ticker'].nunique() if n else 0
                _STATE['last_msg'] = (
                    f'Done — {n:,} signals across {s} sessions.'
                    if n else 'No signals. Try lowering the threshold.')
            except Exception as e:
                _STATE['last_msg'] = f'Error: {e}'
            finally:
                _STATE['running'] = False

        threading.Thread(target=_worker, daemon=True).start()
        return False, 'Running…', {'color': YELLOW, 'fontSize': '12px',
                                   'fontStyle': 'italic'}

    # Poll for completion and refresh all charts
    @app.callback(
        Output('summary-table',     'figure'),
        Output('hitrate-bars',      'figure'),
        Output('per-session-chart', 'figure'),
        Output('fwd-dist',          'figure'),
        Output('time-bucket',       'figure'),
        Output('poll-interval',     'disabled',  allow_duplicate=True),
        Output('status-bar',        'children',  allow_duplicate=True),
        Output('status-bar',        'style',     allow_duplicate=True),
        Input('poll-interval',      'n_intervals'),
        State('signal-selector',    'value'),
        prevent_initial_call=True,
    )
    def poll_and_refresh(_, selected):
        if _STATE['running']:
            prog = _STATE.get('progress', '')
            msg  = f'Running… {prog}' if prog else 'Running…'
            return (no_update,)*5 + (False, msg,
                    {'color': YELLOW, 'fontSize': '12px', 'fontStyle': 'italic'})

        res   = _STATE['results']
        stats = _STATE['stats']
        msg   = _STATE['last_msg']
        sel   = selected or (list(res['signal_name'].unique())
                             if not res.empty else [])
        color = GREEN if 'Done' in msg else (RED if 'Error' in msg else TEXT)

        filt  = res[res['signal_name'].isin(sel)]   if not res.empty   else res
        sfilt = stats[stats['signal_name'].isin(sel)] if not stats.empty else stats

        return (fig_table(filt), fig_bars(filt),
                fig_per_session(sfilt, sel),
                fig_fwd_dist(filt, sel),
                fig_time_bucket(res),
                True,   # stop polling
                msg, {'color': color, 'fontSize': '12px'})

    # Signal checkbox filter — instant, no re-run needed
    @app.callback(
        Output('summary-table',     'figure', allow_duplicate=True),
        Output('hitrate-bars',      'figure', allow_duplicate=True),
        Output('per-session-chart', 'figure', allow_duplicate=True),
        Output('fwd-dist',          'figure', allow_duplicate=True),
        Input('signal-selector',    'value'),
        prevent_initial_call=True,
    )
    def filter_by_signal(selected):
        res   = _STATE['results']
        stats = _STATE['stats']
        sel   = selected or []
        filt  = res[res['signal_name'].isin(sel)]   if not res.empty   else res
        sfilt = stats[stats['signal_name'].isin(sel)] if not stats.empty else stats
        return (fig_table(filt), fig_bars(filt),
                fig_per_session(sfilt, sel), fig_fwd_dist(filt, sel))


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Interactive signal dashboard.')
    ap.add_argument('--data-dir', nargs='+', default=['../data'],
                    help='Directories with Kalshi+BTC CSV pairs '
                         '(default: ../data)')
    ap.add_argument('--gstar',       default='g_star.json')
    ap.add_argument('--recalibrate', action='store_true',
                    help='Re-estimate G* from data before running')
    ap.add_argument('--cache-dir',   default='../results/session_cache',
                    help='Session parquet cache dir (default: ../results/session_cache). '
                         'Pass --cache-dir "" to disable.')
    ap.add_argument('--results',     default='../results/results.parquet',
                    help='Cache path (default: ../results/results.parquet)')
    ap.add_argument('--port',        type=int, default=8050)
    args = ap.parse_args()

    from signal_runner import find_session_pairs, load_session, recalibrate

    # Load sessions
    print(f'[dashboard] Scanning {args.data_dir}…')
    pairs = find_session_pairs(args.data_dir)
    if not pairs:
        print('[dashboard] No session pairs found. Check --data-dir.')
        sys.exit(1)

    cache = Path(args.cache_dir) if args.cache_dir else None
    cached_count = 0
    if cache:
        cached_count = sum(1 for kp, _ in pairs
                           if (cache / f'{kp.stem}.parquet').exists())
    print(f'[dashboard] Loading {len(pairs)} sessions '
          f'({cached_count} from cache, '
          f'{len(pairs)-cached_count} to parse)…')
    for kp, bp in pairs:
        df = load_session(kp, bp, cache_dir=cache)
        if df is not None:
            _STATE['sessions'].append(df)
    print(f'[dashboard] {len(_STATE["sessions"])} sessions in memory.')

    # (Re-)calibrate
    if args.recalibrate:
        print('[dashboard] Recalibrating G*…')
        recalibrate(_STATE['sessions'], gstar_path=args.gstar)

    # Load or compute initial results
    rp = Path(args.results)
    sp = rp.with_name(rp.stem + '_stats.parquet')

    if rp.exists() and sp.exists() and not args.recalibrate:
        print(f'[dashboard] Loading cached results from {rp}')
        _STATE['results'] = pd.read_parquet(rp)
        _STATE['stats']   = pd.read_parquet(sp)
    else:
        print('[dashboard] Running initial signal evaluation…')
        sigs = _build_signals(0.40, 1.0, 0.25, 0.5, 0.001,
                              args.gstar, 60, 840, 'obi3')
        _STATE['results'], _STATE['stats'] = _run_signals(
            sigs, _STATE['sessions'])
        if not _STATE['results'].empty:
            _STATE['results'].to_parquet(rp)
            _STATE['stats'].to_parquet(sp)
            print(f'[dashboard] Cached to {rp}')

    n = len(_STATE['results'])
    s = _STATE['results']['ticker'].nunique() if n else 0
    print(f'[dashboard] {n:,} events across {s} sessions.')

    app = Dash(__name__, suppress_callback_exceptions=True)
    app.layout = build_layout(_STATE['results'], _STATE['stats'])
    register_callbacks(app, gstar_path=args.gstar)

    print(f'[dashboard] → http://127.0.0.1:{args.port}')
    app.run(debug=False, port=args.port)


if __name__ == '__main__':
    main()