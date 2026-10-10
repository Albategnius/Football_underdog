import sys, json, itertools, datetime as dt
import numpy as np
from model import *

def walk(played, targets, xi, ridge):
    """Predict each target match with a model fitted on data before its week."""
    out = []
    by_week = {}
    for m in targets:
        d = dt.date.fromisoformat(m['date']); wk = (d - dt.timedelta(days=d.weekday())).isoformat()
        by_week.setdefault(wk, []).append(m)
    for wk, ms in sorted(by_week.items()):
        mod = Model(xi, ridge).fit(played, wk)
        for m in ms:
            g, lh, la = mod.grid(m['home'], m['away'], m['hprom'], m['aprom'])
            out.append((m, g, lh, la))
    return out

def metrics(preds):
    ll = rps = 0; hit = 0
    for m, g, lh, la in preds:
        p = np.array(outcome(g)); o = 0 if m['hg'] > m['ag'] else (1 if m['hg'] == m['ag'] else 2)
        ll -= np.log(p[o]); e = np.zeros(3); e[o] = 1
        rps += ((np.cumsum(p) - np.cumsum(e))[:2] ** 2).sum() / 2
        hit += int(np.argmax(p) == o)
    n = len(preds); return ll / n, rps / n, hit / n

if __name__ == '__main__':
    pl, fx, pr = load()
    val = [m for m in pl if m['season'] in ('2017-18', '2018-19')]
    for xi, ridge in itertools.product([0.001, 0.0019, 0.003], [1, 3, 8, 15]):
        print(xi, ridge, np.round(metrics(walk(pl, val, xi, ridge)), 4), flush=True)
