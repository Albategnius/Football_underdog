import json, numpy as np
from model import *
from backtest import walk, metrics
XI, RIDGE = 0.0019, 1.0
pl, fx, pr = load()
test = [m for m in pl if m['season'] >= '2019-20']
preds = walk(pl, test, XI, RIDGE)

def rec(m, g, lh, la):
    h, d, a = outcome(g)
    fav_home = h >= a
    r = dict(season=m['season'], date=m['date'], home=m['home'], away=m['away'],
             ph=float(h), pd=float(d), pa=float(a), xh=lh, xa=la,
             fav=m['home'] if fav_home else m['away'], dog=m['away'] if fav_home else m['home'],
             pdog=float(a if fav_home else h), pfav=float(h if fav_home else a))
    if 'hg' in m:
        r['hg'], r['ag'] = m['hg'], m['ag']
        dogwin = (m['ag'] > m['hg']) if fav_home else (m['hg'] > m['ag'])
        favwin = (m['hg'] > m['ag']) if fav_home else (m['ag'] > m['hg'])
        r['res'] = 'dog' if dogwin else ('fav' if favwin else 'draw')
    return r

R = [rec(*p) for p in preds]
hist = [r for r in R if r['season'] < '2026-27']; cur = [r for r in R if r['season'] == '2026-27']
out = {}
H = hist
out['n_hist'] = len(H)
out['metrics_hist'] = dict(zip(['logloss', 'rps', 'acc'], metrics([p for p in preds if p[0]['season'] < '2026-27'])))
# baselines
hw = np.mean([r['hg'] > r['ag'] for r in H]); dr = np.mean([r['hg'] == r['ag'] for r in H]); aw = 1 - hw - dr
out['base'] = dict(home=hw, draw=dr, away=aw)
def rps_const(p):
    s = 0
    for r in H:
        o = 0 if r['hg'] > r['ag'] else (1 if r['hg'] == r['ag'] else 2); e = np.zeros(3); e[o] = 1
        s += ((np.cumsum(p) - np.cumsum(e))[:2] ** 2).sum() / 2
    return s / len(H)
out['rps_base'] = rps_const(np.array([hw, dr, aw]))
out['fav'] = {k: np.mean([r['res'] == k for r in H]) for k in ('fav', 'draw', 'dog')}
bins = [0, .15, .20, .25, .30, .35, 1]
cal = []
for lo, hi in zip(bins, bins[1:]):
    b = [r for r in H if lo <= r['pdog'] < hi]
    cal.append(dict(lo=lo, hi=hi, n=len(b), pred=float(np.mean([r['pdog'] for r in b])), act=float(np.mean([r['res'] == 'dog' for r in b]))))
out['cal'] = cal
for th in (.25, .30, .33):
    f = [r for r in H if r['pdog'] >= th]; nf = [r for r in H if r['pdog'] < th]
    out[f'flag{int(th*100)}'] = dict(n=len(f), share=len(f) / len(H), dogwin=np.mean([r['res'] == 'dog' for r in f]),
        notdogwin=np.mean([r['res'] == 'dog' for r in nf]), favwin=np.mean([r['res'] == 'fav' for r in f]),
        notfavwin=np.mean([r['res'] == 'fav' for r in nf]))
big = sorted([r for r in H if r['res'] == 'dog'], key=lambda r: r['pdog'])[:10]
out['big_hist'] = big
out['cur'] = cur
out['cur_summary'] = {k: sum(r['res'] == k for r in cur) for k in ('fav', 'draw', 'dog')}
out['cur_expected_dog'] = sum(r['pdog'] for r in cur)
# per-season upset counts
out['by_season'] = {s: dict(n=sum(r['season'] == s for r in H), dog=sum(r['season'] == s and r['res'] == 'dog' for r in H),
    exp=sum(r['pdog'] for r in H if r['season'] == s)) for s in sorted({r['season'] for r in H})}
# upcoming
mod = Model(XI, RIDGE).fit(pl, '2026-10-10')
up = []
for f in fx:
    if f['date'] > '2026-10-19': break
    g, lh, la = mod.grid(f['home'], f['away'], f['hprom'], f['aprom'])
    r = rec(f, g, lh, la); r['round'] = f['round']; r['time'] = f['time']
    # most likely upset score
    fav_home = r['fav'] == f['home']
    best = None
    for i in range(7):
        for j in range(7):
            dogw = (j > i) if fav_home else (i > j)
            if dogw and (best is None or g[i, j] > best[0]): best = (float(g[i, j]), i, j)
    r['upset_score'] = [best[1], best[2]]; r['p_upset_score'] = best[0]
    r['p_dog_nolose'] = r['pdog'] + r['pd']
    r['grid'] = np.round(g[:6, :6] * 100, 1).tolist()
    up.append(r)
out['upcoming'] = up
out['ratings'] = sorted([dict(team=t, att=float(mod.att[t]), de=float(mod.de[t])) for t in {f['home'] for f in fx}], key=lambda x: -(x['att'] + x['de']))
out['params'] = dict(mu=mod.mu, home=mod.home, rho=mod.rho, patt=mod.patt, pdef=mod.pdef)
json.dump(out, open('/home/claude/model/out.json', 'w'), default=float, indent=1)
print(json.dumps({k: out[k] for k in ['n_hist', 'metrics_hist', 'base', 'rps_base', 'fav', 'cal', 'flag25', 'flag30', 'flag33', 'cur_summary', 'cur_expected_dog', 'by_season']}, default=float, indent=1))
for r in big: print(r['date'], r['home'], r['hg'], '-', r['ag'], r['away'], 'dog', r['dog'], round(r['pdog'] * 100, 1))
print('--- current season biggest')
for r in sorted([r for r in cur if r['res'] == 'dog'], key=lambda r: r['pdog'])[:8]: print(r['date'], r['home'], r['hg'], '-', r['ag'], r['away'], r['dog'], round(r['pdog'] * 100, 1))
print('--- upcoming')
for r in sorted(up, key=lambda r: -r['pdog']): print(r['date'], r['round'], r['home'], 'v', r['away'], 'dog', r['dog'], round(r['pdog'] * 100), 'draw', round(r['pd'] * 100), 'score', r['upset_score'])
lv = {}
for name, lo, hi in (('red', .33, 1), ('amber', .25, .33), ('low', 0, .25)):
    b = [r for r in H if lo <= r['pdog'] < hi]
    lv[name] = dict(n=len(b), share=len(b) / len(H), dog=np.mean([r['res'] == 'dog' for r in b]),
                    draw=np.mean([r['res'] == 'draw' for r in b]), fav=np.mean([r['res'] == 'fav' for r in b]),
                    pred=np.mean([r['pdog'] for r in b]))
out['levels'] = lv
cl = [r for r in cur]
out['cur_levels'] = {name: dict(n=sum(lo <= r['pdog'] < hi for r in cl), dog=sum(lo <= r['pdog'] < hi and r['res'] == 'dog' for r in cl))
                     for name, lo, hi in (('red', .33, 1), ('amber', .25, .33), ('low', 0, .25))}
json.dump(out, open('/home/claude/model/out.json', 'w'), default=float, indent=1)
print(json.dumps(lv, indent=1, default=float)); print(out['cur_levels'])
print(sum(r['pdog'] for r in H)/len(H))
