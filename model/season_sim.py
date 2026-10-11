"""Simulate the rest of a Premier League season from the current table.

Each simulation draws a small random shift in every team's attack and defence
rating (sigma) so the uncertainty about how good teams really are is carried
into the final table, not just match-day luck.
"""
import json, sys
import numpy as np
from scipy.stats import poisson
from model import Model, load, MAXG

XI, RIDGE = 0.0019, 1.0
G = np.arange(MAXG + 1)


def simulate(mod, done, todo, N, sigma, rng):
    teams = sorted({m['home'] for m in done + todo} | {m['away'] for m in done + todo})
    T = {t: i for i, t in enumerate(teams)}; n = len(teams)
    pts0 = np.zeros(n); gf0 = np.zeros(n); ga0 = np.zeros(n)
    for m in done:
        h, a = T[m['home']], T[m['away']]
        gf0[h] += m['hg']; ga0[h] += m['ag']; gf0[a] += m['ag']; ga0[a] += m['hg']
        if m['hg'] > m['ag']: pts0[h] += 3
        elif m['hg'] < m['ag']: pts0[a] += 3
        else: pts0[h] += 1; pts0[a] += 1
    pts = np.tile(pts0, (N, 1)); gf = np.tile(gf0, (N, 1)); ga = np.tile(ga0, (N, 1))
    ea = rng.normal(0, sigma, (N, n)); ed = rng.normal(0, sigma, (N, n))
    r = mod.rho
    for f in todo:
        h, a = T[f['home']], T[f['away']]
        lh0, la0 = mod.rates(f['home'], f['away'], f['hprom'], f['aprom'])
        lh = lh0 * np.exp(ea[:, h] - ed[:, a]); la = la0 * np.exp(ea[:, a] - ed[:, h])
        g = poisson.pmf(G[None, :, None], lh[:, None, None]) * poisson.pmf(G[None, None, :], la[:, None, None])
        g[:, 0, 0] *= 1 - lh * la * r; g[:, 0, 1] *= 1 + lh * r; g[:, 1, 0] *= 1 + la * r; g[:, 1, 1] *= 1 - r
        c = g.reshape(N, -1).cumsum(1); c /= c[:, -1:]
        k = (c < rng.random((N, 1))).sum(1)
        hg, ag = k // (MAXG + 1), k % (MAXG + 1)
        gf[:, h] += hg; ga[:, h] += ag; gf[:, a] += ag; ga[:, a] += hg
        pts[:, h] += np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
        pts[:, a] += np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
    key = pts * 1e6 + (gf - ga + 500) * 1e3 + gf + rng.random((N, n)) * 0.5
    order = np.argsort(-key, axis=1)
    pos = np.empty_like(order); pos[np.arange(N)[:, None], order] = np.arange(1, n + 1)
    return teams, T, pts0, gf0, ga0, pts, pos


def season_matches(played, fixtures, season):
    return ([m for m in played if m['season'] == season], [f for f in fixtures if f['season'] == season])


def backtest(sigmas, N=2000):
    """From the end of matchday 5 in each past season, project the final table and score it."""
    played, fixtures, _ = load()
    seasons = ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26']
    res = {s: dict(cover=[], err=[], rel_brier=[], t4_brier=[], champ_p=[]) for s in sigmas}
    for season in seasons:
        ms = sorted([m for m in played if m['season'] == season], key=lambda m: m['date'])
        asof = ms[50]['date']  # first date after 50 matches
        done = [m for m in ms if m['date'] < asof]; todo = [m for m in ms if m['date'] >= asof]
        mod = Model(XI, RIDGE).fit(played, asof)
        # actual final table
        teams = sorted({m['home'] for m in ms}); A = {t: 0 for t in teams}; GD = {t: 0 for t in teams}; GF = {t: 0 for t in teams}
        for m in ms:
            GD[m['home']] += m['hg'] - m['ag']; GD[m['away']] += m['ag'] - m['hg']; GF[m['home']] += m['hg']; GF[m['away']] += m['ag']
            if m['hg'] > m['ag']: A[m['home']] += 3
            elif m['hg'] < m['ag']: A[m['away']] += 3
            else: A[m['home']] += 1; A[m['away']] += 1
        fin = sorted(teams, key=lambda t: (-A[t], -GD[t], -GF[t])); fpos = {t: i + 1 for i, t in enumerate(fin)}
        for s in sigmas:
            tm, T, _, _, _, pts, pos = simulate(mod, done, todo, N, s, np.random.default_rng(7))
            for t in tm:
                i = T[t]; lo, hi = np.percentile(pts[:, i], [10, 90])
                res[s]['cover'].append(lo <= A[t] <= hi); res[s]['err'].append(pts[:, i].mean() - A[t])
                pr = (pos[:, i] >= 18).mean(); res[s]['rel_brier'].append((pr - (fpos[t] >= 18)) ** 2)
                p4 = (pos[:, i] <= 4).mean(); res[s]['t4_brier'].append((p4 - (fpos[t] <= 4)) ** 2)
            res[s]['champ_p'].append(float((pos[:, T[fin[0]]] == 1).mean()))
        print(season, asof, 'champion', fin[0], {s: round(res[s]['champ_p'][-1], 2) for s in sigmas}, flush=True)
    summ = {}
    for s in sigmas:
        r = res[s]
        summ[s] = dict(cover80=float(np.mean(r['cover'])), mae=float(np.mean(np.abs(r['err']))),
                       rel_brier=float(np.mean(r['rel_brier'])), top4_brier=float(np.mean(r['t4_brier'])),
                       champ_p=r['champ_p'])
        print(s, {k: (round(v, 4) if isinstance(v, float) else [round(x, 2) for x in v]) for k, v in summ[s].items()})
    return summ


def run(sigma, N=10000, asof='2026-10-10'):
    played, fixtures, _ = load()
    done, todo = season_matches(played, fixtures, '2026-27')
    mod = Model(XI, RIDGE).fit(played, asof)
    teams, T, pts0, gf0, ga0, pts, pos = simulate(mod, done, todo, N, sigma, np.random.default_rng(2026))
    out = []
    for t in teams:
        i = T[t]; p = pos[:, i]
        out.append(dict(team=t, played=sum(t in (m['home'], m['away']) for m in done), pts=int(pts0[i]),
                        gd=int(gf0[i] - ga0[i]), gf=int(gf0[i]),
                        title=float((p == 1).mean()), top4=float((p <= 4).mean()), releg=float((p >= 18).mean()),
                        xpts=float(pts[:, i].mean()), pts_lo=float(np.percentile(pts[:, i], 10)),
                        pts_hi=float(np.percentile(pts[:, i], 90)), avgpos=float(p.mean()),
                        pos_hist=np.bincount(p, minlength=21)[1:].tolist()))
    table = sorted(out, key=lambda r: (-r['pts'], -r['gd'], -r['gf']))
    for k, r in enumerate(table, 1): r['now_pos'] = k
    n = len(teams)
    summary = dict(n=N, sigma=sigma, asof=asof, played=len(done), left=len(todo),
                   safety_median=float(np.median(np.sort(pts, 1)[:, n - 17])),
                   champ_median=float(np.median(np.sort(pts, 1)[:, -1])),
                   checks=dict(title=sum(r['title'] for r in out), top4=sum(r['top4'] for r in out), releg=sum(r['releg'] for r in out)))
    return dict(summary=summary, table=table)


if __name__ == '__main__':
    if sys.argv[1:] == ['backtest']:
        json.dump(backtest([0.0, 0.1, 0.15, 0.2]), open('/home/claude/model/season_backtest.json', 'w'), indent=1)
    else:
        sigma = float(sys.argv[1]) if sys.argv[1:] else 0.1
        o = run(sigma); json.dump(o, open('/home/claude/model/season.json', 'w'), indent=1)
        print(json.dumps(o['summary'], indent=1))
        print(f"{'#':>2} {'team':15} {'Pts':>3} {'GD':>3} | {'xPts':>5} {'10-90':>9} {'title':>6} {'top4':>6} {'rel':>6}")
        for r in o['table']:
            print(f"{r['now_pos']:>2} {r['team']:15} {r['pts']:>3} {r['gd']:>3} | {r['xpts']:5.1f} {r['pts_lo']:4.0f}-{r['pts_hi']:<4.0f} {r['title']*100:5.1f}% {r['top4']*100:5.1f}% {r['releg']*100:5.1f}%")
