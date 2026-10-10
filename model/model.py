"""Dixon-Coles style Poisson model with time decay and a promoted-team effect."""
import json, glob, os, math, datetime as dt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

DATA = '/home/claude/data'
MAXG = 10

SHORT = {
    'AFC Bournemouth': 'Bournemouth', 'Brighton & Hove Albion': 'Brighton',
    'Manchester City': 'Man City', 'Manchester United': 'Man United',
    'Tottenham Hotspur': 'Tottenham', 'Wolverhampton Wanderers': 'Wolves',
    'Nottingham Forest': "Nott'm Forest", 'Newcastle United': 'Newcastle',
    'West Ham United': 'West Ham', 'West Bromwich Albion': 'West Brom',
    'Leeds United': 'Leeds', 'Leicester City': 'Leicester', 'Ipswich Town': 'Ipswich',
    'Coventry City': 'Coventry', 'Hull City': 'Hull', 'Sheffield United': 'Sheffield Utd',
    'Huddersfield Town': 'Huddersfield', 'Norwich City': 'Norwich', 'Luton Town': 'Luton',
    'Cardiff City': 'Cardiff', 'Swansea City': 'Swansea', 'Stoke City': 'Stoke',
}


def norm(n):
    for suf in (' AFC', ' FC'):
        if n.endswith(suf):
            n = n[: -len(suf)]
    return SHORT.get(n, n)


def ft(x):
    s = x.get('score')
    if isinstance(s, dict):
        s = s.get('ft')
    if isinstance(s, list) and len(s) == 2:
        return s
    return None


def load():
    """Return played matches and unplayed fixtures, each a list of dicts."""
    played, fixtures = [], []
    seasons = sorted(glob.glob(f'{DATA}/en1_*.json'))
    teams_by_season = {}
    for f in seasons:
        season = os.path.basename(f)[4:11]
        ms = json.load(open(f))['matches']
        teams_by_season[season] = {norm(x['team1']) for x in ms}
        for x in ms:
            rec = dict(season=season, date=x['date'], round=x.get('round', ''),
                       home=norm(x['team1']), away=norm(x['team2']), time=x.get('time', ''))
            s = ft(x)
            if s:
                rec['hg'], rec['ag'] = int(s[0]), int(s[1])
                played.append(rec)
            else:
                fixtures.append(rec)
    keys = sorted(teams_by_season)
    promoted = {}
    for i, k in enumerate(keys):
        prev = teams_by_season[keys[i - 1]] if i else set()
        promoted[k] = {t for t in teams_by_season[k] if i and t not in prev}
    for r in played + fixtures:
        r['hprom'] = int(r['home'] in promoted[r['season']])
        r['aprom'] = int(r['away'] in promoted[r['season']])
    played.sort(key=lambda r: r['date'])
    fixtures.sort(key=lambda r: (r['date'], r['time']))
    return played, fixtures, promoted


def days(d):
    return dt.date.fromisoformat(d).toordinal()


def tau(hg, ag, lh, la, rho):
    t = np.ones_like(lh)
    m00 = (hg == 0) & (ag == 0); t[m00] = 1 - lh[m00] * la[m00] * rho
    m01 = (hg == 0) & (ag == 1); t[m01] = 1 + lh[m01] * rho
    m10 = (hg == 1) & (ag == 0); t[m10] = 1 + la[m10] * rho
    m11 = (hg == 1) & (ag == 1); t[m11] = 1 - rho
    return t


class Model:
    def __init__(self, xi=0.0019, ridge=2.0, window_days=1500):
        self.xi, self.ridge, self.window = xi, ridge, window_days

    def fit(self, matches, asof):
        """Fit using matches strictly before `asof` (ISO date)."""
        t0 = days(asof)
        ms = [m for m in matches if m['date'] < asof and t0 - days(m['date']) <= self.window]
        teams = sorted({m['home'] for m in ms} | {m['away'] for m in ms})
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        H = np.array([idx[m['home']] for m in ms]); A = np.array([idx[m['away']] for m in ms])
        hg = np.array([m['hg'] for m in ms]); ag = np.array([m['ag'] for m in ms])
        hp = np.array([m['hprom'] for m in ms]); ap = np.array([m['aprom'] for m in ms])
        w = np.exp(-self.xi * (t0 - np.array([days(m['date']) for m in ms])))
        lgh, lga = np.array([math.lgamma(g + 1) for g in hg]), np.array([math.lgamma(g + 1) for g in ag])
        ridge = self.ridge

        def unpack(p):
            return p[0], p[1], p[2], p[3], p[4], p[5:5 + n], p[5 + n:5 + 2 * n]

        def nll(p):
            mu, home, rho, patt, pdef, att, de = unpack(p)
            llh = mu + home + att[H] - de[A] + patt * hp - pdef * ap
            lla = mu + att[A] - de[H] + patt * ap - pdef * hp
            lh, la = np.exp(llh), np.exp(lla)
            ll = hg * llh - lh - lgh + ag * lla - la - lga
            t = tau(hg, ag, lh, la, rho)
            ll = ll + np.log(np.clip(t, 1e-10, None))
            m00 = (hg == 0) & (ag == 0); m01 = (hg == 0) & (ag == 1); m10 = (hg == 1) & (ag == 0)
            obj = -(w * ll).sum() + ridge * ((att ** 2).sum() + (de ** 2).sum())
            # gradient (ignore tau's dependence on lambdas: small, rho handled numerically)
            tc = np.clip(t, 1e-10, None)
            dlh = np.zeros_like(lh); dla = np.zeros_like(la)
            dlh[m00] = -la[m00] * rho * lh[m00]; dla[m00] = -lh[m00] * rho * la[m00]
            dlh[m01] = rho * lh[m01]; dla[m10] = rho * la[m10]
            rh = w * (hg - lh + dlh / tc); ra = w * (ag - la + dla / tc)
            g = np.zeros_like(p)
            g[0] = -(rh.sum() + ra.sum()); g[1] = -rh.sum()
            # d/d rho
            dt_ = np.zeros_like(lh)
            m00 = (hg == 0) & (ag == 0); dt_[m00] = -lh[m00] * la[m00]
            m01 = (hg == 0) & (ag == 1); dt_[m01] = lh[m01]
            m10 = (hg == 1) & (ag == 0); dt_[m10] = la[m10]
            m11 = (hg == 1) & (ag == 1); dt_[m11] = -1
            g[2] = -(w * dt_ / np.clip(t, 1e-10, None)).sum()
            g[3] = -((rh * hp).sum() + (ra * ap).sum())
            g[4] = ((rh * ap).sum() + (ra * hp).sum())
            gatt = -(np.bincount(H, rh, n) + np.bincount(A, ra, n)) + 2 * ridge * att
            gdef = (np.bincount(A, rh, n) + np.bincount(H, ra, n)) + 2 * ridge * de
            g[5:5 + n] = gatt; g[5 + n:] = gdef
            return obj, g

        p0 = np.zeros(5 + 2 * n); p0[0] = 0.3; p0[1] = 0.2
        bounds = [(None, None), (None, None), (-0.2, 0.2)] + [(None, None)] * (2 + 2 * n)
        res = minimize(nll, p0, jac=True, method='L-BFGS-B', bounds=bounds,
                       options=dict(maxiter=5000, ftol=1e-12, gtol=1e-8))
        self.p = res.x; self.teams = teams; self.idx = idx; self.n = n
        mu, home, rho, patt, pdef, att, de = unpack(res.x)
        self.mu, self.home, self.rho, self.patt, self.pdef = mu, home, rho, patt, pdef
        self.att = dict(zip(teams, att)); self.de = dict(zip(teams, de))
        return self

    def rates(self, home, away, hprom, aprom):
        ah, dh = self.att.get(home, 0.0), self.de.get(home, 0.0)
        aa, da = self.att.get(away, 0.0), self.de.get(away, 0.0)
        lh = math.exp(self.mu + self.home + ah - da + self.patt * hprom - self.pdef * aprom)
        la = math.exp(self.mu + aa - dh + self.patt * aprom - self.pdef * hprom)
        return lh, la

    def grid(self, home, away, hprom=0, aprom=0):
        lh, la = self.rates(home, away, hprom, aprom)
        g = np.outer(poisson.pmf(np.arange(MAXG + 1), lh), poisson.pmf(np.arange(MAXG + 1), la))
        r = self.rho
        g[0, 0] *= 1 - lh * la * r; g[0, 1] *= 1 + lh * r; g[1, 0] *= 1 + la * r; g[1, 1] *= 1 - r
        g /= g.sum()
        return g, lh, la


def outcome(g):
    hw = np.tril(g, -1).sum(); d = np.trace(g); aw = np.triu(g, 1).sum()
    return hw, d, aw
