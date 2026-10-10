# Upset Radar

Premier League fixtures ranked by the underdog's chance of winning outright, with weather-style warning levels.

**Live page:** https://albategnius.github.io/football_underdog/

| Warning | Underdog win chance | What happened in backtest (2,660 matches, 2019/20–2025/26) |
|---|---|---|
| Red | 33%+ | Underdog won 38%, favourite won 38% |
| Amber | 25–33% | Underdog won 31% |
| Low | under 25% | Underdog won 17% |

## How it works

- Dixon–Coles Poisson goals model: attack and defence rating per team, home advantage, low-score correction.
- Trained on Premier League seasons from 2015/16. Time decay of 0.0019 per day (a result from a year ago counts half).
- Promoted clubs get an estimated "promoted team" adjustment.
- Walk-forward backtest: each match predicted with a model refit on data from before that week.
- Backtest RPS 0.203 (league-average baseline 0.233), favourite picked correctly 52%.

## Rebuild

```bash
mkdir -p /home/claude/data
for s in 2015-16 2016-17 2017-18 2018-19 2019-20 2020-21 2021-22 2022-23 2023-24 2024-25 2025-26 2026-27; do
  curl -s -o /home/claude/data/en1_$s.json https://raw.githubusercontent.com/openfootball/football.json/master/$s/en.1.json
done
cd model && python3 run.py   # writes out.json
```

`model/template.html` is the page; the `__DATA__` placeholder is replaced with `out.json`.

Data: [openfootball](https://github.com/openfootball/football.json). For information and fun, not betting advice.
