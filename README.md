# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/budi2499arfo2/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/budi2499arfo2/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each Grand Theft Auto VI PlayStation 5 trophy's Exophase earn rate at a
case's target as five quantiles, by decomposing the rate into earners and the
tracker's pool and modelling Exophase's pool refreshes.

## The idea

Exophase prints a rate as `100 * earners / players_tracked`. Both move, in
opposite directions: the pool grows (diluting every rate), players progress
(adding earners). A forecast that fits the rate's logit alone sees one net number
and extrapolates it blindly. This program fits `log(earners)` and `log(pool)`
separately and divides them, so a pool refresh is a measured event rather than a
surprise.

The rules record one refresh exactly: the page sat at 638 players tracked from
2026-11-18T23:48Z through 2026-11-19T04:00Z, then refreshed by 04:30Z to 3,868,
with "every rate but the one at 100% lower than before". A read before that
refresh is on a different denominator than a read after it, so the fit starts at
the last change in `players_tracked`.

The published practice loss of the flat baseline (1.51) pins the net dilution of
that first refresh at about 4.2x. The program carries that as a one-sided prior
for a refresh the history has not shown yet, widening the lower tail rather than
the upper one.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Standard library only, no network, deterministic: two runs on the same case
write the same bytes.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
