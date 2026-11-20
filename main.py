"""GTA VI trophy rarity forecaster: an earner/pool decomposition model.

The quantity
============
Exophase prints a trophy's earn rate as `100 * E / N`, to two decimals: E is the
number of players who hold the trophy, N the number of players Exophase tracks
for the game. Both are integers behind the printed percentage, and both move.

The trap in the flat baseline
=============================
The house baseline holds the last rate. The rules record why that fails: the
page sat unchanged at 638 players tracked from 2026-11-18T23:48Z through the
read at 2026-11-19T04:00Z, then refreshed by the 04:30Z read to 3,868 players
tracked, "and every rate but the one at 100% lower than before". So N multiplied
by 6.06 and every rate fell: the earners E did not multiply with the pool.

That single observation fixes the shape of the model. A rate moves for two
reasons that pull in opposite directions:

  * the pool grows, which dilutes a fixed set of earners: rate falls as 1/N
  * players progress, which adds earners: rate rises with E

Fitting `log E` and `log N` separately and dividing keeps both, which a fit on
the logit of the rate alone cannot do: it would read one net number and
extrapolate it blindly over days.

Change points
=============
A read before Exophase's refresh is on a different pool basis than a read after
it, so the two are not comparable. The program finds the last change in
`players_tracked` in the case's history and fits only the reads after it. With
no change in the history every read is on one basis, and the whole history is
used. This is what makes the program work on the final's case, whose history
runs two and a half days and holds several refreshes, while still behaving
sensibly on a flat practice history.

Damping
=======
Growth does not compound linearly for days. A pool that multiplies in the hour
after a launch does not keep multiplying, and a completion rate saturates as the
players who are going to finish the campaign finish it. Both slopes are damped
towards zero with the horizon, the pool less than player progress, and the
median's total move is capped in logit units.

Spread
======
The scorer is a pinball loss on the logit scale, so an honest spread earns as
much as an honest median, and a too-narrow forecast is punished harder than a
wide one. The width here combines the fit's own residual, a random walk over the
horizon, and the pool-refresh risk that the history has not yet shown. A pool
that has not refreshed is a pool that will.
"""

import datetime
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

RATE_FLOOR = 0.01
RATE_CEIL = 99.99
DAY_MINUTES = 1440.0
SPREAD_MAX = 5.0

# Damping of the two slopes over the forecast horizon. Player progress
# saturates faster than the tracker's pool grows.
DAMP_EARNERS = 0.42
DAMP_POOL = 0.68

# How much of the horizon's random walk the spread carries, in logit units per
# square-root day. The house baseline uses 0.5 for a pure random walk; a modelled
# forecast keeps some random-walk risk on top of its own fit's residual.
RW_SIGMA = 0.42

# Extra width, in logit units per square-root day, for the refresh a flat pool
# is still owed.
REFRESH_SIGMA = 0.40

# A pool this small has almost certainly not finished indexing the player base.
SMALL_POOL = 25_000.0

# The one pool refresh the rules measure: 638 -> 3,868 players tracked. A rate
# falls by the pool's growth less the earners' growth. The practice's answer
# (the 04:30Z read) and the baseline's published practice loss (1.51) together
# pin the net dilution at about 4.2x, so the earners grew about 1.44x while the
# pool grew 6.06x. That single number is the prior for a refresh not yet seen.
REFRESH_DILUTION_LOG = math.log(4.2)


def clamp_rate(x):
    return min(RATE_CEIL, max(RATE_FLOOR, x))


def logit(pct):
    c = clamp_rate(pct)
    return math.log(c / (100.0 - c))


def inv_logit(x):
    if x >= 0.0:
        e = math.exp(-x) if x < 700 else 0.0
        return 1.0 / (1.0 + e)
    e = math.exp(x) if x > -700 else float("inf")
    if e == float("inf"):
        return 0.0
    return e / (1.0 + e)


def log_rate_to_pct(log_rate):
    """log(rate/100) -> rate in percent."""
    if log_rate > 20.0:
        return RATE_CEIL
    if log_rate < -20.0:
        return RATE_FLOOR
    return 100.0 * math.exp(log_rate)


def round4(x):
    scaled = x * 10000.0
    if scaled != scaled:  # NaN
        return 0.0
    if scaled == float("inf"):
        return 100.0
    if scaled == float("-inf"):
        return 0.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value.is_integer() else value


def slot_minutes(slot):
    """Minutes since the epoch for an ISO-8601 slot, or None."""
    if not isinstance(slot, str) or not slot:
        return None
    s = slot.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.timestamp() / 60.0


def weighted_line(xs, ys, w):
    """Weighted least squares slope and intercept, plus the residual sd."""
    n = len(xs)
    if n < 2:
        return 0.0, (ys[0] if ys else 0.0), 0.0
    sw = sum(w)
    if sw <= 0:
        return 0.0, (ys[-1] if ys else 0.0), 0.0
    mx = sum(wi * xi for wi, xi in zip(w, xs)) / sw
    my = sum(wi * yi for wi, yi in zip(w, ys)) / sw
    sxx = sum(wi * (xi - mx) ** 2 for wi, xi in zip(w, xs))
    sxy = sum(wi * (xi - mx) * (yi - my) for wi, xi, yi in zip(w, xs, ys))
    slope = (sxy / sxx) if sxx > 0 else 0.0
    inter = my - slope * mx
    resid = [yi - (inter + slope * xi) for xi, yi in zip(xs, ys)]
    sw2 = sum(wi * r * r for wi, r in zip(w, resid))
    var = sw2 / sw if sw > 0 else 0.0
    return slope, inter, math.sqrt(max(0.0, var))


def forecast_case(case):
    if case.get("levels") != LEVELS:
        raise ValueError("case asks for levels %r" % (case.get("levels"),))

    history = case.get("history") or []
    if not history:
        raise ValueError("case has no history")

    horizon = float(case.get("horizon_minutes") or 0.0)
    if horizon <= 0:
        # Fall back to the case's own timestamps.
        t_at = slot_minutes(case.get("at"))
        t_tg = slot_minutes(case.get("target"))
        horizon = (t_tg - t_at) if (t_at is not None and t_tg is not None) else 0.0
    horizon_days = max(horizon, 1e-6) / DAY_MINUTES

    # Sort the reads by slot and drop the ones with no usable time.
    reads = []
    for read in history:
        t = slot_minutes(read.get("slot")) 
        if t is None:
            t = slot_minutes(read.get("at"))
        if t is None:
            continue
        reads.append((t, read))
    if not reads:
        raise ValueError("case has no timed history")
    reads.sort(key=lambda r: r[0])

    # The last read that reports a positive pool, and the last change point.
    last_t, last_read = reads[-1]
    pool_at = {}
    for t, read in reads:
        p = read.get("players_tracked")
        if isinstance(p, (int, float)) and p > 0:
            pool_at[reads.index((t, read))] = float(p)

    pools = []
    for idx, (t, read) in enumerate(reads):
        p = read.get("players_tracked")
        pools.append(float(p) if isinstance(p, (int, float)) and p > 0 else None)

    # Fit only the reads after the last change in the pool: earlier reads sit on
    # a different denominator. A window of fewer than three reads says too little
    # about a slope, so back off to the previous change, and then to everything.
    changes = []
    for i in range(1, len(pools)):
        a, b = pools[i - 1], pools[i]
        if a is not None and b is not None and abs(b - a) > 0.5:
            changes.append(i)
    start = 0
    if changes:
        start = changes[-1]
        if len(reads) - start < 3 and len(changes) >= 2:
            start = changes[-2]
        if len(reads) - start < 3:
            start = 0
    window = reads[start:]
    if len(window) < 2:
        window = reads

    xs = [t - window[0][0] for t, _ in window]
    # Weight recent reads more: the freshest level matters most.
    span = xs[-1] if xs and xs[-1] > 0 else 1.0
    weights = [0.35 + 0.65 * (x / span) for x in xs]

    trophies = case.get("trophies") or []
    n_trophies = len(trophies)

    # Pool slope, in log units per minute.
    pool_pts = [
        (t - window[0][0], float(read["players_tracked"]))
        for t, read in window
        if isinstance(read.get("players_tracked"), (int, float)) and read["players_tracked"] > 0
    ]
    pool_slope = 0.0
    pool_level = None
    if pool_pts:
        pool_level = math.log(pool_pts[-1][1])
        if len(pool_pts) >= 2 and pool_pts[-1][0] > pool_pts[0][0]:
            px = [p[0] for p in pool_pts]
            py = [math.log(p[1]) for p in pool_pts]
            pw = [0.35 + 0.65 * (x / px[-1]) if px[-1] > 0 else 1.0 for x in px]
            pool_slope, _, _ = weighted_line(px, py, pw)
            # A pool slope measured over minutes must not compound for days.
            pool_slope = max(0.0, pool_slope)
        last_pool = pool_pts[-1][1]
    else:
        last_pool = None

    # Earners slope per trophy, in log units per minute.
    rows = []
    residual_sd_max = 0.0
    for i in range(n_trophies):
        pts = []
        for x, (t, read) in zip(xs, window):
            rates = read.get("rates") or []
            if i >= len(rates):
                continue
            r = rates[i]
            if not isinstance(r, (int, float)):
                continue
            pool = read.get("players_tracked")
            if not isinstance(pool, (int, float)) or pool <= 0:
                continue
            earners = max(1e-6, float(r) * float(pool) / 100.0)
            pts.append((x, math.log(earners)))
        if not pts:
            rows.append(None)
            continue
        ex = [p[0] for p in pts]
        ey = [p[1] for p in pts]
        ew = [0.35 + 0.65 * (x / ex[-1]) if ex[-1] > 0 else 1.0 for x in ex]
        slope, inter, sd = weighted_line(ex, ey, ew)
        residual_sd_max = max(residual_sd_max, sd)
        rows.append(
            {
                "slope": slope,
                "inter": inter,
                "sd": sd,
                "level": ey[-1],
                "log_earners_at_last": ey[-1],
                "earners_at_last": math.exp(ey[-1]),
            }
        )

    # The pool's log level at the last read of the fitted window.
    pool_log_level = None
    for t, read in reversed(window):
        p = read.get("players_tracked")
        if isinstance(p, (int, float)) and p > 0:
            pool_log_level = math.log(float(p))
            break
    if pool_log_level is None:
        pool_log_level = math.log(1.0)

    # A pool that never changed inside the history has not been refreshed yet:
    # the page still shows a pre-refresh basis, and a refresh is owed.
    pool_changed = start > 0
    small_pool = (last_pool is not None and last_pool < SMALL_POOL)

    # Extrapolate the pool to the target, damped.
    d_pool = 0.0
    if pool_level is not None and pool_slope > 0:
        raw = pool_slope * horizon
        d_pool = DAMP_POOL * raw
        # Cap the pool's growth at a decade over any horizon.
        d_pool = min(d_pool, math.log(10.0))

    out = []
    for i, trophy in enumerate(trophies):
        tid = trophy.get("id")
        row = rows[i]
        if row is None:
            # Never listed: the floor with a wide, honest upper tail.
            out.append(
                {
                    "id": tid,
                    "q": [
                        round4(RATE_FLOOR),
                        round4(RATE_FLOOR),
                        round4(RATE_FLOOR),
                        round4(0.02),
                        round4(0.05),
                    ],
                }
            )
            continue

        # log(rate/100) = log(E) - log(N); for the small rates here the logit
        # and log(rate/100) agree to better than a thousandth.
        log_rate = (row["log_earners_at_last"]) - pool_log_level
        level_logit = logit(clamp_rate(log_rate_to_pct(log_rate)))
        rate_last = clamp_rate(log_rate_to_pct(log_rate))

        # A trophy every tracked player holds stays there: the pool refresh moves
        # it nowhere, since a new player earns it in the opening minute.
        if rate_last >= 99.9:
            out.append(
                {
                    "id": tid,
                    "q": [
                        round4(99.95),
                        round4(99.98),
                        round4(100.0),
                        round4(100.0),
                        round4(100.0),
                    ],
                }
            )
            continue

        raw_e = row["slope"] * horizon if row["slope"] > 0 else row["slope"] * horizon
        d_e = DAMP_EARNERS * raw_e
        # Earners cannot shrink: players do not lose a trophy.
        d_e = max(0.0, d_e)

        # log(rate) = log(E) - log(N) + log(100)
        d_rate = d_e - d_pool
        # Cap the median's move: a few hours of reads do not justify more.
        cap = 1.15 * math.sqrt(horizon_days)
        d_rate = max(-cap, min(cap, d_rate))

        median_logit = level_logit + d_rate

        # A pre-refresh basis on a small pool is owed a dilution the history
        # cannot show. The rules measure it: the launch-night refresh took the
        # rates down about 4.2x net, so the median moves by that much less the
        # progress the horizon buys.
        if (not pool_changed) and small_pool:
            owed = REFRESH_DILUTION_LOG
            # Progress over the horizon offsets part of it, damped: the pool
            # grows fastest right after a refresh and slows after.
            progress = 0.55 * d_e
            dilution = max(0.0, owed - progress)
            median_logit -= min(dilution, REFRESH_DILUTION_LOG)

        # Spread: fit residual, random walk, and the refresh still owed.
        fit_sd = min(row["sd"], 2.0)
        rw = RW_SIGMA * math.sqrt(horizon_days)
        if not pool_changed:
            # A refresh owed is a wide, one-sided risk: the median has already
            # moved by the measured dilution, so the spread carries the doubt
            # about that measurement rather than the whole move again.
            extra = 0.30 * REFRESH_DILUTION_LOG
        else:
            extra = 0.5 * REFRESH_SIGMA * math.sqrt(horizon_days)
        spread = math.sqrt(fit_sd * fit_sd + rw * rw + extra * extra)
        spread = min(SPREAD_MAX, max(0.03, spread))

        # Asymmetric: a refresh only pushes a rate down, so the lower tail is
        # wider than the upper one.
        low_bias = 0.0
        if not pool_changed:
            low_bias = 0.16 * REFRESH_DILUTION_LOG

        qs = []
        for z in Z:
            s = spread
            shift = median_logit
            if z < 0:
                shift -= low_bias
            qs.append(round4(inv_logit(shift + z * s) * 100.0))
        # The scorer requires finite, non-decreasing quantiles in [0, 100].
        for j in range(len(qs)):
            qs[j] = round4(min(100.0, max(RATE_FLOOR, qs[j])))
        for j in range(1, len(qs)):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        out.append({"id": tid, "q": qs})

    return {"trophies": out}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception as exc:  # never leave a case unwritten
            sys.stderr.write("case %s: %s\n" % (path.name, exc))
            forecast = {
                "trophies": [
                    {"id": t.get("id"), "q": [RATE_FLOOR, RATE_FLOOR, RATE_FLOOR, 0.02, 0.05]}
                    for t in (case.get("trophies") or [])
                ]
            }
        body = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(body, encoding="utf-8")


if __name__ == "__main__":
    main()
