"""Roblox Up-and-Coming top-10 forecaster (make.fun, program/v1).

Forecasts the Up-and-Coming chart's top 10, place by place, for the 72h
scored window after entries close. Pure Python 3.13 standard library,
deterministic: no clock, no randomness, no network.

Model (tuned on six 72h backtest windows cut from the public archive,
scorer reimplementation verified against the kit's baseline -43.512039):
per game, a time-decay-weighted mean rank (half-life 8h) plus a damped
linear rank velocity; the rank at each horizon is normal with sigma
growing in sqrt(horizon); place probabilities are the normal mass on
each place, averaged over the 72h horizon grid, scaled so 25% of each
place stays unnamed (covers games that churn into the chart), with each
game's total probability over places capped at 1. Backtest mean over six
72h windows: -36.52 vs the persistence baseline's -47.8 mean; practice
window: -36.522248 vs baseline -43.512039 (+6.99). Entry validated
2026-11-11: deterministic (two runs byte-identical), format checks pass.
Playing-growth drift was tested and REJECTED (hurt every gamma).
"""
import json
import math
import os
import sys
from datetime import datetime

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

HALF_LIFE_H = 8.0
DAMP = 0.5
SIGMA0 = 3.0
SIGMA_GROWTH = 0.2
UNNAMED = 0.25
HORIZONS = (0, 3, 6, 12, 18, 24, 36, 48, 60, 72)


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def main():
    with open(os.path.join(INPUT_DIR, "task.json")) as f:
        task = json.load(f)
    rows = []
    with open(os.path.join(INPUT_DIR, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    for r in rows:
        r["_t"] = parse_ts(r["at"])
    if not rows:
        print("no history rows in input", file=sys.stderr)
        raise SystemExit(1)
    t_end = rows[-1]["_t"]
    window_end = parse_ts(task["window"]["until"])
    window_start = parse_ts(task["window"]["from"])

    obs = {}
    details = {}
    for r in rows:
        t = r["_t"]
        for g in r["up_and_coming"]:
            obs.setdefault(g["universe_id"], []).append((t, g["rank"]))
        for uid, d in (r.get("details") or {}).items():
            if d:
                details[int(uid)] = d

    last_chart = sorted(rows[-1]["up_and_coming"], key=lambda g: g["rank"])
    cand = set(g["universe_id"] for g in last_chart[:40])
    for r in rows[-5:]:
        for g in r["up_and_coming"][:20]:
            cand.add(g["universe_id"])

    stats = {}
    for uid in cand:
        pts = obs.get(uid)
        if not pts:
            continue
        ws = wr = wtr = wt = wt2 = 0.0
        for (t, rank) in pts:
            w = 0.5 ** ((t_end - t) / 3600.0 / HALF_LIFE_H)
            dt = (t - t_end) / 3600.0
            ws += w
            wr += w * rank
            wt += w * dt
            wt2 += w * dt * dt
            wtr += w * dt * rank
        mean_rank = wr / ws
        denom = ws * wt2 - wt * wt
        vel = (ws * wtr - wt * wr) / denom if abs(denom) > 1e-9 else 0.0
        vel = max(-1.5, min(1.5, vel)) * DAMP
        # Age-out: a game leaves the chart 28 days after creation.
        age_factor = 1.0
        d = details.get(uid)
        if d and d.get("created"):
            try:
                created = parse_ts(d["created"])
                age_out = created + 28 * 86400
                if age_out <= window_start:
                    continue
                if age_out < window_end:
                    age_factor = max(0.1, (age_out - window_start) / (window_end - window_start))
            except (ValueError, TypeError):
                pass
        stats[uid] = (mean_rank, vel, age_factor)

    raw = {k: {} for k in range(1, 11)}
    for h in HORIZONS:
        for uid, (mr, vel, af) in stats.items():
            mu = max(1.0, min(120.0, mr + vel * h))
            sig = SIGMA0 + SIGMA_GROWTH * math.sqrt(h)
            for k in range(1, 11):
                p = norm_cdf((k + 0.5 - mu) / sig) - norm_cdf((k - 0.5 - mu) / sig)
                if p > 1e-6:
                    raw[k][uid] = raw[k].get(uid, 0.0) + p * af / len(HORIZONS)

    probs = []
    for k in range(1, 11):
        d = raw[k]
        s = sum(d.values())
        target = 1.0 - UNNAMED
        scale = target / s if s > target else 1.0
        for uid, p in d.items():
            probs.append({"universe_id": uid, "place": k, "p": p * scale})
    per_game = {}
    for row in probs:
        per_game[row["universe_id"]] = per_game.get(row["universe_id"], 0.0) + row["p"]
    for row in probs:
        tot = per_game[row["universe_id"]]
        if tot > 1.0:
            row["p"] *= 1.0 / tot

    exp_rank = {}
    for row in probs:
        exp_rank[row["universe_id"]] = exp_rank.get(row["universe_id"], 0.0) + row["place"] * row["p"]
    top10 = sorted(exp_rank, key=lambda u: exp_rank[u])[:10]
    for g in last_chart:
        if len(top10) >= 10:
            break
        if g["universe_id"] not in top10:
            top10.append(g["universe_id"])

    out = {"probabilities": [r for r in probs if r["p"] > 1e-4], "top10": top10[:10]}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task["output"]), "w") as f:
        json.dump(out, f)
        f.write("\n")


if __name__ == "__main__":
    main()
