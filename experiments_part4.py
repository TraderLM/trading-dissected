"""
Trading Dissected, Part 4: two reproducible experiments.

Experiment 1 (Part 4a): the touch-fill illusion.
A price path is simulated at one-minute resolution and aggregated to hourly
candles. A mean reversion rule places a limit buy 2*ATR below the previous
close and exits at the 20-period SMA (or after 48 hours). The same rule is
backtested twice on the same path:
  (a) optimistic: fill at the limit whenever the candle low touches it,
  (b) realistic:  fill only when the minute-level price trades through the
      level by 0.1%, on the reasoning that a resting order sits behind the
      queue that already existed at that price.
The simulation is used deliberately: only there is the strategy's true
edge known to exist, because the process mean-reverts by construction, so
the gap between the two runs is attributable to the fill assumption alone.
The trade-through margin stands in for the queue a resting order waits
behind; it is a stipulated convention, not an order book model.

Experiment 2 (Part 4b): the expected quality of the best worthless strategy.
100 configurations, each taking a random long or short position every day,
are applied to two years of zero-drift daily returns, so none has edge by
construction. The distribution of resulting Sharpe ratios, and in particular
the maximum, shows what selection alone produces from noise.

Both experiments use fixed seeds. Rerun with `python experiments_part4.py`.
The run ends with a robustness check that repeats Experiment 1 across 25
independent seeds and three trade-through margins, confirming that the
effect is a property of the mechanism rather than of one particular path.
"""

import numpy as np


# Experiment 1: touch fill vs trade-through fill

def simulate_minute_path(n_days=730, seed=42):
    """Log price with a slowly drifting fair value and mild mean reversion
    around it, fat-tailed minute innovations. Returns minute prices."""
    r = np.random.default_rng(seed)
    n = n_days * 24 * 60
    # fair value: random walk, ~0.30% hourly vol
    sigma_m = 0.0030 / np.sqrt(60)
    m = np.cumsum(sigma_m * r.standard_normal(n))
    # price: mean-reverts to fair value, half-life ~4 hours,
    # ~0.50% hourly idiosyncratic vol, Student-t(4) innovations
    kappa = np.log(2) / 240.0
    sigma_p = 0.0050 / np.sqrt(60)
    t_scale = np.sqrt(2.0)  # var of t(4) is 2, rescale to unit variance
    eps = r.standard_t(4, n) / t_scale
    p = np.empty(n)
    p[0] = m[0]
    for i in range(1, n):
        p[i] = p[i - 1] + kappa * (m[i - 1] - p[i - 1]) + sigma_p * eps[i]
    return np.exp(p + 9.0)  # arbitrary level ~ 8100


def to_hourly(prices):
    n_h = len(prices) // 60
    px = prices[: n_h * 60].reshape(n_h, 60)
    return {
        "open": px[:, 0],
        "high": px.max(axis=1),
        "low": px.min(axis=1),
        "close": px[:, -1],
        "minutes": px,
    }


def atr(candles, n=14):
    h, l, c = candles["high"], candles["low"], candles["close"]
    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    out = np.full(len(tr), np.nan)
    out[n - 1] = tr[:n].mean()
    for i in range(n, len(tr)):
        out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out


def sma(x, n=20):
    out = np.full(len(x), np.nan)
    c = np.cumsum(x)
    out[n - 1 :] = (c[n - 1 :] - np.concatenate([[0], c[:-n]])) / n
    return out


def run_backtest(candles, fill_mode, k_atr=2.0, delta=0.001,
                 sma_n=20, max_hold=48):
    """fill_mode: 'touch' (candle low <= limit) or
    'through' (minute price <= limit * (1 - delta))."""
    close, low = candles["close"], candles["low"]
    minutes = candles["minutes"]
    a = atr(candles)
    s = sma(close, sma_n)
    n = len(close)
    trades = []          # (entry_hour, entry_px, exit_hour, exit_px)
    in_pos = False
    entry_px = entry_t = 0
    for t in range(21, n):
        if in_pos:
            if close[t] >= s[t] or (t - entry_t) >= max_hold:
                trades.append((entry_t, entry_px, t, close[t]))
                in_pos = False
            continue
        if np.isnan(a[t - 1]) or np.isnan(s[t - 1]):
            continue
        limit = close[t - 1] - k_atr * a[t - 1]
        if fill_mode == "touch":
            filled = low[t] <= limit
        else:
            filled = minutes[t].min() <= limit * (1 - delta)
        if filled:
            in_pos = True
            entry_px = limit
            entry_t = t
    if in_pos:
        trades.append((entry_t, entry_px, n - 1, close[n - 1]))
    return trades


def trade_stats(trades, n_hours):
    r = np.array([(x[3] - x[1]) / x[1] for x in trades])
    equity_ret = np.zeros(n_hours)
    for (t0, p0, t1, p1) in trades:
        equity_ret[t1] += (p1 - p0) / p0  # attribute at exit, cost-free
    daily = equity_ret[: (n_hours // 24) * 24].reshape(-1, 24).sum(axis=1)
    sharpe_ann = daily.mean() / daily.std() * np.sqrt(365) if daily.std() > 0 else 0
    return {
        "n_trades": len(r),
        "win_rate": (r > 0).mean(),
        "avg_ret": r.mean(),
        "total_ret": np.prod(1 + r) - 1,
        "sharpe_ann": sharpe_ann,
        "returns": r,
        "trades": trades,
    }


def experiment_1():
    prices = simulate_minute_path()
    candles = to_hourly(prices)
    n_hours = len(candles["close"])

    opt = run_backtest(candles, "touch")
    real = run_backtest(candles, "through")
    so = trade_stats(opt, n_hours)
    sr = trade_stats(real, n_hours)

    # trades the optimistic backtest fills that the realistic one denies:
    # entries where the low touched the limit but never traded 0.1% through
    real_entries = {t[0] for t in real}
    phantom = [t for t in opt if t[0] not in real_entries]
    pr = np.array([(x[3] - x[1]) / x[1] for x in phantom])

    print("EXPERIMENT 1: touch fill vs trade-through fill")
    for name, s in [("optimistic (touch)", so), ("realistic (through)", sr)]:
        print(f"{name:22s} trades={s['n_trades']:4d}  "
              f"win={s['win_rate']:.1%}  avg/trade={s['avg_ret']:+.3%}  "
              f"total={s['total_ret']:+.1%}  Sharpe(ann)={s['sharpe_ann']:.2f}")
    print(f"phantom trades (touched, never traded through): {len(phantom)}")
    if len(pr):
        print(f"  their win rate: {(pr > 0).mean():.1%}, "
              f"avg return: {pr.mean():+.3%}")
    return so, sr, pr


def experiment_1_robustness(n_seeds=25, deltas=(0.0005, 0.001, 0.002)):
    """Repeats Experiment 1 across independent seeds and several
    trade-through margins. Reported per margin: how much of the optimistic
    total return survives the realistic fill rule, and the share of trades
    that exist only under the touch assumption."""
    print(f"EXPERIMENT 1 ROBUSTNESS: {n_seeds} seeds, "
          f"margins {', '.join(f'{d:.2%}' for d in deltas)}")
    survive = {d: [] for d in deltas}   # realistic / optimistic total return
    phantom_share = {d: [] for d in deltas}
    phantom_best = {d: 0 for d in deltas}  # phantoms beat realistic avg
    for seed in range(1, n_seeds + 1):
        candles = to_hourly(simulate_minute_path(seed=seed))
        n_hours = len(candles["close"])
        opt = run_backtest(candles, "touch")
        so = trade_stats(opt, n_hours)
        for d in deltas:
            real = run_backtest(candles, "through", delta=d)
            sr = trade_stats(real, n_hours)
            survive[d].append(sr["total_ret"] / so["total_ret"])
            real_entries = {t[0] for t in real}
            phantom = [t for t in opt if t[0] not in real_entries]
            phantom_share[d].append(len(phantom) / len(opt))
            pr = np.array([(x[3] - x[1]) / x[1] for x in phantom])
            if len(pr) and pr.mean() > sr["avg_ret"]:
                phantom_best[d] += 1
    for d in deltas:
        s, ps = np.array(survive[d]), np.array(phantom_share[d])
        print(f"margin {d:.2%}:  realistic/optimistic total return "
              f"median {np.median(s):.2f}, range [{s.min():.2f}, {s.max():.2f}]  "
              f"|  phantom share median {np.median(ps):.1%}")
        print(f"  optimistic overstates the result in "
              f"{(s < 1).sum()}/{n_seeds} seeds; phantom trades beat the "
              f"realistic average in {phantom_best[d]}/{n_seeds} seeds")


# Experiment 2: best of ~100 worthless configurations

def experiment_2(n_configs=100, T=730, n_paths=500):
    """100 independent long/short configurations, each taking a random
    direction every day, applied to zero-drift daily returns (3% daily vol,
    two years). None has edge by construction, and the random directions
    make the configurations statistically independent of each other and of
    the path's realized drift. Reported: the Sharpe distribution on one
    fixed path, and the distribution of the *best* Sharpe across many
    independent paths."""
    r2 = np.random.default_rng(7)

    def all_sharpes(rr):
        s = []
        for _ in range(n_configs):
            sig = r2.choice([-1.0, 1.0], T)
            strat = np.concatenate([[0], sig[:-1]]) * rr  # signal lagged 1 day
            s.append(strat.mean() / strat.std() * np.sqrt(365))
        return np.array(s)

    ret0 = 0.03 * r2.standard_normal(T)
    sharpes = all_sharpes(ret0)

    best_across_paths = []
    for _ in range(n_paths):
        rr = 0.03 * r2.standard_normal(T)
        best_across_paths.append(all_sharpes(rr).max())
    best_across_paths = np.array(best_across_paths)

    print("EXPERIMENT 2: best of 100 worthless configurations")
    print(f"single path:  median Sharpe {np.median(sharpes):+.2f}, "
          f"best Sharpe {sharpes.max():+.2f}")
    print(f"across {n_paths} independent paths, best-of-100 Sharpe:")
    print(f"  median {np.median(best_across_paths):.2f}, "
          f"5th-95th pct [{np.percentile(best_across_paths, 5):.2f}, "
          f"{np.percentile(best_across_paths, 95):.2f}]")
    print(f"  share of paths where best-of-100 > 1.0: "
          f"{(best_across_paths > 1.0).mean():.1%}")
    print(f"  share of paths where best-of-100 > 1.4: "
          f"{(best_across_paths > 1.4).mean():.1%}")
    return sharpes, best_across_paths


if __name__ == "__main__":
    experiment_1()
    print()
    experiment_2()
    print()
    experiment_1_robustness()
