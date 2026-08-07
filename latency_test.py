#!/usr/bin/env python3
"""
FAZ 0 - LATENCY TEST
====================
تست تعیین‌کننده‌ی نهایی.

تست قبلی فرض کرده بود در همان میلی‌ثانیه‌ای که سیگنال آمد، معامله می‌کنی.
در واقعیت سفارش تو باید تا صرافی برود و برگردد.

این اسکریپت همان تست را با تأخیر واقعی اجرا می‌کند:
    سیگنال در t  →  ورود در t+delay  →  خروج در t+delay+horizon

اگر لبه با تأخیر ۳۰۰ms هم زنده بماند، واقعی است.
اگر با تأخیر بمیرد، آنچه دیدیم فقط «تازه شدن فید کند ما» بوده، نه فرصت.

اجرا:
    venv/bin/python latency_test.py data/ticks_XXXX.csv
    venv/bin/python latency_test.py data/ticks_XXXX.csv --maker 0.0002
"""

import argparse

import numpy as np
import pandas as pd

DELAYS_MS = [0, 100, 200, 300, 500, 800]
HORIZONS_MS = [400, 800, 1500, 3000]
BASIS_WINDOW = 300
K_SIGMA = 2.0
MIN_SIGNALS = 40
BPS = 1e4


def load(path):
    df = pd.read_csv(path)
    if "kind" not in df.columns:
        df["kind"] = "book"
    for c in ("bid", "ask"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["bid", "ask"])
    df = df[(df.bid > 0) & (df.ask >= df.bid)]
    df["mid"] = (df.bid + df.ask) / 2
    df["key"] = df.symbol.str.replace("_", "", regex=False).str.lower()
    return df.sort_values("ts_ms")


def px_at(ts, px, t):
    i = np.searchsorted(ts, t, side="right") - 1
    return np.where(i >= 0, px[np.clip(i, 0, len(px) - 1)], np.nan)


def run_pair(bn, lb):
    bts = bn.ts_ms.values.astype(np.int64)
    bpx = bn.mid.values

    g = lb.groupby("ts_ms").last()
    lts = g.index.values.astype(np.int64)
    lmid = g["mid"].values
    spread_bps = float(np.median((g["ask"].values - g["bid"].values) / lmid * BPS))

    if len(lts) < 300:
        return None

    bn_at = px_at(bts, bpx, lts)
    ok = np.isfinite(bn_at) & (bn_at > 0)
    if ok.sum() < 300:
        return None

    basis = np.full(len(lts), np.nan)
    basis[ok] = (bn_at[ok] - lmid[ok]) / lmid[ok]
    s = pd.Series(basis)
    dev = ((s - s.rolling(BASIS_WINDOW, min_periods=50).median()) * BPS).values
    sd = float(np.nanstd(dev))
    thr = K_SIGMA * sd

    side = np.where(dev > thr, 1.0, np.where(dev < -thr, -1.0, 0.0))
    fired = np.isfinite(dev) & (side != 0)

    grid = {}
    for d in DELAYS_MS:
        entry = px_at(lts, lmid, lts + d)
        for h in HORIZONS_MS:
            exit_ = px_at(lts, lmid, lts + d + h)
            m = fired & np.isfinite(entry) & np.isfinite(exit_) & (entry > 0)
            n = int(m.sum())
            if n < MIN_SIGNALS:
                grid[(d, h)] = (n, np.nan, np.nan)
                continue
            r = (exit_[m] - entry[m]) / entry[m] * BPS * side[m]
            grid[(d, h)] = (n, float(np.mean(r)), float(np.mean(r > 0)))

    return {"spread": spread_bps, "sd": sd, "thr": thr,
            "n_obs": len(lts), "grid": grid}


def main(path, maker):
    df = load(path)
    hrs = (df.ts_ms.max() - df.ts_ms.min()) / 3.6e6
    fee = 2 * maker * BPS

    print("=" * 78)
    print(f"فایل: {path}   |   بازه: {hrs:.2f} ساعت")
    print(f"آستانه: {K_SIGMA} سیگما   |   کارمزد میکر رفت‌وبرگشت: {fee:.2f} bps")
    print("=" * 78)
    print("اعداد جدول: سود خام بر حسب bps، بعد از تأخیر ورود")
    print("ستون‌ها = مدت نگهداری   |   ردیف‌ها = تأخیر رسیدن سفارش تو")

    survivors = []

    for key in sorted(df.key.unique()):
        bn = df[(df.venue == "binance") & (df.key == key)]
        lb = df[(df.venue == "lbank") & (df.key == key) & (df.kind == "book")]
        if len(bn) < 1000 or len(lb) < 300:
            continue

        res = run_pair(bn, lb)
        if not res:
            continue

        print(f"\n{'─'*78}")
        print(f"{key.upper()}   اسپرد={res['spread']:.1f}bps  "
              f"آستانه={res['thr']:.1f}bps  مشاهده={res['n_obs']:,}")

        hdr = "  تأخیر │" + "".join(f"{h:>9}ms" for h in HORIZONS_MS) + "     n"
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))

        d0 = None
        for d in DELAYS_MS:
            cells, nn = [], 0
            for h in HORIZONS_MS:
                n, g, hit = res["grid"][(d, h)]
                nn = max(nn, n)
                cells.append("      n/a" if np.isnan(g) else f"{g - fee:>+9.2f}")
            line = f"  {d:>5}ms │" + "".join(cells) + f"  {nn:>6,}"
            if d == 0:
                d0 = [res["grid"][(0, h)][1] for h in HORIZONS_MS]
            print(line)

        # نرخ بقا: چقدر از لبه با تأخیر ۳۰۰ms باقی می‌ماند
        d3 = [res["grid"][(300, h)][1] for h in HORIZONS_MS]
        pairs = [(a, b) for a, b in zip(d0, d3)
                 if np.isfinite(a) and np.isfinite(b) and a > 0]
        if pairs:
            surv = np.mean([b / a for a, b in pairs]) * 100
            best300 = max(b for _, b in pairs) - fee
            print(f"  → با تأخیر ۳۰۰ms، {surv:.0f}٪ از لبه باقی می‌ماند "
                  f"(بهترین: {best300:+.2f} bps خالص)")
            survivors.append((key, surv, best300, res["spread"]))

    print("\n" + "=" * 78)
    print("خلاصه — بقای لبه بعد از تأخیر واقعی")
    print("=" * 78)
    if not survivors:
        print("داده‌ی کافی نبود.")
        return

    for k, surv, best, sp in sorted(survivors, key=lambda x: -x[2]):
        if best > 1.0 and surv > 50:
            tag = "✅ زنده مانده"
        elif best > 0:
            tag = "🟡 مرزی"
        else:
            tag = "❌ مرده"
        print(f"  {k.upper():10s} اسپرد={sp:>5.1f}bps  بقا={surv:>5.0f}٪  "
              f"بهترین خالص={best:>+7.2f}bps  {tag}")

    alive = [s for s in survivors if s[2] > 1.0 and s[1] > 50]
    print("\n" + "─" * 78)
    if alive:
        print(f"✅ {len(alive)} جفت با تأخیر واقعی هم لبه دارد.")
        print("   → قدم بعد: شبیه‌سازی پرشدن سفارش (آیا اصلاً فیل می‌خوری؟)")
    else:
        print("❌ با تأخیر واقعی، لبه از بین می‌رود.")
        print("   یعنی آنچه دیدیم عمدتاً تازه شدن فید کند ما بود، نه فرصت معاملاتی.")
        print("   → از آمستردام این کار شدنی نیست. سرور آسیایی یا تغییر استراتژی.")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--maker", type=float, default=0.0)
    a = ap.parse_args()
    main(a.csv, a.maker)
