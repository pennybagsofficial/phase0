#!/usr/bin/env python3
"""
FAZ 0 - ECONOMICS
=================
تست تعیین‌کننده: آیا تأخیری که پیدا کردیم از «اسپرد + کارمزد» بزرگ‌تر است؟

همبستگی نشان می‌دهد قیمت‌ها مرتبط‌اند.
این اسکریپت نشان می‌دهد آیا آن ارتباط قابل *برداشت* است یا نه.

سه هزینه‌ای که باید شکست بخورند:
  ۱. نصف اسپرد هنگام ورود
  ۲. نصف اسپرد هنگام خروج
  ۳. کارمزد رفت و برگشت

اجرا:
    venv/bin/python economics.py data/ticks_XXXX.csv
    venv/bin/python economics.py data/ticks_XXXX.csv --taker 0.001 --maker 0.0
"""

import argparse

import numpy as np
import pandas as pd

HORIZONS_MS = [200, 400, 800, 1500, 3000]
BASIS_WINDOW = 300        # تعداد مشاهده برای حذف انحراف پایدار بین دو صرافی
MIN_SIGNALS = 30
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


def analyse_pair(key, bn, lb, taker, maker):
    bts = bn.ts_ms.values.astype(np.int64)
    bpx = bn.mid.values

    l = lb.groupby("ts_ms").last()
    lts = l.index.values.astype(np.int64)
    lmid = l["mid"].values
    lbid = l["bid"].values
    lask = l["ask"].values

    if len(lts) < 200:
        return None

    # ---- اسپرد واقعی LBank ----
    spread_bps = (lask - lbid) / lmid * BPS
    sp_med = float(np.median(spread_bps))
    sp_p25 = float(np.percentile(spread_bps, 25))

    # ---- انحراف از بایننس ----
    bn_at = px_at(bts, bpx, lts)
    ok = np.isfinite(bn_at) & (bn_at > 0)
    if ok.sum() < 200:
        return None

    basis = np.full(len(lts), np.nan)
    basis[ok] = (bn_at[ok] - lmid[ok]) / lmid[ok]
    s = pd.Series(basis)
    dev = (s - s.rolling(BASIS_WINDOW, min_periods=50).median()).values  # انحراف خالص
    dev_bps = dev * BPS
    dev_sd = float(np.nanstd(dev_bps))

    out = {
        "n": len(lts),
        "spread_med": sp_med,
        "spread_p25": sp_p25,
        "dev_sd": dev_sd,
        "rows": [],
    }

    # ---- برای هر افق، بازده آتی مید LBank ----
    for h in HORIZONS_MS:
        fwd = px_at(lts, lmid, lts + h)
        r = (fwd - lmid) / lmid * BPS
        valid = np.isfinite(r) & np.isfinite(dev_bps)

        for k in (1.0, 1.5, 2.0, 3.0):
            thr = k * dev_sd
            long_m = valid & (dev_bps > thr)      # بایننس بالاتر → LBank باید بالا برود
            short_m = valid & (dev_bps < -thr)
            n_sig = int(long_m.sum() + short_m.sum())
            if n_sig < MIN_SIGNALS:
                continue

            gross = np.concatenate([r[long_m], -r[short_m]])
            g = float(np.mean(gross))
            hit = float(np.mean(gross > 0))

            # سناریو الف: تهاجمی — ورود و خروج با عبور از اسپرد
            cost_taker = sp_med + 2 * taker * BPS
            # سناریو ب: منفعل — سفارش لیمیت، فرض خوش‌بینانه‌ی پرشدن
            cost_maker = 2 * maker * BPS

            out["rows"].append({
                "h": h, "k": k, "thr": thr, "n": n_sig,
                "gross": g, "hit": hit,
                "net_taker": g - cost_taker,
                "net_maker": g - cost_maker,
            })
    return out


def main(path, taker, maker):
    df = load(path)
    hrs = (df.ts_ms.max() - df.ts_ms.min()) / 3.6e6

    print("=" * 78)
    print(f"فایل: {path}   |   بازه: {hrs:.2f} ساعت")
    print(f"کارمزد تیکر: {taker*100:.3f}%   کارمزد میکر: {maker*100:.3f}%")
    print("=" * 78)
    print("gross = حرکت خام LBank بعد از سیگنال (bps)")
    print("تیکر  = بعد از اسپرد کامل + کارمزد رفت‌وبرگشت")
    print("میکر  = فقط کارمزد (خوش‌بینانه — فرض می‌کند لیمیت‌ت پر می‌شود)")

    verdicts = []

    for key in sorted(df.key.unique()):
        bn = df[(df.venue == "binance") & (df.key == key)]
        lb = df[(df.venue == "lbank") & (df.key == key) & (df.kind == "book")]
        if len(bn) < 1000 or len(lb) < 200:
            continue

        res = analyse_pair(key, bn, lb, taker, maker)
        if not res or not res["rows"]:
            continue

        print(f"\n{'─'*78}")
        print(f"{key.upper()}")
        print(f"  اسپرد LBank: میانه {res['spread_med']:.1f}bps   "
              f"(چارک اول {res['spread_p25']:.1f}bps)")
        print(f"  نوسان انحراف: {res['dev_sd']:.1f}bps   |  مشاهده: {res['n']:,}")

        if res["dev_sd"] < res["spread_med"] / 2:
            print("  ⚠ انحراف از نصف اسپرد کوچک‌تر است — از پایه جای امیدی نیست")

        print(f"  {'افق':>6} {'آستانه':>8} {'n':>6} {'خام':>9} {'برد':>6} "
              f"{'تیکر':>10} {'میکر':>10}")
        best = None
        for r in res["rows"]:
            mark = ""
            if r["net_maker"] > 0:
                mark = " ◄ میکر مثبت"
            if r["net_taker"] > 0:
                mark = " ◄◄ تیکر مثبت!"
            print(f"  {r['h']:>5}ms {r['thr']:>7.1f} {r['n']:>6,} "
                  f"{r['gross']:>+8.2f} {r['hit']:>5.0%} "
                  f"{r['net_taker']:>+9.2f} {r['net_maker']:>+9.2f}{mark}")
            if best is None or r["net_maker"] > best["net_maker"]:
                best = r

        verdicts.append((key, res, best))

    # ---------------- خلاصه ----------------
    print("\n" + "=" * 78)
    print("خلاصه — آیا لبه از هزینه بزرگ‌تر است؟")
    print("=" * 78)

    taker_ok = [v for v in verdicts if v[2]["net_taker"] > 0]
    maker_ok = [v for v in verdicts if v[2]["net_maker"] > 0]

    for key, res, b in sorted(verdicts, key=lambda x: -x[2]["net_maker"]):
        if b["net_taker"] > 0:
            tag = "✅ حتی تهاجمی هم سودده"
        elif b["net_maker"] > 0:
            tag = "🟡 فقط منفعل (باید لیمیتت پر شود)"
        else:
            tag = "❌ زیر هزینه"
        print(f"  {key.upper():10s} اسپرد={res['spread_med']:>6.1f}bps  "
              f"خام={b['gross']:>+7.2f}  میکر={b['net_maker']:>+7.2f}  {tag}")

    print("\n" + "─" * 78)
    if taker_ok:
        print(f"✅ {len(taker_ok)} جفت حتی با عبور از اسپرد سودده است.")
        print("   این نادر است — دوباره چک کن که خطایی نباشد.")
    elif maker_ok:
        print(f"🟡 {len(maker_ok)} جفت فقط در حالت منفعل سودده است.")
        print("   یعنی باید سفارش لیمیت بگذاری و منتظر بمانی پر شود.")
        print("   → قدم بعد: شبیه‌سازی جایگاه صف. سوال اصلی این است که")
        print("     آیا اصلاً پر می‌شوی، یا فقط وقتی پر می‌شوی که ضرر کنی.")
    else:
        print("❌ در هیچ جفتی لبه از هزینه بزرگ‌تر نیست.")
        print("   تأخیر واقعی است ولی از اسپرد کوچک‌تر — قابل برداشت نیست.")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--taker", type=float, default=0.001, help="کارمزد تیکر، مثلا 0.001")
    ap.add_argument("--maker", type=float, default=0.0, help="کارمزد میکر")
    a = ap.parse_args()
    main(a.csv, a.taker, a.maker)
