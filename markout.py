#!/usr/bin/env python3
"""
MARKOUT — تست انتخاب معکوس
===========================
تنها عددی که بین «بازارگردان سودده» و «بازارگردان ورشکسته» فرق می‌گذارد.

منطق:
    هر معامله دو طرف دارد. طرف تهاجمی (taker) و طرف منفعل (maker).
    اگر تو بازارگردان باشی، تو همیشه طرف منفعلی:
        خریدار تهاجمی آمد  →  تو روی ASK فروختی
        فروشنده تهاجمی آمد →  تو روی BID خریدی

    سود تو = نصف اسپرد  +  حرکت قیمت به نفعت  −  کارمزد

    مسئله اینجاست که طرف تهاجمی معمولاً چیزی می‌داند. اگر بعد از هر
    فروش او قیمت پایین برود، تو که خریدی ضرر کرده‌ای. این «انتخاب
    معکوس» است و اسپرد را می‌بلعد.

اجرا:
    python markout.py data/mm_XXXX.csv
    python markout.py data/mm_XXXX.csv --fee 2.0
"""

import argparse

import numpy as np
import pandas as pd

HORIZONS_S = [1, 5, 30, 60]
MIN_TRADES = 100
BPS = 1e4


def load(path):
    df = pd.read_csv(path)
    for c in ("ts_ms", "ts_srv", "bid", "ask", "price", "qty"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["side"] = df.get("side", "").astype(str).str.lower().str.strip()
    return df.sort_values("ts_ms")


def norm_side(s):
    """جهت تهاجمی → سمت بازارگردان.
    خریدار تهاجمی ⇒ بازارگردان فروخت (-1)
    فروشنده تهاجمی ⇒ بازارگردان خرید (+1)"""
    if s.startswith("b"):      # buy
        return -1.0
    if s.startswith("s"):      # sell
        return 1.0
    return 0.0


def px_at(ts, arr, t):
    i = np.searchsorted(ts, t, side="right") - 1
    return np.where(i >= 0, arr[np.clip(i, 0, len(arr) - 1)], np.nan)


def analyse(sym, book, trades, fee_rt):
    b = book.dropna(subset=["bid", "ask"])
    b = b[(b.bid > 0) & (b.ask > b.bid)]
    if len(b) < 200:
        return None

    bts = b.ts_ms.values.astype(np.int64)
    bid = b.bid.values
    ask = b.ask.values
    mid = (bid + ask) / 2

    t = trades.dropna(subset=["price"])
    t = t[t.price > 0]
    t["mm"] = t.side.map(norm_side)
    t = t[t.mm != 0]
    if len(t) < MIN_TRADES:
        return None

    tts = t.ts_ms.values.astype(np.int64)
    mm = t.mm.values

    # وضعیت دفتر درست قبل از معامله
    i = np.searchsorted(bts, tts, side="right") - 1
    ok = i >= 0
    i = np.clip(i, 0, len(bts) - 1)
    m0 = mid[i]
    hs_bps = (ask[i] - bid[i]) / 2 / m0 * BPS      # نصف اسپرد = لبه‌ی خام تو
    ok &= np.isfinite(m0) & (m0 > 0) & np.isfinite(hs_bps)

    res = {"n": int(ok.sum()), "half_spread": float(np.median(hs_bps[ok])),
           "rows": []}

    for h in HORIZONS_S:
        mfut = px_at(bts, mid, tts + h * 1000)
        good = ok & np.isfinite(mfut) & (mfut > 0)
        if good.sum() < MIN_TRADES:
            continue
        dmid = (mfut[good] - m0[good]) / m0[good] * BPS
        # حرکت قیمت از دید بازارگردان: مثبت یعنی به نفع او
        adverse = -(mm[good] * dmid)            # هزینه‌ی انتخاب معکوس
        gross = hs_bps[good] + mm[good] * dmid  # سود ناخالص هر فیل
        net = gross - fee_rt

        res["rows"].append({
            "h": h,
            "n": int(good.sum()),
            "adverse": float(np.mean(adverse)),
            "gross": float(np.mean(gross)),
            "net": float(np.mean(net)),
            "hit": float(np.mean(net > 0)),
            "med_net": float(np.median(net)),
        })

    # تفکیک خرید و فروش — نامتقارنی نشانه‌ی روند است
    for lbl, msk in (("خرید MM", mm > 0), ("فروش MM", mm < 0)):
        g = ok & msk
        mfut = px_at(bts, mid, tts + 5000)
        g = g & np.isfinite(mfut) & (mfut > 0)
        if g.sum() >= 30:
            dmid = (mfut[g] - m0[g]) / m0[g] * BPS
            res.setdefault("split", {})[lbl] = (
                int(g.sum()),
                float(np.mean(hs_bps[g] + mm[g] * dmid - fee_rt)))
    return res


def main(path, fee):
    df = load(path)
    fee_rt = 2 * fee
    hrs = (df.ts_ms.max() - df.ts_ms.min()) / 3.6e6

    print("=" * 82)
    print(f"فایل: {path}   |   بازه: {hrs:.2f} ساعت")
    print(f"کارمزد میکر: {fee:.1f}bps هر طرف → {fee_rt:.1f}bps رفت‌وبرگشت")
    print("=" * 82)
    print("انتخاب معکوس = چقدر قیمت بعد از فیل، علیه تو حرکت می‌کند")
    print("خالص = نصف اسپرد − انتخاب معکوس − کارمزد")

    summary = []

    for sym in sorted(df.symbol.dropna().unique()):
        sub = df[df.symbol == sym]
        r = analyse(sym, sub[sub.kind == "book"], sub[sub.kind == "trade"].copy(),
                    fee_rt)
        if not r or not r["rows"]:
            print(f"\n{sym.upper()}: داده کافی نیست")
            continue

        print(f"\n{'─'*82}")
        print(f"{sym.upper()}   معاملات: {r['n']:,}   "
              f"نصف اسپرد: {r['half_spread']:.1f}bps")
        print(f"  {'افق':>6}{'n':>9}{'انتخاب معکوس':>16}{'ناخالص':>11}"
              f"{'خالص':>10}{'برد':>8}")
        for row in r["rows"]:
            mark = " ◄ مثبت" if row["net"] > 0 else ""
            print(f"  {row['h']:>4}s {row['n']:>9,} {row['adverse']:>+14.2f} "
                  f"{row['gross']:>+10.2f} {row['net']:>+9.2f} "
                  f"{row['hit']:>7.0%}{mark}")

        if "split" in r:
            parts = "   ".join(f"{k}: n={v[0]:,} خالص={v[1]:+.2f}"
                               for k, v in r["split"].items())
            print(f"  تفکیک (۵s): {parts}")

        best = max(r["rows"], key=lambda x: x["net"])
        summary.append((sym, r["half_spread"], best["adverse"],
                        best["net"], best["h"], r["n"]))

    # ---------------- خلاصه ----------------
    print("\n" + "=" * 82)
    print("خلاصه — آیا بازارگردانی روی این جفت سودده است؟")
    print("=" * 82)
    if not summary:
        print("داده‌ی کافی نبود.")
        return

    print(f"  {'جفت':<14}{'نصف اسپرد':>11}{'انتخاب معکوس':>15}"
          f"{'خالص':>10}{'افق':>7}{'n':>9}")
    print("  " + "─" * 68)
    for sym, hs, adv, net, h, n in sorted(summary, key=lambda x: -x[3]):
        tag = "✅ سودده" if net > 1.0 else ("🟡 مرزی" if net > 0 else "❌ ضررده")
        print(f"  {sym.upper():<14}{hs:>10.1f}{adv:>+14.2f}{net:>+10.2f}"
              f"{h:>6}s{n:>9,}  {tag}")

    good = [s for s in summary if s[3] > 1.0]
    print("\n" + "─" * 82)
    if good:
        print(f"✅ {len(good)} جفت بعد از انتخاب معکوس و کارمزد هنوز مثبت است.")
        print("   → این اولین بار است که یک عدد واقعاً مثبت می‌بینیم.")
        print("   → قدم بعد: شبیه‌سازی جایگاه صف — آیا واقعاً فیل می‌خوری؟")
    else:
        print("❌ انتخاب معکوس کل اسپرد را می‌خورد.")
        print("   → یعنی هر بار که فیل می‌شوی، طرف مقابل حق داشته.")
        print("   → راه‌حل: کوتیشن عریض‌تر، یا کنار کشیدن هنگام جریان سمّی.")
    print("\n⚠ این تست فرض می‌کند تو طرف منفعل هر معامله بودی — یعنی سقف نظری.")
    print("  در عمل فقط بخشی از معاملات به تو می‌رسد.\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--fee", type=float, default=2.0, help="کارمزد میکر هر طرف (bps)")
    a = p.parse_args()
    main(a.csv, a.fee)
