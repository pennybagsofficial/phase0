#!/usr/bin/env python3
"""
CONSENSUS LAB — تقویت سیگنال «مرشد»
====================================
سوال: می‌شود همبستگی ۰.۴۵ را بالاتر برد؟

سه اهرم تست می‌شود:

۱. توافق چند صرافی
   یک صرافی نویز دارد. وقتی بایننس، OKX و بایبیت هم‌جهت حرکت کنند،
   احتمال اینکه حرکت واقعی باشد بسیار بیشتر است.

۲. فیلتر بزرگی حرکت
   همبستگی ۰.۴۵ روی *همه‌ی* حرکات است، شامل نویز ریز.
   اگر فقط حرکات بزرگ را حساب کنیم، همبستگی معمولاً جهش می‌کند.
   ولی تعداد سیگنال کم می‌شود — این معامله را می‌سنجیم.

۳. اندازه‌ی واقعی لبه
   همبستگی بالا به تنهایی بی‌فایده است. سوال نهایی این است:
   وقتی مرشد حرف می‌زند، LBank چند bps حرکت می‌کند؟

اجرا:
    python consensus_lab.py data/multi_XXXX.csv
    python consensus_lab.py data/multi_XXXX.csv --taker-fee 10
"""

import argparse

import numpy as np
import pandas as pd

GRID_MS = 100
LOOKBACK_MS = 500          # پنجره‌ی محاسبه‌ی حرکت رهبرها
HORIZON_MS = [200, 300, 500, 1000]
LEADERS = ["binance", "okx", "bybit"]
BPS = 1e4


def load(path):
    df = pd.read_csv(path)
    for c in ("ts_ms", "bid", "ask"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ts_ms", "bid", "ask"])
    df = df[(df.bid > 0) & (df.ask > df.bid)]
    df["mid"] = (df.bid + df.ask) / 2
    df["ts"] = pd.to_datetime(df.ts_ms, unit="ms")
    return df


def grid(sub):
    s = sub.set_index("ts")["mid"].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.resample(f"{GRID_MS}ms").last().ffill()


def main(a):
    df = load(a.csv)
    hrs = (df.ts_ms.max() - df.ts_ms.min()) / 3.6e6
    syms = sorted(df.symbol.dropna().unique())
    venues = sorted(df.venue.unique())

    print("=" * 94)
    print(f"فایل: {a.csv}   |   {hrs:.2f} ساعت   |   صرافی‌ها: {', '.join(venues)}")
    print("=" * 94)
    for v in venues:
        n = (df.venue == v).sum()
        print(f"   {v:<9} {n:>10,} تیک   ({n/max(hrs*3600,1):.1f}/ثانیه)")

    lb_shift = int(LOOKBACK_MS / GRID_MS)

    for sym in syms:
        sub = df[df.symbol == sym]
        if "lbank" not in set(sub.venue):
            continue

        series = {}
        for v in venues:
            s = sub[sub.venue == v]
            if len(s) < 500:
                continue
            series[v] = grid(s)
        have = [v for v in LEADERS if v in series]
        if not have or "lbank" not in series:
            print(f"\n{sym.upper()}: داده‌ی کافی از رهبرها نیست")
            continue

        idx = series["lbank"].index
        for v in have:
            idx = idx.intersection(series[v].index)
        if len(idx) < 3000:
            print(f"\n{sym.upper()}: همپوشانی کم ({len(idx)})")
            continue

        lb = series["lbank"].loc[idx]
        rets = {}
        for v in have:
            p = series[v].loc[idx]
            rets[v] = np.log(p / p.shift(lb_shift))

        R = pd.DataFrame(rets).dropna()
        lbr = np.log(lb / lb.shift(lb_shift)).reindex(R.index)

        # سیگنال = میانگین حرکت رهبرها ؛ توافق = چند تا هم‌جهت‌اند
        sig = R.mean(axis=1)
        sgn = np.sign(sig)
        agree = (np.sign(R).eq(sgn, axis=0)).sum(axis=1)
        sd = sig.std()

        print(f"\n{'─'*94}")
        print(f"{sym.upper()}   نقاط: {len(R):,}   رهبرها: {', '.join(have)}")
        print(f"  انحراف معیار سیگنال: {sd*BPS:.2f}bps")

        # ---- همبستگی پایه با هر صرافی و با اجماع ----
        print(f"\n  همبستگی خام (بدون فیلتر)، افق {HORIZON_MS[1]}ms:")
        h = int(HORIZON_MS[1] / GRID_MS)
        fwd = np.log(lb.shift(-h) / lb).reindex(R.index)
        base = {}
        for v in have:
            m = R[v].notna() & fwd.notna()
            if m.sum() > 500:
                base[v] = float(np.corrcoef(R[v][m], fwd[m])[0, 1])
        m = sig.notna() & fwd.notna()
        base["اجماع"] = float(np.corrcoef(sig[m], fwd[m])[0, 1]) \
            if m.sum() > 500 else np.nan
        for k, v in base.items():
            star = "  ◄" if k == "اجماع" else ""
            print(f"     {k:<10} {v:+.3f}{star}")

        # ---- جدول: آستانه × توافق ----
        print(f"\n  اثر فیلتر — لبه بر حسب bps (و تعداد سیگنال در ساعت)")
        print("  " + "─" * 88)
        hdr = f"  {'آستانه':>8}{'توافق':>8}" + \
              "".join(f"{h_:>13}ms" for h_ in HORIZON_MS) + f"{'سیگنال/ساعت':>14}"
        print(hdr)
        print("  " + "─" * 88)

        best = None
        for k in (0.0, 1.0, 1.5, 2.0, 3.0):
            for ag in (len(have), max(len(have) - 1, 1)):
                mask = (np.abs(sig) >= k * sd) & (agree >= ag)
                n = int(mask.sum())
                if n < 100:
                    continue
                cells, corr0 = "", None
                for h_ms in HORIZON_MS:
                    hh = int(h_ms / GRID_MS)
                    f = np.log(lb.shift(-hh) / lb).reindex(R.index)
                    mm = mask & f.notna()
                    if mm.sum() < 100:
                        cells += f"{'—':>15}"
                        continue
                    edge = float((np.sign(sig[mm]) * f[mm]).mean() * BPS)
                    cells += f"{edge:>+15.2f}"
                    if h_ms == HORIZON_MS[1]:
                        corr0 = float(np.corrcoef(sig[mm], f[mm])[0, 1])
                        if best is None or edge > best[0]:
                            best = (edge, k, ag, corr0, n)
                rate = n / max(hrs, 0.01)
                print(f"  {k:>7.1f}σ{ag:>8}{cells}{rate:>14.0f}")

        if best:
            edge, k, ag, corr, n = best
            print(f"\n  بهترین: آستانه {k:.1f}σ با توافق {ag} صرافی")
            print(f"     لبه {edge:+.2f}bps   |   همبستگی {corr:+.3f}   |   "
                  f"{n/max(hrs,0.01):.0f} سیگنال در ساعت")
            rt = 2 * a.taker_fee
            print(f"     بعد از کارمزد تیکر رفت‌وبرگشت ({rt:.0f}bps): "
                  f"{edge - rt:+.2f}bps", end="")
            print("  ✅" if edge > rt else "  ❌")
            mk = 2 * a.maker_fee
            print(f"     اگر ورود میکر و خروج تیکر ({a.maker_fee+a.taker_fee:.0f}bps): "
                  f"{edge - a.maker_fee - a.taker_fee:+.2f}bps", end="")
            print("  ✅" if edge > a.maker_fee + a.taker_fee else "  ❌")

    print("\n" + "=" * 94)
    print("چطور بخوانی:")
    print("  • اگر لبه با بالا رفتن آستانه رشد کند → فیلتر کار می‌کند")
    print("  • توافق کامل معمولاً لبه را بالا می‌برد ولی تعداد را کم")
    print("  • عدد تعیین‌کننده: لبه منهای کارمزد. زیر صفر یعنی غیرقابل برداشت")
    print("\n⚠ این سیگنال برای معامله‌ی تهاجمی طراحی نشده — کاربرد اصلی‌اش")
    print("  اریب کردن کوتیشن بازارگردان و جلوگیری از فیل بد است.\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--taker-fee", type=float, default=10.0)
    p.add_argument("--maker-fee", type=float, default=2.0)
    main(p.parse_args())
