#!/usr/bin/env python3
"""
FAZ 0 - CLOCK ANALYSIS
======================
جداسازی «تأخیر واقعی بازار» از «تأخیر شبکه‌ی خودت».

سه عدد بیرون می‌دهد:
  ۱. تأخیر شبکه‌ی تو به هر صرافی   (از مقایسه‌ی زمان محلی و زمان سرور)
  ۲. تأخیر واقعی بازار              (روی ساعت صرافی‌ها — یعنی نتیجه‌ی توکیو)
  ۳. پنجره‌ی قابل استفاده از آمستردام و از توکیو

منطقه‌ی زمانی سرورها خودکار تشخیص داده می‌شود.

اجرا:
    venv/bin/python clock_analysis.py data/srv_XXXX.csv
"""

import argparse

import numpy as np
import pandas as pd

MAX_LAG_MS = 4000
LAG_STEP_MS = 50
MAX_GAP_MS = 15000
MIN_EVENTS = 120
TOKYO_RTT_MS = 10.0        # تأخیر تخمینی سرور توکیو (رفت و برگشت)
HOUR = 3600_000


def load(path):
    df = pd.read_csv(path)
    for c in ("bid", "ask", "ts_ms", "ts_srv"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df = df.dropna(subset=["bid", "ask", "ts_ms"])
    df = df[(df.bid > 0) & (df.ask >= df.bid)]
    df["mid"] = (df.bid + df.ask) / 2
    df["key"] = df.symbol.str.replace("_", "", regex=False).str.lower()
    return df.sort_values("ts_ms")


def clock_offsets(df):
    """منطقه‌ی زمانی + تأخیر شبکه را برای هر صرافی تخمین می‌زند."""
    info = {}
    for v in ("binance", "lbank"):
        s = df[(df.venue == v) & df.ts_srv.notna()]
        if len(s) < 100:
            info[v] = None
            continue
        raw = (s.ts_ms - s.ts_srv).values
        med = float(np.median(raw))
        tz = round(med / HOUR) * HOUR          # جزء منطقه‌ی زمانی
        lat = raw - tz                          # باقیمانده = تأخیر واقعی شبکه
        info[v] = {
            "n": len(s),
            "coverage": len(s) / max(len(df[df.venue == v]), 1),
            "tz_hours": tz / HOUR,
            "p05": float(np.percentile(lat, 5)),
            "p50": float(np.median(lat)),
            "p95": float(np.percentile(lat, 95)),
        }
    return info


def px_at(ts, px, t):
    i = np.searchsorted(ts, t, side="right") - 1
    return np.where(i >= 0, px[np.clip(i, 0, len(px) - 1)], np.nan)


def leadlag(bts, bpx, lts, lpx, lags):
    dt = np.diff(lts)
    ok = (dt > 0) & (dt < MAX_GAP_MS)
    if ok.sum() < MIN_EVENTS:
        return None, 0
    t1, t2 = lts[:-1][ok], lts[1:][ok]
    r_l = np.log(lpx[1:][ok] / lpx[:-1][ok])
    live = np.abs(r_l) > 1e-9
    t1, t2, r_l = t1[live], t2[live], r_l[live]
    if len(r_l) < MIN_EVENTS:
        return None, len(r_l)

    out = []
    for lag in lags:
        a = px_at(bts, bpx, t1 - lag)
        b = px_at(bts, bpx, t2 - lag)
        m = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        if m.sum() < MIN_EVENTS:
            out.append(np.nan)
            continue
        r_b = np.log(b[m] / a[m])
        x, y = r_l[m], r_b
        out.append(np.nan if x.std() == 0 or y.std() == 0
                   else float(np.corrcoef(x, y)[0, 1]))
    return np.array(out), len(r_l)


def series(sub, col):
    g = sub.dropna(subset=[col]).groupby(col)["mid"].last()
    return g.index.values.astype(np.int64), g.values.astype(float)


def main(path):
    df = load(path)
    hrs = (df.ts_ms.max() - df.ts_ms.min()) / 3.6e6

    print("=" * 78)
    print(f"فایل: {path}   |   بازه: {hrs:.2f} ساعت   |   ردیف: {len(df):,}")
    print("=" * 78)

    # ---------- ۱. تأخیر شبکه ----------
    print("\n۱) تأخیر شبکه‌ی تو (از مقایسه‌ی ساعت محلی و ساعت صرافی)")
    print("─" * 78)
    info = clock_offsets(df)
    for v in ("binance", "lbank"):
        i = info[v]
        if not i:
            print(f"   {v:8s}: زمان سرور در دسترس نیست")
            continue
        print(f"   {v:8s}: منطقه‌ی زمانی سرور UTC{i['tz_hours']:+.0f}  "
              f"| پوشش {i['coverage']:.0%}")
        print(f"             تأخیر → میانه {i['p50']:>7.0f}ms   "
              f"(۵٪ بهترین {i['p05']:.0f}ms، ۹۵٪ بدترین {i['p95']:.0f}ms)")

    lat_b = info["binance"]["p50"] if info["binance"] else np.nan
    lat_l = info["lbank"]["p50"] if info["lbank"] else np.nan

    # اصلاح منطقه‌ی زمانی: ساعت هر دو صرافی را به یک مبنا می‌آوریم
    df["ts_adj"] = df.ts_srv
    for v in ("binance", "lbank"):
        if info[v]:
            m = df.venue == v
            df.loc[m, "ts_adj"] = df.loc[m, "ts_srv"] + info[v]["tz_hours"] * HOUR

    if info["binance"] and info["lbank"]:
        skew = lat_l - lat_b
        print(f"\n   → اختلاف تأخیر (LBank منهای بایننس): {skew:+.0f}ms")
        print("     این عددی است که تا حالا با «تأخیر بازار» قاطی شده بود.")

    # ---------- ۲. تأخیر بازار روی ساعت صرافی ----------
    print("\n\n۲) تأخیر واقعی بازار — روی ساعت خود صرافی‌ها")
    print("   (این همان چیزی است که یک سرور در توکیو می‌دید)")
    print("─" * 78)

    lags = list(range(-MAX_LAG_MS, MAX_LAG_MS + 1, LAG_STEP_MS))
    rows = []

    for key in sorted(df.key.unique()):
        bn = df[(df.venue == "binance") & (df.key == key)]
        lb = df[(df.venue == "lbank") & (df.key == key)]
        if len(bn) < 500 or len(lb) < 200:
            continue

        res = {}
        for clock, col in (("محلی", "ts_ms"), ("سرور", "ts_adj")):
            b = bn.dropna(subset=[col])
            l = lb.dropna(subset=[col])
            if len(b) < 300 or len(l) < 150:
                continue
            bts, bpx = series(b, col)
            lts, lpx = series(l, col)
            c, n = leadlag(bts, bpx, lts, lpx, lags)
            if c is None or np.all(np.isnan(c)):
                continue
            i = int(np.nanargmax(c))
            res[clock] = (lags[i], float(c[i]), n)

        if not res:
            continue

        print(f"\n  {key.upper()}")
        for clock, (lg, cr, n) in res.items():
            print(f"     ساعت {clock}: تأخیر={lg:+5d}ms  همبستگی={cr:+.3f}  n={n:,}")

        if "سرور" in res and "محلی" in res:
            true_lag = res["سرور"][0]
            seen_lag = res["محلی"][0]
            print(f"     → تفاوت: {seen_lag - true_lag:+d}ms از تأخیر شبکه‌ی تو بود")
            rows.append((key, true_lag, res["سرور"][1], seen_lag))

    # ---------- ۳. پنجره‌ی قابل استفاده ----------
    print("\n\n۳) پنجره‌ی قابل استفاده")
    print("─" * 78)
    if not rows:
        print("   داده‌ی کافی روی ساعت سرور نبود.")
        print("   اگر پوشش زمان سرور پایین بود، یعنی صرافی آن فیلد را نمی‌فرستد.")
        return

    amst_cost = (lat_b + lat_l) if np.isfinite(lat_b) and np.isfinite(lat_l) else np.nan

    print(f"   هزینه‌ی تأخیر از آمستردام: {amst_cost:.0f}ms  "
          f"(دریافت از بایننس + ارسال به LBank)")
    print(f"   هزینه‌ی تأخیر از توکیو   : ~{TOKYO_RTT_MS:.0f}ms\n")

    print(f"   {'جفت':<10} {'تأخیر بازار':>12} {'همبستگی':>9} "
          f"{'پنجره آمستردام':>16} {'پنجره توکیو':>14}")
    print("   " + "─" * 68)

    good_tokyo, good_amst = [], []
    for key, tl, cr, sl in sorted(rows, key=lambda x: -x[1]):
        w_amst = tl - amst_cost if np.isfinite(amst_cost) else np.nan
        w_tok = tl - TOKYO_RTT_MS
        fa = "✅" if w_amst > 50 else "❌"
        ft = "✅" if w_tok > 50 else "❌"
        print(f"   {key.upper():<10} {tl:>+10d}ms {cr:>+9.3f} "
              f"{w_amst:>+13.0f}ms {fa} {w_tok:>+11.0f}ms {ft}")
        if w_tok > 50 and cr > 0.15:
            good_tokyo.append(key)
        if np.isfinite(w_amst) and w_amst > 50 and cr > 0.15:
            good_amst.append(key)

    print("\n" + "─" * 78)
    if good_amst:
        print(f"✅ {len(good_amst)} جفت حتی از آمستردام هم پنجره دارد.")
        print("   → لازم نیست سرور عوض کنی. برو سراغ شبیه‌سازی پرشدن سفارش.")
    elif good_tokyo:
        print(f"🟢 {len(good_tokyo)} جفت از توکیو پنجره دارد، ولی از آمستردام نه.")
        print("   → این یعنی مشکل فقط موقعیت سرور است، نه نبودِ لبه.")
        print("   → سرور رایگان Oracle در توکیو مسئله را حل می‌کند.")
    else:
        print("❌ حتی روی ساعت صرافی هم تأخیر بازار کافی نیست.")
        print("   → آنچه می‌دیدیم عمدتاً تأخیر شبکه‌ی خودمان بود. لبه‌ای نیست.")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    main(ap.parse_args().csv)
