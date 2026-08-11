#!/usr/bin/env python3
"""
MURSHID FILTER — آیا مرشد جلوی فیل بد را می‌گیرد؟
==================================================
مرشد برای معامله کردن ضعیف است (لبه ۰.۹bps در برابر کارمزد ۱۲bps).
ولی برای **معامله نکردن** ممکن است ارزشمند باشد — و کنسل کردن رایگان است.

سوال دقیق:
    وقتی رهبرها دارند می‌ریزند و بید من روی LBank هنوز آنجاست،
    آیا آن خریدِ من واقعاً بدتر از حالت عادی است؟
    و اگر آن لحظه‌ها بید را بکشم، markout چقدر بهتر می‌شود؟

روش:
    برای هر لحظه فرض می‌کنیم روی بید خریدیم و روی اسک فروختیم،
    و markout هر کدام را جدا حساب می‌کنیم — تفکیک‌شده بر اساس
    اینکه مرشد در آن لحظه چه می‌گفت.

    این «بازده مشروط» است: تفاوت بین حالت هم‌جهت و خلاف‌جهت،
    همان چیزی است که فیلتر به دست می‌آورد.

اجرا:
    python murshid_filter.py data/multi_XXXX.csv
    python murshid_filter.py data/multi_XXXX.csv --maker-fee 0
"""

import argparse

import numpy as np
import pandas as pd

GRID_MS = 100
LOOKBACK_MS = 500
HORIZONS_S = [1, 5, 30]
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


def grid(sub, col):
    s = sub.set_index("ts")[col].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.resample(f"{GRID_MS}ms").last().ffill()


def analyse(sub, fee):
    venues = set(sub.venue)
    have = [v for v in LEADERS if v in venues]
    if not have or "lbank" not in venues:
        return None

    lb = sub[sub.venue == "lbank"]
    if len(lb) < 500:
        return None
    lb_mid = grid(lb, "mid")
    lb_bid = grid(lb, "bid")
    lb_ask = grid(lb, "ask")

    ser = {}
    for v in have:
        s = sub[sub.venue == v]
        if len(s) >= 500:
            ser[v] = grid(s, "mid")
    have = list(ser)
    if not have:
        return None

    idx = lb_mid.index
    for v in have:
        idx = idx.intersection(ser[v].index)
    if len(idx) < 3000:
        return None

    k = int(LOOKBACK_MS / GRID_MS)
    R = pd.DataFrame({v: np.log(ser[v].loc[idx] / ser[v].loc[idx].shift(k))
                      for v in have}).dropna()
    sig = R.mean(axis=1)
    agree = (np.sign(R).eq(np.sign(sig), axis=0)).sum(axis=1)
    sd = sig.std()
    if not np.isfinite(sd) or sd <= 0:
        return None
    z = sig / sd

    bid = lb_bid.reindex(R.index)
    ask = lb_ask.reindex(R.index)
    mid = lb_mid.reindex(R.index)

    out = {"n": len(R), "sd_bps": sd * BPS, "leaders": have, "rows": [],
           "filter": []}

    # ---- بازده مشروط بر سیگنال ----
    buckets = [("سقوط شدید", z <= -2), ("سقوط", (z > -2) & (z <= -1)),
               ("خنثی", (z > -1) & (z < 1)),
               ("صعود", (z >= 1) & (z < 2)), ("صعود شدید", z >= 2)]

    for name, m in buckets:
        if m.sum() < 100:
            continue
        row = {"name": name, "n": int(m.sum()), "share": m.mean()}
        for h in HORIZONS_S:
            hh = int(h * 1000 / GRID_MS)
            f = mid.shift(-hh)
            mm = m & f.notna() & bid.notna() & ask.notna()
            if mm.sum() < 50:
                row[h] = (np.nan, np.nan)
                continue
            buy = ((f[mm] - bid[mm]) / bid[mm] * BPS - fee).mean()
            sell = ((ask[mm] - f[mm]) / ask[mm] * BPS - fee).mean()
            row[h] = (float(buy), float(sell))
        out["rows"].append(row)

    # ---- اثر فیلتر کنسل ----
    hh = int(5000 / GRID_MS)
    f5 = mid.shift(-hh)
    ok = f5.notna() & bid.notna() & ask.notna()
    buy_all = ((f5 - bid) / bid * BPS - fee)[ok]
    sell_all = ((ask - f5) / ask * BPS - fee)[ok]
    base = float((buy_all.mean() + sell_all.mean()) / 2)

    for thr in (0.5, 1.0, 1.5, 2.0, 3.0):
        zz = z[ok]
        keep_bid = zz > -thr          # وقتی رهبرها می‌ریزند، بید را بکش
        keep_ask = zz < thr           # وقتی بالا می‌روند، اسک را بکش
        if keep_bid.sum() < 100 or keep_ask.sum() < 100:
            continue
        b = float(buy_all[keep_bid].mean())
        s = float(sell_all[keep_ask].mean())
        both = (b + s) / 2
        uptime = float((keep_bid.mean() + keep_ask.mean()) / 2)
        out["filter"].append({"thr": thr, "buy": b, "sell": s,
                              "both": both, "gain": both - base,
                              "uptime": uptime})
    out["base"] = base
    out["base_buy"] = float(buy_all.mean())
    out["base_sell"] = float(sell_all.mean())
    return out


def main(a):
    df = load(a.csv)
    hrs = (df.ts_ms.max() - df.ts_ms.min()) / 3.6e6
    print("=" * 92)
    print(f"فایل: {a.csv}   |   {hrs:.2f} ساعت   |   کارمزد میکر {a.maker_fee:.1f}bps")
    print("=" * 92)
    print("سوال: آیا فیل‌های خلاف جهت مرشد واقعاً بدترند؟")

    any_ok = False
    for sym in sorted(df.symbol.dropna().unique()):
        r = analyse(df[df.symbol == sym], a.maker_fee)
        if not r:
            print(f"\n{sym.upper()}: داده کافی نیست")
            continue
        any_ok = True

        print(f"\n{'─'*92}")
        print(f"{sym.upper()}   نقاط {r['n']:,}   "
              f"نوسان سیگنال {r['sd_bps']:.2f}bps   رهبرها {', '.join(r['leaders'])}")
        print(f"\n  بازده خرید روی بید / فروش روی اسک (bps، بعد از کارمزد میکر)")
        print("  " + "─" * 86)
        hd = f"  {'وضعیت مرشد':<14}{'سهم':>7}" + \
             "".join(f"{'خرید'+str(h)+'s':>13}{'فروش'+str(h)+'s':>13}"
                     for h in HORIZONS_S)
        print(hd)
        print("  " + "─" * 86)
        for row in r["rows"]:
            cells = ""
            for h in HORIZONS_S:
                b, s = row[h]
                cells += (f"{b:>+13.2f}" if np.isfinite(b) else f"{'—':>13}")
                cells += (f"{s:>+13.2f}" if np.isfinite(s) else f"{'—':>13}")
            print(f"  {row['name']:<14}{row['share']:>6.0%}{cells}")

        # تفسیر
        rows = {x["name"]: x for x in r["rows"]}
        if "سقوط شدید" in rows and "صعود شدید" in rows:
            bd, _ = rows["سقوط شدید"][5]
            bu, _ = rows["صعود شدید"][5]
            if np.isfinite(bd) and np.isfinite(bu):
                print(f"\n  → خرید هنگام سقوط شدید: {bd:+.2f}bps")
                print(f"  → خرید هنگام صعود شدید: {bu:+.2f}bps")
                print(f"  → تفاوت: {bu-bd:+.2f}bps  "
                      f"{'✅ مرشد تمایز می‌دهد' if bu-bd > 1 else '❌ تمایز ناچیز'}")

        print(f"\n  اثر فیلتر کنسل (افق ۵ ثانیه)")
        print("  " + "─" * 86)
        print(f"  {'آستانه':>9}{'خرید':>12}{'فروش':>12}{'میانگین':>12}"
              f"{'بهبود':>12}{'زمان فعال':>13}")
        print("  " + "─" * 86)
        print(f"  {'بدون فیلتر':>9}{r['base_buy']:>+11.2f}"
              f"{r['base_sell']:>+11.2f}{r['base']:>+11.2f}"
              f"{'—':>12}{'100%':>13}")
        best = None
        for f in r["filter"]:
            print(f"  {f['thr']:>8.1f}σ{f['buy']:>+11.2f}{f['sell']:>+11.2f}"
                  f"{f['both']:>+11.2f}{f['gain']:>+11.2f}{f['uptime']:>12.0%}")
            if best is None or f["gain"] > best["gain"]:
                best = f
        if best:
            print(f"\n  بهترین: آستانه {best['thr']:.1f}σ → "
                  f"بهبود {best['gain']:+.2f}bps، "
                  f"زمان فعال {best['uptime']:.0%}")
            if best["gain"] > 0.5:
                print("  ✅ فیلتر ارزش پیاده‌سازی دارد — و کنسل کردن رایگان است.")
            else:
                print("  ❌ بهبود ناچیز. مرشد را به ربات وصل نکن.")

    if not any_ok:
        return
    print("\n" + "=" * 92)
    print("چطور بخوانی:")
    print("  • «خرید» = اگر روی بید بخری، بعد از h ثانیه چقدر جلویی")
    print("  • اگر خرید هنگام سقوط خیلی بدتر از صعود باشد → فیلتر ارزش دارد")
    print("  • «زمان فعال» = چه کسری از وقت کوتیشن روی میز است")
    print("\n⚠ این تست فیل‌های *فرضی* را می‌سنجد، نه فیل واقعی.")
    print("  فیل واقعی انتخابی است و معمولاً بدتر — پس ارزش فیلتر")
    print("  در عمل بیشتر از این عدد است، نه کمتر.\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--maker-fee", type=float, default=2.0)
    main(p.parse_args())
