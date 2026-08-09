#!/usr/bin/env python3
"""
PATIENT MM v2 — نسخه‌ی اصلاح‌شده
=================================
سه باگ نسخه‌ی قبل رفع شد. نسخه‌ی قبل ۶۰۶٪ ماهانه می‌داد که واقعی نبود.

باگ ۱ — نرخ بستن ۱۹۸٪
    موقعیت‌هایی بسته می‌شدند که وجود نداشتند. حالا موجودی باز
    به‌صورت یک صف FIFO نگه داشته می‌شود و نمی‌شود بیشتر از آنچه
    باز شده بست.

باگ ۲ — خروج فقط وقتی سودده باشد
    شرط `good` باعث می‌شد شبیه‌ساز موقعیت‌های ضررده را هرگز نبندد و
    فقط سودها را برداشت کند. این «انتخاب گیلاس» است، نه شبیه‌سازی.
    حالا خروج بی‌طرف است: هر جا سفارش خروج پر شود، پر می‌شود.

باگ ۳ — صف خروج مدل نشده بود
    فرض می‌شد هر معامله‌ای که به قیمت ما رسید، ما را پر می‌کند.
    ولی ما هم در صف هستیم. حالا سفارش خروج هم صف دارد.

اضافه: حجم معامله بین ورود و خروج تقسیم می‌شود (یک معامله نمی‌تواند
       همزمان کل سفارش ورود و کل سفارش خروج را پر کند).

اجرا:
    python patient_mm2.py data/q_XXXX.csv --size-usd 50
    python patient_mm2.py data/q_XXXX.csv --sweep
"""

import argparse
import itertools
from collections import deque

import numpy as np
import pandas as pd

BPS = 1e4
SLOPE_H = [1, 5, 30, 120, 300]


def load(path):
    df = pd.read_csv(path)
    for c in ("ts_ms", "b1p", "b1q", "a1p", "a1q", "price", "qty"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["side"] = df.get("side", "").astype(str).str.lower().str.strip()
    return df.sort_values("ts_ms").reset_index(drop=True)


def arrays(sub):
    b = sub[sub.kind == "book"].dropna(subset=["b1p", "a1p"])
    b = b[(b.b1p > 0) & (b.a1p > b.b1p)]
    t = sub[sub.kind == "trade"].dropna(subset=["price", "qty"])
    t = t[(t.price > 0) & (t.qty > 0)]
    if len(b) < 100 or len(t) < 50:
        return None
    return {
        "bts": b.ts_ms.values.astype(np.int64),
        "bid": b.b1p.values, "ask": b.a1p.values,
        "bq": np.nan_to_num(b.b1q.values), "aq": np.nan_to_num(b.a1q.values),
        "tts": t.ts_ms.values.astype(np.int64),
        "tpx": t.price.values, "tq": t.qty.values,
        "tsell": t.side.str.startswith("s").values,
    }


def mid_at(A, t):
    i = np.searchsorted(A["bts"], t, side="right") - 1
    return (A["bid"][i] + A["ask"][i]) / 2 if i >= 0 else np.nan


def slope_score(A, size_usd, fee):
    """منحنی markout — بدون تغییر، این بخش باگ نداشت."""
    st = {"bid": None, "ask": None}
    fills = []
    bi = ti = 0
    nb, nt = len(A["bts"]), len(A["tts"])
    while bi < nb or ti < nt:
        if ti >= nt or (bi < nb and A["bts"][bi] <= A["tts"][ti]):
            for k, p, q in (("bid", A["bid"][bi], A["bq"][bi]),
                            ("ask", A["ask"][bi], A["aq"][bi])):
                s = st[k]
                if s is None or s["price"] != p:
                    st[k] = {"price": p, "ahead": q, "left": size_usd / p}
            bi += 1
        else:
            sell = A["tsell"][ti]
            k = "bid" if sell else "ask"
            s = st[k]
            if s is not None:
                hit = (A["tpx"][ti] <= s["price"] + 1e-12) if sell \
                    else (A["tpx"][ti] >= s["price"] - 1e-12)
                if hit:
                    v = A["tq"][ti]
                    e = min(v, s["ahead"])
                    s["ahead"] -= e
                    v -= e
                    if v > 0 and s["left"] > 0:
                        f = min(v, s["left"])
                        s["left"] -= f
                        fills.append((A["tts"][ti], k, s["price"]))
                        if s["left"] <= 1e-12:
                            st[k] = None
            ti += 1
    if len(fills) < 40:
        return None
    if len(fills) > 3000:
        fills = fills[::len(fills) // 3000 + 1]
    curve = []
    for h in SLOPE_H:
        v = [ (mid_at(A, ts + int(h*1000)) - px) / px * BPS
              * (1 if k == "bid" else -1) - fee
              for ts, k, px in fills
              if np.isfinite(mid_at(A, ts + int(h*1000))) ]
        curve.append(np.mean(v) if v else np.nan)
    if any(not np.isfinite(c) for c in curve):
        return None
    return {"curve": curve, "slope": curve[-1] - curve[1], "long": curve[-1]}


def simulate(A, size_usd, off, fee, max_inv_usd):
    """
    حسابداری درست:
      • موقعیت‌های باز در یک صف FIFO
      • خروج بی‌طرف — سود و ضرر هر دو ثبت می‌شوند
      • سفارش خروج هم صف دارد
      • حجم هر معامله بین ورود و خروج تقسیم می‌شود
    """
    cash = 0.0
    lots = deque()                 # (ts, sign, price, qty باقیمانده)
    inv = 0.0
    entry = {"bid": None, "ask": None}
    exit_o = {"long": None, "short": None}   # سفارش‌های خروج
    n_open = n_close = 0
    hold_sum = 0.0
    inv_peak = 0.0
    bi = ti = 0
    nb, nt = len(A["bts"]), len(A["tts"])
    last_mid = np.nan

    while bi < nb or ti < nt:
        if ti >= nt or (bi < nb and A["bts"][bi] <= A["tts"][ti]):
            b1, a1 = A["bid"][bi], A["ask"][bi]
            sp = a1 - b1
            tick = sp if sp > 0 else b1 * 1e-4
            last_mid = (b1 + a1) / 2
            inv_usd = inv * last_mid
            inv_peak = max(inv_peak, abs(inv_usd))

            # --- سفارش‌های ورود ---
            for k, price, allow in (
                    ("bid", b1 - off * tick, inv_usd < max_inv_usd),
                    ("ask", a1 + off * tick, inv_usd > -max_inv_usd)):
                if not allow or price <= 0:
                    entry[k] = None
                    continue
                s = entry[k]
                if s is None or abs(s["price"] - price) > 1e-12:
                    ahead = 0.0 if off > 0 else \
                        (A["bq"][bi] if k == "bid" else A["aq"][bi])
                    entry[k] = {"price": price, "ahead": ahead,
                                "left": size_usd / price}

            # --- سفارش‌های خروج، روی تاچ مقابل، با صف ---
            long_q = sum(l[3] for l in lots if l[1] > 0)
            short_q = sum(l[3] for l in lots if l[1] < 0)
            for tag, qty, price, qd in (("long", long_q, a1, A["aq"][bi]),
                                        ("short", short_q, b1, A["bq"][bi])):
                if qty <= 1e-15:
                    exit_o[tag] = None
                    continue
                s = exit_o[tag]
                if s is None or abs(s["price"] - price) > 1e-12:
                    exit_o[tag] = {"price": price, "ahead": qd, "left": qty}
                else:
                    s["left"] = qty
            bi += 1

        else:
            ts = A["tts"][ti]
            sell = A["tsell"][ti]
            px_t = A["tpx"][ti]
            vol = A["tq"][ti]

            # ---- ۱. خروج اول (اولویت با بستن ریسک) ----
            tag = "long" if not sell else "short"   # لانگ با خریدار تهاجمی بسته می‌شود
            s = exit_o[tag]
            if s is not None and vol > 0:
                hit = (px_t >= s["price"] - 1e-12) if tag == "long" \
                    else (px_t <= s["price"] + 1e-12)
                if hit:
                    e = min(vol, s["ahead"])
                    s["ahead"] -= e
                    vol -= e
                    if vol > 0 and s["left"] > 0:
                        f = min(vol, s["left"])
                        vol -= f
                        s["left"] -= f
                        p = s["price"]
                        rem = f
                        want = 1 if tag == "long" else -1
                        while rem > 1e-15 and lots:
                            found = None
                            for idx, l in enumerate(lots):
                                if l[1] == want and l[3] > 1e-15:
                                    found = idx
                                    break
                            if found is None:
                                break
                            l = lots[found]
                            take = min(rem, l[3])
                            l[3] -= take
                            rem -= take
                            if want > 0:
                                cash += p * take * (1 - fee / BPS)
                                inv -= take
                            else:
                                cash -= p * take * (1 + fee / BPS)
                                inv += take
                            n_close += 1
                            hold_sum += (ts - l[0]) / 1000
                            if l[3] <= 1e-15:
                                del lots[found]
                        # اگر لاتی نبود، سفارش خروج بی‌مورد بوده
                        if rem > 1e-15:
                            s["left"] = 0.0

            # ---- ۲. ورود با حجم باقیمانده ----
            k = "bid" if sell else "ask"
            s = entry[k]
            if s is not None and vol > 0:
                hit = (px_t <= s["price"] + 1e-12) if sell \
                    else (px_t >= s["price"] - 1e-12)
                if hit:
                    e = min(vol, s["ahead"])
                    s["ahead"] -= e
                    vol -= e
                    if vol > 0 and s["left"] > 0:
                        f = min(vol, s["left"])
                        s["left"] -= f
                        p = s["price"]
                        if k == "bid":
                            cash -= p * f * (1 + fee / BPS)
                            inv += f
                            lots.append([ts, 1, p, f])
                        else:
                            cash += p * f * (1 - fee / BPS)
                            inv -= f
                            lots.append([ts, -1, p, f])
                        n_open += 1
                        if s["left"] <= 1e-12:
                            entry[k] = None
            ti += 1

    if n_open < 20 or not np.isfinite(last_mid):
        return None
    hrs = (max(A["bts"][-1], A["tts"][-1]) -
           min(A["bts"][0], A["tts"][0])) / 3.6e6
    if hrs <= 0:
        return None

    open_qty = sum(abs(l[3]) for l in lots)
    equity = cash + inv * last_mid
    return {
        "hours": hrs, "opens": n_open, "closes": n_close,
        "close_rate": n_close / max(n_open, 1),
        "opens_hr": n_open / hrs,
        "avg_hold_s": hold_sum / max(n_close, 1),
        "pnl_hr": equity / hrs,
        "inv_peak": inv_peak,
        "open_left": open_qty * last_mid,
    }


def main(a):
    df = load(a.csv)
    syms = sorted(df.symbol.dropna().unique())
    maxinv = a.max_inv_usd if a.max_inv_usd > 0 else a.size_usd * 4

    print("=" * 96)
    print(f"فایل: {a.csv}   |   {len(syms)} جفت   |   سایز ${a.size_usd:.0f}"
          f"   میکر {a.fee:.1f}bps   سقف ±${maxinv:,.0f}")
    print("نسخه‌ی اصلاح‌شده: حسابداری FIFO، خروج بی‌طرف، صف خروج")
    print("=" * 96)

    A = {}
    for s in syms:
        x = arrays(df[df.symbol == s])
        if x:
            A[s] = x

    print("\n۱) شیب منحنی markout (این بخش باگ نداشت)")
    print("  " + "─" * 88)
    print(f"  {'جفت':<14}" + "".join(f"{h:>9}s" for h in SLOPE_H) + f"{'شیب':>10}")
    print("  " + "─" * 88)
    picked = []
    for s, x in A.items():
        sc = slope_score(x, a.size_usd, a.fee)
        if not sc:
            continue
        good = sc["slope"] > a.min_slope and sc["long"] > 0
        if good:
            picked.append(s)
        print(f"  {s.upper():<14}" + "".join(f"{c:>+10.1f}" for c in sc["curve"])
              + f"{sc['slope']:>+10.1f}  {'✅' if good else '❌'}")

    if not picked:
        print("\n  هیچ جفتی شیب صعودی ندارد.")
        return
    print(f"\n  انتخاب‌شده: {', '.join(s.upper() for s in picked)}")

    print(f"\n\n۲) اثر فاصله‌ی کوتیشن — با حسابداری درست")
    print("  " + "─" * 88)
    print(f"  {'فاصله':>7}{'ورود/ساعت':>12}{'نرخ بستن':>11}{'نگهداری':>11}"
          f"{'باز مانده':>12}{'سود/ساعت':>13}{'روزانه':>13}")
    print("  " + "─" * 88)
    best = None
    for off in (0, 1, 2, 3, 5):
        tot, oh, cr, hd, lf, n = 0.0, [], [], [], [], 0
        for s in picked:
            r = simulate(A[s], a.size_usd, off, a.fee, maxinv)
            if not r:
                continue
            tot += r["pnl_hr"]
            oh.append(r["opens_hr"]); cr.append(r["close_rate"])
            hd.append(r["avg_hold_s"]); lf.append(r["open_left"]); n += 1
        if not n:
            continue
        warn = " ⚠" if np.mean(cr) > 1.001 else ""
        print(f"  {off:>7}{np.mean(oh):>11.1f}{np.mean(cr):>10.0%}"
              f"{np.mean(hd):>10.0f}s{np.mean(lf):>11,.0f}$"
              f"{tot:>+12.3f}${tot*24:>+12.2f}$"
              f" {'✅' if tot > 0 else '❌'}{warn}")
        if best is None or tot > best[1]:
            best = (off, tot)

    if a.sweep:
        print(f"\n\n۳) جستجو")
        print("  " + "─" * 88)
        print(f"  {'فاصله':>7}{'سقف':>12}{'سود/ساعت':>14}{'روزانه':>14}"
              f"{'سرمایه':>12}{'ماهانه٪':>12}")
        print("  " + "─" * 88)
        out = []
        for off, mi in itertools.product((0, 1, 2, 3),
                                         [a.size_usd * k for k in (2, 4, 8)]):
            tot = 0.0
            for s in picked:
                r = simulate(A[s], a.size_usd, off, a.fee, mi)
                if r:
                    tot += r["pnl_hr"]
            cap = len(picked) * 2 * mi
            out.append((off, mi, tot, cap))
        out.sort(key=lambda x: -x[2])
        for off, mi, tot, cap in out[:8]:
            print(f"  {off:>7}{mi:>11,.0f}${tot:>+13.3f}${tot*24:>+13.2f}$"
                  f"{cap:>11,.0f}${tot*24*30/cap*100:>11.1f}%")

    print("\n" + "=" * 96)
    if best and best[1] > 0:
        off, tot = best
        cap = len(picked) * 2 * maxinv
        print(f"✅ بهترین: کوتیشن {off} تیک عقب‌تر")
        print(f"   روزانه {tot*24:+,.2f}$ روی سرمایه‌ی {cap:,.0f}$"
              f"  = {tot*24*30/cap*100:+.1f}% ماهانه")
        print(f"   با {len(picked)} جفت، بدون عبور از اسپرد.")
    else:
        print("❌ با حسابداری درست، سودده نیست.")
        print("   نسخه‌ی قبلی ۶۰۶٪ می‌داد — آن عدد از باگ می‌آمد، نه استراتژی.")
    print("\n⚠ ستون «باز مانده» = موجودی تسویه‌نشده در پایان.")
    print("  اگر بزرگ باشد، سود کاغذی است نه تحقق‌یافته.\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--size-usd", type=float, default=50)
    p.add_argument("--fee", type=float, default=2.0)
    p.add_argument("--max-inv-usd", type=float, default=0)
    p.add_argument("--min-slope", type=float, default=1.0)
    p.add_argument("--sweep", action="store_true")
    main(p.parse_args())
