#!/usr/bin/env python3
"""
EXIT LAB — منحنی markout و قانون خروج
======================================
تشخیص از دور قبل:

    همه‌ی جفت‌ها markout مثبت داشتند (+۱۲ تا +۱۶bps در ۵ ثانیه)
    ولی نصفشان ضرر می‌دادند. و اریب موجودی ضرر را بیشتر کرد.

    یعنی سود در لحظه‌ی فیل هست، ولی جایی در ادامه از بین می‌رود.

این اسکریپت دو چیز را جواب می‌دهد:

۱. منحنی markout در افق‌های مختلف
   اگر در ۵ ثانیه مثبت و در ۶۰ ثانیه منفی باشد → باید سریع خارج شد.
   اگر همه‌جا مثبت باشد → مشکل جای دیگری است.

۲. شبیه‌سازی قانون خروج
   بعد از هر فیل، سفارش خروج روی سمت مقابل می‌گذاریم.
   اگر تا T ثانیه پر نشد، با عبور از اسپرد خارج می‌شویم.
   برای هر T، سود کل را حساب می‌کند.

   این تفاوت «بگیر و رها کن» با «کوتیشن بده و صبر کن» است.

اجرا:
    python exit_lab.py data/q_XXXX.csv --size-usd 50
    python exit_lab.py data/q_XXXX.csv --taker-fee 10
"""

import argparse

import numpy as np
import pandas as pd

HORIZONS = [0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300]
EXIT_T = [1, 2, 5, 10, 20, 30, 60, 120]
BPS = 1e4
MAX_FILLS = 4000        # سقف برای سرعت


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
    if len(b) < 100 or len(t) < 100:
        return None
    return {
        "bts": b.ts_ms.values.astype(np.int64),
        "bid": b.b1p.values, "ask": b.a1p.values,
        "bq": np.nan_to_num(b.b1q.values), "aq": np.nan_to_num(b.a1q.values),
        "tts": t.ts_ms.values.astype(np.int64),
        "tpx": t.price.values, "tq": t.qty.values,
        "tsell": t.side.str.startswith("s").values,
    }


def entry_fills(A, size_usd):
    """فیل‌های ورود با منطق صف — کوتیشن متقارن روی بهترین سطح."""
    st = {"bid": None, "ask": None}
    fills, reposts = [], 0
    bi, ti = 0, 0
    nb, nt = len(A["bts"]), len(A["tts"])

    while bi < nb or ti < nt:
        take_book = ti >= nt or (bi < nb and A["bts"][bi] <= A["tts"][ti])
        if take_book:
            p_b, p_a = A["bid"][bi], A["ask"][bi]
            for k, price, lvlq in (("bid", p_b, A["bq"][bi]),
                                   ("ask", p_a, A["aq"][bi])):
                s = st[k]
                if s is None or s["price"] != price:
                    st[k] = {"price": price, "ahead": lvlq,
                             "left": size_usd / price}
                    reposts += 1
            bi += 1
        else:
            sell = A["tsell"][ti]
            k = "bid" if sell else "ask"
            s = st[k]
            if s is not None:
                hit = (A["tpx"][ti] <= s["price"] + 1e-12) if sell \
                    else (A["tpx"][ti] >= s["price"] - 1e-12)
                if hit:
                    vol = A["tq"][ti]
                    eat = min(vol, s["ahead"])
                    s["ahead"] -= eat
                    vol -= eat
                    if vol > 0 and s["left"] > 0:
                        f = min(vol, s["left"])
                        s["left"] -= f
                        fills.append((A["tts"][ti], k, s["price"], f))
                        if s["left"] <= 1e-12:
                            st[k] = None
            ti += 1
    return fills, reposts


def mid_at(A, t):
    i = np.searchsorted(A["bts"], t, side="right") - 1
    if i < 0:
        return np.nan
    return (A["bid"][i] + A["ask"][i]) / 2


def touch_at(A, t):
    i = np.searchsorted(A["bts"], t, side="right") - 1
    if i < 0:
        return np.nan, np.nan
    return A["bid"][i], A["ask"][i]


def markout_curve(A, fills, maker_fee):
    out = {}
    for h in HORIZONS:
        vals = []
        for ts, k, px, q in fills:
            m = mid_at(A, ts + int(h * 1000))
            if not np.isfinite(m) or m <= 0:
                continue
            sign = 1.0 if k == "bid" else -1.0
            vals.append((m - px) / px * BPS * sign - maker_fee)
        out[h] = (np.mean(vals), np.mean(np.array(vals) > 0), len(vals)) \
            if vals else (np.nan, np.nan, 0)
    return out


def exit_rule(A, fills, T, maker_fee, taker_fee):
    """
    بعد از هر فیل: سفارش خروج روی سمت مقابل (میکر).
    اگر تا T ثانیه پر نشد → عبور از اسپرد (تیکر).
    """
    pnl, passive, forced = [], 0, 0
    for ts, k, px, q in fills:
        b0, a0 = touch_at(A, ts)
        if not (np.isfinite(b0) and np.isfinite(a0)):
            continue
        long = (k == "bid")
        target = a0 if long else b0          # قیمت سفارش خروج ما
        t_end = ts + int(T * 1000)

        lo = np.searchsorted(A["tts"], ts, side="right")
        hi = np.searchsorted(A["tts"], t_end, side="right")
        done = False
        if hi > lo:
            seg_px = A["tpx"][lo:hi]
            seg_sell = A["tsell"][lo:hi]
            if long:
                # برای فروش ما، خریدار تهاجمی لازم است در قیمت ≥ هدف
                ok = (~seg_sell) & (seg_px >= target - 1e-12)
            else:
                ok = seg_sell & (seg_px <= target + 1e-12)
            if ok.any():
                done = True

        if done:
            ex, fee_out = target, maker_fee
            passive += 1
        else:
            b1, a1 = touch_at(A, t_end)
            if not (np.isfinite(b1) and np.isfinite(a1)):
                continue
            ex, fee_out = (b1 if long else a1), taker_fee
            forced += 1

        sign = 1.0 if long else -1.0
        r = (ex - px) / px * BPS * sign - maker_fee - fee_out
        pnl.append((r, px * q))

    if not pnl:
        return None
    r = np.array([p[0] for p in pnl])
    notional = np.array([p[1] for p in pnl])
    return {
        "n": len(r),
        "bps": float(r.mean()),
        "win": float((r > 0).mean()),
        "usd": float((notional * r / BPS).sum()),
        "passive_rate": passive / max(passive + forced, 1),
    }


def main(a):
    df = load(a.csv)
    syms = sorted(df.symbol.dropna().unique())
    hrs = (df.ts_ms.max() - df.ts_ms.min()) / 3.6e6

    print("=" * 96)
    print(f"فایل: {a.csv}   |   {len(syms)} جفت   |   {hrs:.1f} ساعت")
    print(f"سایز ${a.size_usd:.0f}   میکر {a.fee:.1f}bps   "
          f"تیکر {a.taker_fee:.1f}bps")
    print("=" * 96)

    data = {}
    for s in syms:
        A = arrays(df[df.symbol == s])
        if not A:
            continue
        f, rp = entry_fills(A, a.size_usd)
        if len(f) < 50:
            continue
        if len(f) > MAX_FILLS:
            step = len(f) // MAX_FILLS + 1
            f = f[::step]
        data[s] = (A, f, rp)

    if not data:
        print("داده‌ی کافی نیست.")
        return

    # ---------- بخش ۱: منحنی markout ----------
    print("\n۱) منحنی markout — سود هر فیل در افق‌های مختلف (bps)")
    print("   سوال: سود با گذشت زمان می‌ماند یا آب می‌رود؟")
    print("  " + "─" * 92)
    hdr = f"  {'جفت':<13}" + "".join(f"{h:>8}s" for h in HORIZONS)
    print(hdr)
    print("  " + "─" * 92)

    peaks = {}
    agg = {h: [] for h in HORIZONS}
    for s, (A, f, rp) in data.items():
        c = markout_curve(A, f, a.fee)
        cells = ""
        best_h, best_v = None, -1e9
        for h in HORIZONS:
            v = c[h][0]
            cells += f"{v:>+9.1f}" if np.isfinite(v) else f"{'—':>9}"
            if np.isfinite(v):
                agg[h].append(v)
                if v > best_v:
                    best_v, best_h = v, h
        peaks[s] = (best_h, best_v)
        print(f"  {s.upper():<13}{cells}")

    print("  " + "─" * 92)
    means = [np.mean(agg[h]) if agg[h] else np.nan for h in HORIZONS]
    print(f"  {'میانگین':<13}" + "".join(f"{m:>+9.1f}" for m in means))

    ok_h = [h for h, m in zip(HORIZONS, means) if np.isfinite(m) and m > 0]
    if ok_h:
        print(f"\n  → markout تا {max(ok_h)} ثانیه مثبت می‌ماند.")
        bi = int(np.nanargmax(means))
        print(f"  → اوج در {HORIZONS[bi]} ثانیه با {means[bi]:+.1f}bps")
    else:
        print("\n  → markout در هیچ افقی مثبت نیست.")

    # ---------- بخش ۲: قانون خروج ----------
    print(f"\n\n۲) شبیه‌سازی قانون خروج")
    print("   بعد از فیل، لیمیت روی سمت مقابل. اگر تا T پر نشد → عبور از اسپرد.")
    print("  " + "─" * 92)
    print(f"  {'T':>5}{'خروج منفعل':>13}{'سود/دور':>12}{'برد':>8}"
          f"{'سود کل':>13}{'روزانه':>13}{'':>4}")
    print("  " + "─" * 92)

    best_row = None
    for T in EXIT_T:
        tot_usd, tot_n, bps_l, win_l, pas_l = 0.0, 0, [], [], []
        for s, (A, f, rp) in data.items():
            r = exit_rule(A, f, T, a.fee, a.taker_fee)
            if not r:
                continue
            tot_usd += r["usd"]
            tot_n += r["n"]
            bps_l.append(r["bps"])
            win_l.append(r["win"])
            pas_l.append(r["passive_rate"])
        if not bps_l:
            continue
        daily = tot_usd / hrs * 24
        tag = "✅" if daily > 0 else "❌"
        print(f"  {T:>4}s{np.mean(pas_l):>12.0%}{np.mean(bps_l):>+11.2f}"
              f"{np.mean(win_l):>8.0%}{tot_usd:>+12.2f}${daily:>+12.2f}$ {tag}")
        if best_row is None or daily > best_row[1]:
            best_row = (T, daily, np.mean(bps_l), np.mean(pas_l))

    # ---------- بخش ۳: حکم ----------
    print("\n" + "=" * 96)
    print("۳) حکم")
    print("=" * 96)
    if best_row and best_row[1] > 0:
        T, daily, bps, pas = best_row
        cap = a.size_usd * len(data) * 2
        print(f"  ✅ بهترین قانون خروج: حداکثر {T} ثانیه نگه دار")
        print(f"     سود هر دور: {bps:+.2f}bps   |   "
              f"خروج منفعل: {pas:.0%} مواقع")
        print(f"     روزانه {daily:+,.2f}$ روی سرمایه‌ی تقریبی {cap:,.0f}$"
              f"  ({daily*30/cap*100:+.1f}% ماهانه)")
        print(f"\n     معماری ربات: «بگیر و رها کن» — نه کوتیشن دوطرفه و صبر.")
    else:
        print("  ❌ هیچ قانون خروجی سودده نیست.")
        if ok_h:
            print(f"     ولی markout تا {max(ok_h)} ثانیه مثبت است — یعنی سود")
            print("     در لحظه وجود دارد ولی هزینه‌ی خروج آن را می‌خورد.")
            print(f"     با کارمزد تیکر {a.taker_fee:.0f}bps، هر خروج اجباری")
            print("     گران است. → یا باید خروج منفعل تضمین شود، یا رها کرد.")
        else:
            print("     و markout هم منفی است. این جفت‌ها قابل بازارگردانی نیستند.")

    print("\n⚠ خروج منفعل اینجا خوش‌بینانه مدل شده (صف خروج حساب نشده).")
    print("  عدد واقعی کمتر است.\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--size-usd", type=float, default=50)
    p.add_argument("--fee", type=float, default=2.0, help="کارمزد میکر bps")
    p.add_argument("--taker-fee", type=float, default=10.0,
                   help="کارمزد تیکر bps (اسپات LBank = ۱۰)")
    main(p.parse_args())
