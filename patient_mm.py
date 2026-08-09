#!/usr/bin/env python3
"""
PATIENT MM — بازارگردانی صبور
==============================
چهار تغییر، همه از خود داده:

۱. کوتیشن عقب‌تر، نه روی بهترین قیمت
   قبلاً همیشه روی بهترین بید/اسک می‌ایستادیم → ته صف، نرخ فیل بالا،
   و فیل فقط وقتی که طرف مقابل می‌دانست چه می‌کند.
   حالا N تیک عقب‌تر: کمتر فیل، ولی هر فیل حاشیه‌ی بزرگ‌تر.

۲. خروج فقط منفعل — هرگز عبور از اسپرد
   کارمزد تیکر ۱۰bps کل لبه‌ی ۱۰.۶bps را می‌خورد. پس هیچ‌وقت تیکر نمی‌شویم.
   سفارش خروج را روی سمت مقابل می‌گذاریم و با حرکت قیمت جابه‌جایش می‌کنیم.

۳. صبر بلند
   داده نشان داد markout در ۳۰۰ ثانیه از ۵ ثانیه بیشتر است
   (SHELL از +۱۷ به +۴۴). پس افق نگهداری را بلند می‌کنیم.

۴. انتخاب جفت بر پایه‌ی شیب منحنی markout
   نه اسپرد، نه نرخ فیل. شیب صعودی = صبر پاداش دارد.

اجرا:
    python patient_mm.py data/q_XXXX.csv --size-usd 50
    python patient_mm.py data/q_XXXX.csv --sweep
"""

import argparse
import itertools

import numpy as np
import pandas as pd

BPS = 1e4
SLOPE_H = [1, 5, 30, 120, 300]     # افق‌های سنجش شیب


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


def slope_score(A, size_usd, maker_fee):
    """شیب منحنی markout: آیا صبر پاداش دارد؟"""
    st = {"bid": None, "ask": None}
    fills = []
    bi = ti = 0
    nb, nt = len(A["bts"]), len(A["tts"])
    while bi < nb or ti < nt:
        if ti >= nt or (bi < nb and A["bts"][bi] <= A["tts"][ti]):
            for k, price, q in (("bid", A["bid"][bi], A["bq"][bi]),
                                ("ask", A["ask"][bi], A["aq"][bi])):
                s = st[k]
                if s is None or s["price"] != price:
                    st[k] = {"price": price, "ahead": q, "left": size_usd / price}
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
        v = []
        for ts, k, px in fills:
            m = mid_at(A, ts + int(h * 1000))
            if np.isfinite(m) and m > 0:
                v.append((m - px) / px * BPS * (1 if k == "bid" else -1) - maker_fee)
        curve.append(np.mean(v) if v else np.nan)
    if any(not np.isfinite(c) for c in curve):
        return None
    return {"curve": curve, "short": curve[1], "long": curve[-1],
            "slope": curve[-1] - curve[1], "n": len(fills)}


def simulate(A, size_usd, offset_ticks, max_hold_s, maker_fee, max_inv_usd):
    """
    بازارگردانی صبور:
      • کوتیشن offset_ticks تیک عقب‌تر از بهترین قیمت
      • خروج فقط منفعل، سفارش با قیمت جابه‌جا می‌شود
      • هیچ عبوری از اسپرد
    """
    cash = inv = 0.0
    open_pos = []          # [(ts, side, entry_px, qty)]
    st = {"bid": None, "ask": None}
    n_fill = n_close = 0
    bi = ti = 0
    nb, nt = len(A["bts"]), len(A["tts"])
    last_mid = np.nan
    inv_peak = 0.0
    hold_sum = 0.0

    while bi < nb or ti < nt:
        book_first = ti >= nt or (bi < nb and A["bts"][bi] <= A["tts"][ti])

        if book_first:
            b1, a1 = A["bid"][bi], A["ask"][bi]
            spread = a1 - b1
            tick = spread if spread > 0 else b1 * 1e-4
            last_mid = (b1 + a1) / 2
            inv_usd = inv * last_mid
            inv_peak = max(inv_peak, abs(inv_usd))

            # کوتیشن ورود، عقب‌تر از بهترین قیمت
            want_bid = b1 - offset_ticks * tick
            want_ask = a1 + offset_ticks * tick

            allow_bid = inv_usd < max_inv_usd
            allow_ask = inv_usd > -max_inv_usd

            for k, price, allow in (("bid", want_bid, allow_bid),
                                    ("ask", want_ask, allow_ask)):
                if not allow or price <= 0:
                    st[k] = None
                    continue
                s = st[k]
                if s is None or abs(s["price"] - price) > 1e-12:
                    # عقب‌تر از تاچ ⇒ صف جلویی نداریم
                    ahead = 0.0 if offset_ticks > 0 else \
                        (A["bq"][bi] if k == "bid" else A["aq"][bi])
                    st[k] = {"price": price, "ahead": ahead,
                             "left": size_usd / price}
            bi += 1

        else:
            ts = A["tts"][ti]
            sell = A["tsell"][ti]
            px_t = A["tpx"][ti]
            vol = A["tq"][ti]

            # --- ورود ---
            k = "bid" if sell else "ask"
            s = st[k]
            if s is not None:
                hit = (px_t <= s["price"] + 1e-12) if sell \
                    else (px_t >= s["price"] - 1e-12)
                if hit:
                    e = min(vol, s["ahead"])
                    s["ahead"] -= e
                    rem = vol - e
                    if rem > 0 and s["left"] > 0:
                        f = min(rem, s["left"])
                        s["left"] -= f
                        p = s["price"]
                        if k == "bid":
                            cash -= p * f * (1 + maker_fee / BPS)
                            inv += f
                            open_pos.append([ts, 1, p, f])
                        else:
                            cash += p * f * (1 - maker_fee / BPS)
                            inv -= f
                            open_pos.append([ts, -1, p, f])
                        n_fill += 1
                        if s["left"] <= 1e-12:
                            st[k] = None
                        vol = max(rem - f, 0.0)

            # --- خروج منفعل موقعیت‌های باز ---
            if open_pos and vol > 0:
                j = np.searchsorted(A["bts"], ts, side="right") - 1
                if j >= 0:
                    b1, a1 = A["bid"][j], A["ask"][j]
                    still = []
                    for pos in open_pos:
                        p_ts, sgn, e_px, q = pos
                        if q <= 1e-15:
                            continue
                        # لانگ ⇒ فروش روی اسک، نیازمند خریدار تهاجمی
                        tgt = a1 if sgn > 0 else b1
                        can = ((not sell) and px_t >= tgt - 1e-12) if sgn > 0 \
                            else (sell and px_t <= tgt + 1e-12)
                        # فقط اگر سود بدهد
                        good = (tgt > e_px) if sgn > 0 else (tgt < e_px)
                        if can and good and vol > 0:
                            f = min(vol, q)
                            vol -= f
                            if sgn > 0:
                                cash += tgt * f * (1 - maker_fee / BPS)
                                inv -= f
                            else:
                                cash -= tgt * f * (1 + maker_fee / BPS)
                                inv += f
                            pos[3] -= f
                            n_close += 1
                            hold_sum += (ts - p_ts) / 1000
                        if pos[3] > 1e-15:
                            still.append(pos)
                    open_pos = still
            ti += 1

    if n_fill < 20 or not np.isfinite(last_mid):
        return None

    hrs = (max(A["bts"][-1], A["tts"][-1]) - min(A["bts"][0], A["tts"][0])) / 3.6e6
    if hrs <= 0:
        return None

    equity = cash + inv * last_mid
    return {
        "hours": hrs,
        "fills": n_fill,
        "closes": n_close,
        "close_rate": n_close / max(n_fill, 1),
        "fills_hr": n_fill / hrs,
        "avg_hold_s": hold_sum / max(n_close, 1),
        "pnl_hr": equity / hrs,
        "inv_peak": inv_peak,
        "inv_final": inv * last_mid,
    }


def main(a):
    df = load(a.csv)
    syms = sorted(df.symbol.dropna().unique())
    maxinv = a.max_inv_usd if a.max_inv_usd > 0 else a.size_usd * 4

    print("=" * 96)
    print(f"فایل: {a.csv}   |   {len(syms)} جفت")
    print(f"سایز ${a.size_usd:.0f}   میکر {a.fee:.1f}bps   "
          f"سقف موجودی ±${maxinv:,.0f}   |   بدون عبور از اسپرد")
    print("=" * 96)

    A = {}
    for s in syms:
        x = arrays(df[df.symbol == s])
        if x:
            A[s] = x
    if not A:
        print("داده کافی نیست.")
        return

    # ---------- بخش ۱: شیب منحنی ----------
    print("\n۱) شیب منحنی markout — آیا صبر پاداش دارد؟")
    print("  " + "─" * 92)
    print(f"  {'جفت':<14}" + "".join(f"{h:>9}s" for h in SLOPE_H) +
          f"{'شیب':>10}{'':>5}")
    print("  " + "─" * 92)

    scores = {}
    for s, x in A.items():
        sc = slope_score(x, a.size_usd, a.fee)
        if not sc:
            continue
        scores[s] = sc
        good = sc["slope"] > a.min_slope and sc["long"] > 0
        cells = "".join(f"{c:>+10.1f}" for c in sc["curve"])
        print(f"  {s.upper():<14}{cells}{sc['slope']:>+10.1f}"
              f"  {'✅' if good else '❌'}")

    picked = [s for s, sc in scores.items()
              if sc["slope"] > a.min_slope and sc["long"] > 0]
    print(f"\n  انتخاب‌شده (شیب > {a.min_slope:.0f} و افق بلند مثبت): "
          f"{len(picked)} جفت")
    if not picked:
        print("  هیچ جفتی شیب صعودی ندارد. --min-slope را پایین بیاور.")
        return
    print(f"  {', '.join(s.upper() for s in picked)}")

    # ---------- بخش ۲: اثر فاصله‌ی کوتیشن ----------
    print(f"\n\n۲) اثر عقب‌تر گذاشتن کوتیشن — روی جفت‌های انتخاب‌شده")
    print("   ۰ = روی بهترین قیمت (رفتار قبلی)، بالاتر = عقب‌تر و صبورتر")
    print("  " + "─" * 92)
    print(f"  {'فاصله':>8}{'فیل/ساعت':>12}{'نرخ بستن':>12}"
          f"{'نگهداری':>11}{'اوج موجودی':>13}{'سود/ساعت':>13}{'روزانه':>13}")
    print("  " + "─" * 92)

    best = None
    for off in (0, 1, 2, 3, 5):
        tot, fh, cr, hold, pk, n = 0.0, [], [], [], [], 0
        for s in picked:
            r = simulate(A[s], a.size_usd, off, a.max_hold, a.fee, maxinv)
            if not r:
                continue
            tot += r["pnl_hr"]
            fh.append(r["fills_hr"])
            cr.append(r["close_rate"])
            hold.append(r["avg_hold_s"])
            pk.append(r["inv_peak"])
            n += 1
        if n == 0:
            continue
        tag = "✅" if tot > 0 else "❌"
        print(f"  {off:>8}{np.mean(fh):>11.1f}{np.mean(cr):>11.0%}"
              f"{np.mean(hold):>10.0f}s{np.mean(pk):>12,.0f}$"
              f"{tot:>+12.3f}${tot*24:>+12.2f}$ {tag}")
        if best is None or tot > best[1]:
            best = (off, tot)

    # ---------- بخش ۳: جستجو ----------
    if a.sweep:
        print(f"\n\n۳) جستجوی ترکیب فاصله و سقف موجودی")
        print("  " + "─" * 92)
        print(f"  {'فاصله':>8}{'سقف موجودی':>14}{'سود/ساعت':>14}"
              f"{'روزانه':>14}{'ماهانه٪':>14}")
        print("  " + "─" * 92)
        out = []
        for off, mi in itertools.product((0, 1, 2, 3, 5),
                                         (a.size_usd * k for k in (2, 4, 8))):
            tot = 0.0
            for s in picked:
                r = simulate(A[s], a.size_usd, off, a.max_hold, a.fee, mi)
                if r:
                    tot += r["pnl_hr"]
            cap = len(picked) * 2 * mi
            out.append((off, mi, tot, cap))
        out.sort(key=lambda x: -x[2])
        for off, mi, tot, cap in out[:10]:
            print(f"  {off:>8}{mi:>13,.0f}${tot:>+13.3f}${tot*24:>+13.2f}$"
                  f"{tot*24*30/cap*100:>13.1f}%")

    # ---------- حکم ----------
    print("\n" + "=" * 96)
    if best and best[1] > 0:
        off, tot = best
        cap = len(picked) * 2 * maxinv
        print(f"✅ بهترین: کوتیشن {off} تیک عقب‌تر → "
              f"{tot*24:+,.2f}$ روزانه")
        print(f"   روی سرمایه‌ی {cap:,.0f}$  = {tot*24*30/cap*100:+.1f}% ماهانه")
        print(f"   با {len(picked)} جفت، بدون هیچ عبوری از اسپرد.")
    else:
        print("❌ حتی با صبر و کوتیشن عقب‌تر هم مثبت نشد.")
        print("   بخش ۲ نشان می‌دهد کدام اهرم بیشترین اثر را داشت.")
    print("\n⚠ خروج منفعل خوش‌بینانه است (صف خروج مدل نشده).")
    print("  موجودی باز در پایان با قیمت میانه ارزش‌گذاری شده.\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--size-usd", type=float, default=50)
    p.add_argument("--fee", type=float, default=2.0)
    p.add_argument("--max-inv-usd", type=float, default=0)
    p.add_argument("--max-hold", type=float, default=900)
    p.add_argument("--min-slope", type=float, default=1.0,
                   help="حداقل رشد markout از ۵s تا ۳۰۰s")
    p.add_argument("--sweep", action="store_true")
    main(p.parse_args())
