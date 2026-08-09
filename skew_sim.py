#!/usr/bin/env python3
"""
SKEW SIM — بازارگردانی با اریب موجودی
======================================
تشخیص از داده‌ی واقعی:

    markout مثبت بود (+۱۰ تا +۱۶bps) ولی سود نقدی منفی.
    کل تفاوت = ضرر موجودی.

    ربات می‌خرید، به سقف می‌رسید، گیر می‌کرد، و قیمت علیه‌اش می‌رفت.
    نامتعادلی ۷۰ تا ۱۰۰٪ در جفت‌های ضررده.

سه مکانیزم اضافه شد:

۱. اریب موجودی (skew)
   کوتیشن متقارن نمی‌دهیم. اگر لانگ شدیم، سمت فروش را به قیمت
   بهتری می‌بریم تا زودتر تخلیه شود، و سمت خرید را عقب می‌کشیم.
       اریب = gamma × (موجودی ÷ سقف)

۲. فیلتر سمّی بودن
   قانونی که در داده پیدا شد: نرخ فیل بالا ⇒ markout منفی، بدون استثنا.
   جفت‌هایی با نرخ فیل بالای آستانه اصلاً وارد نمی‌شویم.

۳. تخلیه‌ی اجباری
   اگر موجودی از حد بحرانی گذشت، با عبور از اسپرد صاف می‌کنیم.
   ضرر کوچک الان، بهتر از ضرر بزرگ بعداً.

اجرا:
    python skew_sim.py data/q_XXXX.csv --size-usd 50
    python skew_sim.py data/q_XXXX.csv --gamma 1.0 --max-fill-rate 0.15
    python skew_sim.py data/q_XXXX.csv --sweep       # جستجوی بهترین تنظیم
"""

import argparse
import itertools

import numpy as np
import pandas as pd

MARKOUT_S = 5
BPS = 1e4


def load(path):
    df = pd.read_csv(path)
    for c in ("ts_ms", "b1p", "b1q", "a1p", "a1q", "price", "qty"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["side"] = df.get("side", "").astype(str).str.lower().str.strip()
    return df.sort_values("ts_ms").reset_index(drop=True)


def simulate(sub, size_usd, max_inv, fee, gamma, flat_at, tick_frac=0.5):
    """
    gamma    : شدت اریب. ۰ = متقارن (رفتار قبلی). ۱ = اریب کامل.
    flat_at  : اگر |موجودی| از این کسر سقف گذشت، اجباری صاف کن.
    """
    st = {"bid": None, "ask": None}
    cash = inv = 0.0
    fills, reposts, flattens = [], 0, 0
    mts, mpx = [], []
    last_mid = None
    inv_peak = 0.0
    inv_sum = inv_n = 0.0
    bq = aq = 0.0

    rows = sub[["ts_ms", "kind", "b1p", "b1q", "a1p", "a1q",
                "price", "qty", "side"]].itertuples(index=False)

    for r in rows:
        if r.kind == "book":
            if not (np.isfinite(r.b1p) and np.isfinite(r.a1p)) or r.a1p <= r.b1p:
                continue
            bid1, ask1 = r.b1p, r.a1p
            mid = (bid1 + ask1) / 2
            last_mid = mid
            mts.append(r.ts_ms)
            mpx.append(mid)
            bq = r.b1q if np.isfinite(r.b1q) else 0.0
            aq = r.a1q if np.isfinite(r.a1q) else 0.0

            inv_usd = inv * mid
            inv_peak = max(inv_peak, abs(inv_usd))
            inv_sum += abs(inv_usd)
            inv_n += 1
            ratio = inv_usd / max_inv if max_inv > 0 else 0.0

            # ---- ۳. تخلیه‌ی اجباری ----
            if abs(ratio) >= flat_at:
                if inv > 0:                       # لانگیم → بفروش روی بید
                    cash += bid1 * inv * (1 - fee / BPS)
                    inv = 0.0
                else:
                    cash -= ask1 * (-inv) * (1 + fee / BPS)
                    inv = 0.0
                flattens += 1
                st = {"bid": None, "ask": None}
                continue

            # ---- ۱. اریب موجودی ----
            # لانگ (ratio>0) ⇒ سمت فروش را پایین بیاور تا زودتر پر شود،
            #                  سمت خرید را عقب بکش تا کمتر بخری.
            spread = ask1 - bid1
            step = spread * tick_frac
            skew = gamma * ratio * step

            if skew > 0:                          # لانگ
                want_bid = bid1 - skew            # کمتر مشتاق خرید
                want_ask = max(ask1 - skew, bid1 + spread * 0.1)  # مشتاق فروش
            elif skew < 0:                        # شورت
                want_ask = ask1 - skew            # کمتر مشتاق فروش
                want_bid = min(bid1 - skew, ask1 - spread * 0.1)  # مشتاق خرید
            else:
                want_bid, want_ask = bid1, ask1

            for k, price in (("bid", want_bid), ("ask", want_ask)):
                if price <= 0:
                    continue
                # اگر از بهترین سطح بهتر بودیم، اول صف می‌ایستیم
                at_touch = (abs(price - bid1) < 1e-12) if k == "bid" \
                    else (abs(price - ask1) < 1e-12)
                ahead = (bq if k == "bid" else aq) if at_touch else 0.0
                s = st[k]
                if s is None or abs(s["price"] - price) > 1e-12:
                    st[k] = {"price": price, "ahead": ahead,
                             "left": size_usd / price}
                    reposts += 1

        elif r.kind == "trade":
            if not (np.isfinite(r.price) and np.isfinite(r.qty)) or r.qty <= 0:
                continue
            agg_sell = r.side.startswith("s")
            k = "bid" if agg_sell else "ask"
            s = st[k]
            if s is None:
                continue
            hit = (r.price <= s["price"] + 1e-12) if agg_sell \
                else (r.price >= s["price"] - 1e-12)
            if not hit:
                continue

            vol = r.qty
            eat = min(vol, s["ahead"])
            s["ahead"] -= eat
            vol -= eat
            if vol > 0 and s["left"] > 0:
                f = min(vol, s["left"])
                s["left"] -= f
                px = s["price"]
                if k == "bid":
                    cash -= px * f * (1 + fee / BPS)
                    inv += f
                else:
                    cash += px * f * (1 - fee / BPS)
                    inv -= f
                fills.append((r.ts_ms, k, px, f))
                if s["left"] <= 1e-12:
                    st[k] = None

    if not fills or last_mid is None or len(mts) < 50:
        return None

    hrs = (sub.ts_ms.max() - sub.ts_ms.min()) / 3.6e6
    if hrs <= 0:
        return None

    mts = np.array(mts, dtype=np.int64)
    mpx = np.array(mpx, dtype=float)

    marks = []
    for ts, k, px, q in fills:
        i = np.searchsorted(mts, ts + MARKOUT_S * 1000, side="right") - 1
        if i >= 0:
            sign = 1.0 if k == "bid" else -1.0
            marks.append((mpx[i] - px) / px * BPS * sign - fee)
    marks = np.array(marks) if marks else np.array([0.0])

    nb = sum(1 for f in fills if f[1] == "bid")
    notional = sum(f[2] * f[3] for f in fills)

    return {
        "hours": hrs,
        "fills": len(fills),
        "fills_hr": len(fills) / hrs,
        "fill_rate": len(fills) / max(reposts, 1),
        "imbalance": abs(2 * nb - len(fills)) / max(len(fills), 1),
        "notional_hr": notional / hrs,
        "pnl_hr": (cash + inv * last_mid) / hrs,
        "inv_peak": inv_peak,
        "inv_avg": inv_sum / max(inv_n, 1),
        "flattens_hr": flattens / hrs,
        "win": float((marks > 0).mean()),
        "markout": float(marks.mean()),
    }


def run(df, syms, **kw):
    out = {}
    for s in syms:
        sub = df[df.symbol == s]
        if len(sub) < 200:
            continue
        r = simulate(sub, **kw)
        if r:
            out[s] = r
    return out


def main(a):
    df = load(a.csv)
    maxinv = a.max_inv_usd if a.max_inv_usd > 0 else a.size_usd * 3
    syms = sorted(df.symbol.dropna().unique())
    t0, t1 = df.ts_ms.min(), df.ts_ms.max()
    split = t0 + (t1 - t0) * a.train_frac
    tr, te = df[df.ts_ms <= split], df[df.ts_ms > split]

    print("=" * 94)
    print(f"فایل: {a.csv}   |   {len(syms)} جفت   |   {(t1-t0)/3.6e6:.1f} ساعت")
    print(f"سایز ${a.size_usd:.0f}   سقف موجودی ±${maxinv:,.0f}   "
          f"کارمزد {a.fee:.1f}bps")
    print("=" * 94)

    base = dict(size_usd=a.size_usd, max_inv=maxinv, fee=a.fee)

    # ---------- گام ۱: فیلتر سمّی بودن روی دوره‌ی آموزش ----------
    probe = run(tr, syms, gamma=0.0, flat_at=1.0, **base)
    clean = [s for s, r in probe.items() if r["fill_rate"] <= a.max_fill_rate]
    toxic = [s for s in probe if s not in clean]

    print(f"\n۱) فیلتر سمّی بودن (نرخ فیل > {a.max_fill_rate:.0%} = رد)")
    print(f"   قانون کشف‌شده از داده: هر جا راحت فیل می‌شوی، طرف مقابل می‌داند.")
    print(f"   پاک: {len(clean)} جفت   |   سمّی (حذف‌شده): "
          f"{', '.join(s.upper() for s in toxic) if toxic else '—'}")
    if not clean:
        print("   هیچ جفت پاکی نماند. آستانه را شل کن.")
        return

    # ---------- گام ۲: مقایسه‌ی متقارن و اریب ----------
    print(f"\n\n۲) اثر اریب موجودی — روی دوره‌ی آزمون")
    print("  " + "─" * 90)
    print(f"  {'حالت':<24}{'سود/ساعت':>13}{'نامتعادلی':>12}"
          f"{'موجودی متوسط':>15}{'تخلیه/ساعت':>13}{'فیل/ساعت':>12}")
    print("  " + "─" * 90)

    variants = [("متقارن (رفتار قبلی)", 0.0, 1.0)]
    for g in (a.gamma, a.gamma * 2):
        variants.append((f"اریب gamma={g:.1f}", g, a.flat_at))
    variants.append((f"اریب + تخلیه‌ی زودتر", a.gamma, a.flat_at * 0.6))

    best = None
    for name, g, fl in variants:
        rr = run(te, clean, gamma=g, flat_at=fl, **base)
        if not rr:
            continue
        pnl = sum(v["pnl_hr"] for v in rr.values())
        imb = np.mean([v["imbalance"] for v in rr.values()])
        iav = np.mean([v["inv_avg"] for v in rr.values()])
        fl_h = np.mean([v["flattens_hr"] for v in rr.values()])
        fh = np.mean([v["fills_hr"] for v in rr.values()])
        tag = "✅" if pnl > 0 else "❌"
        print(f"  {name:<24}{pnl:>+12.3f}${imb:>11.0%}{iav:>14,.0f}$"
              f"{fl_h:>12.1f}{fh:>11.1f} {tag}")
        if best is None or pnl > best[1]:
            best = (name, pnl, g, fl, rr)

    # ---------- گام ۳: جستجوی شبکه‌ای ----------
    if a.sweep:
        print(f"\n\n۳) جستجوی بهترین تنظیم (روی آموزش، ارزیابی روی آزمون)")
        print("  " + "─" * 90)
        print(f"  {'gamma':>8}{'flat_at':>10}{'سود آموزش':>14}"
              f"{'سود آزمون':>14}{'نامتعادلی':>12}")
        print("  " + "─" * 90)
        results = []
        for g, fl in itertools.product((0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
                                       (0.4, 0.6, 0.8, 1.0)):
            a_tr = run(tr, clean, gamma=g, flat_at=fl, **base)
            a_te = run(te, clean, gamma=g, flat_at=fl, **base)
            if not a_tr or not a_te:
                continue
            p_tr = sum(v["pnl_hr"] for v in a_tr.values())
            p_te = sum(v["pnl_hr"] for v in a_te.values())
            imb = np.mean([v["imbalance"] for v in a_te.values()])
            results.append((g, fl, p_tr, p_te, imb))
        results.sort(key=lambda x: -x[2])          # رتبه بر اساس آموزش
        for g, fl, p_tr, p_te, imb in results[:10]:
            mark = "  ◄ بهترین در آموزش" if (g, fl) == (results[0][0], results[0][1]) else ""
            print(f"  {g:>8.1f}{fl:>10.1f}{p_tr:>+13.3f}${p_te:>+13.3f}$"
                  f"{imb:>11.0%}{mark}")
        if results:
            g, fl, p_tr, p_te, _ = results[0]
            print(f"\n  بهترین تنظیم آموزش: gamma={g}, flat_at={fl}")
            print(f"  عملکردش در داده‌ی ندیده: {p_te*24:+.2f}$ روزانه")
            if p_te > 0:
                print("  ✅ در داده‌ی ندیده هم مثبت ماند.")
            else:
                print("  ❌ در داده‌ی ندیده مثبت نماند — بیش‌برازش.")

    # ---------- گام ۴: جزئیات بهترین حالت ----------
    if best and best[4]:
        name, pnl, g, fl, rr = best
        print(f"\n\n۴) جزئیات — {name}")
        print("  " + "─" * 90)
        print(f"  {'جفت':<14}{'فیل/ساعت':>10}{'نرخ فیل':>9}{'markout':>10}"
              f"{'برد':>7}{'نامتعادلی':>11}{'سود/ساعت':>12}")
        print("  " + "─" * 90)
        for s, r in sorted(rr.items(), key=lambda x: -x[1]["pnl_hr"]):
            print(f"  {s.upper():<14}{r['fills_hr']:>9.1f}{r['fill_rate']:>8.1%}"
                  f"{r['markout']:>+9.2f}{r['win']:>7.0%}{r['imbalance']:>10.0%}"
                  f"{r['pnl_hr']:>+11.3f}$")
        cap = len(rr) * 2 * maxinv
        print("  " + "─" * 90)
        print(f"  سرمایه ${cap:,.0f}  →  روزانه {pnl*24:+,.2f}$  "
              f"→  ماهانه {pnl*24*30:+,.2f}$ ({pnl*24*30/cap*100:+.1f}%)")

    print("\n⚠ اریب موجودی یک معامله است: نامتعادلی کمتر، ولی فیل کمتر هم.")
    print("  اگر سود با اریب بالا نرفت، یعنی ضرر موجودی مشکل اصلی نبوده.\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--size-usd", type=float, default=50)
    p.add_argument("--max-inv-usd", type=float, default=0)
    p.add_argument("--fee", type=float, default=2.0)
    p.add_argument("--gamma", type=float, default=1.0, help="شدت اریب موجودی")
    p.add_argument("--flat-at", type=float, default=0.8,
                   help="کسری از سقف که در آن اجباری صاف می‌کنیم")
    p.add_argument("--max-fill-rate", type=float, default=0.15,
                   help="نرخ فیل بالاتر از این = جفت سمّی، حذف")
    p.add_argument("--train-frac", type=float, default=0.4)
    p.add_argument("--sweep", action="store_true", help="جستجوی شبکه‌ای تنظیمات")
    main(p.parse_args())
