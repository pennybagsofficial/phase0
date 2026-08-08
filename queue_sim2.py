#!/usr/bin/env python3
"""
QUEUE SIM v2 — با سقف موجودی و انتخاب رو به جلو
================================================
دو ایراد نسخه‌ی قبل رفع شد:

۱. سقف موجودی
   نسخه‌ی قبل اجازه می‌داد بی‌نهایت بخری. SCRT با سایز ۵۰ دلار
   ۸۷۴ خرید خالص انباشت — یعنی ۴۳ هزار دلار موقعیت با سرمایه‌ی ۳۰۰ دلار.
   آن «سود» شرط جهت‌دار بود، نه بازارگردانی.
   حالا وقتی به سقف برسی، آن سمت را کوتیشن نمی‌دهی.

۲. حسابداری نقدی واقعی
   به جای جمع زدن markout هر فیل، نقد و موجودی را دنبال می‌کنیم:
       خرید  → نقد کم، موجودی زیاد
       فروش  → نقد زیاد، موجودی کم
   سود نهایی = نقد + ارزش موجودی باقیمانده
   این تنها روشی است که ریسک موجودی را پنهان نمی‌کند.

۳. انتخاب رو به جلو (walk-forward)
   جفت‌ها بر اساس بخش اول داده انتخاب می‌شوند و روی بخش دوم
   ارزیابی می‌شوند. بدون این، فیلتر کردن بر اساس نرخ برد یعنی
   تقلب — از آینده خبر داشته‌ای.

اجرا:
    python queue_sim2.py data/q_XXXX.csv --size-usd 50 --max-inv-usd 150
    python queue_sim2.py data/q_XXXX.csv --train-frac 0.4 --min-win 0.60
"""

import argparse

import numpy as np
import pandas as pd

BPS = 1e4


def load(path):
    df = pd.read_csv(path)
    for c in ("ts_ms", "b1p", "b1q", "a1p", "a1q", "price", "qty"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["side"] = df.get("side", "").astype(str).str.lower().str.strip()
    return df.sort_values("ts_ms").reset_index(drop=True)


def simulate(sub, size_usd, max_inv_usd, fee):
    """
    حسابداری نقدی کامل با سقف موجودی.
    برمی‌گرداند: سود واقعی، آمار فیل، و بیشترین موجودی رسیده.
    """
    st = {"bid": None, "ask": None}
    cash = 0.0
    inv = 0.0                    # موجودی به واحد ارز پایه
    fills = []                   # (ts, side, px, qty, mid_at_fill)
    reposts = 0
    blocked = {"bid": 0, "ask": 0}
    last_mid = None
    inv_peak = 0.0
    best_bq = best_aq = 0.0

    rows = sub[["ts_ms", "kind", "b1p", "b1q", "a1p", "a1q",
                "price", "qty", "side"]].itertuples(index=False)

    for r in rows:
        if r.kind == "book":
            if not (np.isfinite(r.b1p) and np.isfinite(r.a1p)) or r.a1p <= r.b1p:
                continue
            mid = (r.b1p + r.a1p) / 2
            last_mid = mid
            best_bq = r.b1q if np.isfinite(r.b1q) else 0.0
            best_aq = r.a1q if np.isfinite(r.a1q) else 0.0

            inv_usd = inv * mid
            inv_peak = max(inv_peak, abs(inv_usd))

            # سقف موجودی: اگر پر شدی، آن سمت را کوتیشن نده
            allow = {"bid": inv_usd < max_inv_usd,
                     "ask": inv_usd > -max_inv_usd}

            for k, price, lvlq in (("bid", r.b1p, best_bq),
                                   ("ask", r.a1p, best_aq)):
                if not allow[k]:
                    if st[k] is not None:
                        st[k] = None          # سفارش را می‌کشیم
                    blocked[k] += 1
                    continue
                s = st[k]
                if s is None or s["price"] != price:
                    q = size_usd / price if price > 0 else 0.0
                    st[k] = {"price": price, "ahead": lvlq, "left": q}
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
                if k == "bid":                       # ما خریدیم
                    cash -= px * f * (1 + fee / BPS)
                    inv += f
                else:                                # ما فروختیم
                    cash += px * f * (1 - fee / BPS)
                    inv -= f
                fills.append((r.ts_ms, k, px, f, last_mid))
                if s["left"] <= 1e-12:
                    s["ahead"] = best_bq if k == "bid" else best_aq
                    s["left"] = size_usd / px if px > 0 else 0.0
                    reposts += 1

    if not fills or last_mid is None:
        return None

    # تسویه‌ی موجودی باقیمانده با قیمت میانه
    equity = cash + inv * last_mid
    hrs = (sub.ts_ms.max() - sub.ts_ms.min()) / 3.6e6
    if hrs <= 0:
        return None

    # سود هر فیل نسبت به میانه — فقط برای تشخیص، نه حسابداری
    marks = []
    for ts, k, px, q, mid in fills:
        if mid and mid > 0:
            sign = 1.0 if k == "bid" else -1.0
            marks.append((mid - px) / px * BPS * sign - fee)
    marks = np.array(marks) if marks else np.array([0.0])

    nb = sum(1 for f in fills if f[1] == "bid")
    notional = sum(f[2] * f[3] for f in fills)

    return {
        "hours": hrs,
        "fills": len(fills),
        "fills_hr": len(fills) / hrs,
        "bid": nb, "ask": len(fills) - nb,
        "imbalance": abs(nb - (len(fills) - nb)) / max(len(fills), 1),
        "reposts": reposts,
        "fill_rate": len(fills) / max(reposts, 1),
        "notional": notional,
        "pnl": equity,
        "pnl_hr": equity / hrs,
        "inv_peak": inv_peak,
        "inv_final": inv * last_mid,
        "blocked": blocked["bid"] + blocked["ask"],
        "win": float((marks > 0).mean()),
        "avg_bps": float(marks.mean()),
    }


def run_all(df, syms, size, maxinv, fee):
    out = {}
    for s in syms:
        sub = df[df.symbol == s]
        if len(sub) < 200:
            continue
        r = simulate(sub, size, maxinv, fee)
        if r:
            out[s] = r
    return out


def table(res, title, capital):
    print(f"\n{title}")
    print("  " + "─" * 84)
    print(f"  {'جفت':<14}{'فیل/ساعت':>10}{'نرخ فیل':>9}{'برد':>7}"
          f"{'نامتعادلی':>11}{'اوج موجودی':>13}{'سود/ساعت':>12}")
    print("  " + "─" * 84)
    tot = 0.0
    for s, r in sorted(res.items(), key=lambda x: -x[1]["pnl_hr"]):
        tot += r["pnl_hr"]
        tag = "✅" if r["pnl_hr"] > 0 else "❌"
        print(f"  {s.upper():<14}{r['fills_hr']:>9.1f}{r['fill_rate']:>8.1%}"
              f"{r['win']:>7.0%}{r['imbalance']:>10.0%}"
              f"{r['inv_peak']:>12,.0f}${r['pnl_hr']:>+11.3f}$ {tag}")
    print("  " + "─" * 84)
    print(f"  {'مجموع':<14}{'':>44}{tot:>+11.3f}$")
    print(f"  → روزانه {tot*24:+,.2f}$   ماهانه {tot*24*30:+,.2f}$"
          f"   ({tot*24*30/capital*100:+.1f}% ماهانه روی {capital:,.0f}$)")
    return tot


def main(a):
    df = load(a.csv)
    maxinv = a.max_inv_usd if a.max_inv_usd > 0 else a.size_usd * 3
    capital = maxinv * 2 + a.size_usd * 2
    syms = sorted(df.symbol.dropna().unique())

    t0, t1 = df.ts_ms.min(), df.ts_ms.max()
    split = t0 + (t1 - t0) * a.train_frac

    print("=" * 88)
    print(f"فایل: {a.csv}")
    print(f"سایز کوتیشن: ${a.size_usd:.0f}   |   سقف موجودی: ±${maxinv:,.0f}"
          f"   |   کارمزد میکر: {a.fee:.1f}bps")
    print(f"سرمایه‌ی لازم: ${capital:,.0f}")
    print("=" * 88)
    print("«نامتعادلی» = اختلاف خرید و فروش. بالای ۳۰٪ یعنی شرط جهت‌دار،")
    print("               نه بازارگردانی — حتی اگر سودده باشد.")

    # ---------- بخش ۱: کل دوره ----------
    full = run_all(df, syms, a.size_usd, maxinv, a.fee)
    if not full:
        print("\nهیچ فیلی رخ نداد.")
        return
    table(full, "۱) کل دوره — همه‌ی جفت‌ها", capital)

    # ---------- بخش ۲: انتخاب رو به جلو ----------
    tr = df[df.ts_ms <= split]
    te = df[df.ts_ms > split]
    hrs_tr = (split - t0) / 3.6e6
    hrs_te = (t1 - split) / 3.6e6

    print("\n\n" + "=" * 88)
    print(f"۲) انتخاب رو به جلو — آموزش {hrs_tr:.1f} ساعت، "
          f"آزمون {hrs_te:.1f} ساعت")
    print("=" * 88)
    print("سوال: آیا جفت‌های خوبِ ساعت اول، در ساعت‌های بعد هم خوب می‌مانند؟")
    print("این تنها راه تشخیص است که اسکن روزانه واقعاً کار می‌کند یا نه.")

    r_tr = run_all(tr, syms, a.size_usd, maxinv, a.fee)
    r_te = run_all(te, syms, a.size_usd, maxinv, a.fee)

    picked = [s for s, r in r_tr.items()
              if r["win"] >= a.min_win
              and r["pnl_hr"] > 0
              and r["imbalance"] <= a.max_imbalance
              and r["fills_hr"] >= a.min_fills]

    print(f"\nمعیار انتخاب: برد ≥ {a.min_win:.0%}، سود مثبت، "
          f"نامتعادلی ≤ {a.max_imbalance:.0%}، فیل/ساعت ≥ {a.min_fills}")
    print(f"انتخاب‌شده در دوره‌ی آموزش: "
          f"{', '.join(s.upper() for s in picked) if picked else '— هیچ‌کدام —'}")

    if not picked:
        print("\n❌ هیچ جفتی معیارها را در دوره‌ی آموزش پاس نکرد.")
        print("   → یا معیارها را شل کن، یا این سبد اصلاً مناسب نیست.")
        return

    sel_te = {s: r for s, r in r_te.items() if s in picked}
    all_te = r_te

    if sel_te:
        t_sel = table(sel_te, "\nنتیجه در دوره‌ی آزمون — فقط جفت‌های انتخاب‌شده",
                      capital)
    else:
        print("\nجفت‌های انتخاب‌شده در دوره‌ی آزمون فیلی نگرفتند.")
        t_sel = 0.0
    t_all = sum(r["pnl_hr"] for r in all_te.values())

    print("\n" + "=" * 88)
    print("۳) حکم")
    print("=" * 88)
    print(f"  بدون فیلتر (همه‌ی جفت‌ها) در آزمون : {t_all*24:>+10,.2f}$ در روز")
    print(f"  با فیلتر (انتخاب از آموزش)        : {t_sel*24:>+10,.2f}$ در روز")

    print()
    if t_sel > 0 and t_sel > t_all:
        print("  ✅ فیلتر کار کرد — و در داده‌ای که ندیده بود.")
        print("     یعنی انتخاب روزانه‌ی جفت‌ها منطق واقعی دارد.")
        print(f"     ماهانه: {t_sel*24*30:+,.2f}$ روی سرمایه‌ی {capital:,.0f}$"
              f"  ({t_sel*24*30/capital*100:+.1f}%)")
    elif t_sel > 0:
        print("  🟡 فیلتر سودده است ولی از حالت بدون فیلتر بهتر نشد.")
    else:
        print("  ❌ جفت‌های خوبِ دوره‌ی آموزش، در دوره‌ی آزمون سودده نبودند.")
        print("     یعنی نرخ برد گذشته، آینده را پیش‌بینی نمی‌کند.")
        print("     → استراتژی انتخاب روزانه بر پایه‌ی نرخ برد جواب نمی‌دهد.")

    print("\n⚠ همچنان مدل نشده: تأخیر ارسال سفارش، کارمزد خروج اضطراری،")
    print("  و اینکه رقبا به کوتیشن تو واکنش نشان می‌دهند.\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--size-usd", type=float, default=50)
    p.add_argument("--max-inv-usd", type=float, default=0,
                   help="سقف موجودی هر طرف (۰ = سه برابر سایز)")
    p.add_argument("--fee", type=float, default=2.0)
    p.add_argument("--train-frac", type=float, default=0.4)
    p.add_argument("--min-win", type=float, default=0.60)
    p.add_argument("--max-imbalance", type=float, default=0.30)
    p.add_argument("--min-fills", type=float, default=5.0)
    main(p.parse_args())
