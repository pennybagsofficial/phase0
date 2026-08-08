#!/usr/bin/env python3
"""
PORTFOLIO SIM — مقیاس‌دهی به چند جفت
=====================================
سه تغییر نسبت به queue_sim2:

۱. باگ markout رفع شد
   قبلاً قیمت میانه را در *لحظه‌ی* فیل می‌گرفتم. چون همیشه روی بید
   می‌خریم، (میانه − بید) همیشه مثبت است → نرخ برد ۱۰۰٪ که بی‌معناست.
   حالا میانه را در t+5s می‌گیرد. این عدد واقعی است.

۲. منحنی مقیاس
   بزرگ‌ترین اهرم استفاده‌نشده: تعداد جفت‌ها. اسکنر ۹۹ جفت واجد شرایط
   پیدا کرد؛ ما روی ۱ تا تست کردیم. این اسکریپت نشان می‌دهد بازده با
   افزودن جفت چطور تغییر می‌کند — و کجا اشباع می‌شود.

۳. رصد حجم
   کارمزد صفر با حجم می‌آید نه سرمایه. گزارش می‌دهد ماهانه چقدر حجم
   تولید می‌کنی، تا بدانی چقدر تا رده‌ی کارمزد صفر فاصله داری.

اجرا:
    python portfolio_sim.py data/q_XXXX.csv --size-usd 50
    python portfolio_sim.py data/q_XXXX.csv --fee 0     # سناریوی کارمزد صفر
"""

import argparse

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


def simulate(sub, size_usd, max_inv_usd, fee):
    """حسابداری نقدی با سقف موجودی. فیل‌ها را با زمان برمی‌گرداند."""
    st = {"bid": None, "ask": None}
    cash = inv = 0.0
    fills, reposts = [], 0
    mts, mpx = [], []
    last_mid = None
    inv_peak = 0.0
    bq = aq = 0.0
    blocked = 0

    rows = sub[["ts_ms", "kind", "b1p", "b1q", "a1p", "a1q",
                "price", "qty", "side"]].itertuples(index=False)

    for r in rows:
        if r.kind == "book":
            if not (np.isfinite(r.b1p) and np.isfinite(r.a1p)) or r.a1p <= r.b1p:
                continue
            mid = (r.b1p + r.a1p) / 2
            last_mid = mid
            mts.append(r.ts_ms)
            mpx.append(mid)
            bq = r.b1q if np.isfinite(r.b1q) else 0.0
            aq = r.a1q if np.isfinite(r.a1q) else 0.0

            inv_usd = inv * mid
            inv_peak = max(inv_peak, abs(inv_usd))
            allow = {"bid": inv_usd < max_inv_usd,
                     "ask": inv_usd > -max_inv_usd}

            for k, price, lvlq in (("bid", r.b1p, bq), ("ask", r.a1p, aq)):
                if not allow[k]:
                    st[k] = None
                    blocked += 1
                    continue
                s = st[k]
                if s is None or s["price"] != price:
                    st[k] = {"price": price, "ahead": lvlq,
                             "left": size_usd / price if price > 0 else 0.0}
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
                    s["ahead"] = bq if k == "bid" else aq
                    s["left"] = size_usd / px if px > 0 else 0.0
                    reposts += 1

    if not fills or last_mid is None or len(mts) < 50:
        return None

    hrs = (sub.ts_ms.max() - sub.ts_ms.min()) / 3.6e6
    if hrs <= 0:
        return None

    mts = np.array(mts, dtype=np.int64)
    mpx = np.array(mpx, dtype=float)

    # --- markout درست: میانه در t + MARKOUT_S، نه لحظه‌ی فیل ---
    marks = []
    for ts, k, px, q in fills:
        i = np.searchsorted(mts, ts + MARKOUT_S * 1000, side="right") - 1
        if i < 0:
            continue
        sign = 1.0 if k == "bid" else -1.0
        marks.append((mpx[i] - px) / px * BPS * sign - fee)
    marks = np.array(marks) if marks else np.array([0.0])

    nb = sum(1 for f in fills if f[1] == "bid")
    notional = sum(f[2] * f[3] for f in fills)

    return {
        "hours": hrs,
        "fills": len(fills),
        "fills_hr": len(fills) / hrs,
        "imbalance": abs(2 * nb - len(fills)) / max(len(fills), 1),
        "fill_rate": len(fills) / max(reposts, 1),
        "notional_hr": notional / hrs,
        "pnl_hr": (cash + inv * last_mid) / hrs,
        "inv_peak": inv_peak,
        "win": float((marks > 0).mean()),
        "markout": float(marks.mean()),
        "blocked": blocked,
    }


def run(df, syms, size, maxinv, fee):
    out = {}
    for s in syms:
        sub = df[df.symbol == s]
        if len(sub) < 200:
            continue
        r = simulate(sub, size, maxinv, fee)
        if r:
            out[s] = r
    return out


def main(a):
    df = load(a.csv)
    maxinv = a.max_inv_usd if a.max_inv_usd > 0 else a.size_usd * 3
    cap_pair = 2 * maxinv
    syms = sorted(df.symbol.dropna().unique())

    t0, t1 = df.ts_ms.min(), df.ts_ms.max()
    split = t0 + (t1 - t0) * a.train_frac

    print("=" * 92)
    print(f"فایل: {a.csv}   |   {len(syms)} جفت   |   "
          f"{(t1-t0)/3.6e6:.1f} ساعت")
    print(f"سایز: ${a.size_usd:.0f}   سقف موجودی: ±${maxinv:,.0f}   "
          f"کارمزد میکر: {a.fee:.1f}bps   سرمایه هر جفت: ${cap_pair:,.0f}")
    print("=" * 92)

    tr = df[df.ts_ms <= split]
    te = df[df.ts_ms > split]
    r_tr = run(tr, syms, a.size_usd, maxinv, a.fee)
    r_te = run(te, syms, a.size_usd, maxinv, a.fee)

    if not r_tr or not r_te:
        print("داده‌ی کافی نیست.")
        return

    # ---------------- بخش ۱: همه‌ی جفت‌ها در آموزش ----------------
    print(f"\n۱) دوره‌ی آموزش ({(split-t0)/3.6e6:.1f} ساعت) — رتبه‌بندی")
    print("  " + "─" * 88)
    print(f"  {'جفت':<14}{'فیل/ساعت':>10}{'نرخ فیل':>9}{'markout':>10}"
          f"{'برد':>7}{'نامتعادلی':>11}{'سود/ساعت':>12}{'':>4}")
    print("  " + "─" * 88)

    ok = []
    for s, r in sorted(r_tr.items(), key=lambda x: -x[1]["pnl_hr"]):
        good = (r["pnl_hr"] > 0 and r["win"] >= a.min_win
                and r["imbalance"] <= a.max_imbalance
                and r["fills_hr"] >= a.min_fills)
        if good:
            ok.append(s)
        print(f"  {s.upper():<14}{r['fills_hr']:>9.1f}{r['fill_rate']:>8.1%}"
              f"{r['markout']:>+9.2f}{r['win']:>7.0%}{r['imbalance']:>10.0%}"
              f"{r['pnl_hr']:>+11.3f}$ {'✅' if good else '❌'}")

    print(f"\n  واجد شرایط: {len(ok)} از {len(r_tr)}")
    if not ok:
        print("  هیچ جفتی معیارها را پاس نکرد. معیارها را شل کن یا داده‌ی بیشتر.")
        return

    # ---------------- بخش ۲: منحنی مقیاس ----------------
    ranked = [s for s in sorted(r_tr, key=lambda x: -r_tr[x]["pnl_hr"])
              if s in ok]
    print(f"\n\n۲) منحنی مقیاس — روی دوره‌ی آزمون ({(t1-split)/3.6e6:.1f} ساعت)")
    print("   آیا افزودن جفت واقعاً سود را بالا می‌برد، یا رقیقش می‌کند؟")
    print("  " + "─" * 88)
    print(f"  {'تعداد جفت':>10}{'سرمایه':>12}{'سود روزانه':>14}"
          f"{'بازده روزانه':>14}{'بازده ماهانه':>15}{'حجم ماهانه':>16}")
    print("  " + "─" * 88)

    steps = sorted({1, 2, 3, 5, 8, 12, 16, 20, 25, len(ranked)})
    curve = []
    for k in steps:
        if k > len(ranked):
            continue
        sel = ranked[:k]
        pnl = sum(r_te[s]["pnl_hr"] for s in sel if s in r_te)
        vol = sum(r_te[s]["notional_hr"] for s in sel if s in r_te)
        cap = k * cap_pair
        daily = pnl * 24
        print(f"  {k:>10}{cap:>11,.0f}${daily:>+13.2f}$"
              f"{daily/cap*100:>13.2f}%{daily*30/cap*100:>14.1f}%"
              f"{vol*24*30:>15,.0f}$")
        curve.append((k, cap, daily, vol * 24 * 30))

    # ---------------- بخش ۳: حساسیت به کارمزد ----------------
    print(f"\n\n۳) اثر کارمزد — همان جفت‌ها، کارمزدهای مختلف")
    print("  " + "─" * 88)
    print(f"  {'کارمزد میکر':>14}{'سود روزانه':>16}{'بازده ماهانه':>16}"
          f"{'نسبت به فعلی':>18}")
    print("  " + "─" * 88)
    base = None
    kbest = curve[-1][0] if curve else 1
    sel = ranked[:kbest]
    cap = kbest * cap_pair
    for fee in (a.fee, 1.0, 0.5, 0.0, -0.5):
        rr = run(te, sel, a.size_usd, maxinv, fee)
        pnl = sum(v["pnl_hr"] for v in rr.values()) * 24
        if base is None:
            base = pnl
        mult = (pnl / base) if base and base != 0 else float("nan")
        lbl = f"{fee:.1f}bps" + (" (فعلی)" if fee == a.fee else
                                 " (ریبیت)" if fee < 0 else "")
        print(f"  {lbl:>14}{pnl:>+15.2f}${pnl*30/cap*100:>15.1f}%"
              f"{mult:>17.2f}x")

    # ---------------- بخش ۴: مسیر کارمزد صفر ----------------
    if curve:
        k, cap, daily, monthly_vol = curve[-1]
        print(f"\n\n۴) مسیر رسیدن به کارمزد صفر")
        print("  " + "─" * 88)
        print(f"  با {k} جفت، حجم ماهانه‌ی تولیدی: ${monthly_vol:,.0f}")
        print(f"  رده‌های VIP معمولاً بر پایه‌ی حجم ۳۰ روزه‌اند.")
        for tier in (1e6, 5e6, 2e7, 1e8):
            need = tier / monthly_vol if monthly_vol > 0 else 0
            print(f"     برای حجم ${tier:>13,.0f} → "
                  f"{need:>6.1f} برابر مقیاس فعلی "
                  f"(یعنی ~{need*k:.0f} جفت یا سایز {need:.0f} برابر)")
        print("\n  ⚠ آستانه‌های واقعی LBank را از پنل حسابت چک کن —")
        print("    اعداد بالا فقط برای مقیاس‌سنجی‌اند.")

    print("\n" + "=" * 92)
    if curve:
        best = max(curve, key=lambda c: c[2] / c[1])
        print(f"بهترین بازده: {best[0]} جفت → "
              f"{best[2]*30/best[1]*100:+.1f}% ماهانه روی ${best[1]:,.0f}")
        big = curve[-1]
        print(f"بیشترین سود مطلق: {big[0]} جفت → "
              f"${big[2]*30:+,.2f} ماهانه روی ${big[1]:,.0f}")
    print("\n⚠ مدل‌نشده: تأخیر ارسال سفارش، واکنش رقبا، و اینکه سفارش تو")
    print("  خودش بخشی از دفتر می‌شود. عدد واقعی معمولاً کمتر است.\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--size-usd", type=float, default=50)
    p.add_argument("--max-inv-usd", type=float, default=0)
    p.add_argument("--fee", type=float, default=2.0)
    p.add_argument("--train-frac", type=float, default=0.4)
    p.add_argument("--min-win", type=float, default=0.50)
    p.add_argument("--max-imbalance", type=float, default=0.40)
    p.add_argument("--min-fills", type=float, default=3.0)
    main(p.parse_args())
