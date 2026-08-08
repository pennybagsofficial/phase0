#!/usr/bin/env python3
"""
QUEUE SIMULATOR — آیا واقعاً فیل می‌خوری؟
==========================================
همه‌ی تست‌های قبلی یک فرض خوش‌بینانه داشتند: که سفارش لیمیت تو پر می‌شود.
این اسکریپت آن فرض را برمی‌دارد و واقعیت را شبیه‌سازی می‌کند.

منطق صف:
    وقتی روی بهترین بید سفارش می‌گذاری، پشت همه‌ی کسانی می‌ایستی که
    از قبل آنجا بودند. تا حجم آن‌ها مصرف نشود، نوبت تو نمی‌رسد.

    • فقط **معامله** صف را جلو می‌برد (کنسل دیگران کمکت نمی‌کند —
      فرض محافظه‌کارانه، چون نمی‌دانیم کنسل جلوی تو بوده یا پشتت)
    • هر بار قیمت بهترین سطح عوض شود، سفارشت را می‌کشی و دوباره
      می‌گذاری → صف از صفر شروع می‌شود
    • در بازار تندرو، هیچ‌وقت به جلوی صف نمی‌رسی

خروجی: تعداد فیل واقعی در ساعت، و درآمد روزانه با سرمایه‌ی مشخص.

اجرا:
    python queue_sim.py data/q_XXXX.csv --size-usd 50 --fee 2.0
"""

import argparse

import numpy as np
import pandas as pd

MARKOUT_S = 5          # افق ارزش‌گذاری بعد از فیل
BPS = 1e4


def load(path):
    df = pd.read_csv(path)
    num = ["ts_ms", "b1p", "b1q", "a1p", "a1q", "price", "qty"]
    for c in num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["side"] = df.get("side", "").astype(str).str.lower().str.strip()
    return df.sort_values("ts_ms").reset_index(drop=True)


def simulate(sub, size_usd, fee_rt, join_only_if_alone=False):
    """
    شبیه‌سازی رویداد به رویداد برای یک جفت.
    سفارش دوطرفه روی بهترین بید و بهترین اسک.
    """
    ev = sub[["ts_ms", "kind", "b1p", "b1q", "a1p", "a1q",
              "price", "qty", "side"]].itertuples(index=False)

    # وضعیت هر سمت: قیمت ما، حجم جلوی ما، حجم باقیمانده‌ی ما
    st = {"bid": None, "ask": None}
    mid_ts, mid_px = [], []
    fills = []          # (ts, side, price, qty)
    reposts = {"bid": 0, "ask": 0}
    best = {"b": None, "a": None, "bq": 0.0, "aq": 0.0}

    for r in ev:
        if r.kind == "book":
            if not (np.isfinite(r.b1p) and np.isfinite(r.a1p)):
                continue
            best["b"], best["a"] = r.b1p, r.a1p
            best["bq"] = r.b1q if np.isfinite(r.b1q) else 0.0
            best["aq"] = r.a1q if np.isfinite(r.a1q) else 0.0
            mid_ts.append(r.ts_ms)
            mid_px.append((r.b1p + r.a1p) / 2)

            for k, price, lvlq in (("bid", r.b1p, best["bq"]),
                                   ("ask", r.a1p, best["aq"])):
                s = st[k]
                if s is None or s["price"] != price:
                    # قیمت عوض شد → کنسل و ثبت مجدد، صف از صفر
                    qty = size_usd / price if price > 0 else 0
                    st[k] = {"price": price, "ahead": lvlq, "left": qty}
                    reposts[k] += 1
                else:
                    # همان سطح: کاهش حجم را به کنسل نسبت می‌دهیم، نه پیشرفت صف
                    s["ahead"] = min(s["ahead"], max(lvlq - s["left"], 0.0)) \
                        if lvlq > 0 else s["ahead"]

        elif r.kind == "trade":
            if not (np.isfinite(r.price) and np.isfinite(r.qty)) or r.qty <= 0:
                continue
            aggressive_sell = r.side.startswith("s")
            k = "bid" if aggressive_sell else "ask"
            s = st[k]
            if s is None:
                continue
            # معامله باید در سطح ما (یا بهتر برای مهاجم) رخ داده باشد
            hit = (r.price <= s["price"] + 1e-12) if aggressive_sell \
                else (r.price >= s["price"] - 1e-12)
            if not hit:
                continue

            vol = r.qty
            eaten = min(vol, s["ahead"])
            s["ahead"] -= eaten
            vol -= eaten
            if vol > 0 and s["left"] > 0:
                f = min(vol, s["left"])
                s["left"] -= f
                fills.append((r.ts_ms, k, s["price"], f))
                if s["left"] <= 1e-12:
                    # کامل پر شد → دوباره ته صف
                    s["ahead"] = best["bq"] if k == "bid" else best["aq"]
                    s["left"] = size_usd / s["price"] if s["price"] > 0 else 0
                    reposts[k] += 1

    if not fills or len(mid_ts) < 50:
        return None

    mts = np.array(mid_ts, dtype=np.int64)
    mpx = np.array(mid_px, dtype=float)

    hrs = (sub.ts_ms.max() - sub.ts_ms.min()) / 3.6e6
    rows = []
    for ts, k, px, q in fills:
        i = np.searchsorted(mts, ts + MARKOUT_S * 1000, side="right") - 1
        if i < 0:
            continue
        fut = mpx[i]
        sign = 1.0 if k == "bid" else -1.0          # bid ⇒ خریدیم
        bps = (fut - px) / px * BPS * sign - fee_rt
        rows.append((k, px * q, bps))

    if not rows:
        return None

    notional = np.array([r[1] for r in rows])
    bpsv = np.array([r[2] for r in rows])
    pnl_usd = notional * bpsv / BPS

    n_bid = sum(1 for r in rows if r[0] == "bid")
    return {
        "hours": hrs,
        "fills": len(rows),
        "fills_hr": len(rows) / hrs,
        "bid_fills": n_bid,
        "ask_fills": len(rows) - n_bid,
        "reposts": reposts["bid"] + reposts["ask"],
        "fill_rate": len(rows) / max(reposts["bid"] + reposts["ask"], 1),
        "volume_usd": float(notional.sum()),
        "vol_hr": float(notional.sum()) / hrs,
        "avg_bps": float(bpsv.mean()),
        "med_bps": float(np.median(bpsv)),
        "win": float((bpsv > 0).mean()),
        "pnl_usd": float(pnl_usd.sum()),
        "pnl_hr": float(pnl_usd.sum()) / hrs,
    }


def main(a):
    df = load(a.csv)
    fee_rt = 2 * a.fee
    capital = a.size_usd * 2 * a.buffer

    print("=" * 88)
    print(f"فایل: {a.csv}")
    print(f"سایز کوتیشن: ${a.size_usd:.0f} هر طرف   |   "
          f"کارمزد: {fee_rt:.1f}bps رفت‌وبرگشت")
    print(f"سرمایه‌ی لازم: ${capital:,.0f}  "
          f"(دو طرف × ضریب اطمینان {a.buffer:.1f})")
    print("=" * 88)
    print("«ثبت مجدد» = چند بار مجبور شدی سفارش را جابه‌جا کنی")
    print("«نرخ فیل»  = چه کسری از سفارش‌ها واقعاً پر شد")

    res = []
    for sym in sorted(df.symbol.dropna().unique()):
        sub = df[df.symbol == sym]
        r = simulate(sub, a.size_usd, fee_rt)
        if not r:
            print(f"\n{sym.upper()}: هیچ فیلی رخ نداد "
                  f"(صف هیچ‌وقت به تو نرسید)")
            continue

        print(f"\n{'─'*88}")
        print(f"{sym.upper()}   ({r['hours']:.1f} ساعت)")
        print(f"   ثبت مجدد: {r['reposts']:,}      فیل: {r['fills']:,}      "
              f"نرخ فیل: {r['fill_rate']:.2%}")
        print(f"   فیل در ساعت: {r['fills_hr']:.1f}   "
              f"(خرید {r['bid_fills']:,} / فروش {r['ask_fills']:,})")
        print(f"   حجم معامله‌شده: ${r['vol_hr']:,.0f} در ساعت")
        print(f"   سود هر فیل: میانگین {r['avg_bps']:+.2f}bps   "
              f"میانه {r['med_bps']:+.2f}bps   برد {r['win']:.0%}")
        print(f"   ➜ سود: ${r['pnl_hr']:+.3f} در ساعت   "
              f"= ${r['pnl_hr']*24:+.2f} در روز")
        res.append((sym, r))

    # ---------------- خلاصه ----------------
    print("\n" + "=" * 88)
    print("خلاصه — درآمد واقعی بعد از شبیه‌سازی صف")
    print("=" * 88)
    if not res:
        print("هیچ جفتی فیل نگرفت. یعنی صف هیچ‌وقت به تو نمی‌رسد.")
        print("→ سایز کوچک‌تر امتحان کن، یا جفت‌های کم‌رقابت‌تر.")
        return

    print(f"  {'جفت':<14}{'فیل/ساعت':>10}{'bps':>9}{'برد':>7}"
          f"{'روزانه':>11}{'بازده روز':>12}")
    print("  " + "─" * 66)
    total = 0.0
    for sym, r in sorted(res, key=lambda x: -x[1]["pnl_hr"]):
        daily = r["pnl_hr"] * 24
        total += daily
        ret = daily / capital * 100
        tag = "✅" if daily > 0 else "❌"
        print(f"  {sym.upper():<14}{r['fills_hr']:>9.1f}{r['avg_bps']:>+9.2f}"
              f"{r['win']:>6.0%}{daily:>+10.2f}${ret:>+11.2f}% {tag}")

    print("  " + "─" * 66)
    print(f"  {'مجموع':<14}{'':>9}{'':>9}{'':>6}{total:>+10.2f}$"
          f"{total/capital*100:>+11.2f}%")

    print(f"\n  سرمایه‌ی درگیر: ${capital:,.0f}")
    print(f"  درآمد روزانه:   ${total:+,.2f}")
    print(f"  درآمد ماهانه:   ${total*30:+,.2f}   "
          f"({total*30/capital*100:+.1f}% ماهانه)")

    if total > 0:
        be = a.server_cost / (total * 30) if total > 0 else 0
        print(f"\n  هزینه‌ی سرور ماهانه: ${a.server_cost:.0f}")
        if total * 30 > a.server_cost:
            print(f"  ✅ سود ماهانه از هزینه‌ی سرور بیشتر است "
                  f"({total*30/a.server_cost:.1f} برابر)")
        else:
            need = a.server_cost / (total * 30) * capital
            print(f"  ❌ برای پوشش هزینه‌ی سرور، سرمایه باید حدود "
                  f"${need:,.0f} باشد")

    print("\n⚠ این شبیه‌سازی هنوز خوش‌بینانه است:")
    print("  • تأخیر ارسال/کنسل سفارش را حساب نمی‌کند")
    print("  • فرض می‌کند سفارشت بلافاصله جابه‌جا می‌شود")
    print("  • ریسک موجودی (گیر کردن روی یک طرف) را مدل نمی‌کند")
    print("  عدد واقعی معمولاً نصف تا یک‌سوم این است.\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--size-usd", type=float, default=50,
                   help="سایز هر کوتیشن به دلار")
    p.add_argument("--fee", type=float, default=2.0, help="کارمزد میکر هر طرف bps")
    p.add_argument("--buffer", type=float, default=3.0,
                   help="ضریب سرمایه (برای موجودی و نوسان)")
    p.add_argument("--server-cost", type=float, default=10.0,
                   help="هزینه‌ی ماهانه‌ی سرور به دلار")
    main(p.parse_args())
