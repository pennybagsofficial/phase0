#!/usr/bin/env python3
"""
SPREAD SCANNER — پیدا کردن جفت‌هایی که واقعاً ارزش بازارگردانی دارند
=====================================================================
درس اصلی از فاز صفر:
    ما دنبال سیگنال ۲bps بودیم، در حالی که اسپرد ۵bps همان‌جا نشسته بود.
    و هشت جفتی که انتخاب کردیم، نقدشونده‌ترین بازارهای دنیا بودند —
    یعنی جایی که بازارگردان حرفه‌ای از قبل نشسته و اسپرد را صفر کرده.

این اسکریپت تمام جفت‌های LBank را جارو می‌کند و می‌پرسد:
    کجا اسپرد از کارمزد بزرگ‌تر است، و همزمان حجم کافی برای پرشدن هست؟

هیچ سفارشی نمی‌گذارد. فقط می‌خواند.

اجرا:
    python spread_scanner.py
    python spread_scanner.py --fee 2.0 --top 200 --samples 3
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import median

BASES = ["https://api.lbank.info", "https://api.lbkex.com"]
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
TIMEOUT = 15
WORKERS = 8


# ---------------------------------------------------------------------------
# لایه‌ی شبکه
# ---------------------------------------------------------------------------

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def unwrap(js):
    """LBank گاهی {'result':true,'data':[...]} می‌دهد و گاهی خود آرایه."""
    if isinstance(js, dict):
        for k in ("data", "result"):
            v = js.get(k)
            if isinstance(v, (list, dict)):
                return v
        return js
    return js


def pick_base():
    for b in BASES:
        try:
            get(f"{b}/v2/timestamp.do")
            print(f"✓ API در دسترس: {b}")
            return b
        except Exception as e:
            print(f"✗ {b} → {type(e).__name__}")
    print("\nهیچ اندپوینتی جواب نداد. اینترنت یا فیلترینگ را چک کن.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# جمع‌آوری
# ---------------------------------------------------------------------------

def fetch_tickers(base):
    """یک فراخوانی، تمام جفت‌ها با حجم ۲۴ ساعته."""
    js = unwrap(get(f"{base}/v2/ticker/24hr.do?symbol=all"))
    out = {}
    for row in js if isinstance(js, list) else []:
        sym = row.get("symbol")
        t = row.get("ticker") or {}
        if not sym:
            continue
        try:
            out[sym] = {
                "last": float(t.get("latest") or 0),
                "vol": float(t.get("vol") or 0),          # حجم به ارز پایه
                "turnover": float(t.get("turnover") or 0),  # حجم به ارز مظنه
                "change": float(t.get("change") or 0),
            }
        except (TypeError, ValueError):
            continue
    return out


def fetch_depth(base, sym):
    js = unwrap(get(f"{base}/v2/depth.do?symbol={sym}&size=10"))
    if not isinstance(js, dict):
        return None
    bids, asks = js.get("bids"), js.get("asks")
    if not bids or not asks:
        return None
    try:
        b = sorted(((float(x[0]), float(x[1])) for x in bids if float(x[1]) > 0),
                   reverse=True)
        a = sorted((float(x[0]), float(x[1])) for x in asks if float(x[1]) > 0)
    except (ValueError, IndexError):
        return None
    if not b or not a or a[0][0] <= b[0][0]:
        return None

    bid, ask = b[0][0], a[0][0]
    mid = (bid + ask) / 2
    return {
        "bid": bid, "ask": ask, "mid": mid,
        "spread_bps": (ask - bid) / mid * 1e4,
        # عمق نزدیک قیمت: ۵ سطح اول، به ارز مظنه
        "bid_depth": sum(p * q for p, q in b[:5]),
        "ask_depth": sum(p * q for p, q in a[:5]),
    }


def fetch_trade_rate(base, sym):
    """نرخ معامله در ساعت، از فاصله‌ی زمانی ۱۰۰ معامله‌ی آخر."""
    js = unwrap(get(f"{base}/v2/trades.do?symbol={sym}&size=100"))
    if not isinstance(js, list) or len(js) < 10:
        return None
    ts = []
    for t in js:
        v = t.get("date_ms") or t.get("ts") or t.get("date")
        if v:
            v = float(v)
            ts.append(v * 1000 if v < 1e11 else v)   # ثانیه یا میلی‌ثانیه
    if len(ts) < 10:
        return None
    span_s = (max(ts) - min(ts)) / 1000
    if span_s <= 0:
        return None
    return len(ts) / span_s * 3600


def probe_pair(base, sym, samples, gap):
    """چند بار دفتر را نمونه‌برداری می‌کند تا اسپرد پایدار به دست آید."""
    snaps = []
    for i in range(samples):
        try:
            d = fetch_depth(base, sym)
            if d:
                snaps.append(d)
        except Exception:
            pass
        if i < samples - 1:
            time.sleep(gap)
    if not snaps:
        return None

    try:
        rate = fetch_trade_rate(base, sym)
    except Exception:
        rate = None

    return {
        "symbol": sym,
        "spread_bps": median(s["spread_bps"] for s in snaps),
        "spread_min": min(s["spread_bps"] for s in snaps),
        "spread_max": max(s["spread_bps"] for s in snaps),
        "mid": snaps[-1]["mid"],
        "bid_depth": median(s["bid_depth"] for s in snaps),
        "ask_depth": median(s["ask_depth"] for s in snaps),
        "trades_hr": rate,
        "n_snaps": len(snaps),
    }


# ---------------------------------------------------------------------------


def main(a):
    base = pick_base()

    print("\nدریافت لیست جفت‌ها و حجم ۲۴ ساعته...")
    tk = fetch_tickers(base)
    print(f"   {len(tk):,} جفت پیدا شد")

    # فقط جفت‌های مظنه‌شده به USDT — مقایسه‌ی حجم بین آن‌ها معنادار است
    cands = []
    for sym, t in tk.items():
        if not sym.endswith(f"_{a.quote}"):
            continue
        if t["turnover"] < a.min_volume:
            continue
        cands.append((sym, t))

    cands.sort(key=lambda x: -x[1]["turnover"])
    cands = cands[:a.top]

    print(f"   {len(cands):,} جفت {a.quote.upper()} با حجم روزانه بالای "
          f"${a.min_volume:,.0f}")
    print(f"\nنمونه‌برداری دفتر سفارش ({a.samples} بار هر جفت)... "
          f"حدود {len(cands)*a.samples/WORKERS*0.4:.0f} ثانیه\n")

    rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(probe_pair, base, s, a.samples, a.gap): (s, t)
                for s, t in cands}
        for f in as_completed(futs):
            sym, t = futs[f]
            done += 1
            if done % 25 == 0:
                print(f"   {done}/{len(cands)}...", flush=True)
            try:
                r = f.result()
            except Exception:
                r = None
            if not r:
                continue
            r["turnover_24h"] = t["turnover"]
            rows.append(r)

    if not rows:
        print("\nهیچ داده‌ای جمع نشد.")
        return

    # ---------------- اقتصاد ----------------
    fee_rt = 2 * a.fee                       # کارمزد رفت و برگشت (bps)
    for r in rows:
        r["net_bps"] = r["spread_bps"] - fee_rt
        hourly_vol = r["turnover_24h"] / 24
        r["hourly_vol"] = hourly_vol
        # سهم تو از حجم — فرض محافظه‌کارانه، قابل تنظیم
        r["est_hourly_usd"] = (r["net_bps"] / 1e4) * hourly_vol * a.share
        r["est_daily_usd"] = r["est_hourly_usd"] * 24

    rows.sort(key=lambda r: -r["est_daily_usd"])

    # ---------------- گزارش ----------------
    print("\n" + "=" * 100)
    print(f"کارمزد میکر: {a.fee:.1f}bps هر طرف  →  {fee_rt:.1f}bps رفت‌وبرگشت"
          f"   |   سهم فرضی از حجم: {a.share:.0%}")
    print("=" * 100)
    print(f"{'جفت':<16}{'اسپرد':>8}{'خالص':>8}{'حجم/ساعت':>13}"
          f"{'معامله/ساعت':>13}{'عمق بید':>11}{'تخمین روزانه':>14}")
    print("─" * 100)

    shown = 0
    for r in rows:
        if r["net_bps"] <= 0:
            continue
        if r["trades_hr"] is not None and r["trades_hr"] < a.min_trades:
            continue
        shown += 1
        if shown > a.show:
            break
        tr = f"{r['trades_hr']:.0f}" if r["trades_hr"] else "?"
        print(f"{r['symbol']:<16}{r['spread_bps']:>7.1f}{r['net_bps']:>8.1f}"
              f"{r['hourly_vol']:>12,.0f}${tr:>13}"
              f"{r['bid_depth']:>10,.0f}${r['est_daily_usd']:>13,.2f}$")

    # ---------------- خلاصه ----------------
    viable = [r for r in rows if r["net_bps"] > 0
              and (r["trades_hr"] or 0) >= a.min_trades]
    dead = [r for r in rows if r["net_bps"] <= 0]

    print("\n" + "=" * 100)
    print(f"جفت‌های با اسپرد بزرگ‌تر از کارمزد و حجم کافی: {len(viable)}")
    print(f"جفت‌هایی که اسپردشان از کارمزد کمتر است:        {len(dead)}")

    if viable:
        top = viable[0]
        print(f"\nبهترین: {top['symbol']}")
        print(f"   اسپرد {top['spread_bps']:.1f}bps  منهای کارمزد {fee_rt:.1f}bps"
              f"  =  {top['net_bps']:.1f}bps حاشیه")
        print(f"   حاشیه‌ی تو {top['net_bps']/fee_rt:.1f} برابر کارمزد است")
        print(f"   تخمین درآمد روزانه با {a.share:.0%} سهم: "
              f"${top['est_daily_usd']:,.2f}")

    # ذخیره
    import csv
    out = "scan_results.csv"
    keys = ["symbol", "spread_bps", "spread_min", "spread_max", "net_bps",
            "mid", "bid_depth", "ask_depth", "trades_hr", "turnover_24h",
            "hourly_vol", "est_hourly_usd", "est_daily_usd", "n_snaps"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nهمه‌ی نتایج در {out} ذخیره شد ({len(rows)} جفت).")

    print("\n⚠ این اعداد سقف نظری‌اند، نه پیش‌بینی درآمد.")
    print("  فرض می‌کنند سفارش لیمیتت پر می‌شود و انتخاب معکوس نمی‌خوری.")
    print("  در عمل معمولاً بخش بزرگی از این حاشیه از دست می‌رود.")
    print("  قدم بعد: روی ۳ جفت برتر داده ضبط کن و انتخاب معکوس را بسنج.\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fee", type=float, default=2.0,
                   help="کارمزد میکر هر طرف بر حسب bps (۲.۰ = ۰.۰۲٪)")
    p.add_argument("--quote", default="usdt", help="ارز مظنه")
    p.add_argument("--top", type=int, default=250, help="چند جفت پرحجم بررسی شود")
    p.add_argument("--show", type=int, default=40, help="چند ردیف چاپ شود")
    p.add_argument("--samples", type=int, default=3, help="چند بار نمونه‌برداری دفتر")
    p.add_argument("--gap", type=float, default=1.5, help="فاصله‌ی نمونه‌ها (ثانیه)")
    p.add_argument("--min-volume", type=float, default=20000,
                   help="حداقل حجم ۲۴ ساعته به دلار")
    p.add_argument("--min-trades", type=float, default=20,
                   help="حداقل معامله در ساعت")
    p.add_argument("--share", type=float, default=0.05,
                   help="سهم فرضی تو از حجم جفت (۰.۰۵ = ۵٪)")
    main(p.parse_args())
