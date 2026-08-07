#!/usr/bin/env python3
"""
FAZ 0 - RECORDER  (نسخه ۴ — با زمان سرور صرافی)
================================================
ایده‌ی کلیدی:
    آنچه از آمستردام می‌بینیم = تأخیر واقعی بازار + تأخیر شبکه‌ی ما

هر پیام، زمان سرور خود صرافی را هم دارد. اگر هر دو را ذخیره کنیم،
می‌توانیم این دو را از هم جدا کنیم — یعنی بدون داشتن سرور توکیو،
نتیجه‌ی توکیو را حساب کنیم.

ستون جدید: ts_srv  (زمان سرور صرافی، ms)

اجرا:
    venv/bin/python recorder_v4.py --hours 6
"""

import argparse
import asyncio
import csv
import json
import os
import signal
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone

import websockets

PAIRS = [
    ("btcusdt", "btc_usdt"),
    ("ethusdt", "eth_usdt"),
    ("solusdt", "sol_usdt"),
    ("xrpusdt", "xrp_usdt"),
    ("dogeusdt", "doge_usdt"),
    ("adausdt", "ada_usdt"),
    ("linkusdt", "link_usdt"),
    ("avaxusdt", "avax_usdt"),
]

BINANCE_WS = "wss://stream.binance.com:9443/stream?streams="
LBANK_WS = "wss://www.lbkex.net/ws/V2/"

OPEN_TIMEOUT = 30.0
SILENCE_TIMEOUT = 25.0
KEEPALIVE_SEC = 5.0
FLUSH_EVERY_SEC = 2.0


def iso_ms(s):
    """'2026-08-06T20:15:12.463' → epoch ms (بدون فرض منطقه‌ی زمانی).
    اختلاف منطقه‌ی زمانی را تحلیلگر خودش تشخیص می‌دهد."""
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return ""


class TickWriter:
    def __init__(self, outdir="data"):
        os.makedirs(outdir, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        self.path = os.path.join(outdir, f"srv_{stamp}.csv")
        self._fh = open(self.path, "w", newline="", buffering=1024 * 1024)
        self._w = csv.writer(self._fh)
        self._w.writerow(["ts_ms", "ts_srv", "venue", "symbol", "kind", "bid", "ask"])
        self.buf = []
        self.counts = defaultdict(int)
        self.no_srv = defaultdict(int)
        self.reconnects = defaultdict(int)

    def add(self, ts, srv, venue, sym, kind, bid, ask):
        self.buf.append((ts, srv, venue, sym, kind, bid, ask))
        self.counts[(venue, kind)] += 1
        if srv == "":
            self.no_srv[venue] += 1

    def flush(self):
        if self.buf:
            self._w.writerows(self.buf)
            self.buf.clear()
            self._fh.flush()

    def close(self):
        self.flush()
        self._fh.close()


async def recv_or_die(ws):
    try:
        return await asyncio.wait_for(ws.recv(), timeout=SILENCE_TIMEOUT)
    except asyncio.TimeoutError:
        raise ConnectionError(f"سکوت {SILENCE_TIMEOUT:.0f}s")


# ---------------------------------------------------------------------------
# بایننس — bookTicker (اگر زمان داشت) + aggTrade (حتماً زمان دارد)
# ---------------------------------------------------------------------------


async def binance_stream(writer, stop_evt):
    parts = []
    for s, _ in PAIRS:
        parts.append(f"{s}@bookTicker")
        parts.append(f"{s}@aggTrade")
    url = BINANCE_WS + "/".join(parts)
    backoff = 1

    while not stop_evt.is_set():
        try:
            async with websockets.connect(
                url, ping_interval=15, ping_timeout=10, open_timeout=OPEN_TIMEOUT
            ) as ws:
                print(f"[binance] متصل شد ({len(PAIRS)} جفت × ۲ کانال)", flush=True)
                backoff = 1
                while not stop_evt.is_set():
                    raw = await recv_or_die(ws)
                    ts = int(time.time() * 1000)
                    msg = json.loads(raw)
                    d = msg.get("data")
                    if not d:
                        continue
                    sym = d.get("s")
                    if sym:
                        sym = sym.lower()
                    else:
                        sym = msg.get("stream", "").split("@")[0]
                    if not sym:
                        continue

                    if d.get("e") == "aggTrade":
                        # این کانال زمان سرور دارد → مبنای تحلیل ساعت
                        srv = d.get("T") or d.get("E") or ""
                        writer.add(ts, srv, "binance", sym, "trade", d["p"], d["p"])
                    elif "b" in d and "a" in d:
                        # bookTicker — زمان سرور ندارد، فقط برای تحلیل ساعت محلی
                        writer.add(ts, "", "binance", sym, "book", d["b"], d["a"])
        except Exception as e:
            if stop_evt.is_set():
                break
            writer.reconnects["binance"] += 1
            print(f"[binance] قطع ({e}) — {backoff}s", flush=True)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 20)


# ---------------------------------------------------------------------------
# LBank — فیلد TS زمان سرور آنهاست
# ---------------------------------------------------------------------------


async def lbank_keepalive(ws, stop_evt):
    while not stop_evt.is_set():
        await asyncio.sleep(KEEPALIVE_SEC)
        try:
            await ws.send(json.dumps({"action": "ping", "ping": str(uuid.uuid4())}))
        except Exception:
            return


async def lbank_stream(writer, stop_evt, pairs=None, tag=""):
    """هر نمونه فقط زیرمجموعه‌ای از جفت‌ها را می‌گیرد.
    چند اتصال موازی = وقتی یکی قطع می‌شود، بقیه ادامه می‌دهند."""
    pairs = pairs if pairs is not None else PAIRS
    backoff = 1
    while not stop_evt.is_set():
        ka = None
        try:
            print(f"[lbank{tag}]  در حال اتصال...", flush=True)
            async with websockets.connect(
                LBANK_WS, ping_interval=None, open_timeout=OPEN_TIMEOUT
            ) as ws:
                for _, pair in pairs:
                    await ws.send(json.dumps({
                        "action": "subscribe", "subscribe": "depth",
                        "depth": "10", "pair": pair}))
                    await asyncio.sleep(0.10)
                    await ws.send(json.dumps({
                        "action": "subscribe", "subscribe": "trade", "pair": pair}))
                    await asyncio.sleep(0.10)

                print(f"[lbank{tag}]  متصل شد ({len(pairs)} جفت × ۲ کانال)", flush=True)
                backoff = 1
                ka = asyncio.create_task(lbank_keepalive(ws, stop_evt))

                while not stop_evt.is_set():
                    raw = await recv_or_die(ws)
                    ts = int(time.time() * 1000)
                    m = json.loads(raw)
                    act, typ = m.get("action"), m.get("type")

                    if act == "ping":
                        await ws.send(json.dumps({"action": "pong", "pong": m["ping"]}))
                        continue
                    if act == "pong":
                        continue

                    srv = iso_ms(m["TS"]) if m.get("TS") else ""

                    if typ == "depth":
                        d = m.get("depth") or {}
                        bids, asks = d.get("bids"), d.get("asks")
                        if not bids or not asks:
                            continue
                        try:
                            bb = max(float(b[0]) for b in bids if float(b[1]) > 0)
                            ba = min(float(a[0]) for a in asks if float(a[1]) > 0)
                        except (ValueError, IndexError):
                            continue
                        if bb > 0 and ba > bb:
                            writer.add(ts, srv, "lbank", m["pair"], "book", bb, ba)

                    elif typ == "trade":
                        t = m.get("trade") or {}
                        tsrv = iso_ms(t["TS"]) if t.get("TS") else srv
                        try:
                            p = float(t["price"])
                        except (KeyError, TypeError, ValueError):
                            continue
                        if p > 0:
                            writer.add(ts, tsrv, "lbank", m["pair"], "trade", p, p)

        except Exception as e:
            if stop_evt.is_set():
                break
            writer.reconnects["lbank"] += 1
            print(f"[lbank{tag}]  قطع ({e}) — {backoff}s", flush=True)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 20)
        finally:
            if ka:
                ka.cancel()


# ---------------------------------------------------------------------------


async def housekeeper(writer, stop_evt, run_seconds):
    started, last = time.time(), 0
    while not stop_evt.is_set():
        await asyncio.sleep(FLUSH_EVERY_SEC)
        writer.flush()
        el = time.time() - started
        if el - last >= 60:
            last = el
            c = writer.counts
            b = c[("binance", "book")] + c[("binance", "trade")]
            l = c[("lbank", "book")] + c[("lbank", "trade")]
            mb = os.path.getsize(writer.path) / 1e6
            miss_b = writer.no_srv["binance"]
            miss_l = writer.no_srv["lbank"]
            print(f"[{el/60:6.1f}د] بایننس={b:,} LBank={l:,} | {mb:.0f}MB | "
                  f"بدون‌زمان B={miss_b:,} L={miss_l:,} | "
                  f"قطعی B={writer.reconnects['binance']} L={writer.reconnects['lbank']}",
                  flush=True)
        if run_seconds and el >= run_seconds:
            print("\nزمان ضبط تمام شد.", flush=True)
            stop_evt.set()


async def main(run_seconds):
    writer = TickWriter()
    stop_evt = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sg in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sg, stop_evt.set)
        except NotImplementedError:
            pass

    print(f"شروع ضبط → {writer.path}\nتوقف با Ctrl+C\n")
    N_CONN = 3
    chunks = [PAIRS[i::N_CONN] for i in range(N_CONN)]
    tasks = [binance_stream(writer, stop_evt),
             housekeeper(writer, stop_evt, run_seconds)]
    for i, ch in enumerate(chunks):
        if ch:
            tasks.append(lbank_stream(writer, stop_evt, ch, f"-{i+1}"))
    await asyncio.gather(*tasks, return_exceptions=True)
    writer.close()
    print(f"\nذخیره شد: {writer.path}")
    for k, v in sorted(writer.counts.items()):
        print(f"   {k[0]:8s} {k[1]:6s} {v:>12,}")
    print(f"بدون زمان سرور → بایننس: {writer.no_srv['binance']:,}  "
          f"LBank: {writer.no_srv['lbank']:,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=6)
    a = ap.parse_args()
    try:
        asyncio.run(main(int(a.hours * 3600)))
    except KeyboardInterrupt:
        print("\nمتوقف شد.")
        sys.exit(0)
