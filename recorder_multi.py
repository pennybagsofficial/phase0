#!/usr/bin/env python3
"""
MULTI-VENUE RECORDER — «مرشد»
==============================
سه صرافی رهبر + LBank را همزمان ضبط می‌کند.

چرا چند صرافی؟
    با یک صرافی، همبستگی ۰.۴۵ بود. بخش بزرگی از آن ۰.۵۵ باقی‌مانده،
    نویز خود بایننس است نه بی‌ربطی بازار. وقتی سه صرافی مستقل هم‌جهت
    حرکت کنند، سیگنال بسیار تمیزتر است.

چرا دو جفت؟
    ETH بالاترین همبستگی (۰.۵۱) و AVAX بیشترین تأخیر (۳۵۰ms) را داشت.
    هزینه‌ی ضبط هر دو یکسان است — بگذار داده برنده را انتخاب کند.

اجرا:
    python recorder_multi.py --hours 3
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

# نماد متعارف → نماد هر صرافی
PAIRS = {
    "ethusdt":  {"bin": "ethusdt",  "okx": "ETH-USDT",
                 "byb": "ETHUSDT",  "lb": "eth_usdt"},
    "avaxusdt": {"bin": "avaxusdt", "okx": "AVAX-USDT",
                 "byb": "AVAXUSDT", "lb": "avax_usdt"},
}

BINANCE = "wss://stream.binance.com:9443/stream?streams="
OKX = "wss://ws.okx.com:8443/ws/v5/public"
BYBIT = "wss://stream.bybit.com/v5/public/spot"
LBANK = "wss://www.lbkex.net/ws/V2/"

OPEN_T = 30.0
SILENCE = 25.0
FLUSH = 2.0


def now_ms():
    return int(time.time() * 1000)


def iso_ms(s):
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return ""


class W:
    def __init__(self, outdir="data"):
        os.makedirs(outdir, exist_ok=True)
        st = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        self.path = os.path.join(outdir, f"multi_{st}.csv")
        self.fh = open(self.path, "w", newline="", buffering=1 << 20)
        self.w = csv.writer(self.fh)
        self.w.writerow(["ts_ms", "ts_srv", "venue", "symbol", "bid", "ask"])
        self.buf = []
        self.n = defaultdict(int)
        self.rc = defaultdict(int)

    def add(self, srv, venue, sym, bid, ask):
        self.buf.append((now_ms(), srv, venue, sym, bid, ask))
        self.n[(venue, sym)] += 1

    def flush(self):
        if self.buf:
            self.w.writerows(self.buf)
            self.buf.clear()
            self.fh.flush()

    def close(self):
        self.flush()
        self.fh.close()


async def recv_t(ws, t=SILENCE):
    try:
        return await asyncio.wait_for(ws.recv(), timeout=t)
    except asyncio.TimeoutError:
        raise ConnectionError(f"سکوت {t:.0f}s")


async def loop(name, w, stop, connect, handle, setup=None, ping=None):
    """اسکلت مشترک: اتصال، اشتراک، پینگ، بازیابی."""
    backoff = 1
    while not stop.is_set():
        pt = None
        try:
            async with connect() as ws:
                if setup:
                    await setup(ws)
                print(f"[{name}] متصل", flush=True)
                backoff = 1
                if ping:
                    pt = asyncio.create_task(ping(ws, stop))
                while not stop.is_set():
                    raw = await recv_t(ws)
                    await handle(ws, raw)
        except Exception as e:
            if stop.is_set():
                break
            w.rc[name] += 1
            print(f"[{name}] قطع ({e}) — {backoff}s", flush=True)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 20)
        finally:
            if pt:
                pt.cancel()


# ---------------------------------------------------------------- بایننس
async def binance(w, stop):
    streams = "/".join(f"{p['bin']}@bookTicker" for p in PAIRS.values())
    rev = {p["bin"]: k for k, p in PAIRS.items()}

    async def handle(ws, raw):
        d = json.loads(raw).get("data")
        if d and "b" in d and "a" in d:
            sym = rev.get(d["s"].lower())
            if sym:
                w.add("", "binance", sym, d["b"], d["a"])

    await loop("binance", w, stop,
               lambda: websockets.connect(BINANCE + streams, ping_interval=15,
                                          ping_timeout=10, open_timeout=OPEN_T),
               handle)


# ---------------------------------------------------------------- OKX
async def okx(w, stop):
    rev = {p["okx"]: k for k, p in PAIRS.items()}

    async def setup(ws):
        await ws.send(json.dumps({"op": "subscribe", "args": [
            {"channel": "bbo-tbt", "instId": p["okx"]} for p in PAIRS.values()]}))

    async def ping(ws, stop_):
        while not stop_.is_set():
            await asyncio.sleep(20)
            try:
                await ws.send("ping")
            except Exception:
                return

    async def handle(ws, raw):
        if raw == "pong":
            return
        m = json.loads(raw)
        arg, data = m.get("arg"), m.get("data")
        if not arg or not data:
            return
        sym = rev.get(arg.get("instId"))
        if not sym:
            return
        for d in data:
            b, a = d.get("bids"), d.get("asks")
            if b and a:
                try:
                    w.add(int(d.get("ts", 0)) or "", "okx", sym,
                          float(b[0][0]), float(a[0][0]))
                except (ValueError, IndexError, TypeError):
                    pass

    await loop("okx", w, stop,
               lambda: websockets.connect(OKX, ping_interval=None,
                                          open_timeout=OPEN_T),
               handle, setup, ping)


# ---------------------------------------------------------------- Bybit
async def bybit(w, stop):
    rev = {p["byb"]: k for k, p in PAIRS.items()}

    async def setup(ws):
        await ws.send(json.dumps({"op": "subscribe", "args": [
            f"orderbook.1.{p['byb']}" for p in PAIRS.values()]}))

    async def ping(ws, stop_):
        while not stop_.is_set():
            await asyncio.sleep(20)
            try:
                await ws.send(json.dumps({"op": "ping"}))
            except Exception:
                return

    async def handle(ws, raw):
        m = json.loads(raw)
        d = m.get("data")
        if not d or not isinstance(d, dict):
            return
        sym = rev.get(d.get("s", ""))
        if not sym:
            return
        b, a = d.get("b"), d.get("a")
        if b and a:
            try:
                w.add(m.get("cts") or m.get("ts") or "", "bybit", sym,
                      float(b[0][0]), float(a[0][0]))
            except (ValueError, IndexError, TypeError):
                pass

    await loop("bybit", w, stop,
               lambda: websockets.connect(BYBIT, ping_interval=None,
                                          open_timeout=OPEN_T),
               handle, setup, ping)


# ---------------------------------------------------------------- LBank
async def lbank(w, stop):
    rev = {p["lb"]: k for k, p in PAIRS.items()}

    async def setup(ws):
        for p in PAIRS.values():
            await ws.send(json.dumps({"action": "subscribe",
                                      "subscribe": "depth", "depth": "10",
                                      "pair": p["lb"]}))
            await asyncio.sleep(0.12)

    async def ping(ws, stop_):
        while not stop_.is_set():
            await asyncio.sleep(5)
            try:
                await ws.send(json.dumps({"action": "ping",
                                          "ping": str(uuid.uuid4())}))
            except Exception:
                return

    async def handle(ws, raw):
        m = json.loads(raw)
        if m.get("action") == "ping":
            await ws.send(json.dumps({"action": "pong", "pong": m["ping"]}))
            return
        if m.get("action") == "pong" or m.get("type") != "depth":
            return
        sym = rev.get(m.get("pair", ""))
        if not sym:
            return
        d = m.get("depth") or {}
        bids, asks = d.get("bids"), d.get("asks")
        if not bids or not asks:
            return
        try:
            bb = max(float(x[0]) for x in bids if float(x[1]) > 0)
            ba = min(float(x[0]) for x in asks if float(x[1]) > 0)
        except (ValueError, IndexError):
            return
        if bb > 0 and ba > bb:
            w.add(iso_ms(m["TS"]) if m.get("TS") else "", "lbank", sym, bb, ba)

    await loop("lbank", w, stop,
               lambda: websockets.connect(LBANK, ping_interval=None,
                                          open_timeout=OPEN_T),
               handle, setup, ping)


# ----------------------------------------------------------------
async def house(w, stop, secs):
    t0, last = time.time(), 0
    while not stop.is_set():
        await asyncio.sleep(FLUSH)
        w.flush()
        el = time.time() - t0
        if el - last >= 60:
            last = el
            per = defaultdict(int)
            for (v, _), n in w.n.items():
                per[v] += n
            mb = os.path.getsize(w.path) / 1e6
            s = "  ".join(f"{k}={v:,}" for k, v in sorted(per.items()))
            print(f"[{el/60:5.1f}د] {s} | {mb:.0f}MB | "
                  f"قطعی={dict(w.rc)}", flush=True)
        if secs and el >= secs:
            print("\nپایان ضبط.", flush=True)
            stop.set()


async def main(a):
    w = W()
    stop = asyncio.Event()
    lp = asyncio.get_running_loop()
    for sg in (signal.SIGINT, signal.SIGTERM):
        try:
            lp.add_signal_handler(sg, stop.set)
        except NotImplementedError:
            pass
    print(f"جفت‌ها: {', '.join(PAIRS)}")
    print(f"ضبط → {w.path}\n")
    await asyncio.gather(binance(w, stop), okx(w, stop), bybit(w, stop),
                         lbank(w, stop), house(w, stop, int(a.hours * 3600)),
                         return_exceptions=True)
    w.close()
    print(f"\nذخیره: {w.path}")
    for (v, s), n in sorted(w.n.items()):
        print(f"   {v:<9} {s:<10} {n:>10,}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=float, default=3)
    try:
        asyncio.run(main(p.parse_args()))
    except KeyboardInterrupt:
        print("\nمتوقف شد.")
        sys.exit(0)
