#!/usr/bin/env python3
"""
MM RECORDER v2 — با حجم صف
===========================
تفاوت با نسخه‌ی قبل: **حجم** هر سطح را هم ذخیره می‌کند.

بدون حجم نمی‌شود فهمید چند نفر جلوی تو در صف‌اند — و کل شبیه‌سازی
پرشدن سفارش به همین بستگی دارد. قیمت به تنهایی کافی نیست.

اجرا:
    python recorder_mm2.py --hours 4
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

LBANK_WS = "wss://www.lbkex.net/ws/V2/"
OPEN_TIMEOUT = 30.0
SILENCE_TIMEOUT = 25.0
KEEPALIVE_SEC = 5.0
FLUSH_EVERY_SEC = 2.0
PAIRS_PER_CONN = 4
LEVELS = 3


def load_pairs(path):
    if not os.path.exists(path):
        print(f"فایل {path} پیدا نشد. اول spread_scanner را اجرا کن.")
        sys.exit(1)
    out = [l.strip() for l in open(path)
           if l.strip() and not l.startswith("#")]
    if not out:
        print(f"{path} خالی است.")
        sys.exit(1)
    return out


def iso_ms(s):
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return ""


class Writer:
    def __init__(self, outdir="data"):
        os.makedirs(outdir, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        self.path = os.path.join(outdir, f"q_{stamp}.csv")
        self._fh = open(self.path, "w", newline="", buffering=1024 * 1024)
        self._w = csv.writer(self._fh)
        cols = ["ts_ms", "ts_srv", "symbol", "kind"]
        for i in range(1, LEVELS + 1):
            cols += [f"b{i}p", f"b{i}q", f"a{i}p", f"a{i}q"]
        cols += ["price", "qty", "side"]
        self._w.writerow(cols)
        self.ncols = len(cols)
        self.buf = []
        self.counts = defaultdict(int)
        self.sides = defaultdict(int)
        self.reconnects = 0

    def book(self, ts, srv, sym, bids, asks):
        row = [ts, srv, sym, "book"]
        for i in range(LEVELS):
            bp, bq = bids[i] if i < len(bids) else ("", "")
            ap, aq = asks[i] if i < len(asks) else ("", "")
            row += [bp, bq, ap, aq]
        row += ["", "", ""]
        self.buf.append(row)
        self.counts[(sym, "book")] += 1

    def trade(self, ts, srv, sym, price, qty, side):
        row = [ts, srv, sym, "trade"] + [""] * (LEVELS * 4) + [price, qty, side]
        self.buf.append(row)
        self.counts[(sym, "trade")] += 1
        self.sides[side] += 1

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


async def keepalive(ws, stop):
    while not stop.is_set():
        await asyncio.sleep(KEEPALIVE_SEC)
        try:
            await ws.send(json.dumps({"action": "ping", "ping": str(uuid.uuid4())}))
        except Exception:
            return


def parse_levels(raw, reverse):
    out = []
    for x in raw:
        try:
            p, q = float(x[0]), float(x[1])
        except (ValueError, IndexError, TypeError):
            continue
        if p > 0 and q > 0:
            out.append((p, q))
    out.sort(key=lambda t: t[0], reverse=reverse)
    return out[:LEVELS]


async def stream(w, stop, pairs, tag):
    backoff = 1
    while not stop.is_set():
        ka = None
        try:
            async with websockets.connect(
                LBANK_WS, ping_interval=None, open_timeout=OPEN_TIMEOUT
            ) as ws:
                for p in pairs:
                    await ws.send(json.dumps({
                        "action": "subscribe", "subscribe": "depth",
                        "depth": "10", "pair": p}))
                    await asyncio.sleep(0.10)
                    await ws.send(json.dumps({
                        "action": "subscribe", "subscribe": "trade", "pair": p}))
                    await asyncio.sleep(0.10)

                print(f"[{tag}] متصل: {', '.join(pairs)}", flush=True)
                backoff = 1
                ka = asyncio.create_task(keepalive(ws, stop))

                while not stop.is_set():
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
                    sym = m.get("pair", "")

                    if typ == "depth":
                        d = m.get("depth") or {}
                        bids = parse_levels(d.get("bids") or [], reverse=True)
                        asks = parse_levels(d.get("asks") or [], reverse=False)
                        if bids and asks and asks[0][0] > bids[0][0]:
                            w.book(ts, srv, sym, bids, asks)

                    elif typ == "trade":
                        t = m.get("trade") or {}
                        tsrv = iso_ms(t["TS"]) if t.get("TS") else srv
                        try:
                            px = float(t["price"])
                            qty = float(t.get("amount") or t.get("volume") or 0)
                        except (KeyError, TypeError, ValueError):
                            continue
                        side = str(t.get("direction") or t.get("type") or "").lower()
                        if px > 0 and qty > 0:
                            w.trade(ts, tsrv, sym, px, qty, side)

        except Exception as e:
            if stop.is_set():
                break
            w.reconnects += 1
            print(f"[{tag}] قطع ({e}) — {backoff}s", flush=True)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 20)
        finally:
            if ka:
                ka.cancel()


async def house(w, stop, secs):
    t0, last = time.time(), 0
    while not stop.is_set():
        await asyncio.sleep(FLUSH_EVERY_SEC)
        w.flush()
        el = time.time() - t0
        if el - last >= 60:
            last = el
            nb = sum(v for (_, k), v in w.counts.items() if k == "book")
            nt = sum(v for (_, k), v in w.counts.items() if k == "trade")
            mb = os.path.getsize(w.path) / 1e6
            print(f"[{el/60:5.1f}د] عمق={nb:,} معامله={nt:,} | {mb:.0f}MB | "
                  f"جهت={dict(w.sides)} | قطعی={w.reconnects}", flush=True)
        if secs and el >= secs:
            print("\nزمان ضبط تمام شد.", flush=True)
            stop.set()


async def main(a):
    global PAIRS_PER_CONN
    PAIRS_PER_CONN = max(1, a.per_conn)
    pairs = load_pairs(a.pairs)
    print(f"جفت‌ها ({len(pairs)}): {', '.join(pairs)}\n")
    w = Writer()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sg in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sg, stop.set)
        except NotImplementedError:
            pass

    print(f"ضبط → {w.path}\n")
    chunks = [pairs[i:i + PAIRS_PER_CONN]
              for i in range(0, len(pairs), PAIRS_PER_CONN)]
    tasks = [house(w, stop, int(a.hours * 3600))]
    for i, c in enumerate(chunks):
        tasks.append(stream(w, stop, c, f"c{i+1}"))
    await asyncio.gather(*tasks, return_exceptions=True)

    w.close()
    print(f"\nذخیره: {w.path}")
    for (sym, kind), n in sorted(w.counts.items()):
        print(f"   {sym:<14} {kind:<6} {n:>10,}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=float, default=4)
    p.add_argument("--pairs", default="top_pairs.txt")
    p.add_argument("--per-conn", type=int, default=4,
                   help="چند جفت روی هر اتصال (کمتر = پایدارتر)")
    a = p.parse_args()
    try:
        asyncio.run(main(a))
    except KeyboardInterrupt:
        print("\nمتوقف شد.")
        sys.exit(0)
