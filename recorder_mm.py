#!/usr/bin/env python3
"""
MM RECORDER — ضبط برای سنجش انتخاب معکوس
=========================================
با recorder قبلی یک تفاوت کلیدی دارد: **جهت معامله** را هم ذخیره می‌کند.

بدون جهت، نمی‌شود فهمید بازارگردان خریدار بوده یا فروشنده —
و کل سنجش انتخاب معکوس به همین بستگی دارد.

جفت‌ها را از top_pairs.txt می‌خواند (خروجی spread_scanner).
بایننس لازم نیست: این تست کاملاً درون‌LBank است.

اجرا:
    python spread_scanner.py --fee 2.0        # top_pairs.txt را می‌سازد
    python recorder_mm.py --hours 4
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
PAIRS_PER_CONN = 3


def load_pairs(path):
    if not os.path.exists(path):
        print(f"فایل {path} پیدا نشد.")
        print("اول این را اجرا کن:  python spread_scanner.py --fee 2.0")
        sys.exit(1)
    out = []
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
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
        self.path = os.path.join(outdir, f"mm_{stamp}.csv")
        self._fh = open(self.path, "w", newline="", buffering=1024 * 1024)
        self._w = csv.writer(self._fh)
        self._w.writerow(["ts_ms", "ts_srv", "symbol", "kind",
                          "bid", "ask", "price", "qty", "side"])
        self.buf = []
        self.counts = defaultdict(int)
        self.sides = defaultdict(int)
        self.reconnects = 0

    def book(self, ts, srv, sym, bid, ask):
        self.buf.append((ts, srv, sym, "book", bid, ask, "", "", ""))
        self.counts[(sym, "book")] += 1

    def trade(self, ts, srv, sym, price, qty, side):
        self.buf.append((ts, srv, sym, "trade", "", "", price, qty, side))
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
                        bids, asks = d.get("bids"), d.get("asks")
                        if not bids or not asks:
                            continue
                        try:
                            bb = max(float(x[0]) for x in bids if float(x[1]) > 0)
                            ba = min(float(x[0]) for x in asks if float(x[1]) > 0)
                        except (ValueError, IndexError):
                            continue
                        if bb > 0 and ba > bb:
                            w.book(ts, srv, sym, bb, ba)

                    elif typ == "trade":
                        t = m.get("trade") or {}
                        tsrv = iso_ms(t["TS"]) if t.get("TS") else srv
                        try:
                            px = float(t["price"])
                            qty = float(t.get("amount") or t.get("volume") or 0)
                        except (KeyError, TypeError, ValueError):
                            continue
                        # جهت = سمت تهاجمی. کلید اصلی این تست.
                        side = str(t.get("direction") or t.get("type") or "").lower()
                        if px > 0:
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
            sd = dict(w.sides)
            print(f"[{el/60:5.1f}د] عمق={nb:,} معامله={nt:,} | {mb:.0f}MB | "
                  f"جهت‌ها={sd} | قطعی={w.reconnects}", flush=True)
        if secs and el >= secs:
            print("\nزمان ضبط تمام شد.", flush=True)
            stop.set()


async def main(a):
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
    print(f"جهت معاملات: {dict(w.sides)}")
    if not any(k for k in w.sides if k):
        print("⚠ هیچ جهتی ثبت نشد — تست انتخاب معکوس کار نخواهد کرد.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=float, default=4)
    p.add_argument("--pairs", default="top_pairs.txt")
    a = p.parse_args()
    try:
        asyncio.run(main(a))
    except KeyboardInterrupt:
        print("\nمتوقف شد.")
        sys.exit(0)
