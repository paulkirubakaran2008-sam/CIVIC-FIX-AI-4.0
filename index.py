"""CivicFix AI v4 - Vercel serverless endpoint.

GET  /api          health + model info
POST /api          {"complaint": "..."}                -> one analysis
POST /api/batch    {"complaints": ["...", "..."]}      -> ranked list (max 200)
"""
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _core  # noqa: E402

HEADERS = {
    "Content-Type": "application/json; charset=utf-8",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}


class handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        for k, v in HEADERS.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, {})

    def do_GET(self):
        b = _core.get_bundle()
        self._send(200, {"status": "ok", "version": "4.0", "trained_on": b["n_train"],
                         "categories": list(b["category"].classes_)})

    def do_POST(self):
        t0 = time.perf_counter()
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > 200_000:
                return self._send(413, {"error": "Payload too large."})
            data = json.loads(self.rfile.read(n) or b"{}")
            if self.path.rstrip("/").endswith("/batch"):
                items = data.get("complaints")
                if not isinstance(items, list) or not items:
                    return self._send(400, {"error": "Send {\"complaints\": [..]}"})
                texts = [_core.validate(x) for x in items[:200] if str(x).strip()]
                res = _core.analyze_many(texts)
                res.sort(key=lambda r: -r["priority_score"])
                return self._send(200, {"count": len(res), "results": res,
                                        "ms": round((time.perf_counter() - t0) * 1000, 1)})
            res = _core.analyze_one(_core.validate(data.get("complaint")))
            res["ms"] = round((time.perf_counter() - t0) * 1000, 1)
            self._send(200, res)
        except ValueError as e:
            self._send(400, {"error": str(e)})
        except Exception as e:  # never leak a stack trace
            self._send(500, {"error": "Analysis failed: " + type(e).__name__})
