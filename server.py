#!/usr/bin/env python3
"""Small standard-library server for the TensorX chat app."""

import json
import os
import ssl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "public"


def load_env():
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_env()
API_URL = os.getenv("TENSORX_API_URL", "https://api.tensorx.ai/v1").rstrip("/")
MODEL = os.getenv("TENSORX_MODEL", "z-ai/glm-5.3-flash")
PORT = int(os.getenv("PORT", "3000"))
HOST = os.getenv("HOST", "0.0.0.0")


def https_context():
    """Use the system CA bundle; some macOS Python builds point at a missing file."""
    candidates = [
        os.getenv("SSL_CERT_FILE"),
        "/etc/ssl/cert.pem",
        "/etc/ssl/certs/ca-certificates.crt",
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return ssl.create_default_context(cafile=candidate)
    return ssl.create_default_context()


TLS_CONTEXT = https_context()


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):  # noqa: N802
        if self.path != "/api/chat":
            self.send_json(404, {"error": "Not found"})
            return
        if not os.getenv("TENSORX_API_KEY"):
            self.send_json(500, {"error": "Add your TENSORX_API_KEY to .env first."})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            request_body = json.dumps({
                "model": body.get("model", MODEL),
                "messages": body.get("messages", []),
                "temperature": body.get("temperature", 0.7),
                "stream": False,
            }).encode()
            request = Request(
                f"{API_URL}/chat/completions",
                data=request_body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {os.environ['TENSORX_API_KEY']}",
                },
                method="POST",
            )
            with urlopen(request, timeout=120, context=TLS_CONTEXT) as response:
                data = response.read()
                self.send_response(response.status)
                self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except HTTPError as error:
            self.send_response(error.code)
            self.send_header("Content-Type", error.headers.get("Content-Type", "application/json"))
            self.end_headers()
            self.wfile.write(error.read())
        except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
            self.send_json(502, {"error": f"TensorX request failed: {error}"})

    def do_GET(self):  # noqa: N802
        relative = self.path.split("?", 1)[0]
        relative = "/index.html" if relative == "/" else relative
        requested = (PUBLIC / relative.lstrip("/")).resolve()
        if PUBLIC not in requested.parents:
            self.send_json(404, {"error": "Not found"})
            return
        try:
            data = requested.read_bytes()
        except OSError:
            self.send_json(404, {"error": "Not found"})
            return
        types = {".html": "text/html", ".css": "text/css", ".js": "text/javascript"}
        self.send_response(200)
        self.send_header("Content-Type", types.get(requested.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        print(f"{self.address_string()} - {format % args}")


if __name__ == "__main__":
    print(f"TensorX chat running at http://localhost:{PORT}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
