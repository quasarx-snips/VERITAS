import base64
import json
import threading
import time
from http.client import HTTPConnection
from pathlib import Path

import pytest
from veritas.api import VeritasAPIHandler, ThreadingHTTPServer

TEST_PORT = 8791


@pytest.fixture(scope="module")
def api_server():
    server = ThreadingHTTPServer(("127.0.0.1", TEST_PORT), VeritasAPIHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.15)
    yield f"http://127.0.0.1:{TEST_PORT}"
    server.shutdown()
    thread.join(timeout=2.0)
    server.server_close()


def test_api_health(api_server):
    conn = HTTPConnection("127.0.0.1", TEST_PORT)
    conn.request("GET", "/api/health")
    res = conn.getresponse()
    assert res.status == 200
    data = json.loads(res.read().decode("utf-8"))
    assert data["status"] == "healthy"
    assert "SIFT" in data["detectors"]


def test_static_page_served(api_server):
    conn = HTTPConnection("127.0.0.1", TEST_PORT)
    conn.request("GET", "/pages/page_zero.html")
    res = conn.getresponse()
    assert res.status == 200
    body = res.read().decode("utf-8")
    assert "Veritas" in body


def test_analysis_page_served(api_server):
    conn = HTTPConnection("127.0.0.1", TEST_PORT)
    conn.request("GET", "/pages/analysis_page.html")
    res = conn.getresponse()
    assert res.status == 200
    body = res.read().decode("utf-8")
    assert "VERIFICATION REPORT" in body
    assert "CANDIDATE MATCHES" in body


def test_api_verify_samples_base64(api_server):
    before_file = Path("samples/sample_before.png")
    after_file = Path("samples/sample_after.jpg")

    if not before_file.exists() or not after_file.exists():
        pytest.skip("Sample files not present")

    b64_before = base64.b64encode(before_file.read_bytes()).decode("utf-8")
    b64_after = base64.b64encode(after_file.read_bytes()).decode("utf-8")

    payload = json.dumps({
        "before_base64": b64_before,
        "after_base64": b64_after,
    })

    conn = HTTPConnection("127.0.0.1", TEST_PORT)
    conn.request("POST", "/api/verify", body=payload, headers={"Content-Type": "application/json"})
    res = conn.getresponse()
    assert res.status == 200
    data = json.loads(res.read().decode("utf-8"))
    assert data["status"] == "success"
    assert "verdict" in data
    assert "gate" in data
    assert "audit" in data
