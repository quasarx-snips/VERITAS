import base64
import json
import threading
import time
import unittest
from http.client import HTTPConnection
from pathlib import Path

from veritas.api import VeritasAPIHandler, ThreadingHTTPServer

TEST_PORT = 8791


class TestVeritasAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", TEST_PORT), VeritasAPIHandler)
        cls.server.daemon_threads = True
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.15)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=2.0)
        cls.server.server_close()

    def setUp(self):
        # Each test represents a fresh client session; keep shared rate-limit
        # state from coupling otherwise independent integration tests.
        with VeritasAPIHandler._chat_rate_lock:
            VeritasAPIHandler._last_chat_request_by_ip.clear()

    def test_health(self):
        conn = HTTPConnection("127.0.0.1", TEST_PORT)
        conn.request("GET", "/api/health")
        res = conn.getresponse()
        self.assertEqual(res.status, 200)
        data = json.loads(res.read().decode("utf-8"))
        self.assertEqual(data["status"], "healthy")
        self.assertIn("SIFT", data["detectors"])

    def test_analysis_page_served(self):
        conn = HTTPConnection("127.0.0.1", TEST_PORT)
        conn.request("GET", "/pages/analysis_page.html")
        res = conn.getresponse()
        self.assertEqual(res.status, 200)
        body = res.read().decode("utf-8")
        self.assertIn("VERIFICATION REPORT", body)
        self.assertIn("CANDIDATE MATCHES", body)

    def test_counsel_agent_page_served(self):
        conn = HTTPConnection("127.0.0.1", TEST_PORT)
        conn.request("GET", "/pages/counsel_agent.html")
        res = conn.getresponse()
        self.assertEqual(res.status, 200)
        body = res.read().decode("utf-8")
        self.assertIn("VERITAS Platform UI", body)
        self.assertIn("saveChatsToCookies", body)


    def test_verify_samples(self):
        before_file = Path("samples/sample_before.png")
        after_file = Path("samples/sample_after.jpg")
        if not before_file.exists() or not after_file.exists():
            self.skipTest("Sample files not present")

        b64_before = base64.b64encode(before_file.read_bytes()).decode("utf-8")
        b64_after = base64.b64encode(after_file.read_bytes()).decode("utf-8")

        payload = json.dumps({"before_base64": b64_before, "after_base64": b64_after})
        conn = HTTPConnection("127.0.0.1", TEST_PORT)
        conn.request("POST", "/api/verify", body=payload, headers={"Content-Type": "application/json"})
        res = conn.getresponse()
        self.assertEqual(res.status, 200)
        data = json.loads(res.read().decode("utf-8"))
        self.assertEqual(data["status"], "success")
        self.assertIn("verdict", data)
        self.assertIn("gate", data)

    def test_chat_basic(self):
        payload = json.dumps({"messages": [{"role": "user", "content": "Hello VERITAS"}]})
        conn = HTTPConnection("127.0.0.1", TEST_PORT)
        conn.request("POST", "/api/chat", body=payload, headers={"Content-Type": "application/json"})
        res = conn.getresponse()
        self.assertEqual(res.status, 200)
        data = json.loads(res.read().decode("utf-8"))
        self.assertEqual(data["status"], "success")
        self.assertIn("text", data)
        self.assertTrue(len(data["text"]) > 0)

    def test_chat_rate_limit_returns_retry_after(self):
        payload = json.dumps({"messages": [{"role": "user", "content": "Explain satellite imagery."}]})
        headers = {"Content-Type": "application/json"}

        first = HTTPConnection("127.0.0.1", TEST_PORT)
        first.request("POST", "/api/chat", body=payload, headers=headers)
        self.assertEqual(first.getresponse().status, 200)

        second = HTTPConnection("127.0.0.1", TEST_PORT)
        second.request("POST", "/api/chat", body=payload, headers=headers)
        response = second.getresponse()
        self.assertEqual(response.status, 429)
        self.assertGreaterEqual(int(response.getheader("Retry-After", "0")), 1)

    def test_chat_with_images_and_payload(self):
        before_file = Path("samples/sample_before.png")
        after_file = Path("samples/sample_after.jpg")
        if not before_file.exists() or not after_file.exists():
            self.skipTest("Sample files not present")

        b64_before = base64.b64encode(before_file.read_bytes()).decode("utf-8")
        b64_after = base64.b64encode(after_file.read_bytes()).decode("utf-8")

        payload = json.dumps({
            "before_base64": b64_before,
            "after_base64": b64_after,
            "prompt": "Why did VERITAS reach this verdict?",
        })
        conn = HTTPConnection("127.0.0.1", TEST_PORT)
        conn.request("POST", "/api/chat", body=payload, headers={"Content-Type": "application/json"})
        res = conn.getresponse()
        self.assertEqual(res.status, 200)
        data = json.loads(res.read().decode("utf-8"))
        self.assertEqual(data["status"], "success")
        self.assertIn("text", data)
        self.assertIn("payload", data)
        self.assertIsNotNone(data["payload"])
        self.assertIn("verdict", data["payload"])
        # Check depth and rich markdown formatting
        self.assertIn("###", data["text"])
        self.assertTrue(len(data["text"]) > 200, "Response should be comprehensive and in-depth")

    def test_counsel_context_building(self):
        from veritas.api import build_counsel_context

        sample_payload = {
            "verdict": {"verdict": "VERIFIED", "threshold_profile": "strict", "rationale_codes": ["AFFINE_LOCK"]},
            "gate": {"action": "ALLOW", "reasons": ["QUORUM_SATISFIED"]},
            "explanation": {
                "headline": "Geometric correspondence verified.",
                "key_reasons": ["Consensus across detectors", "Low residual"],
                "recommended_next_action": "Clear for downstream consumption."
            },
            "partial_correspondence": {"status": "GLOBAL", "confidence": 0.94, "rationale": "Even distribution"},
            "detectors": {
                "sift": {"before_count": 210, "after_count": 195},
                "orb": {"before_count": 180, "after_count": 172},
                "akaze": {"before_count": 150, "after_count": 140}
            },
            "summary": {
                "evidence": {
                    "geometry": {
                        "is_valid": True,
                        "inlier_count": 88,
                        "residual_mean": 0.82,
                        "residual_std": 0.31,
                        "residual_max": 2.15,
                        "condition_number": 1.042,
                        "determinant": 0.998,
                        "scale_change": 1.01,
                        "rotation_deg": 0.45,
                        "affine_matrix": [[1.0, 0.0, 5.0], [0.0, 1.0, -2.0]]
                    },
                    "spatial": {"coverage_fraction": 0.78, "active_cell_count": 14},
                    "quorum": {"consensus_status": "UNANIMOUS", "agreed_detectors": ["SIFT", "ORB", "AKAZE"], "quorum_score": 1.0},
                    "matches": {
                        "sift": {"candidate_count": 120, "inlier_count": 88, "inlier_ratio": 0.733},
                        "orb": {"candidate_count": 95, "inlier_count": 64, "inlier_ratio": 0.674},
                        "akaze": {"candidate_count": 80, "inlier_count": 55, "inlier_ratio": 0.687}
                    }
                }
            },
            "audit": {"run_id": "run-test-12345", "timestamp": "2026-09-08T00:00:00Z"}
        }

        context = build_counsel_context(sample_payload)
        self.assertIn("ACTIVE VERITAS FORENSIC EVIDENCE DOSSIER", context)
        self.assertIn("**Official Verdict**: VERIFIED", context)
        self.assertIn("**Safety Gate Action**: ALLOW", context)
        self.assertIn("Certified Affine Lock", context)
        self.assertIn("Mean = 0.820 px", context)
        self.assertIn("Condition Number: 1.0420", context)
        self.assertIn("SIFT: Reference Features = 210", context)
        self.assertIn("Spatial Grid Coverage Fraction: 78.0%", context)
        self.assertIn("run-test-12345", context)

    def test_chat_poll_scouts_table(self):
        sample_payload = {
            "verdict": {"verdict": "VERIFIED", "threshold_profile": "default", "rationale_codes": ["AFFINE_LOCK"]},
            "gate": {"action": "ALLOW", "reasons": ["QUORUM_MET"]},
            "explanation": {"headline": "Quorum certified.", "recommended_next_action": "Safe to proceed."},
            "detectors": {
                "sift": {"before_count": 100, "after_count": 90},
                "orb": {"before_count": 80, "after_count": 75},
                "akaze": {"before_count": 60, "after_count": 55}
            }
        }
        payload = json.dumps({
            "prompt": "Poll the scouts on this pair and check quorum.",
            "payload": sample_payload
        })
        conn = HTTPConnection("127.0.0.1", TEST_PORT)
        conn.request("POST", "/api/chat", body=payload, headers={"Content-Type": "application/json"})
        res = conn.getresponse()
        self.assertEqual(res.status, 200)
        data = json.loads(res.read().decode("utf-8"))
        # Must return rich markdown response analyzing detector scouts
        self.assertIn("SIFT", data["text"])
        self.assertIn("ORB", data["text"])
        self.assertIn("AKAZE", data["text"])
        self.assertTrue(len(data["text"]) > 100)


if __name__ == "__main__":
    unittest.main()

