"""Local web interface API tests; no LLM or network calls."""

import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import web_app


class WebAppTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(web_app.app)
        self.headers = {"X-App-Token": web_app.APP_TOKEN}

    def test_health_and_bootstrap(self):
        self.assertTrue(self.client.get("/api/health").json()["ok"])
        data = self.client.get("/api/bootstrap").json()
        self.assertEqual(data["token"], web_app.APP_TOKEN)
        self.assertNotIn("api_key", data["settings"])

    def test_mutation_requires_token(self):
        response = self.client.post("/api/index")
        self.assertEqual(response.status_code, 403)

    @patch("web_app.run_agent")
    def test_chat_event_stream(self, run_agent):
        async def answer(*_args):
            return "润色后的段落。"
        run_agent.side_effect = answer
        response = self.client.post(
            "/api/chat", headers=self.headers,
            json={"message": "请润色", "thread_id": "test-thread"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("event: status", response.text)
        self.assertIn("润色后的段落", response.text)
        self.assertIn("event: done", response.text)


if __name__ == "__main__":
    unittest.main()
