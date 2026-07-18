"""
Tests for RateLimitMiddleware (Milestone 10).

Uses FastAPI's TestClient (real ASGI request/response cycle through the
actual middleware stack, no mocking of the middleware itself) against a
minimal throwaway app, so the limiter's logic is proven directly rather
than only asserted about in isolation.
"""
import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

os.environ["RATE_LIMIT_REQUESTS_PER_MINUTE"] = "5"
from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.core.rate_limit import RateLimitMiddleware  # noqa: E402


@pytest.fixture
def app():
    test_app = FastAPI()
    test_app.add_middleware(RateLimitMiddleware)

    @test_app.get("/health")
    async def health():
        return {"status": "ok"}

    @test_app.get("/limited")
    async def limited():
        return {"ok": True}

    return test_app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestRateLimitMiddleware:
    def test_requests_within_limit_succeed(self, client):
        for _ in range(5):
            response = client.get("/limited")
            assert response.status_code == 200

    def test_request_beyond_limit_returns_429(self, client):
        for _ in range(5):
            client.get("/limited")
        response = client.get("/limited")
        assert response.status_code == 429
        assert "Retry-After" in response.headers

    def test_exempt_path_is_never_rate_limited(self, client):
        for _ in range(20):
            response = client.get("/health")
            assert response.status_code == 200

    def test_different_clients_have_independent_limits(self, app):
        client_a = TestClient(app, headers={"x-forwarded-for": "1.1.1.1"})
        client_b = TestClient(app, headers={"x-forwarded-for": "2.2.2.2"})

        for _ in range(5):
            assert client_a.get("/limited").status_code == 200
        # Client A is now at its limit...
        assert client_a.get("/limited").status_code == 429
        # ...but client B, a different IP, has its own independent budget.
        assert client_b.get("/limited").status_code == 200

    def test_disabled_rate_limiting_never_blocks(self, monkeypatch):
        monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
        get_settings.cache_clear()

        test_app = FastAPI()
        test_app.add_middleware(RateLimitMiddleware)

        @test_app.get("/limited")
        async def limited():
            return {"ok": True}

        disabled_client = TestClient(test_app)
        for _ in range(20):
            assert disabled_client.get("/limited").status_code == 200

        monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
        get_settings.cache_clear()
