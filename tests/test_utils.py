# SPDX-License-Identifier: MPL-2.0

"""Utility functions and fixtures for testing."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from services.api_gateway.main import app


@pytest.fixture
def api_client():
    """Provide a test client for API testing."""
    return TestClient(app)


@pytest.fixture
def authenticated_client(api_client):
    """Provide an authenticated test client."""
    response = api_client.post(
        "/token",
        data={"username": "admin", "password": "password"}
    )
    token = response.json()["access_token"]

    class AuthenticatedClient:
        def __init__(self, client, token):
            self.client = client
            self.headers = {"Authorization": f"Bearer {token}"}

        def get(self, *args, **kwargs):
            kwargs.setdefault("headers", self.headers)
            return self.client.get(*args, **kwargs)

        def post(self, *args, **kwargs):
            kwargs.setdefault("headers", self.headers)
            return self.client.post(*args, **kwargs)

        def put(self, *args, **kwargs):
            kwargs.setdefault("headers", self.headers)
            return self.client.put(*args, **kwargs)

        def delete(self, *args, **kwargs):
            kwargs.setdefault("headers", self.headers)
            return self.client.delete(*args, **kwargs)

    return AuthenticatedClient(api_client, token)


def assert_successful_response(response, expected_status_code=200):
    """Assert that response is successful."""
    assert response.status_code == expected_status_code, \
        f"Expected {expected_status_code}, got {response.status_code}: {response.text}"


def assert_error_response(response, expected_status_code=400):
    """Assert that response is an error."""
    assert response.status_code == expected_status_code, \
        f"Expected {expected_status_code}, got {response.status_code}"


def assert_required_fields_in_response(response, required_fields):
    """Assert that required fields are present in response."""
    data = response.json()
    for field in required_fields:
        assert field in data, f"Required field '{field}' missing from response"
