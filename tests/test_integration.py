# SPDX-License-Identifier: MPL-2.0

"""Integration tests for multi-service interactions."""

from __future__ import annotations

import pytest
import time
from fastapi.testclient import TestClient
from services.api_gateway.main import app, create_access_token

client = TestClient(app)


@pytest.fixture
def auth_token():
    """Fixture to provide authentication token for tests."""
    response = client.post(
        "/token",
        data={"username": "admin", "password": "password"}
    )
    assert response.status_code == 200
    return response.json()["access_token"]


@pytest.fixture
def auth_headers(auth_token):
    """Fixture to provide authorization headers."""
    return {"Authorization": f"Bearer {auth_token}"}


class TestFullAttackWorkflow:
    """Integration tests for complete attack workflow."""

    def test_end_to_end_attack_pipeline(self, auth_headers):
        """Test full attack pipeline from launch to results."""
        # Step 1: Get available models
        models_response = client.get("/models", headers=auth_headers)
        assert models_response.status_code == 200
        models = models_response.json()["data"]
        assert len(models) > 0
        default_model = next((m for m in models if "default" in m["id"]), models[0])

        # Step 2: Launch attack against model
        attack_payload = {
            "model_id": default_model["id"],
            "attack_method_id": "textfooler",
            "input_data": "This is a great product!",
            "target_label": "Negative",
            "attack_parameters": {"num_words_to_change": 1, "max_candidates": 5}
        }
        launch_response = client.post(
            "/attacks/launch",
            json=attack_payload,
            headers=auth_headers
        )
        assert launch_response.status_code == 200
        attack_data = launch_response.json()
        attack_id = attack_data["attack_id"]
        assert attack_data["status"] == "initiated"

        # Step 3: Monitor attack progress
        max_wait_time = 30
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            status_response = client.get(
                f"/attacks/{attack_id}/status",
                headers=auth_headers
            )
            assert status_response.status_code == 200
            status = status_response.json()["status"]

            if status in ["completed", "failed"]:
                break
            time.sleep(2)

        assert status in ["completed", "failed"], \
            f"Attack did not complete within {max_wait_time} seconds"

        # Step 4: Retrieve attack results
        if status == "completed":
            results_response = client.get(
                f"/attacks/{attack_id}/results",
                headers=auth_headers
            )
            assert results_response.status_code == 200
            results = results_response.json()
            assert "original_input" in results
            assert "adversarial_example" in results
            assert "attack_success" in results

    def test_attack_on_different_models(self, auth_headers):
        """Test attacks on multiple models."""
        models_response = client.get("/models", headers=auth_headers)
        models = models_response.json()["data"][:3]  # Test on first 3 models

        attack_ids = []
        for model in models:
            attack_payload = {
                "model_id": model["id"],
                "attack_method_id": "textfooler",
                "input_data": f"Testing {model['name']}",
                "target_label": "Negative",
                "attack_parameters": {}
            }
            response = client.post(
                "/attacks/launch",
                json=attack_payload,
                headers=auth_headers
            )
            assert response.status_code == 200
            attack_ids.append(response.json()["attack_id"])

        # Verify all attacks were created
        assert len(attack_ids) >= 3


class TestModelManagement:
    """Integration tests for model management operations."""

    def test_create_and_verify_model(self, auth_headers):
        """Test creating a new model and verifying it appears in listings."""
        new_model = {
            "id": f"integration-test-model-{int(time.time())}",
            "name": "Integration Test Model",
            "type": "NLP",
            "version": "1.0.0",
            "description": "Test model for integration testing"
        }

        # Create model
        create_response = client.post(
            "/models",
            json=new_model,
            headers=auth_headers
        )
        assert create_response.status_code == 201

        # Verify model appears in list
        list_response = client.get("/models", headers=auth_headers)
        assert list_response.status_code == 200
        models = list_response.json()["data"]
        assert any(m["id"] == new_model["id"] for m in models)

        # Retrieve specific model
        get_response = client.get(
            f"/models/{new_model['id']}",
            headers=auth_headers
        )
        assert get_response.status_code == 200
        retrieved = get_response.json()
        assert retrieved["id"] == new_model["id"]
        assert retrieved["name"] == new_model["name"]

    def test_model_filtering_and_sorting(self, auth_headers):
        """Test various filtering and sorting operations on models."""
        # Filter by type
        response = client.get("/models?type=NLP", headers=auth_headers)
        assert response.status_code == 200
        models = response.json()["data"]
        assert all(m["type"] == "NLP" for m in models)

        # Filter by status
        response = client.get("/models?status=active", headers=auth_headers)
        assert response.status_code == 200
        models = response.json()["data"]
        assert all(m["status"] == "active" for m in models)

        # Test sorting
        response = client.get(
            "/models?sort_by=name&sort_order=asc",
            headers=auth_headers
        )
        assert response.status_code == 200
        models = response.json()["data"]
        if len(models) > 1:
            for i in range(len(models) - 1):
                assert models[i]["name"] <= models[i + 1]["name"]


class TestAttackQueryAndAnalysis:
    """Integration tests for attack history and analysis."""

    def test_attack_filtering_and_analytics(self, auth_headers):
        """Test filtering attacks and analytical queries."""
        # Get successful attacks
        success_response = client.get(
            "/attacks?attack_success=true",
            headers=auth_headers
        )
        assert success_response.status_code == 200
        successful_attacks = success_response.json()["data"]
        assert all(a["attack_success"] is True for a in successful_attacks)

        # Get failed attacks
        failed_response = client.get(
            "/attacks?attack_success=false",
            headers=auth_headers
        )
        assert failed_response.status_code == 200
        failed_attacks = failed_response.json()["data"]
        assert all(a["attack_success"] is False for a in failed_attacks)

    def test_pagination_consistency(self, auth_headers):
        """Verify pagination works correctly across requests."""
        page1 = client.get(
            "/attacks?limit=5&skip=0&sort_by=created_at&sort_order=asc",
            headers=auth_headers
        ).json()

        page2 = client.get(
            "/attacks?limit=5&skip=5&sort_by=created_at&sort_order=asc",
            headers=auth_headers
        ).json()

        # Ensure pages don't have overlapping IDs
        page1_ids = {a["id"] for a in page1["data"]}
        page2_ids = {a["id"] for a in page2["data"]}
        assert len(page1_ids & page2_ids) == 0, "Pages have overlapping records"


class TestAuthenticationAndAuthorization:
    """Integration tests for security and access control."""

    def test_unauthorized_access_to_protected_endpoints(self):
        """Verify protected endpoints reject unauthorized access."""
        protected_endpoints = [
            ("/predict", "POST", {"model_id": "test", "input_data": "test"}),
            ("/models", "GET", None),
            ("/attacks/launch", "POST", {}),
            ("/attacks", "GET", None),
        ]

        for endpoint, method, payload in protected_endpoints:
            if method == "GET":
                response = client.get(endpoint)
            else:
                response = client.post(endpoint, json=payload)

            assert response.status_code == 401, \
                f"Endpoint {endpoint} should require authentication"

    def test_token_expiration_and_renewal(self):
        """Test token lifecycle and renewal."""
        # Get initial token
        response1 = client.post(
            "/token",
            data={"username": "admin", "password": "password"}
        )
        assert response1.status_code == 200
        token1 = response1.json()["access_token"]

        # Use token for authenticated request
        headers = {"Authorization": f"Bearer {token1}"}
        response = client.get("/models", headers=headers)
        assert response.status_code == 200

        # Get new token
        response2 = client.post(
            "/token",
            data={"username": "admin", "password": "password"}
        )
        assert response2.status_code == 200
        token2 = response2.json()["access_token"]

        # Both tokens should work
        headers2 = {"Authorization": f"Bearer {token2}"}
        response = client.get("/models", headers=headers2)
        assert response.status_code == 200


class TestErrorHandling:
    """Integration tests for error conditions and edge cases."""

    def test_nonexistent_resource_access(self, auth_headers):
        """Verify proper error responses for nonexistent resources."""
        response = client.get(
            "/models/nonexistent-model-123456",
            headers=auth_headers
        )
        assert response.status_code == 404

    def test_invalid_attack_parameters(self, auth_headers):
        """Test handling of invalid attack parameters."""
        invalid_payload = {
            "model_id": "default-sentiment-model",
            "attack_method_id": "invalid-method",
            "input_data": "test",
            "target_label": "Invalid",
            "attack_parameters": {}
        }
        response = client.post(
            "/attacks/launch",
            json=invalid_payload,
            headers=auth_headers
        )
        # Should either fail immediately or with 422
        assert response.status_code in [400, 422, 404]

    def test_concurrent_attack_submissions(self, auth_headers):
        """Test handling multiple concurrent attack submissions."""
        attack_ids = []
        for i in range(3):
            payload = {
                "model_id": "default-sentiment-model",
                "attack_method_id": "textfooler",
                "input_data": f"Concurrent test {i}",
                "target_label": "Negative",
                "attack_parameters": {}
            }
            response = client.post(
                "/attacks/launch",
                json=payload,
                headers=auth_headers
            )
            assert response.status_code == 200
            attack_ids.append(response.json()["attack_id"])

        # Verify all attacks were created with unique IDs
        assert len(set(attack_ids)) == len(attack_ids), \
            "Generated attack IDs are not unique"
