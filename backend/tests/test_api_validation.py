from fastapi.testclient import TestClient
import httpx
from openai import AuthenticationError

from app.main import app
from app.api import routes


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_request_validation_rejects_missing_fields():
    response = client.post("/chat", json={"question": "x"})
    assert response.status_code == 422


def test_evaluation_request_requires_non_empty_dataset():
    response = client.post(
        "/evaluation/run",
        json={"api_key": "x" * 20, "dataset": [], "top_k": 4},
    )
    assert response.status_code == 422


def test_invalid_api_key_is_returned_as_a_graceful_client_error(monkeypatch):
    def raise_auth_error(_api_key):
        request = httpx.Request("GET", "https://api.openai.com/v1/models")
        response = httpx.Response(status_code=401, request=request)
        raise AuthenticationError("invalid api key", response=response, body=None)

    monkeypatch.setattr(routes.rag_service, "list_documents", raise_auth_error)

    response = client.get("/documents", params={"api_key": "sk-invalid"})

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid API key"}
