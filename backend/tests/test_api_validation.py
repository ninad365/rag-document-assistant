from fastapi.testclient import TestClient

from app.main import app


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
