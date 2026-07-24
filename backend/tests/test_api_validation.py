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
