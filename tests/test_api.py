"""End-to-end API tests, including the full upload -> report workflow.

Also asserts the backward-compatibility contract: every original route still
responds, and POST /predict still returns plain text for the existing jQuery
handler.
"""
from __future__ import annotations

import io

import pytest

ORIGINAL_ROUTES = ["/", "/home", "/about", "/service", "/faq", "/contact", "/appointment"]
NEW_PAGES = ["/research", "/dataset", "/history"]


def _upload(client, payload: bytes, filename: str = "scan.png", endpoint: str = "/api/predict"):
    return client.post(
        endpoint,
        data={"file": (io.BytesIO(payload), filename)},
        content_type="multipart/form-data",
    )


# ------------------------------------------------------- backward compatibility


@pytest.mark.parametrize("route", ORIGINAL_ROUTES)
def test_original_routes_still_render(client, route):
    assert client.get(route).status_code == 200


@pytest.mark.parametrize("route", NEW_PAGES)
def test_new_pages_render(client, route):
    assert client.get(route).status_code == 200


def test_legacy_predict_returns_plain_text(client, png_bytes):
    """The original endpoint's contract (text, not JSON) is preserved."""
    response = _upload(client, png_bytes, endpoint="/predict")
    if response.status_code == 503:
        pytest.skip("No model available in this environment.")
    assert response.status_code == 200
    assert "application/json" not in response.content_type
    body = response.get_data(as_text=True)
    assert "Model classification:" in body
    # Phase 10: must not assert a diagnosis.
    assert "You have" not in body


# ---------------------------------------------------------------- validation


def test_predict_without_file_returns_400(client):
    assert client.post("/api/predict", data={}, content_type="multipart/form-data").status_code == 400


def test_predict_rejects_bad_extension(client, png_bytes):
    response = _upload(client, png_bytes, "bad.exe")
    assert response.status_code == 400
    assert response.get_json()["error_type"] == "validation"


def test_predict_rejects_non_image(client):
    response = _upload(client, b"definitely not an image", "fake.png")
    assert response.status_code == 400
    assert "does not appear to be" in response.get_json()["error"]


def test_errors_do_not_leak_stack_traces(client):
    body = _upload(client, b"nope", "x.png").get_data(as_text=True)
    for leak in ("Traceback", "File \"", ".py\", line"):
        assert leak not in body


# ------------------------------------------------------------ full workflow


def test_full_prediction_workflow(client, png_bytes):
    """upload -> validate -> preprocess -> infer -> explain -> report."""
    response = _upload(client, png_bytes)
    if response.status_code == 503:
        pytest.skip("No model available in this environment.")
    assert response.status_code == 200
    report = response.get_json()

    classification = report["classification"]
    assert classification["label"] in ("benign", "malignant")
    assert 0.0 <= classification["probability_positive"] <= 1.0
    assert abs(sum(classification["probabilities"].values()) - 1.0) < 1e-4
    assert "predicted probability" in classification["statement"]

    confidence = report["confidence"]
    assert confidence["band"] in ("high", "moderate", "low")
    assert confidence["calibrated"] is False

    assert report["model"]["name"]
    assert "validated_performance" in report["model"]
    assert report["limitations"]
    assert "not a medical device" in report["disclaimer"].lower()

    explanation = report["explanation"]
    if explanation and explanation.get("overlay_png"):
        assert explanation["overlay_png"].startswith("data:image/png;base64,")
        assert explanation["method"] == "Grad-CAM"
        assert "segmentation" in explanation["disclaimer"]


def test_explanation_can_be_disabled(client, png_bytes):
    response = client.post(
        "/api/predict",
        data={"file": (io.BytesIO(png_bytes), "scan.png"), "explain": "0"},
        content_type="multipart/form-data",
    )
    if response.status_code == 503:
        pytest.skip("No model available in this environment.")
    assert response.get_json()["explanation"] is None


def test_prediction_is_recorded_in_history(client, png_bytes):
    before = client.get("/api/history").get_json()["summary"]["total"]
    if _upload(client, png_bytes).status_code == 503:
        pytest.skip("No model available in this environment.")
    after = client.get("/api/history").get_json()["summary"]["total"]
    assert after == before + 1


def test_history_can_be_cleared(client, png_bytes):
    if _upload(client, png_bytes).status_code == 503:
        pytest.skip("No model available in this environment.")
    assert client.post("/api/history/clear").status_code == 200
    assert client.get("/api/history").get_json()["summary"]["total"] == 0


def test_history_does_not_store_images_by_default(client, png_bytes):
    """PRIVACY: image retention must be opt-in."""
    if _upload(client, png_bytes).status_code == 503:
        pytest.skip("No model available in this environment.")
    rows = client.get("/api/history").get_json()["rows"]
    assert rows
    assert client.get(f"/api/history/{rows[0]['id']}/thumbnail").status_code == 404


def test_text_report_is_downloadable(client, png_bytes):
    response = _upload(client, png_bytes)
    if response.status_code == 503:
        pytest.skip("No model available in this environment.")
    row_id = response.get_json()["history_id"]
    text = client.get(f"/api/report/{row_id}.txt")
    assert text.status_code == 200
    assert "research prototype" in text.get_data(as_text=True)


# ------------------------------------------------------------------- JSON API


def test_health_endpoint(client):
    payload = client.get("/api/health").get_json()
    assert payload["status"] == "ok"
    assert "models_available" in payload


def test_models_endpoint_lists_profiles(client):
    payload = client.get("/api/models").get_json()
    for model in payload["models"]:
        assert "preprocessing_profile" in model


def test_results_endpoint_reports_missing_experiments_honestly(client):
    """HONESTY: no fabricated results when nothing has been run."""
    payload = client.get("/api/results").get_json()
    assert "any_results" in payload
    if not payload["any_results"]:
        assert payload["comparison"]["available"] is False
        assert "not yet executed" in payload["comparison"]["message"]
        assert payload["next_steps"]


def test_unknown_api_route_returns_json_404(client):
    assert client.get("/api/does-not-exist").get_json()["error"] == "Endpoint not found."


def test_figure_route_blocks_path_traversal(client):
    """SECURITY: the figure server must stay inside results/figures."""
    for attempt in ["../../app.py", "..%2F..%2Fapp.py", "....//app.py"]:
        assert client.get(f"/results/figures/{attempt}").status_code in (400, 404)
