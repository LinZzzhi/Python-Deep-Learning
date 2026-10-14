import io

import numpy as np
import pytest

from aimc.signals import synthesize
from aimc.web import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_experiment_metadata_and_assets(client):
    response = client.get("/api/experiments")
    assert response.status_code == 200
    rows = response.json["experiments"]
    assert [row["config"]["preprocess"]["n_samples"] for row in rows] == [1024, 1000]
    assert rows[0]["metrics"]["accuracy"] == 0.71
    assert client.get("/api/assets/synthetic/confusion_matrix.png").mimetype == "image/png"
    assert client.get("/api/assets/synthetic/best.pt").status_code == 404


def test_sample_prediction_contains_real_signal_preview(client):
    response = client.post("/api/sample", json={"modulation": "LFM_UP", "snr": 6, "seed": 2026})
    assert response.status_code == 200
    result = response.json
    assert result["ground_truth"] == "LFM_UP"
    assert result["n_samples"] == 1024
    assert result["spectrogram"].startswith("data:image/png;base64,")
    assert len(result["waveform"]["i"]) == len(result["waveform"]["q"]) == 320
    scores = result["predictions"][0]["scores"]
    assert sum(scores.values()) == pytest.approx(1, abs=1e-6)
    assert result["predictions"][0]["best_class"] == max(scores, key=scores.get)


def test_uploaded_iq_matches_sample_prediction(client):
    iq, _ = synthesize("LFM_UP", np.random.default_rng(2026), snr_db=6)
    buffer = io.BytesIO()
    np.save(buffer, iq)
    buffer.seek(0)
    result = client.post("/api/predict", data={"model": "synthetic", "sample_rate": "1000000",
                                             "file": (buffer, "pulse.npy")})
    expected = client.post("/api/sample", json={"modulation": "LFM_UP", "snr": 6, "seed": 2026})
    assert result.status_code == 200
    assert result.json["predictions"] == expected.json["predictions"]
    assert result.json["ground_truth"] is None


def test_reject_mismatched_sampling_rate_and_pulse_length(client):
    iq = np.ones(1024, dtype=np.complex64)
    buffer = io.BytesIO()
    np.savez(buffer, iq=iq, sample_rate_hz=1_000_000)
    buffer.seek(0)
    result = client.post("/api/predict", data={"model": "synthetic", "sample_rate": "50000000",
                                             "file": (buffer, "pulse.npz")})
    assert result.status_code == 400
    assert "differs from file" in result.json["error"]
    buffer = io.BytesIO()
    np.save(buffer, iq)
    buffer.seek(0)
    result = client.post("/api/predict", data={"model": "official", "sample_rate": "50000000",
                                             "file": (buffer, "pulse.npy")})
    assert result.status_code == 400
    assert "Expected 1000" in result.json["error"]


def test_reject_invalid_input(client):
    assert client.post("/api/predict").status_code == 400
    assert client.post("/api/sample", json={"modulation": "invalid"}).status_code == 400
    response = client.post("/api/predict", data={"model": "synthetic",
                                               "file": (io.BytesIO(b"invalid"), "pulse.txt")})
    assert response.status_code == 400
    assert client.post("/api/sample", json=["invalid"]).status_code == 400


def test_official_checkpoint_uses_thirty_classes(client):
    buffer = io.BytesIO()
    rng = np.random.default_rng(7)
    iq = (rng.standard_normal(1000) + 1j * rng.standard_normal(1000)).astype(np.complex64)
    np.save(buffer, iq)
    buffer.seek(0)
    response = client.post("/api/predict", data={"model": "official", "sample_rate": "50000000",
                                               "file": (buffer, "pulse.npy")})
    assert response.status_code == 200
    assert response.json["n_samples"] == 1000
    assert len(response.json["predictions"][0]["scores"]) == 30
