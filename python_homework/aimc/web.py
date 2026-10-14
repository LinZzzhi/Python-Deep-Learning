"""Local web workbench for the existing STFT-CNN checkpoints."""

import argparse
import base64
from functools import lru_cache
import io
import json
from pathlib import Path
import tempfile
import threading
import time

from flask import Flask, jsonify, request, send_from_directory
import matplotlib
import numpy as np
from PIL import Image
import torch

from .data import encode_iq, load_iq
from .engine import check_sample_rate, load_checkpoint, probabilities, setup
from .signals import CLASSES, synthesize
from .upstream import ROOT

WEB_ROOT = ROOT / "web"
EXPERIMENTS = {
    "synthetic": ("stft_cnn", "8 类合成信号", "本地合成数据", "1024 点 · 1 MHz"),
    "official": ("official_demo", "30 类官方信号", "官方测试数据重新划分", "1000 点 · 50 MHz"),
}
INFERENCE_LOCK = threading.Lock()


def experiment_root(key):
    if key not in EXPERIMENTS:
        raise ValueError("请选择有效的模型。")
    return ROOT / "outputs" / EXPERIMENTS[key][0]


@lru_cache(maxsize=2)
def get_model(key):
    return load_checkpoint(experiment_root(key) / "best.pt", torch.device("cpu"))


def spectrogram_preview(image):
    pixels = matplotlib.colormaps["turbo"](image.numpy(), bytes=True)[:, :, :3]
    picture = Image.fromarray(np.flipud(pixels)).resize((720, 360), Image.Resampling.NEAREST)
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def infer(key, iq, filename, truth=None):
    started = time.perf_counter()
    with INFERENCE_LOCK:
        model, config, checkpoint = get_model(key)
        if len(iq) > 64:
            raise ValueError("一次最多识别 64 条脉冲。")
        images = encode_iq(iq, config)
        scores = probabilities(model, images, torch.device("cpu"), batch_size=16)
    classes = checkpoint["classes"]
    predictions = []
    for index, score in enumerate(scores):
        best = int(score.argmax())
        predictions.append({"index": index, "best_class": classes[best],
                            "confidence": float(score[best]),
                            "scores": {name: float(score[j]) for j, name in enumerate(classes)}})
    indices = np.linspace(0, iq.shape[-1] - 1, 320, dtype=int)
    return {"model": key, "filename": filename, "ground_truth": truth,
            "predictions": predictions, "duration_ms": round((time.perf_counter() - started) * 1000),
            "n_samples": config.n_samples, "sample_rate_hz": config.sample_rate_hz,
            "spectrogram": spectrogram_preview(images[0, 0]),
            "waveform": {"i": iq[0, 0, indices].tolist(), "q": iq[0, 1, indices].tolist()},
            "created_at": time.time()}


def create_app():
    app = Flask(__name__, static_folder=str(WEB_ROOT), static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
    setup(threads=1)

    @app.get("/")
    def index():
        return send_from_directory(WEB_ROOT, "index.html")

    @app.get("/api/experiments")
    def experiments():
        rows = []
        for key, (_, name, source, window) in EXPERIMENTS.items():
            root = experiment_root(key)
            config = json.loads((root / "config.json").read_text())
            metrics = json.loads((root / "metrics.json").read_text())
            history = json.loads((root / "history.json").read_text())
            rows.append({"id": key, "name": name, "source": source, "window": window,
                         "config": config, "metrics": metrics, "history": history,
                         "checkpoint_available": (root / "best.pt").is_file(),
                         "data_available": (ROOT / "data" / ("synthetic" if key == "synthetic" else "official_demo")).is_dir()})
        return jsonify({"experiments": rows, "device": "CPU", "threads": 1})

    @app.get("/api/assets/<key>/<filename>")
    def asset(key, filename):
        if filename not in {"confusion_matrix.png", "training_curves.png", "spectrograms.png"}:
            return jsonify(error="文件不存在。"), 404
        return send_from_directory(experiment_root(key), filename)

    @app.post("/api/sample")
    def sample():
        body = request.get_json() or {}
        if not isinstance(body, dict):
            raise ValueError("样例参数必须是 JSON 对象。")
        modulation = body.get("modulation", "LFM_UP")
        snr = float(body.get("snr", 6))
        seed = int(body.get("seed", 2026))
        if modulation not in CLASSES or not -20 <= snr <= 30 or not 0 <= seed <= 2**32 - 1:
            raise ValueError("样例参数无效。")
        noisy, _ = synthesize(modulation, np.random.default_rng(seed), snr_db=snr)
        from .data import canonical_iq
        result = infer("synthetic", canonical_iq(noisy), f"{modulation}_{snr:g}dB.npy", modulation)
        result["snr_db"] = snr
        return jsonify(result)

    @app.post("/api/predict")
    def predict_file():
        key = request.form.get("model", "synthetic")
        root = experiment_root(key)
        file = request.files.get("file")
        if not file or not file.filename:
            raise ValueError("请先选择 IQ 文件。")
        suffix = Path(file.filename).suffix.lower()
        if suffix not in {".npy", ".npz", ".csv"}:
            raise ValueError("支持 .npy、.npz 和无表头 I,Q 两列 .csv 文件。")
        config = json.loads((root / "config.json").read_text())["preprocess"]
        supplied = request.form.get("sample_rate")
        supplied = float(supplied) if supplied else None
        with tempfile.TemporaryDirectory(prefix="aimc-web-") as directory:
            path = Path(directory) / ("input" + suffix)
            file.save(path)
            check_sample_rate(path, config["sample_rate_hz"], supplied)
            iq = load_iq(path)
        return jsonify(infer(key, iq, Path(file.filename).name))

    @app.errorhandler(ValueError)
    @app.errorhandler(KeyError)
    @app.errorhandler(OSError)
    def invalid_input(error):
        return jsonify(error=str(error)), 400

    @app.errorhandler(413)
    def too_large(error):
        return jsonify(error="文件过大，上传上限为 8 MB。"), 413

    @app.errorhandler(500)
    def server_error(error):
        return jsonify(error="处理失败，请检查文件内容或查看服务日志。"), 500

    return app


def main():
    parser = argparse.ArgumentParser(description="AIMC local signal workbench")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    create_app().run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
