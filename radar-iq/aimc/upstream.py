"""Load the two upstream modules without their optional training dependencies."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party" / "AIMC-image-classification"
UPSTREAM_COMMIT = "6d50291fd3ad29389dff872b3ce73c06fb6f5c7c"


def _load(name, relative_path):
    path = UPSTREAM / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Missing upstream source: {path}. See README.md.")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MainModel = _load("aimc_stft_cnn", "src/STFT_CNN/STFT_CNN.py").MainModel
power_spectrogram = _load(
    "aimc_spectrogram", "src/common/spectrogram.py"
).batched_iq_to_power_spectrogram
