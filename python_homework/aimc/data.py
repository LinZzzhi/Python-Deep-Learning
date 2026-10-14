from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .upstream import power_spectrogram


@dataclass
class PreprocessConfig:
    n_samples: int = 1024
    sample_rate_hz: float = 1_000_000
    n_fft: int = 64
    win_len: int = 64
    hop_len: int = 32
    image_size: int = 96
    dyn_range_db: float = 40.0
    power_mapping: str = "dynamic_range"
    quantize_uint8: bool = False

    def __post_init__(self):
        if self.n_samples < 128 or self.image_size < 64:
            raise ValueError("Need n_samples >= 128 and image_size >= 64.")
        if self.win_len < 1 or self.n_fft < self.win_len or self.hop_len < 1:
            raise ValueError("Invalid STFT window, FFT size or hop length.")
        if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0:
            raise ValueError("Sample rate must be finite and positive.")
        if not np.isfinite(self.dyn_range_db) or self.dyn_range_db <= 0:
            raise ValueError("Dynamic range must be finite and positive.")
        if self.power_mapping not in {"dynamic_range", "upstream_fixed_noise"}:
            raise ValueError("Unknown power mapping.")

    def to_dict(self):
        return asdict(self)


def canonical_iq(array):
    array = np.asarray(array)
    if np.iscomplexobj(array):
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2:
            raise ValueError("Complex IQ must have shape (N,) or (B,N).")
        result = np.stack([array.real, array.imag], axis=1)
    else:
        if array.ndim == 2 and array.shape[0] == 2:
            result = array[None, :, :]
        elif array.ndim == 2 and array.shape[1] == 2:
            result = array.T[None, :, :]
        elif array.ndim == 3 and array.shape[1] == 2:
            result = array
        elif array.ndim == 3 and array.shape[2] == 2:
            result = array.transpose(0, 2, 1)
        else:
            raise ValueError("Real IQ must have shape (2,N), (N,2), (B,2,N) or (B,N,2).")
    result = np.ascontiguousarray(result, dtype=np.float32)
    if not result.shape[0] or result.shape[-1] < 128:
        raise ValueError("Input must contain pulses of at least 128 samples.")
    if not np.isfinite(result).all():
        raise ValueError("IQ contains NaN or infinity.")
    if np.any(np.sum(result.astype(np.float64) ** 2, axis=(1, 2)) == 0):
        raise ValueError("IQ contains an all-zero pulse.")
    return result


def load_iq(path):
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return canonical_iq(archive["iq"])
    if path.suffix.lower() == ".npy":
        return canonical_iq(np.load(path, allow_pickle=False))
    if path.suffix.lower() == ".csv":
        return canonical_iq(np.loadtxt(path, delimiter=","))
    raise ValueError("Input format must be .npy, .npz (iq key), or headerless I,Q .csv.")


def encode_iq(iq, config, batch_size=128):
    iq = canonical_iq(iq)
    if iq.shape[-1] != config.n_samples:
        raise ValueError(
            f"Expected {config.n_samples} samples per pulse, got {iq.shape[-1]}. "
            "Segment/resample pulses consistently before training and inference."
        )
    if batch_size < 1:
        raise ValueError("Batch size must be positive.")
    images = []
    for start in range(0, len(iq), batch_size):
        batch = torch.from_numpy(iq[start:start + batch_size])
        power = power_spectrogram(
            batch, n_fft=config.n_fft, win_len=config.win_len,
            hop_len=config.hop_len,
            dyn_range_db=config.dyn_range_db if config.power_mapping == "dynamic_range" else None,
        )
        power = F.interpolate(
            power[:, None], size=(config.image_size, config.image_size),
            mode="bilinear", align_corners=False,
        )
        if config.quantize_uint8:
            power = (power.clamp(0, 1) * 255).round() / 255
        images.append(power)
    return torch.cat(images)


def load_split(path):
    with np.load(path, allow_pickle=False) as archive:
        iq = canonical_iq(archive["iq"])
        raw_labels = archive["labels"]
        classes = archive["classes"].tolist()
        snr = archive["snr_db"].astype(float) if "snr_db" in archive else None
    if raw_labels.dtype.kind not in "iu":
        raise ValueError("labels must be integer class indices.")
    labels = raw_labels.astype(np.int64)
    if labels.shape != (len(iq),) or np.any(labels < 0) or np.any(labels >= len(classes)):
        raise ValueError(f"Invalid labels in {path}.")
    if not isinstance(classes, list) or len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("classes must contain at least two unique class names.")
    if any(not isinstance(name, str) or not name for name in classes):
        raise ValueError("classes must be nonempty strings.")
    if snr is not None and (snr.shape != labels.shape or not np.isfinite(snr).all()):
        raise ValueError("Invalid snr_db metadata.")
    return iq, labels, classes, snr
