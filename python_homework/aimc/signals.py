import numpy as np

CLASSES = ["CW", "LFM_UP", "LFM_DOWN", "NLFM", "BPSK", "BFSK", "4FSK", "SFM"]
SNR_LEVELS = [-6, -3, 0, 3, 6]


def synthesize(modulation, rng, n_samples=1024, snr_db=6.0):
    """Generate one complex baseband pulse; frequencies are cycles per sample.

    SNR is mean clean-pulse power / mean complex AWGN power in expectation.
    Each observation has independently drawn carrier, bandwidth and phase.
    """
    if modulation not in CLASSES:
        raise ValueError(f"Unsupported modulation: {modulation}")
    if n_samples < 128 or not np.isfinite(snr_db):
        raise ValueError("Need at least 128 samples and a finite SNR.")
    t = np.arange(n_samples) / n_samples
    carrier = rng.uniform(-0.10, 0.10)
    bandwidth = rng.uniform(0.20, 0.42)
    phase_code = np.zeros(n_samples)
    freq = np.full(n_samples, carrier)
    if modulation in {"LFM_UP", "LFM_DOWN"}:
        direction = 1 if modulation == "LFM_UP" else -1
        freq = carrier + direction * bandwidth * (t - 0.5)
    elif modulation == "NLFM":
        freq = carrier + bandwidth * (t ** rng.uniform(1.8, 3.2) - 0.5)
    elif modulation == "SFM":
        freq = carrier + bandwidth / 2 * np.sin(
            2 * np.pi * rng.uniform(1.5, 4.0) * t + rng.uniform(0, 2 * np.pi)
        )
    elif modulation in {"BFSK", "4FSK", "BPSK"}:
        chips = int(rng.integers(8, 25))
        chip_index = np.minimum((t * chips).astype(int), chips - 1)
        n_levels = 4 if modulation == "4FSK" else 2
        # Guarantee all symbol levels appear, while varying symbol order.
        symbols = np.resize(np.arange(n_levels), chips)
        rng.shuffle(symbols)
        symbols = symbols[chip_index]
        if modulation == "BPSK":
            phase_code = np.pi * symbols
        else:
            freq = carrier + bandwidth * (symbols / (n_levels - 1) - 0.5)
    phase = 2 * np.pi * np.cumsum(freq) + phase_code + rng.uniform(0, 2 * np.pi)
    amplitude = rng.uniform(0.7, 1.3)
    clean = amplitude * np.exp(1j * phase)
    noise_std = amplitude / np.sqrt(2 * 10 ** (snr_db / 10))
    noisy = clean + noise_std * (
        rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    )
    return noisy.astype(np.complex64), clean.astype(np.complex64)


def generate_split(per_class, seed, n_samples=1024, snrs=SNR_LEVELS):
    if per_class < 1 or not snrs or not np.isfinite(snrs).all():
        raise ValueError("Samples per class must be positive; SNRs must be finite.")
    rng = np.random.default_rng(seed)
    signals, labels, levels = [], [], []
    for label, modulation in enumerate(CLASSES):
        for i in range(per_class):
            snr = float(snrs[i % len(snrs)])
            noisy, _ = synthesize(modulation, rng, n_samples, snr)
            signals.append(noisy)
            labels.append(label)
            levels.append(snr)
    order = rng.permutation(len(labels))
    return {
        "iq": np.stack(signals)[order],
        "labels": np.asarray(labels, dtype=np.int64)[order],
        "snr_db": np.asarray(levels, dtype=np.float32)[order],
        "classes": np.asarray(CLASSES),
    }
