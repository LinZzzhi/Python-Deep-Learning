import torch
import torch.nn.functional as F
import numpy as np


# =========================
# Batched POWER spectrogram
# =========================
@torch.no_grad()
def batched_iq_to_power_spectrogram(
    x_b2n: torch.Tensor,            # (B, 2, N)
    *,
    n_fft: int = 64,
    win_len: int = 64,
    hop_len: int = 32,
    center_dc: bool = True,
    eps: float = 1e-12,
    # --- two compatible normalisation modes ---------------------------------
    # If dyn_range_db is not None, do per-frame dynamic-range mapping to [0,1]
    dyn_range_db: float | None = None,
    # Otherwise, use the SNR-like mapping you already expose in your single
    # sample function (noise-normalised, to_db, return_mode='snr01', etc.)
    normalise_noise: bool = True,
    noise_floor_mode: str = "fixed",  # 'percentile'|'median'|'fixed'|'none'
    noise_percentile: float = 50.0,
    noise_floor_db: float = 1.0,
    to_db: bool = True,
    db_min: float = -100.0,
    db_max: float = 20.0,
    return_mode: str = "snr01",       # 'db'|'snr_db'|'snr01' (when to_db=True)
) -> torch.Tensor:
    """
    Returns: (B, F, T) float32
      - If dyn_range_db is not None: per-frame max reference, clamp to
        [-dyn_range_db, 0], map to [0,1]  (matches your exporter behaviour).
      - Else: noise-floor + SNR mapping with return_mode (matches your
        single-sample iq_to_power_spectrogram behaviour).
    """
    assert x_b2n.dim() == 3 and x_b2n.size(1) == 2, "expected (B,2,N)"
    B, _, N = x_b2n.shape
    I, Q = x_b2n[:, 0, :], x_b2n[:, 1, :]
    z = torch.complex(I, Q)  # (B, N) complex

    if N < win_len:
        z = F.pad(z, (0, win_len - N))

    win = torch.hann_window(win_len, periodic=True, dtype=torch.float32, device=z.device)
    frames = z.unfold(-1, win_len, hop_len)               # (B, n_frames, win_len)
    frames = frames * win.view(1, 1, -1)
    if n_fft > win_len:
        frames = F.pad(frames, (0, n_fft - win_len))

    S = torch.fft.fft(frames, n=n_fft, dim=-1)            # (B, n_frames, n_fft)
    if center_dc:
        S = torch.fft.fftshift(S, dim=-1)
    S = S.transpose(1, 2)                                  # (B, F, Tspec)

    P = (S.abs() ** 2).clamp_min_(eps)                    # (B,F,T)

    # ---- Path A: per-frame dynamic range → [0,1] (exporter parity) ----------
    if dyn_range_db is not None:
        X_db = 10.0 * torch.log10(P)                      # (B,F,T)
        frame_max = X_db.amax(dim=1, keepdim=True)        # (B,1,T)
        X_rel = (X_db - frame_max).clamp(min=-dyn_range_db, max=0.0)
        out = (X_rel + dyn_range_db) / dyn_range_db       # [0,1]
        return out.to(torch.float32)

    # ---- Path B: noise-floor SNR mapping (single-sample API parity) ----------
    if not to_db:
        if not normalise_noise or noise_floor_mode == "none":
            return P.to(torch.float32)
        # subtract a floor in linear power
        if noise_floor_mode in ("percentile", "median"):
            q = (noise_percentile / 100.0) if noise_floor_mode == "percentile" else 0.5
            floor = torch.quantile(P.reshape(B, -1), q, dim=1, keepdim=True)  # (B,1)
            floor = floor.view(B, 1, 1)
        elif noise_floor_mode == "fixed":
            floor = torch.tensor(10.0 ** (noise_floor_db / 10.0), device=P.device, dtype=P.dtype)
            floor = floor.view(1, 1, 1)
        else:
            floor = torch.zeros(1, 1, 1, device=P.device, dtype=P.dtype)
        out = torch.clamp(P - floor, min=0.0).to(torch.float32)
        return out

    # dB path
    X_db = 10.0 * torch.log10(P)
    if not normalise_noise:
        return X_db.to(torch.float32)

    # choose a floor per (B,*) tensor
    if noise_floor_mode == "percentile":
        nf = torch.quantile(X_db.reshape(B, -1), noise_percentile / 100.0, dim=1, keepdim=True)
        floor_db = nf.view(B, 1, 1)
    elif noise_floor_mode == "median":
        floor_db = X_db.median(dim=1, keepdim=True).values.view(B, 1, 1)
    elif noise_floor_mode == "fixed":
        floor_db = torch.full((B, 1, 1), float(noise_floor_db), device=X_db.device, dtype=X_db.dtype)
    elif noise_floor_mode == "none":
        floor_db = torch.zeros((B, 1, 1), device=X_db.device, dtype=X_db.dtype)
    else:
        raise ValueError("noise_floor_mode must be 'percentile'|'median'|'fixed'|'none'")

    SNR_db = torch.clamp(X_db - floor_db, db_min, db_max)
    if return_mode == "db":
        out = X_db
    elif return_mode == "snr_db":
        out = SNR_db
    elif return_mode == "snr01":
        out = (SNR_db - db_min) / max(db_max - db_min, 1e-6)
    else:
        raise ValueError("return_mode must be 'db'|'snr_db'|'snr01'")
    return out.clamp_(0.0, 1.0).to(torch.float32)


# ========================
# Batched PHASE-EDGE spec
# ========================
@torch.no_grad()
def batched_iq_to_phase_spectrogram(
    x_b2n: torch.Tensor,           # (B, 2, N)
    *,
    n_fft: int = 64,
    win_len: int = 32,
    hop_len: int = 8,
    center_dc: bool = True,
    power_gate_db: float = -12.0,
    eps: float = 1e-12,
    return_mode: str = "abs01",    # 'raw'|'abs'|'abs01'
) -> torch.Tensor:
    """
    Δφ(f,t) = angle( S(f,t) * conj(S(f,t-1)) ), with per-frame power gating.
    Returns: (B, F, T-1) float32; 'abs01' gives [0,1].
    """
    assert x_b2n.dim() == 3 and x_b2n.size(1) == 2, "expected (B,2,N)"
    B, _, N = x_b2n.shape
    I, Q = x_b2n[:, 0, :], x_b2n[:, 1, :]
    z = torch.complex(I, Q)

    if N < win_len:
        z = F.pad(z, (0, win_len - N))

    win = torch.hann_window(win_len, periodic=True, dtype=torch.float32, device=z.device)
    frames = z.unfold(-1, win_len, hop_len)               # (B, n_frames, win_len)
    frames = frames * win.view(1, 1, -1)
    if n_fft > win_len:
        frames = F.pad(frames, (0, n_fft - win_len))

    S = torch.fft.fft(frames, n=n_fft, dim=-1)            # (B, n_frames, n_fft)
    if center_dc:
        S = torch.fft.fftshift(S, dim=-1)
    S = S.transpose(1, 2)                                  # (B, F, Tspec)

    # power gate
    P_db = 10.0 * torch.log10((S.abs() ** 2) + eps)       # (B,F,T)
    frame_max = P_db.amax(dim=1, keepdim=True)            # (B,1,T)
    mask = P_db >= (frame_max + power_gate_db)            # (B,F,T)

    # Δφ along time
    dphi = torch.angle(S[:, :, 1:] * torch.conj(S[:, :, :-1]))  # (B,F,T-1)
    m = mask[:, :, 1:]
    dphi = torch.where(m, dphi, torch.zeros_like(dphi))

    if return_mode == "raw":
        out = dphi
    elif return_mode == "abs":
        out = dphi.abs()
    else:
        out = (dphi.abs() / np.pi).clamp_(0.0, 1.0)
    return out.to(torch.float32)


# ===================================
# Single-sample wrappers (API parity)
# ===================================
@torch.no_grad()
def iq_to_power_spectrogram(
    x_2_by_T: torch.Tensor,
    n_fft: int = 256,
    win_len: int = 256,
    hop_len: int = 128,
    window: torch.Tensor | None = None,  # kept for signature compatibility
    power: float = 2.0,                  # unused here; kept for compatibility
    center_dc: bool = True,
    one_sided: bool = False,             # ignored (complex baseband)
    normalise_noise: bool = True,
    to_db: bool = True,
    noise_floor_mode: str = "fixed",
    noise_percentile: float = 50.0,
    noise_floor_db: float = 1.0,
    db_min: float = -100.0,
    db_max: float = 20.0,
    return_mode: str = "snr01",
    eps: float = 1e-12,
    make_contiguous_final: bool = False,
):
    """
    Backed by the batched kernel to keep behaviour in lockstep with the encoders.
    Returns (F,T) float32 (default in [0,1] with return_mode='snr01').
    """
    x = x_2_by_T.unsqueeze(0)                         # (1,2,N)
    out = batched_iq_to_power_spectrogram(
        x,
        n_fft=n_fft, win_len=win_len, hop_len=hop_len,
        center_dc=center_dc, eps=eps,
        dyn_range_db=None,                            # use noise-floor path
        normalise_noise=normalise_noise,
        noise_floor_mode=noise_floor_mode,
        noise_percentile=noise_percentile,
        noise_floor_db=noise_floor_db,
        to_db=to_db, db_min=db_min, db_max=db_max,
        return_mode=return_mode,
    )[0]
    return out.contiguous() if make_contiguous_final else out


@torch.no_grad()
def iq_to_phase_spectrogram(
    iq: torch.Tensor,             # (2, N)
    n_fft: int = 64,
    win_len: int = 32,
    hop_len: int = 8,
    window: torch.Tensor | None = None,  # kept for signature compatibility
    center_dc: bool = True,
    power_gate_db: float = -12,
    return_mode: str = "abs01",   # 'raw' | 'abs' | 'abs01'
    make_contiguous_final: bool = False,
) -> torch.Tensor:                # (F, Tspec-1) float32
    """
    Backed by the batched kernel for parity.
    """
    x = iq.unsqueeze(0)                                       # (1,2,N)
    out = batched_iq_to_phase_spectrogram(
        x,
        n_fft=n_fft, win_len=win_len, hop_len=hop_len,
        center_dc=center_dc, power_gate_db=power_gate_db,
        return_mode=return_mode,
    )[0]
    return out.contiguous() if make_contiguous_final else out


# ===========================
# Shared image prep utilities
# ===========================
@torch.no_grad()
def to_resnet_batch(
    spec_bft: torch.Tensor,              # (B, F, T) in [0,1]
    size: int = 224,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
) -> torch.Tensor:                        # (B, 3, size, size) channels_last
    x = spec_bft.unsqueeze(1)  # (B,1,F,T)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    mean = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std  = torch.tensor(std,  device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x.contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def resize_spec01_for_save(
    spec01_bft: torch.Tensor, size: int = 224
) -> torch.Tensor:                         # (B,1,H,W) in [0,1]
    x = spec01_bft.unsqueeze(1)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x

@torch.no_grad()
def spec_to_resnet_input(spec_ft: torch.Tensor, size: int = 224, channels: str = "tile3") -> torch.Tensor:
    """
    Backwards-compat shim for old code that expected spec_to_resnet_input(spec) -> (3,H,W).
    Uses to_resnet_batch under the hood. spec_ft is (F,T) in [0,1].
    """
    if not torch.is_tensor(spec_ft):
        spec_ft = torch.as_tensor(spec_ft, dtype=torch.float32)
    if spec_ft.dim() != 2:
        raise ValueError(f"expected (F,T) tensor, got {tuple(spec_ft.shape)}")
    x = to_resnet_batch(spec_ft.unsqueeze(0), size=size)   # (1,3,size,size)
    img = x[0]
    if img.dim() == 4:
        img = img.contiguous(memory_format=torch.channels_last)
    else:
        img = img.contiguous()
    return img  # (3, size, size)
