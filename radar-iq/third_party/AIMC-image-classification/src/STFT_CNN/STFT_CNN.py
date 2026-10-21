import torch
import torch.nn as nn


class MainModel(nn.Module):
    """
    STFT-CNN (paper architecture, unchanged)

    Architecture:
      - Pre downsample: AvgPool2d(2,2)
      - Conv1: 3 -> 6, k=5
      - Pool1: AvgPool2d(2,2)
      - Conv2: 6 -> 12, k=5
      - Pool2: AvgPool2d(2,2)
      - AdaptiveAvgPool2d(7,7)
      - FC: 12*7*7 -> num_classes

    Only preprocessing behaviour has been made safe and correct.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        use_paper_preprocess: bool = True,
        preproc_iters: int = 6,
        eps: float = 1e-8,
        denom: str = "var",     # "var" (your original) or "std"
        xi: float = 0.0,        # Xi threshold (0.0 = disabled)
    ):
        super(MainModel, self).__init__()

        self.use_paper_preprocess = use_paper_preprocess
        self.preproc_iters = preproc_iters
        self.eps = eps
        self.denom = denom
        self.xi = xi

        if denom not in {"var", "std"}:
            raise ValueError("denom must be 'var' or 'std'")

        # --- Paper-style preprocessing (down-sampling) ---
        self.pre_downsample = nn.AvgPool2d(kernel_size=2, stride=2)

        # --- CNN backbone (UNCHANGED) ---
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        self.relu = nn.ReLU()

        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, stride=1)

        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Force stable feature size
        self.adapt = nn.AdaptiveAvgPool2d((7, 7))

        self.flat = nn.Flatten()
        self.fc = nn.Linear(12 * 7 * 7, num_classes)

    # ------------------------------------------------------------------

    def _xi_threshold(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Xi(.) thresholding implemented as CAP (safe).

        Xi(a) = min(a, xi)

        - If xi == 0 → disabled (identity)
        - This prevents exploding activations without zeroing everything.
        """
        if self.xi <= 0.0:
            return alpha
        return torch.clamp(alpha, max=self.xi)

    # ------------------------------------------------------------------

    def _paper_feature_preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Paper Feature Vector Preprocessing:

          1) Down-sampling
          2) Vectorise
          3) Zero-mean scaling: abs(x - mean)/{var or std}
          4) Xi(.) thresholding
          5) Repeat for N iterations
          6) Reshape back to image

        Expects x: (B, C, H, W)
        """

        # Step 1: down-sampling
        x = self.pre_downsample(x)

        B, C, H, W = x.shape

        for _ in range(self.preproc_iters):
            alpha = x.view(B, C, -1)

            mean = alpha.mean(dim=-1, keepdim=True)
            var = alpha.var(dim=-1, unbiased=False, keepdim=True)

            if self.denom == "var":
                denom = var + self.eps
            else:  # "std"
                denom = torch.sqrt(var + self.eps)

            alpha = torch.abs(alpha - mean) / denom

            # Proper Xi thresholding
            alpha = self._xi_threshold(alpha)

            # Safety check (prevents silent NaN death)
            if not torch.isfinite(alpha).all():
                raise RuntimeError("Non-finite values detected in preprocessing")

            x = alpha.view(B, C, H, W)

        return x

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W)

        IMPORTANT:
        If your inputs are uint8, convert before model:
            x = x.float() / 255.0
        """

        if self.use_paper_preprocess:
            x = self._paper_feature_preprocess(x)

        # --- CNN ---
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.adapt(x)
        x = self.flat(x)
        x = self.fc(x)

        return x