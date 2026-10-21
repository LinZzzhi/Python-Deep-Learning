import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dims, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dims[0], dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dims, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dims, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# --- FeedForward / Attention / Transformer unchanged ---


class MainModel(nn.Module):
    """
    ViT that expects STFT images from your loader, but applies paper-style preprocessing
    inside the model:
      - ensure correct channel count (1 or 3)
      - resize to fixed image_size using bilinear interpolation
      - optional amplitude normalisation (divide by per-image max)
      - optional log compression (useful if STFT is linear magnitude/power)
    """

    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dims,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        # preprocessing controls
        do_paper_preprocess: bool = True,
        do_amp_norm: bool = True,
        do_log: bool = False,          # set True if your STFT is linear and you want log compression
        log_eps: float = 1e-8,
    ):
        super().__init__()
        self.image_size = pair(image_size)
        self.channels = int(channels)

        self.do_paper_preprocess = do_paper_preprocess
        self.do_amp_norm = do_amp_norm
        self.do_log = do_log
        self.log_eps = float(log_eps)

        image_height, image_width = self.image_size
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = self.channels * patch_height * patch_width
        assert pool in {"cls", "mean"}

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dims, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dims[0]),
            nn.ReLU(),
            nn.Linear(mlp_dims[0], mlp_dims[1]),
            nn.ReLU(),
            nn.Linear(mlp_dims[1], num_classes),
        )

    def _ensure_channels(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accept:
          - (B,H,W) -> (B,1,H,W)
          - (B,1,H,W) or (B,3,H,W) -> adapt to self.channels
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B,1,H,W)
        if x.dim() != 4:
            raise ValueError(f"Expected (B,C,H,W) or (B,H,W), got {tuple(x.shape)}")

        C = x.size(1)
        if C == self.channels:
            return x

        if self.channels == 3 and C == 1:
            return x.repeat(1, 3, 1, 1)

        if self.channels == 1 and C == 3:
            # safest deterministic collapse
            return x.mean(dim=1, keepdim=True)

        # Any other mismatch (e.g., C=2) -> reduce to 1 then expand if needed
        x = x.mean(dim=1, keepdim=True)
        if self.channels == 3:
            x = x.repeat(1, 3, 1, 1)
        return x

    def _paper_preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = self._ensure_channels(x)

        # If your STFT is linear magnitude/power, log compression can help.
        # Leave off if you already store dB-scaled images.
        if self.do_log:
            x = torch.log(x.abs().clamp_min(self.log_eps))

        # Resize to ViT expected size
        x = F.interpolate(x, size=self.image_size, mode="bilinear", align_corners=False)

        # Per-image amplitude normalisation (divide by max)
        if self.do_amp_norm:
            denom = x.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
            x = x / denom

        return x

    def forward(self, img: torch.Tensor):
        if self.do_paper_preprocess:
            img = self._paper_preprocess(img)
        else:
            img = self._ensure_channels(img)

        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)
