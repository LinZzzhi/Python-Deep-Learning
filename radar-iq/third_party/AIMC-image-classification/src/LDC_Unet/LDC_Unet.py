import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        x2 = self.conv3(x2)
        return x1 + x2


class decoder(nn.Module):
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(
            in_channels=c1, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=c2, out_channels=64, kernel_size=3, stride=1, padding=1
        )

        self.upsample1 = nn.Upsample(
            scale_factor=(2, 2), mode="bilinear", align_corners=True
        )
        self.conv3 = nn.Conv2d(
            in_channels=c3, out_channels=64, kernel_size=3, stride=1, padding=1
        )

        self.upsample2 = nn.Upsample(
            scale_factor=(2, 2), mode="bilinear", align_corners=True
        )
        self.conv4 = nn.Conv2d(
            in_channels=c4, out_channels=64, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x1, x2, x3, x4):
        x1 = self.conv1(self.pool1(x1))
        x2 = self.conv2(x2)
        x3 = self.conv3(self.upsample1(x3))
        x4 = self.conv4(self.upsample2(x4))
        return torch.cat((x1, x2, x3, x4), dim=1)


class decoder_4(nn.Module):
    def __init__(self, c1, c2, c3):
        super().__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(
            in_channels=c1, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=c2, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.upsample1 = nn.Upsample(
            scale_factor=(2, 2), mode="bilinear", align_corners=True
        )
        self.conv3 = nn.Conv2d(
            in_channels=c3, out_channels=64, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x1, x2, x3):
        x1 = self.conv1(self.pool1(x1))
        x2 = self.conv2(x2)
        x3 = self.conv3(self.upsample1(x3))
        return torch.cat((x1, x2, x3), dim=1)


class LDC_Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = encoder(in_channels=3, out_channels=64)
        self.e2 = encoder(in_channels=64, out_channels=128)
        self.e3 = encoder(in_channels=128, out_channels=256)
        self.e4 = encoder(in_channels=256, out_channels=256)
        self.e5 = encoder(in_channels=256, out_channels=512)

        self.d4 = decoder_4(c1=256, c2=256, c3=512)
        self.d3 = decoder(c1=128, c2=256, c3=256, c4=192)
        self.d2 = decoder(c1=64, c2=128, c3=256, c4=256)
        self.d1 = decoder(c1=3, c2=64, c3=128, c4=256)

        self.upsample = nn.Upsample(
            scale_factor=(2, 2), mode="bilinear", align_corners=True
        )
        self.conv1 = nn.Conv2d(
            in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x0 = x
        x_e1 = self.e1(x0)
        x_e2 = self.e2(x_e1)
        x_e3 = self.e3(x_e2)
        x_e4 = self.e4(x_e3)
        x_e5 = self.e5(x_e4)

        x_d4 = self.d4(x_e3, x_e4, x_e5)
        x_d3 = self.d3(x_e2, x_e3, x_e4, x_d4)
        x_d2 = self.d2(x_e1, x_e2, x_e3, x_d3)
        x_d1 = self.d1(x0, x_e1, x_e2, x_d2)

        x = self.upsample(x_d1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vgg19 = models.vgg19(pretrained=True)
        self.features = self.vgg19.features

        self.classification_block = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 256),
        )

        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dense_last = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        features = self.classification_block(x)
        features = self.relu(features)
        features = self.dropout1(features)
        logits = self.dense_last(features)
        return features, logits


class MainModel(nn.Module):
    """
    Implements the paper-side "preprocessing" INSIDE the network:
      - resize TFIs to 128x128 using bilinear interpolation (Jiang et al., 2022)
      - optional per-image amplitude normalisation (max scaling), common in prior TFI pipelines

    Expects input already as a TFI image tensor:
      x: (B, C, H, W) where C is 1 or 3.
    """

    def __init__(
        self, num_classes: int, target_hw=(128, 128), do_amp_norm: bool = True
    ):
        super().__init__()
        self.target_hw = target_hw
        self.do_amp_norm = do_amp_norm

        self.ldc_unet = LDC_Unet()
        self.dcnn = DCNN(num_classes)

    @staticmethod
    def _ensure_3ch(x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected (B,C,H,W) but got shape {tuple(x.shape)}")
        if x.size(1) == 3:
            return x
        if x.size(1) == 1:
            return x.repeat(1, 3, 1, 1)
        raise ValueError(f"Expected 1 or 3 channels, got C={x.size(1)}")

    def _paper_preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # (A) ensure channels
        x = self._ensure_3ch(x)

        # (B) bilinear resize to 128x128 (paper)
        x = F.interpolate(x, size=self.target_hw, mode="bilinear", align_corners=False)

        # (C) optional amplitude normalisation (per-sample, per-image max)
        # If your TFIs are already normalised upstream, set do_amp_norm=False.
        if self.do_amp_norm:
            # normalise using max absolute value per sample (keep dims for broadcasting)
            denom = x.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
            x = x / denom

        return x

    def forward(self, x: torch.Tensor):
        # --- preprocessing inside model ---
        x = self._paper_preprocess(x)

        # --- LDC-Unet then VGG19-based classifier ---
        x = self.ldc_unet(x)
        f, l = self.dcnn(x)
        return f, l
