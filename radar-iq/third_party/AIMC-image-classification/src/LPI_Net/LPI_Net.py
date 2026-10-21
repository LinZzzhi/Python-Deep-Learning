import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNELU(nn.Module):
    """Conv2d -> BatchNorm2d -> ELU"""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ELU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ModuleThing(nn.Module):
    """
    LPI-Net processing module:
      maxpool(3x3,s=2) -> parallel (1x3, 3x1) -> concat -> 1x1
                      -> parallel (1x3, 3x1) -> concat(with previous concat) -> 1x1
      residual: add pooled input
    Paper notes: conv layers followed by BN + eLU. :contentReference[oaicite:2]{index=2}
    """
    def __init__(self, channels=64):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv1 = ConvBNELU(channels, channels, kernel_size=(1, 3), padding=(0, 1))
        self.conv2 = ConvBNELU(channels, channels, kernel_size=(3, 1), padding=(1, 0))

        self.conv3 = ConvBNELU(channels * 2, channels, kernel_size=1, padding=0)

        self.conv4 = ConvBNELU(channels, channels, kernel_size=(1, 3), padding=(0, 1))
        self.conv5 = ConvBNELU(channels, channels, kernel_size=(3, 1), padding=(1, 0))

        self.conv6 = ConvBNELU(channels * 4, channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.pool(x)
        x0 = x  # residual identity (pooled)

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x_cat1 = torch.cat((x1, x2), dim=1)  # 2C
        x = self.conv3(x_cat1)               # C

        x1 = self.conv4(x)
        x2 = self.conv5(x)
        x_cat2 = torch.cat((x1, x2, x_cat1), dim=1)  # C + C + 2C = 4C
        x = self.conv6(x_cat2)                        # C

        return x + x0


class LPINet(nn.Module):
    """
    LPI-Net classifier (paper):
      input: 50x50x1 grayscale TFI
      conv(7x7) + BN+ELU -> module x3 -> GAP -> FC -> Dropout -> FC
    Paper notes: conv layers followed by BN + eLU, and auto zero-padding. :contentReference[oaicite:3]{index=3}
    """
    def __init__(self, num_classes: int):
        super().__init__()

        # "auto zero-padding" + 7x7 should preserve spatial size -> padding=3
        self.stem = ConvBNELU(in_ch=1, out_ch=64, kernel_size=7, stride=1, padding=3)

        self.m1 = ModuleThing(channels=64)
        self.m2 = ModuleThing(channels=64)
        self.m3 = ModuleThing(channels=64)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.stem(x)
        x = self.m1(x)
        x = self.m2(x)
        x = self.m3(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits


class MainModel(nn.Module):
    """
    LPI-Net wrapper that matches your training pipeline:
      - accepts x as (N, 3, H, W) or (N, 1, H, W)
      - converts to 1-channel
      - resizes to 50x50 (bicubic, per LPI-Net paper)
      - then runs the LPI-Net classifier

    Does NOT generate TFIs. Only alters the incoming image tensor.
    """

    def __init__(self, num_classes: int, target_hw=(50, 50)):
        super().__init__()
        self.target_hw = target_hw
        self.net = LPINet(num_classes=num_classes)  # your existing LPI-Net class

    @staticmethod
    def _ensure_1ch(x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected (N,C,H,W) but got {tuple(x.shape)}")

        if x.size(1) == 1:
            return x
        if x.size(1) == 3:
            # deterministic grayscale collapse
            return x.mean(dim=1, keepdim=True)
        raise ValueError(f"Expected C=1 or C=3, got C={x.size(1)}")

    def _paper_preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = self._ensure_1ch(x)
        x = F.interpolate(x, size=self.target_hw, mode="bicubic", align_corners=False)
        return x

    def forward(self, x: torch.Tensor):
        x = self._paper_preprocess(x)
        return self.net(x)
