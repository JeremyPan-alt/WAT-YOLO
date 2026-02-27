import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['HotspotEnhancer']


def init_sharpen(weight):
    """
    Initialize depthwise 3x3 conv as a sharpen / delta-like kernel
    """
    with torch.no_grad():
        weight.zero_()
        # center
        weight[:, 0, 1, 1] = 1.5
        # neighbors
        weight[:, 0, 0, 1] = -0.25
        weight[:, 0, 2, 1] = -0.25
        weight[:, 0, 1, 0] = -0.25
        weight[:, 0, 1, 2] = -0.25


def init_gaussian(weight):
    """
    Initialize depthwise 3x3 conv as a small Gaussian
    """
    g = torch.tensor([[1, 2, 1],
                      [2, 4, 2],
                      [1, 2, 1]], dtype=torch.float32)
    g = g / g.sum()
    with torch.no_grad():
        for c in range(weight.shape[0]):
            weight[c, 0] = g


class HotspotEnhancer(nn.Module):
    """
    Target-aware enhancement block for photovoltaic hotspot detection
    """
    def __init__(self, channels, pool_kernel=7):
        super().__init__()

        # Point-enhancer: preserves sharp peak
        self.point = nn.Conv2d(
            channels, channels,
            kernel_size=3, padding=1,
            groups=channels, bias=False
        )

        # Gaussian branch: mild shape regularization
        self.gauss = nn.Conv2d(
            channels, channels,
            kernel_size=3, padding=1,
            groups=channels, bias=False
        )

        # Background estimation
        self.bg_pool = nn.AvgPool2d(
            kernel_size=pool_kernel,
            stride=1,
            padding=pool_kernel // 2
        )

        # Residual scaling
        self.alpha = nn.Parameter(torch.tensor(0.1))

        # Initialization
        init_sharpen(self.point.weight)
        init_gaussian(self.gauss.weight)

    def forward(self, x):
        # Peak-preserving enhancement
        pt = self.point(x)

        # Mild Gaussian smoothing
        gs = self.gauss(x)

        # Local background
        bg = self.bg_pool(x)

        # Top-hat-like contrast
        th = pt - bg

        # Fusion
        enhanced = pt + gs + th

        return x + self.alpha * enhanced
