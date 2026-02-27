import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.conv import Conv

__all__ = ['SPDConv', 'Multibranch']


class SPDConv(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x = self.conv(x)
        return x


class FGM(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.conv = nn.Conv2d(dim, dim * 2, 3, 1, 1, groups=dim)

        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        # res = x.clone()
        fft_size = x.size()[2:]
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)

        x2_fft = torch.fft.fft2(x2, norm='backward')

        out = x1 * x2_fft

        out = torch.fft.ifft2(out, dim=(-2, -1), norm='backward')
        out = torch.abs(out)

        return out * self.alpha + x * self.beta


class OmniKernel(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        ker = 31
        pad = ker // 2
        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
            nn.GELU()
        )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
        self.dw_13 = nn.Conv2d(dim, dim, kernel_size=(1, ker), padding=(0, pad), stride=1, groups=dim)
        self.dw_31 = nn.Conv2d(dim, dim, kernel_size=(ker, 1), padding=(pad, 0), stride=1, groups=dim)
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)

        self.act = nn.ReLU()

        ### sca ###
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        ### fca ###
        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fgm = FGM(dim)

    def forward(self, x):
        out = self.in_conv(x)

        ### fca ###
        x_att = self.fac_conv(self.fac_pool(out))
        x_fft = torch.fft.fft2(out, norm='backward')
        x_fft = x_att * x_fft
        x_fca = torch.fft.ifft2(x_fft, dim=(-2, -1), norm='backward')
        x_fca = torch.abs(x_fca)
        ### fca ###

        ### sca ###
        x_att = self.conv(self.pool(x_fca))
        x_sca = x_att * x_fca
        ### sca ###
        x_sca = self.fgm(x_sca)

        out = x + self.dw_13(out) + self.dw_31(out) + self.dw_33(out) + self.dw_11(out) + x_sca
        out = self.act(out)
        return self.out_conv(out)


class LargeKernelDecomposed(nn.Module):
    """Decomposed large kernel branch (LSKA/LKA style) but kept lightweight.
    Uses cascaded depthwise 1xK and Kx1 with a small pointwise head.
    """

    def __init__(self, dim, kernel=31):
        super().__init__()
        pad = kernel // 2
        self.dw1 = nn.Conv2d(dim, dim, kernel_size=(1, kernel), padding=(0, pad), groups=dim)
        self.dw2 = nn.Conv2d(dim, dim, kernel_size=(kernel, 1), padding=(pad, 0), groups=dim)
        self.dw3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pw = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        out = x + self.dw1(x)
        out = out + self.dw2(out)
        out = out + self.dw3(out)
        out = self.pw(self.act(out))
        return out


class FourierGuidedAttentionV2(nn.Module):
    """Improved Fourier-guided attention. Learns a real-valued gating map in frequency domain
    and applies it to the complex spectrum (magnitude scaling). This is more stable than
    multiplying by a full complex tensor and is lightweight.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # per-channel learnable frequency gate (real positive after softplus)
        self.freq_gain = nn.Parameter(torch.zeros(dim, 1, 1))
        self.spatial_pw = nn.Conv2d(dim, dim, 1)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        # x: [B,C,H,W]
        fft = torch.fft.fft2(x, norm='ortho')
        # scale magnitude by learned gain (broadcast over spatial frequencies)
        gain = F.softplus(self.freq_gain).view(1, -1, 1, 1)  # [C,1,1] -> broadcast, finally [1, C, 1, 1]
        fft = fft * gain  # still [B, C, H, W]
        out = torch.fft.ifft2(fft, norm='ortho').real  # [B, C, H, W]
        out = self.spatial_pw(out)
        out = self.bn(out)
        return out + x


class ChannelSpatialSE(nn.Module):
    def __init__(self, dim, r=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // r, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // r, dim, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Multibranch(nn.Module):
    """Drop-in replacement for the original Multibranch:
    - a decomposed large-kernel branch (LSKA/LKA style)
    - an improved Fourier-guided attention branch
    - light channel-spatial SE fusion and 1x1 projection to original dim

    This design is motivated by recent large-kernel and frequency-attention
    works while keeping computational cost reasonable for YOLO-style backbones.
    """

    def __init__(self, dim, e=0.25, kernel=31):
        super().__init__()
        self.e = e
        self.dim = dim
        hidden = int(dim * e)

        self.largek = nn.Sequential(
            LargeKernelDecomposed(dim, kernel=kernel),
            nn.Conv2d(dim, hidden, kernel_size=1),
            nn.GELU()
        )

        self.fourier = nn.Sequential(
            FourierGuidedAttentionV2(dim),
            nn.Conv2d(dim, hidden, kernel_size=1),
            nn.GELU()
        )

        # identity passthrough for remaining channels
        self.proj_in = Conv(dim, dim, 1)
        # now we fuse two hidden branches + identity remainder
        self.fuse_pw = nn.Conv2d(hidden * 2 + (dim - hidden), dim, kernel_size=1)
        self.csse = ChannelSpatialSE(dim, r=4)

    def forward(self, x):
        # project + split to get a stable identity part
        x_proj = self.proj_in(x)
        B, C, H, W = x_proj.shape
        hidden = int(self.dim * self.e)

        l = self.largek(x_proj)  # [B, hidden, H, W]
        f = self.fourier(x_proj)  # [B, hidden, H, W]

        # split identity channels (remaining part)
        identity = x_proj[:, hidden:, :, :]

        fused = torch.cat([l, f, identity], dim=1)
        out = self.fuse_pw(fused)
        out = self.csse(out)
        return out
