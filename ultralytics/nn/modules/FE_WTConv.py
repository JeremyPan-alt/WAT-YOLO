# WTConv_with_SharedMLP_Attn.py
# 基于 WTConv2d_Attn 的变体：
# - 子带注意力替换为“**小型共享 MLP**”实现（参数在通道间共享）
# - 提供注释掉的 LL_comp U-Net 版本（需要时手动启用）
#
# 依赖：PyWavelets (`pip install PyWavelets`)
# 用法示例见文件末尾的 smoke test

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

__all__ = ['C3k2_WTConv_Attn_SharedMLP']


def create_wavelet_filter(wt_type, channels, device=None, dtype=torch.float32):
    w = pywt.Wavelet(wt_type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=dtype, device=device)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=dtype, device=device)
    rec_lo = torch.tensor(w.rec_lo, dtype=dtype, device=device)
    rec_hi = torch.tensor(w.rec_hi, dtype=dtype, device=device)

    def outer2d(a, b):
        return torch.ger(a, b)

    LL = outer2d(dec_lo, dec_lo)
    LH = outer2d(dec_lo, dec_hi)
    HL = outer2d(dec_hi, dec_lo)
    HH = outer2d(dec_hi, dec_hi)
    kernels = torch.stack([LL, LH, HL, HH], dim=0)
    dec_filters = kernels.unsqueeze(1).repeat(channels, 1, 1, 1).reshape(channels * 4, 1, LL.shape[0], LL.shape[1])

    RLL = outer2d(rec_lo, rec_lo)
    RLH = outer2d(rec_lo, rec_hi)
    RHL = outer2d(rec_hi, rec_lo)
    RHH = outer2d(rec_hi, rec_hi)
    rkernels = torch.stack([RLL, RLH, RHL, RHH], dim=0)
    rec_filters = rkernels.unsqueeze(1).repeat(channels, 1, 1, 1).reshape(channels * 4, 1, RLL.shape[0], RLL.shape[1])

    return dec_filters, rec_filters


class SubbandAttentionSharedMLP(nn.Module):
    """
    子带注意力模块（小型共享 MLP 版本）
    - 对每个通道计算统计量 [mean, std, max] -> shape (b, c, 3)
    - 共享 MLP：用 Conv1d(in_channels=3, out_channels=hidden, kernel_size=1) 将每个通道的 3-d 向量
      映射到隐层，再映射回 1，Sigmoid 得到通道权重（参数在通道上共享）
    - per_channel=True: 返回 (b, c, 1, 1) 的通道级权重
      per_channel=False: 将通道权重取平均得到每个样本的单一标量 (b,1,1,1)
    - 这样实现既能保留每通道权重，又显著减少参数（相比完全独立的 fc）
    """
    def __init__(self, channels, hidden=None, per_channel=True):
        super().__init__()
        self.c = channels
        self.per_channel = per_channel
        if hidden is None:
            hidden = max(4, channels // 8)  # 小型 hidden，参数共享所以不用太大

        # Shared MLP implemented as Conv1d on (b,3,c)
        # conv1: 3 -> hidden (shared across channels), conv2: hidden -> 1
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=hidden, out_channels=1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (b, c, h, w)
        returns: (b, c, 1, 1) if per_channel else (b, 1, 1, 1)
        """
        b, c, h, w = x.shape
        # stats: mean, std, max  -> each (b, c)
        mean = x.mean(dim=(2, 3))
        # std: fallback when h*w==1 might be nan, but typical inputs >1
        std = x.flatten(2).std(dim=2, unbiased=False)
        maxv = x.amax(dim=(2, 3))
        # stack -> (b, c, 3)
        stats = torch.stack([mean, std, maxv], dim=2)  # (b, c, 3)
        # permute to (b, 3, c) for Conv1d
        stats = stats.permute(0, 2, 1)  # (b, 3, c)
        # apply shared MLP (Conv1d)
        y = self.mlp(stats)  # (b,1,c)
        # permute back -> (b, c, 1)
        y = y.permute(0, 2, 1)  # (b, c, 1)
        if self.per_channel:
            return y.view(b, c, 1, 1)
        else:
            s = y.mean(dim=1, keepdim=True)  # (b,1,1)
            return s.view(b, 1, 1, 1)


class SubbandAttention(nn.Module):
    """
    子带注意力模块（类似 SENet 但用更多统计量）：
    - Squeeze: 对每个通道计算 [mean, std, max]（均为全局池化）
    - Excitation: 两层 MLP 将 (c * 3) -> c//reduction -> c，然后 Sigmoid
    - 如果 per_channel=False，则把输出在通道维取均值，得到每个样本的单一子带标量权重
    设计上我们遵循用户要求：不要只用 GAP，而使用更适合红外特征的统计描述。
    """

    def __init__(self, channels, reduction=16, per_channel=True):
        super().__init__()
        self.c = channels
        self.reduction = max(1, reduction)
        self.per_channel = per_channel

        # MLP: (c*3) -> hidden -> c
        hidden = max(4, (channels * 3) // self.reduction)
        self.fc1 = nn.Linear(channels * 3, hidden, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, channels, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """
        x: (b, c, h, w)
        returns:
          if per_channel=True: (b, c, 1, 1)
          else: (b, 1, 1, 1)
        """
        b, c, h, w = x.shape
        # Squeeze: per-channel statistics
        # mean, std, max -> shape (b, c)
        mean = x.mean(dim=(2, 3))  # (b, c)
        std = x.flatten(2).std(dim=2, unbiased=False)  # (b, c)
        # max: use amax
        maxv = x.amax(dim=(2, 3))  # (b, c)

        # concat -> (b, c*3)
        cat = torch.cat([mean, std, maxv], dim=1)  # (b, 3c)
        # MLP
        y = self.fc1(cat)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sig(y)  # (b, c)

        if self.per_channel:
            return y.view(b, c, 1, 1)
        else:
            # reduce over channels to single scalar per sample
            s = y.mean(dim=1, keepdim=True)  # (b,1)
            return s.view(b, 1, 1, 1)


# 注释掉的 U-Net LL_comp 版本（需要时取消注释并替换 ll_comp_layers 的构造）
"""
class LLCompUNet(nn.Module):
    # 这是一个极简的 U-Net 风格微网络，用于低频补偿（注：参数更多，默认注释）
    def __init__(self, channels, hidden_channels=None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = max(8, channels // 2)
        self.enc1 = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        # x: (b, c, h, w)
        e1 = self.enc1(x)
        p = self.pool(e1)
        e2 = self.enc2(p)
        u = self.up(e2)
        # 按通道拼接 skip
        cat = torch.cat([u, e1], dim=1)
        d = self.dec1(cat)
        return self.out(d)
# 要启用 U-Net 版本：将 WTConv2d_Attn 中 ll_comp_layers 的构造改为:
# ll_comp = LLCompUNet(in_channels)
"""

class WTConv2d_Attn_SharedMLP(nn.Module):
    """
    核心模块：使用 SubbandAttentionSharedMLP 作为注意力
    其余结构与之前一致（每子带 depthwise conv，LL 用轻量补偿）
    """
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                 groups=None, bias=False, wt_levels=1, wt_type='sym8',
                 reduction=16, per_channel=True, wavelet_scale_init=0.1):
        super().__init__()
        groups = in_channels if groups is None else groups
        assert groups == in_channels, "当前实现假设 groups == in_channels（depthwise）。"

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.wt_levels = wt_levels
        self.wt_type = wt_type
        self.per_channel = per_channel

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=1, padding=padding, dilation=dilation,
                                   groups=in_channels, bias=bias)
        self.base_scale = nn.Parameter(torch.ones(1), requires_grad=True)

        self.wavelet_conv_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.ll_comp_layers = nn.ModuleList()

        for lvl in range(wt_levels):
            convs = nn.ModuleList()
            for _ in range(4):
                conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                 stride=1, padding=padding, dilation=dilation,
                                 groups=in_channels, bias=bias)
                convs.append(conv)
            self.wavelet_conv_layers.append(convs)

            # 使用共享 MLP 注意力
            attn_per_level = nn.ModuleList([SubbandAttentionSharedMLP(in_channels, per_channel=per_channel)
                                            for _ in range(4)])
            # 使用专用 MLP 注意力
            '''attn_per_level = nn.ModuleList([SubbandAttention(in_channels, reduction=reduction, per_channel=per_channel)
             for _ in range(4)])'''
            self.attention_layers.append(attn_per_level)

            # LL compensation: 轻量 1x1 conv -> BN -> ReLU（默认启用）
            ll_comp = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
            )
            # 如果你想启用 U-Net 版本：替换上面 ll_comp 为 LLCompUNet(in_channels)
            self.ll_comp_layers.append(ll_comp)

        self.register_buffer('_dec_filters', None)
        self.register_buffer('_rec_filters', None)
        self.wavelet_scale = nn.Parameter(torch.ones(1) * wavelet_scale_init, requires_grad=True)

    def _ensure_wavelet_filters(self, x):
        if self._dec_filters is None or self._rec_filters is None:
            device = x.device
            dtype = x.dtype
            dec, rec = create_wavelet_filter(self.wt_type, self.in_channels, device=device, dtype=dtype)
            self._dec_filters = dec
            self._rec_filters = rec

    def wavelet_transform(self, x):
        """b, c, h, w = x.shape
        self._ensure_wavelet_filters(x)
        dec = self._dec_filters
        out = F.conv2d(x, weight=dec, bias=None, stride=2, padding=dec.shape[-1] // 2, groups=c)
        h2, w2 = out.shape[-2], out.shape[-1]
        out = out.view(b, c, 4, h2, w2)
        return out"""

        """
        对 x 做一层小波分解（per-channel grouped conv）
        输入 x: (b, c, h, w)
        输出: (b, c, 4, h2, w2) -> 四个子带按索引顺序 [LL, LH, HL, HH]
        兼容性增强：若当前 h 或 w 小于 wavelet kernel 大小 k，会先 pad 到最小尺寸 k。
        """
        b, c, h, w = x.shape
        self._ensure_wavelet_filters(x)
        dec = self._dec_filters  # (c*4, 1, k, k)
        k = dec.shape[-1]

        # 如果当前空间尺寸小于 kernel，做对称填充到至少 k
        pad_h_needed = max(0, k - h)
        pad_w_needed = max(0, k - w)
        if pad_h_needed > 0 or pad_w_needed > 0:
            pad_top = pad_h_needed // 2
            pad_bottom = pad_h_needed - pad_top
            pad_left = pad_w_needed // 2
            pad_right = pad_w_needed - pad_left
            # F.pad 的参数为 (left, right, top, bottom)
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')

        # grouped conv: groups = c
        out = F.conv2d(x, weight=dec, bias=None, stride=2, padding=dec.shape[-1] // 2, groups=c)
        h2, w2 = out.shape[-2], out.shape[-1]
        out = out.view(b, c, 4, h2, w2)
        return out

    def inverse_wavelet_transform(self, bands):
        b, c, _, h, w = bands.shape
        self._ensure_wavelet_filters(bands)
        inp = bands.view(b, c * 4, h, w)
        rec = self._rec_filters
        out = F.conv_transpose2d(inp, weight=rec, bias=None, stride=2, padding=rec.shape[-1] // 2, groups=c)
        return out

    def forward(self, x):
        b, c, h, w = x.shape
        assert c == self.in_channels, "in_channels mismatch"

        base = self.base_conv(x) * self.base_scale

        bands_list = []
        x_curr = x
        for lvl in range(self.wt_levels):
            pad_h = x_curr.shape[2] % 2
            pad_w = x_curr.shape[3] % 2
            if pad_h or pad_w:
                x_curr = F.pad(x_curr, (0, pad_w, 0, pad_h), mode='reflect')

            bands = self.wavelet_transform(x_curr)  # (b,c,4,h2,w2)
            processed_bands = []
            convs = self.wavelet_conv_layers[lvl]
            atts = self.attention_layers[lvl]
            ll_comp = self.ll_comp_layers[lvl]

            for idx in range(4):
                band = bands[:, :, idx, :, :]
                band = convs[idx](band)
                att = atts[idx](band)
                band = band * att
                if idx == 0:
                    band = band + ll_comp(band)
                processed_bands.append(band)

            proc = torch.stack(processed_bands, dim=2)
            bands_list.append(proc)
            x_curr = proc[:, :, 0, :, :]

        next_ll = None
        for lvl in reversed(range(self.wt_levels)):
            curr = bands_list[lvl]
            if next_ll is not None:
                target_h, target_w = curr.shape[-2], curr.shape[-1]
                if next_ll.shape[-2] != target_h or next_ll.shape[-1] != target_w:
                    next_ll = F.interpolate(next_ll, size=(target_h, target_w), mode='bilinear', align_corners=False)
                curr[:, :, 0, :, :] = curr[:, :, 0, :, :] + next_ll
            recon = self.inverse_wavelet_transform(curr)
            next_ll = recon

        wave_out = next_ll
        if wave_out.shape[2] != h or wave_out.shape[3] != w:
            wave_out = wave_out[:, :, :h, :w]

        out = base + wave_out * self.wavelet_scale
        return out


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Bottleneck_WTConv_Attn_SharedMLP(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        if c_ == c2:
            self.cv2 = WTConv2d_Attn_SharedMLP(c_, c2, 5, 1)
        else:
            self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_WTConv_Attn_SharedMLP(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2_WTConv_Attn_SharedMLP(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_WTConv_Attn_SharedMLP(self.c, self.c, shortcut, g) for _ in
            range(n)
        )


# quick smoke test
if __name__ == "__main__":
    x = torch.randn(2, 16, 128, 128)
    m = WTConv2d_Attn_SharedMLP(16, kernel_size=3, padding=1, wt_levels=2, wt_type='sym8', wavelet_scale_init=0.1)
    with torch.no_grad():
        y = m(x)
    print("input:", x.shape, "output:", y.shape)

    image_size = (1, 64, 240, 240)
    image = torch.rand(*image_size)

    # Model
    mobilenet_v1 = C3k2_WTConv_Attn_SharedMLP(64, 64)

    out = mobilenet_v1(image)
    print(out.size())
