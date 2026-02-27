# WTConv_with_SubbandAttention.py
# 基于你原始 WTConv 的改进版本：
# - 默认使用 sym8 小波
# - 对每个子带分别做 depthwise conv
# - 在逆变换前为每个子带加入 SubbandAttention（支持通道级或子带级权重）
# - 为低频子带 LL 加入可学习的补偿网络
# - 支持多层 wt_levels（每层有自己的子带 conv/attn/LL-comp）
#
# 说明：保持 in_channels == out_channels 的简化要求（如果需要支持通道变化可以在外部加 1x1 映射）
# 依赖：PyWavelets (`pip install PyWavelets`)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pywt


def create_wavelet_filter(wt_type, channels, device=None, dtype=torch.float32):
    """
    根据 wt_type（例如 'sym8'）创建 2D 小波分解/重构滤波器。
    返回 dec_filters, rec_filters，形状均为 (channels*4, 1, k, k) 以便于做 grouped conv。
    使用 register_buffer 而不是可训练参数（保持小波基为固定先验）。
    """
    w = pywt.Wavelet(wt_type)
    # 分解低/高通滤波器（1D）
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=dtype, device=device)  # 注意pywt顺序，反转用于conv实现
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=dtype, device=device)
    rec_lo = torch.tensor(w.rec_lo, dtype=dtype, device=device)
    rec_hi = torch.tensor(w.rec_hi, dtype=dtype, device=device)

    # 生成 2D separable filters: outer products
    # LL: low*low, LH: low*high, HL: high*low, HH: high*high
    def outer2d(a, b):
        return torch.ger(a, b)  # outer product -> 2D kernel

    LL = outer2d(dec_lo, dec_lo)
    LH = outer2d(dec_lo, dec_hi)
    HL = outer2d(dec_hi, dec_lo)
    HH = outer2d(dec_hi, dec_hi)

    k = LL.shape[0]
    # 将四个 2D kernel 放到 dec_filters，顺序固定为 [LL, LH, HL, HH]
    kernels = torch.stack([LL, LH, HL, HH], dim=0)  # (4, k, k)
    # For grouped conv handle: we want shape (channels*4, 1, k, k)
    dec_filters = kernels.unsqueeze(1).repeat(channels, 1, 1, 1).reshape(channels * 4, 1, k, k)
    # 同理构造重构核（注意重构滤波器顺序需与分解一致）
    RLL = outer2d(rec_lo, rec_lo)
    RLH = outer2d(rec_lo, rec_hi)
    RHL = outer2d(rec_hi, rec_lo)
    RHH = outer2d(rec_hi, rec_hi)
    rkernels = torch.stack([RLL, RLH, RHL, RHH], dim=0)
    rec_filters = rkernels.unsqueeze(1).repeat(channels, 1, 1, 1).reshape(channels * 4, 1, k, k)

    return dec_filters, rec_filters


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


class WTConv2d_Attn(nn.Module):
    """
    WTConv2d 的增强版本，带子带注意力与 LL 补偿。
    参数与原版兼容（尽量），新增参数:
      - wt_type: 小波类型（默认 'sym8'）
      - reduction: attention MLP 缩减率
      - per_channel: attention 是否为通道级（True）还是子带级标量（False）
    """

    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                 groups=None, bias=False, wt_levels=1, wt_type='sym8',
                 reduction=16, per_channel=True, wavelet_scale_init=0.1):
        super().__init__()
        assert in_channels > 0
        # groups 用于 base depthwise conv
        groups = in_channels if groups is None else groups
        assert groups == in_channels, "当前实现假设 groups == in_channels（depthwise），可按需扩展。"

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.wt_levels = wt_levels
        self.wt_type = wt_type
        self.reduction = reduction
        self.per_channel = per_channel

        # 基础空间分支：depthwise conv（类似原始实现）
        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=1, padding=padding, dilation=dilation,
                                   groups=in_channels, bias=bias)
        self.base_scale = nn.Parameter(torch.ones(1), requires_grad=True)

        # 对每个 wt_level，我们分别建立：
        # - 4 个 depthwise conv（分别处理 LL, LH, HL, HH）
        # - 4 个 SubbandAttention
        # - 1 个 LL compensation (1x1 conv -> BN -> ReLU)
        self.wavelet_conv_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.ll_comp_layers = nn.ModuleList()

        for lvl in range(wt_levels):
            # 子带 conv：每个子带都是 depthwise conv（groups=in_channels）
            convs = nn.ModuleList()
            for _ in range(4):
                conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                 stride=1, padding=padding, dilation=dilation,
                                 groups=in_channels, bias=bias)
                convs.append(conv)
            self.wavelet_conv_layers.append(convs)

            # attention 模块（每个子带一个）
            attn_per_level = nn.ModuleList([SubbandAttention(in_channels, reduction=reduction, per_channel=per_channel)
                                            for _ in range(4)])
            self.attention_layers.append(attn_per_level)

            # LL compensation: 小 1x1 conv 网络，用于低频补偿（防止高频噪声支配）
            ll_comp = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
            )
            self.ll_comp_layers.append(ll_comp)

        # wavelet filters buffers（分解/重构），为保持可移植性随设备创建
        # 这里我们先将 buffers 设为 None，然后在 forward 第一次调用时创建并注册到 module（以便知道device/dtype）
        self.register_buffer('_dec_filters', None)
        self.register_buffer('_rec_filters', None)

        # 小波处理输出缩放（和原版类似），以稳定训练
        self.wavelet_scale = nn.Parameter(torch.ones(1) * wavelet_scale_init, requires_grad=True)

    def _ensure_wavelet_filters(self, x):
        # 创建 wavelet filters 并注册 buffer（只在第一次 forward 时做）
        if self._dec_filters is None or self._rec_filters is None:
            device = x.device
            dtype = x.dtype
            dec, rec = create_wavelet_filter(self.wt_type, self.in_channels, device=device, dtype=dtype)
            # dec/rec 是 shape (channels*4, 1, k, k)
            # register as buffers
            self._dec_filters = dec
            self._rec_filters = rec

    def wavelet_transform(self, x):
        """
        对 x 做一层小波分解（per-channel grouped conv）
        输入 x: (b, c, h, w)
        输出: (b, c, 4, h//2, w//2) -> 四个子带按索引顺序 [LL, LH, HL, HH]
        """
        b, c, h, w = x.shape
        self._ensure_wavelet_filters(x)
        dec = self._dec_filters  # (c*4, 1, k, k)
        # grouped conv: groups = c
        # conv 输出 shape: (b, c*4, h2, w2)
        out = F.conv2d(x, weight=dec, bias=None, stride=2, padding=dec.shape[-1] // 2, groups=c)
        h2, w2 = out.shape[-2], out.shape[-1]
        # reshape到 (b, c, 4, h2, w2)
        out = out.view(b, c, 4, h2, w2)
        return out

    def inverse_wavelet_transform(self, bands):
        """
        按逆变换重建： bands: (b, c, 4, h, w) -> 返回 (b, c, h*2, w*2)
        使用 grouped conv_transpose（groups=c）
        """
        b, c, _, h, w = bands.shape
        self._ensure_wavelet_filters(bands)
        # reshape为 (b, c*4, h, w)
        inp = bands.view(b, c * 4, h, w)
        rec = self._rec_filters  # (c*4,1,k,k)
        # conv_transpose: groups=c
        out = F.conv_transpose2d(inp, weight=rec, bias=None, stride=2, padding=rec.shape[-1] // 2, groups=c)
        return out

    def forward(self, x):
        """
        前向流程：
         - base space conv 分支
         - 递归进行 wt_levels 层的小波分解：对每层子带分别 conv -> attention -> 对 LL 做补偿
         - 逐层重构（自底向上），每层在 LL 上与下一层的 LL 做残差融合（curr_ll + next_ll）
         - 最终 reconstruction * wavelet_scale + base_conv * base_scale
        """
        b, c, h, w = x.shape
        assert c == self.in_channels, "in_channels mismatch"

        base = self.base_conv(x) * self.base_scale  # 基础空间分支

        # multi-level 小波分解与处理
        # 我们维护每层的 bands_list（深到浅），以便在回溯时重构
        bands_list = []
        x_curr = x
        for lvl in range(self.wt_levels):
            # 为当前层 pad 为偶数大小（小波分解 stride=2 需偶数）
            pad_h = x_curr.shape[2] % 2
            pad_w = x_curr.shape[3] % 2
            if pad_h or pad_w:
                x_curr = F.pad(x_curr, (0, pad_w, 0, pad_h), mode='reflect')

            bands = self.wavelet_transform(x_curr)  # (b, c, 4, h2, w2)
            # 对每个子带分别做 depthwise conv -> attention -> LL补偿（对 idx=0）
            processed_bands = []
            convs = self.wavelet_conv_layers[lvl]
            atts = self.attention_layers[lvl]
            ll_comp = self.ll_comp_layers[lvl]

            for idx in range(4):
                band = bands[:, :, idx, :, :]  # (b, c, h2, w2)
                # depthwise conv
                band = convs[idx](band)
                # attention
                att = atts[idx](band)  # (b,c,1,1) or (b,1,1,1)
                band = band * att
                # LL 的可学习补偿（idx==0）
                if idx == 0:
                    band = band + ll_comp(band)  # residual style
                processed_bands.append(band)

            # stack 回去 (b,c,4,h2,w2)
            proc = torch.stack(processed_bands, dim=2)
            bands_list.append(proc)

            # 下一层以当前 LL 作为输入（深度分解）
            x_curr = proc[:, :, 0, :, :]  # LL 子带作为下一层输入

        # 自底向上重构
        next_ll = None
        for lvl in reversed(range(self.wt_levels)):
            curr = bands_list[lvl]  # (b,c,4,h,w)
            # 如果存在更深层的重建结果 next_ll（已被上层重建），将其加到当前层的 LL（残差融合）
            if next_ll is not None:
                # next_ll 是上一次 inverse_wavelet_transform 的输出（尺寸应为当前 LL 的尺寸）
                # 确保尺寸匹配（可能因 pad 导致）
                # 我们先将 next_ll 下采样到 LL 的尺寸（或裁剪/插值），但按设计 next_ll 应与 current LL 尺寸一致
                # 为保险起见，使用中心裁剪/插值策略：
                target_h, target_w = curr.shape[-2], curr.shape[-1]
                if next_ll.shape[-2] != target_h or next_ll.shape[-1] != target_w:
                    next_ll = F.interpolate(next_ll, size=(target_h, target_w), mode='bilinear', align_corners=False)
                # 将其分解回为子带结构再把 LL 加上（这里 next_ll 已是空间级张量）
                # 直接把 next_ll 当作补偿加入 curr LL（channel-wise）
                curr[:, :, 0, :, :] = curr[:, :, 0, :, :] + next_ll

            # 逆变换得到上采样结果
            recon = self.inverse_wavelet_transform(curr)  # (b,c,H*2,W*2)
            # 这次的 recon 成为下一层层级的 next_ll（移动到上一层循环）
            next_ll = recon

        # 最终的 wavelet 路径输出（可能多出 pad 区，需要裁剪到原始尺寸）
        wave_out = next_ll
        if wave_out.shape[2] != h or wave_out.shape[3] != w:
            wave_out = wave_out[:, :, :h, :w]

        # 与 base 分支融合并返回
        out = base + wave_out * self.wavelet_scale
        return out


# 小示例：如何实例化
if __name__ == "__main__":
    # quick smoke test
    x = torch.randn(2, 16, 128, 128)
    m = WTConv2d_Attn(16, kernel_size=3, padding=1, wt_levels=2, wt_type='sym8', reduction=8, per_channel=True)
    with torch.no_grad():
        y = m(x)
    print("input:", x.shape, "output:", y.shape)
