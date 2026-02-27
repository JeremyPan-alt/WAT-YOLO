import torch
import torch.nn as nn
import numpy as np
from ..modules.block import *
from ..modules.conv import Conv
import torch.nn.functional as F


__all__ = ['C3k2_HWT_ViT']


# ------------------------------------------ Haar 子带 + 轻量 cross-attention -------------------------------------------
# ---------- helper ----------
def pad_to_even(x):
    """Pad right and bottom by 1 if width/height is odd. Returns padded tensor and pad tuple (left,right,top,bottom)."""
    _, _, h, w = x.shape
    pad_h = 1 if (h % 2) == 1 else 0
    pad_w = 1 if (w % 2) == 1 else 0
    if pad_h == 0 and pad_w == 0:
        return x, None
    # F.pad uses (left, right, top, bottom)
    pad = (0, pad_w, 0, pad_h)
    x_p = F.pad(x, pad, mode='reflect')  # reflect or constant(0) – reflect often safer
    return x_p, pad

def crop_to_original(x, pad):
    """Crop tensor x back by removing right/bottom padding according to pad tuple."""
    if pad is None:
        return x
    left, right, top, bottom = pad  # we used (0, pad_w, 0, pad_h)
    h = x.size(2) - bottom
    w = x.size(3) - right
    return x[..., :h, :w]

# ---------- HaarDWT with input-evening ----------
class HaarDWT(nn.Module):
    def __init__(self):
        super().__init__()
        base = torch.tensor([[[[1.,1.],[1.,1.]]],
                             [[[1.,1.],[-1.,-1.]]],
                             [[[1.,-1.],[1.,-1.]]],
                             [[[1.,-1.],[-1.,1.]]]]) * 0.5
        self.register_buffer('base_k', base)

    def forward(self, x):
        # pad to even H,W to guarantee consistent halves
        x_p, pad = pad_to_even(x)
        N, C, H, W = x_p.shape
        kernels = self.base_k.repeat(C, 1, 1, 1)  # (4C,1,2,2)
        out = F.conv2d(x_p, weight=kernels, bias=None, stride=2, padding=0, groups=C)
        # out: (N, 4C, H/2, W/2)
        out = out.view(N, C, 4, H//2, W//2).permute(0,2,1,3,4).contiguous()
        LL = out[:,0]
        LH = out[:,1]
        HL = out[:,2]
        HH = out[:,3]
        return LL, LH, HL, HH, pad


# ---------- Learnable HaarDWT with input-evening ----------
class LearnableDWT(nn.Module):
    """Per-channel 1-level learnable DWT using grouped conv.

    The transform produces four subbands: LL, LH, HL, HH each with same number of channels
    as the input (per-channel transform). The kernel tensor has shape (4,1,2,2) and is
    repeated per-channel in the grouped convolution.

    If `learnable=False` the kernels are fixed to classic Haar.
    The method `orth_reg()` returns a scalar regularizer (to add to total loss) that
    encourages the 4 basis kernels to remain approximately orthogonal.
    """
    def __init__(self, in_channels, learnable=False, init='haar'):
        super().__init__()
        self.in_channels = in_channels
        # kernel shape (4,1,2,2)
        # initialize based on Haar
        device = None
        base = torch.tensor([[[[1.,1.],[1.,1.]]],   # LL
                              [[[1.,1.],[-1.,-1.]]], # LH
                              [[[1.,-1.],[1.,-1.]]], # HL
                              [[[1.,-1.],[-1.,1.]]]]) * 0.5
        if init == 'haar':
            kern = base
        else:
            # small random near Haar
            kern = base + 0.01 * torch.randn_like(base)
        if learnable:
            self.kernel = nn.Parameter(kern)
        else:
            self.register_buffer('kernel', kern)
        self.learnable = learnable

    def forward(self, x):
        x_p, pad = pad_to_even(x)
        N, C, H, W = x_p.shape
        kernels = self.kernel.repeat(C, 1, 1, 1)  # (4C,1,2,2)
        out = F.conv2d(x_p, weight=kernels, bias=None, stride=2, padding=0, groups=C)
        # out: (N, 4C, H/2, W/2)
        out = out.view(N, C, 4, H // 2, W // 2).permute(0, 2, 1, 3, 4).contiguous()
        LL = out[:, 0]
        LH = out[:, 1]
        HL = out[:, 2]
        HH = out[:, 3]
        return LL, LH, HL, HH, pad

    def orth_reg(self):
        """Return a small scalar encouraging orthogonality among 4 basis kernels.
        Compute pairwise inner products and penalize off-diagonal magnitude.
        """
        k = self.kernel.view(4, -1)  # (4, 4)
        G = torch.matmul(k, k.t())  # Gram matrix (4,4)
        # zero diag
        diag = torch.diag(G)
        G_off = G - torch.diag(diag)
        reg = G_off.abs().sum()
        return reg


# ---------- SPDConv with padding safety ----------
class SPDConv(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x):
        # pad to even dims first
        x_p, pad = pad_to_even(x)
        # now safe to slice
        x_cat = torch.cat([x_p[..., ::2, ::2], x_p[..., 1::2, ::2], x_p[..., ::2, 1::2], x_p[..., 1::2, 1::2]], 1)
        x_out = self.conv(x_cat)
        # if padded, crop back
        x_out = crop_to_original(x_out, pad)
        return x_out


class SimpleCrossAttn(nn.Module):
    """Q from LL (dim mid), KV from HF_mapped (dim mid).
       token_dim: embedding dim for attention; output remapped to mid."""
    def __init__(self, mid_dim, token_dim=64, pool_k=2):
        super().__init__()
        self.pool_k = pool_k
        self.q_proj = nn.Conv2d(mid_dim, token_dim, 1)
        self.k_proj = nn.Conv2d(mid_dim, token_dim, 1)
        self.v_proj = nn.Conv2d(mid_dim, token_dim, 1)
        self.out = nn.Conv2d(token_dim, mid_dim, 1)
        self.scale = token_dim ** -0.5

    def forward(self, f_ll, f_hf_mapped):
        # f_ll: (N, mid, h, w)
        # f_hf_mapped: (N, mid, h, w)
        N, mid, h, w = f_ll.shape

        q = self.q_proj(f_ll)                 # (N, D, h, w)
        k = self.k_proj(f_hf_mapped)
        v = self.v_proj(f_hf_mapped)

        if self.pool_k > 1:
            k = F.avg_pool2d(k, kernel_size=self.pool_k, stride=self.pool_k)
            v = F.avg_pool2d(v, kernel_size=self.pool_k, stride=self.pool_k)

        S_q = h * w
        S_k = k.shape[2] * k.shape[3]

        q_flat = q.flatten(2).permute(0,2,1)  # (N, S_q, D)
        k_flat = k.flatten(2).permute(0,2,1)  # (N, S_k, D)
        v_flat = v.flatten(2).permute(0,2,1)  # (N, S_k, D)

        attn = torch.matmul(q_flat, k_flat.transpose(-2,-1)) * self.scale  # (N,S_q,S_k)
        attn = torch.softmax(attn, dim=-1)
        out_flat = torch.matmul(attn, v_flat)  # (N, S_q, D)

        out = out_flat.permute(0,2,1).reshape(N, -1, h, w)  # (N, D, h, w)
        out = self.out(out)  # (N, mid, h, w)
        return out


# ---------- HWT_ViT (forward): ensure padding and explicit upsample size ----------
class HWT_ViT(nn.Module):
    def __init__(self, inc, ouc, mid_ratio=0.5, token_dim=64, pool_k=2):
        super().__init__()
        #self.dwt = HaarDWT()
        self.dwt = LearnableDWT(inc, learnable=True, init='haar')
        mid_ch = max(4, int(inc * mid_ratio))

        self.ll_conv = Conv(inc, mid_ch, k=1)
        self.hf_conv1 = Conv(inc, mid_ch, k=1)
        self.hf_conv2 = Conv(inc, mid_ch, k=1)
        self.hf_conv3 = Conv(inc, mid_ch, k=1)

        self.hf_to_mid = Conv(3 * mid_ch, mid_ch, k=1)
        self.attn = SimpleCrossAttn(mid_ch, token_dim=token_dim, pool_k=pool_k)
        self.mid_to_inc = Conv(mid_ch, inc, k=1)

        self.conv1 = Conv(inc * 2, inc, 1)
        self.conv2 = Conv(inc, ouc, 1)

    def forward(self, x):
        # store original size
        H_orig, W_orig = x.size(2), x.size(3)

        # DWT returns pad information so that LL etc are computed on even-size padded input
        LL, LH, HL, HH, pad = self.dwt(x)  # each (N,inc,H2,W2)
        f_ll = self.ll_conv(LL)
        f_lh = self.hf_conv1(LH)
        f_hl = self.hf_conv2(HL)
        f_hh = self.hf_conv3(HH)

        f_hf = torch.cat([f_lh, f_hl, f_hh], dim=1)
        f_hf_mapped = self.hf_to_mid(f_hf)

        attn_out = self.attn(f_ll, f_hf_mapped)

        # upsample explicitly to padded original size, then crop to exact original H_orig, W_orig
        # compute padded original size
        pad_h = 1 if (H_orig % 2 == 1) else 0
        pad_w = 1 if (W_orig % 2 == 1) else 0
        H_padded = H_orig + pad_h
        W_padded = W_orig + pad_w

        up = F.interpolate(attn_out, size=(H_padded, W_padded), mode='bilinear', align_corners=False)
        up = self.mid_to_inc(up)
        # crop to original
        up = up[..., :H_orig, :W_orig]

        x_conv = self.conv1(torch.cat([up, x], dim=1))
        out = self.conv2(x_conv + x)
        return out

    def dwt_regularizer(self):
        return self.dwt.orth_reg() if getattr(self.dwt, 'learnable', False) else torch.tensor(0., device=next(
            self.parameters()).device)


class C3k_HWT_ViT(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(HWT_ViT(c_, c_) for _ in range(n)))


class C3k2_HWT_ViT(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3k_HWT_ViT(self.c, self.c, 2, shortcut, g) if c3k else HWT_ViT(self.c, self.c) for _ in range(n))
# ------------------------------------------ Haar 子带 + 轻量 cross-attention -------------------------------------------
