"""CRNN-style (conv-only) line recognisers for the Hunter alphabet.

Both archs share one layout: a conv backbone that reduces the width by 4 (T = W/4
timesteps) and the height by 32 (mean-collapsed at the end), then a conv-only sequence
head and a 1x1 classifier.  Output is `[B, num_classes, 1, T]`.

Deployment constraints (ESP-DL, int8, per-tensor power-of-two): only Conv (incl.
depthwise), ReLU, Add, ReduceMean and plain pooling; no BatchNorm (folded by `fuse_bn`),
no SE, no hard-swish, no LSTM / attention.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

STRIDE_W = 4  # total stride along the width -> T = W / 4


# ----------------------------------------------------------------------------
# shared head
# ----------------------------------------------------------------------------
class HeadBlock(nn.Module):
    """x + [dw(1,k) -> BN -> ReLU -> 1x1 -> BN -> ReLU]."""

    def __init__(self, c: int, k: int = 5):
        super().__init__()
        self.dw = nn.Conv2d(c, c, (1, k), padding=(0, k // 2), groups=c, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.act1 = nn.ReLU()
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.act2 = nn.ReLU()

    def forward(self, x):
        y = self.act1(self.bn1(self.dw(x)))
        y = self.act2(self.bn2(self.pw(y)))
        return x + y


class SeqHead(nn.Module):
    """Collapse H (mean), 2 residual blocks, dropout, 1x1 classifier."""

    def __init__(self, c: int, num_classes: int, dropout: float = 0.1, k: int = 5):
        super().__init__()
        self.blocks = nn.Sequential(HeadBlock(c, k), HeadBlock(c, k))
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Conv2d(c, num_classes, 1)

    def forward(self, x):  # [B, C, h, T]
        x = x.mean(dim=2, keepdim=True)  # ReduceMean over H -> [B, C, 1, T]
        x = self.blocks(x)
        return self.cls(self.drop(x))


# ----------------------------------------------------------------------------
# pretrained MobileNetV4-Conv-Small 0.5
# ----------------------------------------------------------------------------
class MNV4S(nn.Module):
    """timm `mobilenetv4_conv_small_050.e3000_r224_in1k`, first `keep_stages` stages.

    timm stage list (stride in the original / here):
      conv_stem   3x3 s2 / (2,2)        (1 input channel: RGB kernel summed)
      blocks[0]   ConvBnAct x2, s2 / (2,2)     -> W/4 reached here
      blocks[1]   ConvBnAct x2, s2 / (2,1)
      blocks[2]   6 UIR,        s2 / (2,1)   (dw_mid of the first block)
      blocks[3]   6 UIR,        s2 / (2,1)
      [conv_head 480 + 1280 and the classifier are dropped]
    """

    arch = "mnv4s"

    def __init__(self, num_classes: int, pretrained: bool = True, keep_stages: int = 4,
                 dropout: float = 0.1, width: float = 0.5):
        super().__init__()
        import timm
        name = {0.5: "mobilenetv4_conv_small_050.e3000_r224_in1k", 1.0: "mobilenetv4_conv_small.e2400_r224_in1k"}[width]
        m = timm.create_model(name, pretrained=pretrained)
        stem = m.conv_stem
        new = nn.Conv2d(1, stem.out_channels, stem.kernel_size, stem.stride, stem.padding, bias=False)
        with torch.no_grad():
            new.weight.copy_(stem.weight.sum(dim=1, keepdim=True))
        self.conv_stem, self.bn1 = new, m.bn1
        self.blocks = nn.Sequential(*list(m.blocks)[:keep_stages])
        for si, stage in enumerate(self.blocks):
            if si == 0:
                continue  # keep (2,2) in the first stage: W/4
            for mod in stage[0].modules():
                if isinstance(mod, nn.Conv2d) and mod.stride == (2, 2):
                    mod.stride = (2, 1)
        c = self.blocks[-1][-1].conv.out_channels if hasattr(self.blocks[-1][-1], "conv") \
            else self.blocks[-1][-1].pw_proj.conv.out_channels
        self.out_channels = c
        self.head = SeqHead(c, num_classes, dropout)

    def forward(self, x):
        x = self.bn1(self.conv_stem(x))
        x = self.blocks(x)
        return self.head(x)


# ----------------------------------------------------------------------------
# from-scratch baseline
# ----------------------------------------------------------------------------
def _cbr(cin, cout, k, stride, groups=1):
    p = (k[0] // 2, k[1] // 2) if isinstance(k, tuple) else k // 2
    return nn.Sequential(nn.Conv2d(cin, cout, k, stride, p, groups=groups, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU())


class DSBlock(nn.Module):
    """Depthwise-separable block: dw3x3 (stride) -> BN -> ReLU -> 1x1 -> BN -> ReLU,
    with an Add skip when shape is unchanged."""

    def __init__(self, cin, cout, stride=(1, 1), k=3):
        super().__init__()
        self.dw = _cbr(cin, cin, k, stride, groups=cin)
        self.pw = _cbr(cin, cout, 1, 1)
        self.skip = stride == (1, 1) and cin == cout

    def forward(self, x):
        y = self.pw(self.dw(x))
        return x + y if self.skip else y


class Tiny(nn.Module):
    """Plain conv stem + depthwise-separable stages; same strides / head as MNV4S."""

    arch = "tiny"

    def __init__(self, num_classes: int, pretrained: bool = False, dropout: float = 0.1,
                 chans=(24, 48, 96, 144, 192), reps=(2, 2, 3)):
        super().__init__()
        c0, c1, c2, c3, c4 = chans
        self.stem = _cbr(1, c0, 3, (2, 2))                       # H/2,  W/2
        self.s1 = DSBlock(c0, c1, (2, 2))                        # H/4,  W/4  (T = W/4)
        layers = []
        cin = c1
        for cout, n, st in zip((c2, c3, c4), reps, ((2, 1), (2, 1), (2, 1))):
            layers.append(DSBlock(cin, cout, st, k=3))           # H/8, H/16, H/32
            layers += [DSBlock(cout, cout, k=3) for _ in range(n)]
            cin = cout
        self.body = nn.Sequential(*layers)
        self.out_channels = cin
        self.head = SeqHead(cin, num_classes, dropout)

    def forward(self, x):
        x = self.s1(self.stem(x))
        return self.head(self.body(x))


def build_model(arch: str, num_classes: int = 75, pretrained: bool = True, dropout: float = 0.1,
                **kw) -> nn.Module:
    if arch == "mnv4s":
        return MNV4S(num_classes, pretrained=pretrained, dropout=dropout, **kw)
    if arch == "mnv4s1":  # full-width MobileNetV4-small (32/64/96/128 ch): the "base" reader of the model family
        return MNV4S(num_classes, pretrained=pretrained, dropout=dropout, width=1.0, **kw)
    if arch == "mnv4s3":  # first 3 stages only: ~96k params, ~15M MACs @32x256 (S3 speed option)
        return MNV4S(num_classes, pretrained=pretrained, dropout=dropout, keep_stages=3, **kw)
    if arch == "tiny":
        return Tiny(num_classes, dropout=dropout, **kw)
    raise ValueError(f"unknown arch {arch!r} (mnv4s | mnv4s1 | mnv4s3 | tiny)")


# ----------------------------------------------------------------------------
# BN folding
# ----------------------------------------------------------------------------
def _set_module(root: nn.Module, name: str, new: nn.Module):
    *path, last = name.split(".")
    for p in path:
        root = getattr(root, p)
    setattr(root, last, new)


@torch.no_grad()
def fuse_bn(model: nn.Module, example_hw=(32, 256)) -> nn.Module:
    """Return a deep copy with every BatchNorm folded into its preceding conv.

    Pairs are discovered by tracing one forward pass: a BN must consume exactly the
    output tensor of the conv executed just before it.  timm's BatchNormAct2d is
    replaced by its activation (ReLU / Identity); plain BatchNorm2d by Identity.
    """
    model = copy.deepcopy(model).eval()
    names = {m: n for n, m in model.named_modules()}
    is_bn = lambda m: isinstance(m, nn.modules.batchnorm._BatchNorm)
    events, handles = [], []

    def hook(mod, inp, out):
        events.append((mod, inp[0], out))

    for mod in model.modules():
        if isinstance(mod, nn.Conv2d) or is_bn(mod):
            handles.append(mod.register_forward_hook(hook))
    # BatchNormAct2d also contains an act/drop submodule; hooks above only touch conv/BN.
    model(torch.zeros(1, 1, *example_hw))
    for h in handles:
        h.remove()

    pairs, last = [], None
    for mod, inp, out in events:
        if isinstance(mod, nn.Conv2d):
            last = (mod, out)
        else:
            if last is None or last[1] is not inp:
                raise RuntimeError(f"BatchNorm {names[mod]} does not directly follow a conv")
            pairs.append((last[0], mod))
            last = None
    for conv, bn in pairs:
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        conv.weight.mul_(scale.view(-1, 1, 1, 1))
        b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
        conv.bias = nn.Parameter((b - bn.running_mean) * scale + bn.bias)
        if hasattr(bn, "act"):  # timm BatchNormAct2d: keep activation, drop BN (+ drop)
            _set_module(model, names[bn], bn.act)
        else:
            _set_module(model, names[bn], nn.Identity())
    left = [n for n, m in model.named_modules() if is_bn(m)]
    assert not left, f"BN left after fuse: {left}"
    return model


# ----------------------------------------------------------------------------
# size reporting
# ----------------------------------------------------------------------------
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def count_macs(model: nn.Module, height: int = 32, width: int = 256) -> int:
    model = model.eval()
    with FlopCounterMode(display=False) as fc:
        model(torch.zeros(1, 1, height, width))
    return fc.get_total_flops() // 2


def report(model: nn.Module, sizes=((32, 256), (48, 384))) -> dict:
    out = {"params": count_params(model)}
    for h, w in sizes:
        out[f"macs_{h}x{w}"] = count_macs(model, h, w)
    return out
