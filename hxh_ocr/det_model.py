"""DBNet-lite text-line detector (ReLU / Conv / Add / Resize-nearest-x2 / Concat only).

Backbone: timm `mobilenetv4_conv_small_050.e3000_r224_in1k` (ImageNet-pretrained, ReLU-only),
stem summed to one input channel like the recogniser (hxh_ocr/model.py). Features at strides
4 / 8 / 16 / 32 (16 / 32 / 48 / 64 channels).

FPN-lite: 1x1 laterals to `fpn` channels; top-down nearest Resize x2 + Add; per level
depthwise 3x3 + 1x1 smoothing to `level` channels; nearest Resize x2 (chained) to stride 4;
Concat; 1x1 to `mid`.

Head (deployed): conv3x3 mid->32, Resize x2 (stride 2), conv3x3 32->16, conv1x1 16->2
-> [B, 2, H/2, W/2]: channel 0 = text-kernel logit, channel 1 = vertical-orientation logit.
Threshold head (training only, DBNet's differentiable binarisation): same shape, 1 channel,
dropped before export.

Every conv except the output 1x1s is followed by BatchNorm (folded at export) and ReLU.

Model family (experiments/family): `width` picks the backbone (0.5 = the tiny detector,
`mobilenetv4_conv_small_050`; 1.0 = `mobilenetv4_conv_small`, features 32 / 64 / 96 / 128), and
`head_dw` makes the head's two 3x3 convs depthwise-separable (dw3x3 + 1x1), which removes most
of the head's MACs at high resolution. Same op set either way.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

TIMM_NAME = "mobilenetv4_conv_small_050.e3000_r224_in1k"
TIMM_BY_WIDTH = {0.5: TIMM_NAME, 1.0: "mobilenetv4_conv_small.e2400_r224_in1k"}
CHANS_BY_WIDTH = {0.5: [16, 32, 48, 64], 1.0: [32, 64, 96, 128]}


def cbr(cin, cout, k=1, stride=1, groups=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def up2(x):
    return F.interpolate(x, scale_factor=2.0, mode="nearest")


class Smooth(nn.Module):
    """depthwise 3x3 -> BN -> ReLU -> 1x1 -> BN -> ReLU"""

    def __init__(self, cin, cout):
        super().__init__()
        self.dw = cbr(cin, cin, 3, groups=cin)
        self.pw = cbr(cin, cout, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class Head(nn.Module):
    """conv3x3 cin->c1, Resize x2, conv3x3 c1->c2, 1x1 -> cout (bias, no activation).
    dw=True: each 3x3 conv is depthwise 3x3 + 1x1 (both with BN + ReLU)."""

    def __init__(self, cin, cout, c1=32, c2=16, dw: bool = False):
        super().__init__()
        self.c1 = Smooth(cin, c1) if dw else cbr(cin, c1, 3)
        self.c2 = Smooth(c1, c2) if dw else cbr(c1, c2, 3)
        self.out = nn.Conv2d(c2, cout, 1)

    def forward(self, x):
        return self.out(self.c2(up2(self.c1(x))))


class DBNetLite(nn.Module):
    arch = "dbnet_lite"

    def __init__(self, pretrained: bool = True, fpn: int = 48, level: int = 16, mid: int = 64,
                 head=(32, 16), thresh_head: bool = True, width: float = 0.5, head_dw: bool = False):
        super().__init__()
        import timm
        m = timm.create_model(TIMM_BY_WIDTH[width], pretrained=pretrained)
        stem = m.conv_stem
        new = nn.Conv2d(1, stem.out_channels, stem.kernel_size, stem.stride, stem.padding, bias=False)
        with torch.no_grad():
            new.weight.copy_(stem.weight.sum(dim=1, keepdim=True))
        self.conv_stem, self.bn1 = new, m.bn1
        self.stages = nn.ModuleList(list(m.blocks)[:4])  # strides 4, 8, 16, 32
        chans = CHANS_BY_WIDTH[width]
        self.lat = nn.ModuleList([cbr(c, fpn, 1) for c in chans])
        self.smooth = nn.ModuleList([Smooth(fpn, level) for _ in chans])
        self.fuse = cbr(level * len(chans), mid, 1)
        self.head = Head(mid, 2, *head, dw=head_dw)
        self.thresh = Head(mid, 1, *head, dw=head_dw) if thresh_head else None
        self.cfg = {"fpn": fpn, "level": level, "mid": mid, "head": list(head), "width": width, "head_dw": head_dw}

    def features(self, x):
        x = self.bn1(self.conv_stem(x))
        feats = []
        for s in self.stages:
            x = s(x)
            feats.append(x)
        lat = [l(f) for l, f in zip(self.lat, feats)]
        p = [None] * 4
        p[3] = lat[3]
        for i in (2, 1, 0):
            p[i] = lat[i] + up2(p[i + 1])
        outs = []
        for i in range(4):
            y = self.smooth[i](p[i])
            for _ in range(i):
                y = up2(y)
            outs.append(y)
        return self.fuse(torch.cat(outs, dim=1))

    def forward(self, x):
        """Eval: [B, 2, H/2, W/2] logits. Train (with a threshold head): (logits, thresh logits),
        so BN recalibration in train mode also refreshes the threshold head's statistics."""
        f = self.features(x)
        out = self.head(f)
        if self.training and self.thresh is not None:
            return out, self.thresh(f)
        return out


def build_det_model(pretrained: bool = True, **kw) -> DBNetLite:
    return DBNetLite(pretrained=pretrained, **kw)


def count_params(model: nn.Module, deploy: bool = True) -> int:
    return sum(p.numel() for n, p in model.named_parameters()
               if not (deploy and n.startswith("thresh.")))


@torch.no_grad()
def count_macs(model: nn.Module, size: int = 640) -> int:
    model = model.eval()
    with FlopCounterMode(display=False) as fc:
        model(torch.zeros(1, 1, size, size))
    return fc.get_total_flops() // 2


@torch.no_grad()
def macs_breakdown(model: nn.Module, size: int = 640) -> dict:
    model = model.eval()
    out = {}
    with FlopCounterMode(display=False) as fc:
        model(torch.zeros(1, 1, size, size))
    for mod, d in fc.get_flop_counts().items():
        out[mod] = sum(d.values()) // 2
    return out


def load_det_checkpoint(path, device="cpu", weights: str = "ema"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    mcfg = cfg.get("model_cfg", {})
    model = build_det_model(pretrained=False, fpn=mcfg.get("fpn", 48), level=mcfg.get("level", 16),
                            mid=mcfg.get("mid", 64), head=tuple(mcfg.get("head", (32, 16))),
                            width=mcfg.get("width", 0.5), head_dw=mcfg.get("head_dw", False))
    model.load_state_dict(ck[weights])
    return model.to(device).eval(), cfg
