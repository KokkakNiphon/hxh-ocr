"""Quantization-aware training that mimics ESP-DL int8 on the ESP32-S3 (as simulated by ESP-PPQ).

`QATModel(fused_model)` wraps a BN-folded model (`model.fuse_bn`) in fake quantization that
follows what `espdl_quantize_onnx(target="esp32s3", quant_type="w8a8", setting=espdl_setting())`
will do to the exported ONNX graph (read from esp-ppq 1.3.15, see docs/research/JOURNAL.md):

- Everything is per-tensor, symmetric, power-of-two int8 (real = q * 2^exp), rounded half up
  (floor(x + 0.5)), clamped to [-128, 127].
- Weights: exponent from the current weights, every forward, exactly like ESP-PPQ's parameter
  pass: 99.99th-percentile min/max (`TorchPercentileObserver`; equals min/max below 10k
  elements), scale = 2*max|.|/255, rounded UP to a power of two. Straight-through estimator.
- Bias: int32 at exponent in_exp + w_exp, clamped to the S3's 20-bit range [-2^19, 2^19 - 1].
- Activations ("sites"): the graph input (fixed exponent, -7 on the device), every conv output
  (after the ReLU when a ReLU directly follows, since ESP-DL fuses it), every Add output, every
  ReduceMean / AvgPool output and any standalone ReLU (an ESP-DL LUT op). Exponents come from
  ESP-PPQ's own KL calibration, re-implemented here (two passes: min/max, then a 4096-bin
  histogram, KL search over 128-bin multiples, scale rounded to the nearest power of two), on
  fp32 activations with quantized weights, as ESP-PPQ calibrates. They are then frozen.
- Residual Adds ("Align to Output"): both inputs are requantized to the Add's output exponent.
  A producer whose only consumer is the Add is quantized at the Add's exponent directly (ESP-PPQ
  makes its output config a slave of the Add's); a producer with other consumers keeps its own
  exponent and gets an extra requantization on the Add edge (ESP-PPQ exports a RequantizeLinear).
- Activation gradients use a clipped STE (zero outside the representable range).
- Detector ops (DBNet-lite): nearest Resize is a copy at one exponent, so it is transparent
  (its output shares its input's site); Concat is a site with "Align to Output" like Add:
  inputs whose producer has no other consumer take the Concat's exponent, the others are
  requantized on the Concat edge (ESP-PPQ's RequantizeLinear).

Exponents are not learned on purpose: ESP-PPQ re-derives every activation exponent with its own
KL calibration on the exported float ONNX and ignores anything we learned, so QAT is only
faithful if it uses the exponents that calibration will pick. `calibrate()` can be re-run
during training to follow weight drift.

`strip_qat()` returns the plain folded float model (same structure as `fuse_bn`'s output) with
the trained weights, for ONNX export. All fake-quant math runs in fp32 (don't autocast).
"""
from __future__ import annotations

import copy
import math
import operator
from dataclasses import dataclass, field

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F

QMIN, QMAX = -128, 127
BIAS_BITS = 20  # ESP32-S3: int32 bias limited to 20 bits
BIAS_QMIN, BIAS_QMAX = -(1 << (BIAS_BITS - 1)), (1 << (BIAS_BITS - 1)) - 1
WEIGHT_PERCENTILE = 0.9999  # esp_ppq.core.common.OBSERVER_PERCENTILE
KL_BINS = 32 * (1 << 7)  # EspdlQuantizer: 32 * 2^(bits-1) histogram bins for int8
MIN_SCALE = 1e-8  # esp_ppq OBSERVER_MIN_SCALE


# ----------------------------------------------------------------------------
# fake-quant primitives
# ----------------------------------------------------------------------------
class _FakeQuantAct(torch.autograd.Function):
    """Round half up at `scale`, clamp, clipped straight-through gradient."""

    @staticmethod
    def forward(ctx, x, scale, qmin, qmax):
        q = torch.floor(x / scale + 0.5)
        inside = (q >= qmin) & (q <= qmax)
        ctx.save_for_backward(inside)
        return q.clamp_(qmin, qmax).mul_(scale)

    @staticmethod
    def backward(ctx, g):
        (inside,) = ctx.saved_tensors
        return g * inside, None, None, None


def fake_quant_act(x: torch.Tensor, exp: torch.Tensor, qmin=QMIN, qmax=QMAX) -> torch.Tensor:
    return _FakeQuantAct.apply(x.float(), torch.exp2(exp.float()), qmin, qmax)


def fake_quant_ste(x: torch.Tensor, exp: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
    """Round half up + clamp with a plain straight-through gradient (weights, biases)."""
    s = torch.exp2(exp.float())
    q = torch.floor(x / s + 0.5).clamp(qmin, qmax) * s
    return x + (q - x).detach()


@torch.no_grad()
def weight_exponent(w: torch.Tensor, percentile: float = WEIGHT_PERCENTILE) -> torch.Tensor:
    """ESP-PPQ's per-tensor weight exponent (percentile observer + minmax_to_scale_offset).

    Returns a 0-dim float tensor on w's device (no host sync)."""
    flat = w.detach().flatten().float()
    n = flat.numel()
    min_idx = max(0, int(n * (1 - percentile))) + 1
    max_idx = min(int(n * percentile), n - 1) + 1
    if min_idx == 1 and max_idx == n:  # small tensors: the percentile is the min / max
        lo, hi = flat.min(), flat.max()
    else:
        lo = torch.kthvalue(flat, min_idx).values
        hi = torch.kthvalue(flat, max_idx).values
    r = torch.maximum(hi.clamp(min=0), -lo.clamp(max=0)).double()
    scale = torch.clamp(2 * r / (QMAX - QMIN), min=MIN_SCALE)
    return torch.ceil(torch.log2(scale)).float()


# ----------------------------------------------------------------------------
# KL calibration, as esp_ppq.quantization.observer.range.TorchHistObserver (symmetric)
# ----------------------------------------------------------------------------
class KLObserver:
    """ESP-PPQ activation calibration: 'kl' (espdl_setting default) or 'minmax'."""

    def __init__(self, bins: int = KL_BINS, num_bits: int = 8, algo: str = "kl"):
        assert algo in ("kl", "minmax"), algo
        self.bins, self.num_bits, self.algo = bins, num_bits, algo
        self.reset()

    def reset(self):
        self.phase, self.lo, self.hi, self.hist, self.hist_scale = 1, None, None, None, None

    @torch.no_grad()
    def observe(self, x: torch.Tensor):
        x = x.detach().float()
        if self.phase == 1:
            lo, hi = x.min(), x.max()
            self.lo = lo if self.lo is None else torch.minimum(self.lo, lo)
            self.hi = hi if self.hi is None else torch.maximum(self.hi, hi)
        else:
            if self.hist_scale == 0:
                return
            h = torch.histc(x.abs(), self.bins, min=0, max=self.hist_scale * self.bins)
            self.hist = h.int() if self.hist is None else self.hist + h.int()

    def start_hist(self):
        rng = max(abs(float(self.hi)), abs(float(self.lo)))
        self.hist_scale = rng / self.bins
        self.phase = 2

    def clip_and_exponent(self) -> tuple[float, int]:
        """(clip value, power-of-two exponent), following PPQ's hist_to_scale_offset (kl) or
        minmax_to_scale_offset (minmax: scale 2*max|x|/255 rounded UP to a power of two)."""
        if self.algo == "minmax":
            r = max(abs(min(float(self.lo), 0.0)), abs(max(float(self.hi), 0.0)))
            scale = max(2 * r / (QMAX - QMIN), MIN_SCALE)
            return r, int(math.ceil(math.log2(scale)))
        if not self.hist_scale or self.hist is None:
            return 0.0, -7  # tensor is all zeros on the calibration data; any exponent works
        hist = self.hist.cpu().float()
        quant_bins = 2 ** (self.num_bits - 1)
        hist[: int(self.bins * 0.002)] = 0  # "following code is crucial, do not remove" (PPQ)
        hist[int(self.bins * 0.002)] = 1
        total = hist.sum()
        best, best_kl = None, None
        for bin_range in range(quant_bins, self.bins + quant_bins - 1, quant_bins):
            p = hist[:bin_range].clone()
            p[bin_range - 1] += hist[bin_range:].sum()
            p = p / total
            r = bin_range // quant_bins
            q = hist[:bin_range].clone().reshape(quant_bins, r)
            pos = q > 0
            cnt = pos.sum(dim=1, keepdim=True)
            cnt[cnt == 0] = 1
            q = (q.sum(dim=1, keepdim=True) / cnt).repeat(1, r) * pos
            q = (q / q.sum()).flatten()
            kl = torch.dot(p.double(), torch.log10(p.double() + 1e-30) - torch.log10(q.double() + 1e-30)).item()
            if best_kl is None or kl < best_kl:  # first minimum, as PPQ's stable sort
                best, best_kl = bin_range, kl
        scale = max((best / self.bins) * self.hist_scale * (self.bins / quant_bins), MIN_SCALE)
        # ppq_round_to_power_of_2(scale, ROUND_HALF_UP): ties of log2 go towards +inf
        return scale * quant_bins, int(math.floor(math.log2(scale) + 0.5))


# ----------------------------------------------------------------------------
# modules inserted into the traced graph
# ----------------------------------------------------------------------------
class ActQuant(nn.Module):
    """One activation exponent (one ESP-PPQ master config). Called at every graph position that
    is quantized with this exponent; `observe=True` only where ESP-PPQ calibrates it."""

    def __init__(self, fixed_exp: int | None = None, calib_algo: str = "kl"):
        super().__init__()
        # `exp` is the exponent in use (read by the graph, also as QConv bias input exponent);
        # `exp_cal` the latest calibration. They differ only while exponent sampling is on.
        self.register_buffer("exp", torch.tensor(float(fixed_exp if fixed_exp is not None else 0)))
        self.register_buffer("exp_cal", self.exp.clone())
        self.fixed = fixed_exp is not None
        self.mode = "float"  # float | observe | quant
        self.observer = KLObserver(algo=calib_algo)
        self.clip = 0.0
        self.history: list[int] = []  # recent calibrated exponents (for exponent sampling)

    def forward(self, x, observe: bool = True):
        if self.mode == "quant":
            return fake_quant_act(x, self.exp)
        if self.mode == "observe" and observe and not self.fixed:
            self.observer.observe(x)
        return x


class QConv2d(nn.Conv2d):
    """Conv2d with ESP-DL weight / bias fake quantization; takes the input exponent as 2nd arg."""

    @classmethod
    def from_conv(cls, conv: nn.Conv2d) -> "QConv2d":
        q = cls(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                conv.dilation, conv.groups, conv.bias is not None, conv.padding_mode,
                device=conv.weight.device, dtype=conv.weight.dtype)
        q.weight = conv.weight
        q.bias = conv.bias
        q.mode = "float"
        q.w_exp = None
        return q

    def forward(self, x, in_exp=None):
        if self.mode == "float":
            return self._conv_forward(x, self.weight, self.bias)
        w_exp = weight_exponent(self.weight)
        self.w_exp = w_exp
        w = fake_quant_ste(self.weight.float(), w_exp, QMIN, QMAX)
        b = self.bias
        if b is not None and self.mode == "quant":  # ESP-PPQ calibrates with float biases
            b = fake_quant_ste(b.float(), in_exp + w_exp, BIAS_QMIN, BIAS_QMAX)
        return self._conv_forward(x.float(), w, b)


# ----------------------------------------------------------------------------
# graph analysis
# ----------------------------------------------------------------------------
@dataclass(eq=False)
class Site:
    """A quantized tensor in ESP-PPQ terms (one TensorQuantizationConfig)."""
    kind: str                       # input | conv | conv_relu | lut | add | mean | pool
    node: fx.Node                   # graph node whose output this site quantizes
    conv: str | None = None         # conv module path for conv / conv_relu sites
    parent: "Site | None" = None    # union-find: merged into an Add site
    fixed_exp: int | None = None
    aq_name: str | None = None
    members: list = field(default_factory=list)

    def root(self) -> "Site":
        s = self
        while s.parent is not None:
            s = s.parent
        return s


def _is_mod(gm, n, types):
    return n.op == "call_module" and isinstance(gm.get_submodule(n.target), types)


def _is_relu(gm, n):
    return (_is_mod(gm, n, nn.ReLU) or (n.op == "call_function" and n.target in (F.relu, torch.relu))
            or (n.op == "call_method" and n.target == "relu"))


def _is_add(n):
    return ((n.op == "call_function" and n.target in (operator.add, torch.add))
            or (n.op == "call_method" and n.target == "add"))


def _is_mean(n):
    return (n.op == "call_method" and n.target == "mean") or (n.op == "call_function" and n.target == torch.mean)


def _is_resize(n):
    return n.op == "call_function" and n.target is F.interpolate and n.kwargs.get("mode", "nearest") == "nearest"


def _is_cat(n):
    return n.op == "call_function" and n.target is torch.cat


def _is_pool(gm, n):
    return _is_mod(gm, n, (nn.AvgPool2d, nn.AdaptiveAvgPool2d)) or (
        n.op == "call_function" and n.target in (F.avg_pool2d, F.adaptive_avg_pool2d))


@torch.no_grad()
def equalize(fused: nn.Module, iterations: int = 4, max_scale: float = 64.0) -> int:
    """Cross-layer equalization (Nagel et al. 2019) of a BN-folded model, in place and exact in
    float: for every conv -> [ReLU] -> conv pair whose middle tensor has no other consumer, output
    channel c of the first conv is divided by s_c and input channel c of the second multiplied
    by it (a ReLU commutes with a positive scale), with s_c = sqrt(r1_c / r2_c), so both sides
    end up with range sqrt(r1_c r2_c). Per-tensor power-of-two weights then waste far fewer
    levels on a few large channels; the width-1.0 MobileNetV4 needs it (its depthwise layers lose
    almost every channel to int8 otherwise). Returns the number of pairs."""
    gm = fx.symbolic_trace(fused)
    mods = dict(gm.named_modules())
    conv = lambda n: n.op == "call_module" and isinstance(mods.get(n.target), nn.Conv2d)  # noqa: E731
    skip = lambda n: n.op == "call_module" and isinstance(mods.get(n.target), (nn.Identity, nn.Dropout))  # noqa: E731

    def only_user(n):
        """The single consumer of n, looking through Identity / Dropout (folded BN slots)."""
        while True:
            if len(n.users) != 1:
                return None
            n = next(iter(n.users))
            if not skip(n):
                return n

    pairs = []
    for n in gm.graph.nodes:
        if not conv(n):
            continue
        m = only_user(n)
        if m is not None and _is_relu(gm, m):
            m = only_user(m)
        if m is None or not conv(m):
            continue
        c1, c2 = mods[n.target], mods[m.target]
        C = c1.out_channels
        dw2 = c2.groups == C and c2.in_channels == C and c2.out_channels == C
        if c2.groups != 1 and not dw2:
            continue
        pairs.append((c1, c2, dw2))
    for _ in range(iterations):
        for c1, c2, dw2 in pairs:
            C = c1.out_channels
            r1 = c1.weight.abs().reshape(C, -1).amax(1)
            r2 = (c2.weight.abs().reshape(C, -1) if dw2 else c2.weight.abs().transpose(0, 1).reshape(C, -1)).amax(1)
            ok = (r1 > 1e-12) & (r2 > 1e-12)
            s = torch.where(ok, torch.sqrt(r1 / r2.clamp(min=1e-12)), torch.ones_like(r1)).clamp(1 / max_scale, max_scale)
            c1.weight.div_(s.view(-1, 1, 1, 1))
            if c1.bias is not None:
                c1.bias.div_(s)
            c2.weight.mul_(s.view(-1, 1, 1, 1) if dw2 else s.view(1, -1, 1, 1))
    return len(pairs)


class QATModel(nn.Module):
    """Fake-quantized ESP-DL int8 view of a BN-folded model. See the module docstring."""

    def __init__(self, fused: nn.Module, input_exp: int | None = -7, calib_algo: str = "kl"):
        """input_exp: fixed exponent of the graph input (-7 is what ESP-PPQ's KL picks for
        [-1, 1] images), or None to calibrate it like any other tensor (minmax gives -6).
        calib_algo: activation calibration ESP-PPQ will use ('kl', its default, or 'minmax')."""
        super().__init__()
        self.calib_algo = calib_algo
        self._template = [copy.deepcopy(fused).cpu()]  # list: keep it out of the module tree
        gm = fx.symbolic_trace(copy.deepcopy(fused))
        g = gm.graph
        # 1. drop Identity modules (folded BN without activation, empty timm slots)
        for n in list(g.nodes):
            if _is_mod(gm, n, nn.Identity):
                n.replace_all_uses_with(n.args[0])
                g.erase_node(n)
        # Dropout is not in the exported graph; a nearest Resize copies values at one exponent
        is_pass = lambda n: _is_mod(gm, n, nn.Dropout) or _is_resize(n)  # noqa: E731

        def producer(n):
            while is_pass(n):
                n = n.args[0]
            return n

        def users(n):
            out = []
            for u in n.users:
                out += users(u) if is_pass(u) else [u]
            return out

        # 2. find the quantized tensors (sites)
        sites: dict[fx.Node, Site] = {}
        fused_conv: dict[fx.Node, fx.Node] = {}  # relu -> conv it is fused with
        conv_in: dict[fx.Node, fx.Node] = {}     # conv -> producer of its input
        for n in g.nodes:
            if n.op == "placeholder":
                sites[n] = Site("input", n, fixed_exp=input_exp)
            elif _is_mod(gm, n, nn.Conv2d):
                conv_in[n] = producer(n.args[0])
                us = list(n.users)
                if len(us) == 1 and _is_relu(gm, us[0]):
                    fused_conv[us[0]] = n
                else:
                    sites[n] = Site("conv", n, conv=n.target)
            elif _is_relu(gm, n):
                if n in fused_conv:
                    sites[n] = Site("conv_relu", n, conv=fused_conv[n].target)
                else:
                    sites[n] = Site("lut", n)
            elif _is_add(n):
                if len(n.args) != 2 or not all(isinstance(a, fx.Node) for a in n.args) or n.kwargs:
                    raise NotImplementedError(f"only tensor + tensor Adds are supported: {n.format_node()}")
                sites[n] = Site("add", n)
            elif _is_mean(n):
                sites[n] = Site("mean", n)
            elif _is_pool(gm, n):
                sites[n] = Site("pool", n)
            elif _is_cat(n):
                if not isinstance(n.args[0], (list, tuple)) or not all(isinstance(a, fx.Node) for a in n.args[0]):
                    raise NotImplementedError(f"only cat of a list of tensors is supported: {n.format_node()}")
                sites[n] = Site("concat", n)
            elif is_pass(n) or n.op == "output":
                pass
            else:
                raise NotImplementedError(f"op not supported by the ESP-DL QAT model: {n.format_node()}")
        # 3. "Align to Output" at Adds and Concats: a producer used only by it shares its exponent
        def joined(n):
            return list(n.args) if _is_add(n) else list(n.args[0]) if _is_cat(n) else []

        for n in g.nodes:
            for a in joined(n):
                p = producer(a)
                if p.op != "placeholder" and len(users(p)) == 1 and sites[p].root() is not sites[n]:
                    sites[p].root().parent = sites[n]
        roots = []
        for s in sites.values():
            r = s.root()
            if r not in roots:
                roots.append(r)
            r.members.append(s)
        # 4. one ActQuant per root
        self.aq_names = []
        for i, r in enumerate(roots):
            r.aq_name = f"qat_act{i:03d}"
            gm.add_submodule(r.aq_name, ActQuant(r.fixed_exp, calib_algo))
            self.aq_names.append(r.aq_name)
        # 5. rewrite: quantize every site output, requantize Add inputs, QConv with input exponent
        requant = [(n, i) for n in g.nodes for i, a in enumerate(joined(n))
                   if sites[producer(a)].root() is not sites[n].root()]
        for n, s in sites.items():
            r = s.root()
            with g.inserting_after(n):
                q = g.call_module(r.aq_name, (n,), {"observe": r is s})
            n.replace_all_uses_with(q, delete_user_cb=lambda u, q=q: u is not q)
        for n, i in requant:  # ESP-PPQ exports these as RequantizeLinear (or nothing if equal)
            a = joined(n)[i]
            with g.inserting_before(n):
                rq = g.call_module(sites[n].root().aq_name, (a,), {"observe": False})
            if _is_add(n):
                args = list(n.args)
                args[i] = rq
                n.args = tuple(args)
            else:
                lst = list(n.args[0])
                lst[i] = rq
                n.args = (lst,) + tuple(n.args[1:])
        self.n_requant = len(requant)
        self.conv_names, self.conv_in_site = [], {}
        for n in list(g.nodes):
            if _is_mod(gm, n, nn.Conv2d) and not isinstance(gm.get_submodule(n.target), QConv2d):
                conv = gm.get_submodule(n.target)
                parent, _, leaf = n.target.rpartition(".")
                setattr(gm.get_submodule(parent) if parent else gm, leaf, QConv2d.from_conv(conv))
                r = sites[conv_in[n]].root()
                with g.inserting_before(n):
                    e = g.get_attr(f"{r.aq_name}.exp")
                n.args = (n.args[0], e)
                self.conv_names.append(n.target)
                self.conv_in_site[n.target] = r.aq_name
        g.lint()
        gm.recompile()
        self.gm = gm
        self.sites = list(sites.values())
        self.roots = roots
        self.set_mode("float")  # call calibrate() to get the activation exponents
        self.calibrated = False

    # -- modes ---------------------------------------------------------------
    def act_quants(self):
        return [self.gm.get_submodule(n) for n in self.aq_names]

    def qconvs(self):
        return [self.gm.get_submodule(n) for n in self.conv_names]

    def set_mode(self, mode: str):
        """float: plain model. observe: int8 weights, float biases and activations, KL observers
        on (ESP-PPQ's calibration view). quant: full int8 simulation."""
        assert mode in ("float", "observe", "quant")
        for m in self.act_quants() + self.qconvs():
            m.mode = mode
        self.mode = mode
        return self

    exp_sampling = 0  # >0: in training, draw each exponent from its last N calibrations

    def forward(self, x):
        if self.mode == "quant" and self.exp_sampling:
            with torch.no_grad():
                for a in self.act_quants():
                    if self.training and len(set(a.history)) > 1:
                        a.exp.fill_(float(a.history[torch.randint(len(a.history), ()).item()]))
                    else:
                        a.exp.copy_(a.exp_cal)
        return self.gm(x)

    # -- calibration ---------------------------------------------------------
    @torch.no_grad()
    def calibrate(self, batches) -> dict:
        """Two-pass KL calibration of every non-fixed activation exponent on `batches` (a list of
        image tensors on the model's device), exactly as ESP-PPQ calibrates. Returns
        {aq_name: (old_exp, new_exp)} for the exponents that changed."""
        was_training = self.training
        self.eval()
        aqs = self.act_quants()
        for a in aqs:
            a.observer.reset()
            a.exp.copy_(a.exp_cal)  # undo any sampled exponent: weights' view stays the same
        self.set_mode("observe")
        for x in batches:
            self(x)
        if self.calib_algo == "kl":  # second pass: histograms
            for a in aqs:
                if not a.fixed:
                    a.observer.start_hist()
            for x in batches:
                self(x)
        changed = {}
        for name, a in zip(self.aq_names, aqs):
            if a.fixed:
                continue
            clip, e = a.observer.clip_and_exponent()
            a.clip = clip
            old = int(a.exp_cal.item())
            if self.calibrated and old != e:
                changed[name] = (old, e)
            a.exp.fill_(e)
            a.exp_cal.fill_(e)
            a.history = (a.history + [e])[-max(1, self.exp_sampling):]
        self.calibrated = True
        self.set_mode("quant")
        self.train(was_training)
        return changed

    # -- reporting -----------------------------------------------------------
    @torch.no_grad()
    def exponents(self) -> dict:
        """Per conv: weight / input / output exponents (comparable with ESP-PPQ's model.json),
        plus every activation site."""
        aq ={n: int(self.gm.get_submodule(n).exp.item()) for n in self.aq_names}
        out_site = {}
        for s in self.sites:
            if s.conv:
                out_site[s.conv] = s.root().aq_name
        convs = {}
        for name, c in zip(self.conv_names, self.qconvs()):
            convs[name] = {"w": int(weight_exponent(c.weight).item()), "in": aq[self.conv_in_site[name]],
                           "out": aq[out_site[name]]}
        sites = []
        for s in self.sites:
            r = s.root()
            sites.append({"node": s.node.name, "kind": s.kind, "conv": s.conv, "aq": r.aq_name,
                          "exp": aq[r.aq_name], "merged_into": None if r is s else r.node.name,
                          "clip": float(self.gm.get_submodule(r.aq_name).clip)})
        return {"convs": convs, "sites": sites}

    def summary(self) -> str:
        kinds = {}
        for s in self.sites:
            kinds[s.kind] = kinds.get(s.kind, 0) + 1
        merged = sum(1 for s in self.sites if s.parent is not None)
        return (f"{len(self.conv_names)} convs, {len(self.sites)} sites {kinds}, {len(self.roots)} exponents "
                f"({merged} producers share an Add's exponent, {self.n_requant} requantized Add inputs)")

    # -- export --------------------------------------------------------------
    def strip_qat(self) -> nn.Module:
        """Plain folded float model (no fake quant) with the current weights, on the CPU."""
        m = copy.deepcopy(self._template[0])
        own = m.state_dict()
        sd = {k: v.detach().cpu().clone() for k, v in self.gm.state_dict().items() if k in own}
        missing = set(own) - set(sd)
        assert not missing, f"strip_qat: no trained value for {sorted(missing)[:5]}"
        m.load_state_dict(sd, strict=True)
        return m.eval()
