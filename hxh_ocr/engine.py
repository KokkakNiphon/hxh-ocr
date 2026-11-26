"""Shared train/eval helpers: EMA, checkpoint loading, batched prediction."""
from __future__ import annotations

import copy

import torch

from .decode import decode_text, logits_to_ids
from .model import STRIDE_W, build_model, fuse_bn


class EMA:
    """Exponential moving average of parameters; BN buffers are copied from the live model."""

    def __init__(self, model, decay: float = 0.999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for e, m in zip(self.module.parameters(), model.parameters()):
            e.mul_(d).add_(m.detach(), alpha=1 - d)
        for e, m in zip(self.module.buffers(), model.buffers()):
            e.copy_(m)


@torch.no_grad()
def recalibrate_bn(model, batches, device, amp: bool = False):
    """Re-estimate BatchNorm running stats for `model` (e.g. the EMA copy) on training batches.

    EMA weights lag the live weights by ~1/(1-decay) steps, so the live model's BN statistics
    don't fit them. For the pretrained MobileNetV4 that mismatch took val CER from 2.5% to 35.7%
    at step 5000 (see docs/research/JOURNAL.md). These stats are also what gets folded into the
    convs for export."""
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return model
    saved = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative average over the calibration batches
    model.train()
    for b in batches:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            model(b["images"].to(device, non_blocking=True))
    for m, mom in zip(bns, saved):
        m.momentum = mom
    return model.eval()


def input_lengths(widths: torch.Tensor, T: int) -> torch.Tensor:
    """Valid timesteps per sample for padded batches."""
    return torch.clamp(torch.ceil(widths.float() / STRIDE_W).long(), 1, T)


def pad_like_app(x: torch.Tensor, width: int) -> torch.Tensor:
    """Pad a batch on the right to `width` by repeating its last column, the way web/ocr.js pads
    every line shorter than the reader's input (the collate pads each sample to the batch width
    the same way)."""
    if x.shape[-1] >= width:
        return x
    return torch.cat([x, x[..., -1:].expand(*x.shape[:-1], width - x.shape[-1])], dim=-1)


def ctc_lengths(widths: torch.Tensor, T: int, pad_blank: float) -> torch.Tensor:
    """CTC input lengths: the frames over each line (`input_lengths`), or, with `pad_blank` > 0
    (readers v7), every frame of the batch, so the padding after a short line is supervised as
    blank (int8 readers emitted runs of `!` there; train with `pad_like_app` now and then)."""
    if pad_blank > 0:
        return torch.full((len(widths),), T, dtype=torch.long)
    return input_lengths(widths, T)


@torch.no_grad()
def predict(model, loader, device, tokens, amp: bool = False):
    """Run the model over a loader; returns (refs, hyps, dataset_indices).

    The bucket sampler reorders samples, hence the returned indices."""
    model.eval()
    refs, hyps, idxs = [], [], []
    for batch in loader:
        x = batch["images"].to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            out = model(x)
        lens = input_lengths(batch["widths"], out.shape[-1])
        ids = logits_to_ids(out.float())
        for row, n, ref, i in zip(ids, lens.tolist(), batch["texts"], batch["indices"]):
            hyps.append(decode_text(row[:n], tokens))
            refs.append(ref)
            idxs.append(i)
    return refs, hyps, idxs


def load_checkpoint(path, device="cpu", weights: str = "ema"):
    """Rebuild a model from a checkpoint written by train.py (or train_qat.py, whose weights are
    BN-folded: `folded: True`). Returns (model, config)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = build_model(cfg["arch"], num_classes=cfg["num_classes"], pretrained=False,
                        dropout=cfg.get("dropout", 0.1))
    if ck.get("folded"):  # same structure as fuse_bn's output: convs with bias, no BatchNorm
        h = cfg.get("height", 32)
        model = fuse_bn(model, (h, 8 * h))
    model.load_state_dict(ck[weights])
    return model.to(device).eval(), cfg
