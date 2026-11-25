//! Running a validated model in a caller-provided arena.

use core::ops::Range;

use crate::format::{Error, Model, Op, OpKind};
use crate::kernels::{ConvGeom, Kernels, PoolGeom, ResizeGeom};
use crate::{as_i8, as_i8_mut};

/// Split the arena into the output range (mutable) and up to two input ranges (shared).
/// `parse` guarantees that inputs are either disjoint from the output or handled in place.
fn carve(arena: &mut [i8], out: Range<usize>, a: Range<usize>, b: Range<usize>) -> (&mut [i8], &[i8], &[i8]) {
    let (lo, rest) = arena.split_at_mut(out.start);
    let (o, hi) = rest.split_at_mut(out.end - out.start);
    let (lo, hi): (&[i8], &[i8]) = (lo, hi);
    (o, pick(lo, hi, &out, a), pick(lo, hi, &out, b))
}

fn pick<'s>(lo: &'s [i8], hi: &'s [i8], out: &Range<usize>, r: Range<usize>) -> &'s [i8] {
    if r.is_empty() {
        &[]
    } else if r.end <= out.start {
        &lo[r]
    } else {
        &hi[r.start - out.end..r.end - out.end]
    }
}

impl<'a> Model<'a> {
    fn check_arena(&self, arena: &[u8]) -> Result<(), Error> {
        if arena.len() < self.arena_size() {
            return Err(Error::ArenaTooSmall { need: self.arena_size(), have: arena.len() });
        }
        Ok(())
    }

    /// The int8 input tensor inside `arena` (NHWC), for callers that quantise themselves.
    pub fn input_mut<'b>(&self, arena: &'b mut [u8]) -> Result<&'b mut [i8], Error> {
        self.check_arena(arena)?;
        Ok(&mut as_i8_mut(arena)[self.input().range()])
    }

    /// Write a u8 grayscale image (already resized to the input `h x w`, row-major) into
    /// the input tensor through the model's input LUT (normalisation + quantisation).
    pub fn set_input_u8(&self, pixels: &[u8], arena: &mut [u8]) -> Result<(), Error> {
        let lut = self.input_lut().ok_or(Error::NoInputLut)?;
        let need = self.input().size;
        if pixels.len() != need {
            return Err(Error::InputSize { need, have: pixels.len() });
        }
        let dst = self.input_mut(arena)?;
        for (d, &p) in dst.iter_mut().zip(pixels) {
            *d = lut[p as usize];
        }
        Ok(())
    }

    /// Run all ops; returns the int8 output inside `arena` (CTC logits, NHWC `[1, 1, T, C]`,
    /// or the raw output tensor of a raw-output model).
    pub fn run<'b, K: Kernels + ?Sized>(&self, k: &K, arena: &'b mut [u8]) -> Result<&'b [i8], Error> {
        self.run_with_hook(k, arena, |_, _, _| {})
    }

    /// Like [`Model::run`], calling `hook(op_index, &op, output)` after every op
    /// (debugging, per-layer comparison, profiling).
    pub fn run_with_hook<'b, K, F>(&self, k: &K, arena: &'b mut [u8], mut hook: F) -> Result<&'b [i8], Error>
    where
        K: Kernels + ?Sized,
        F: FnMut(usize, &Op, &[i8]),
    {
        self.check_arena(arena)?;
        let ar = as_i8_mut(arena);
        for i in 0..self.n_ops() {
            let op = self.op(i);
            let x = self.tensor(op.in0 as usize);
            let y = self.tensor(op.out as usize);
            match op.kind {
                OpKind::Conv | OpKind::DwConv => {
                    let g = ConvGeom {
                        in_h: x.h,
                        in_w: x.w,
                        in_c: x.c,
                        out_h: y.h,
                        out_w: y.w,
                        out_c: y.c,
                        cout_padded: op.cout_padded,
                        kh: op.kh,
                        kw: op.kw,
                        sh: op.sh,
                        sw: op.sw,
                        pad_t: op.pad[0],
                        pad_l: op.pad[1],
                        shift: op.shift,
                        relu: op.relu(),
                    };
                    let (yo, xi, _) = carve(ar, y.range(), x.range(), 0..0);
                    if op.kind == OpKind::Conv {
                        k.conv2d(&g, xi, self.weights(&op), self.bias(&op), yo);
                    } else {
                        k.dwconv2d(&g, xi, self.weights(&op), self.bias(&op), yo);
                    }
                }
                OpKind::AvgPool => {
                    let g = PoolGeom {
                        in_h: x.h,
                        in_w: x.w,
                        c: x.c,
                        out_h: y.h,
                        out_w: y.w,
                        kh: op.kh,
                        kw: op.kw,
                        sh: op.sh,
                        sw: op.sw,
                        in_exp: x.exp,
                        out_exp: y.exp,
                    };
                    let (yo, xi, _) = carve(ar, y.range(), x.range(), 0..0);
                    k.avgpool(&g, xi, yo);
                }
                OpKind::Lut | OpKind::Requant => {
                    if x.range() != y.range() {
                        let (yo, xi, _) = carve(ar, y.range(), x.range(), 0..0);
                        yo.copy_from_slice(xi);
                    }
                    let buf = &mut ar[y.range()];
                    if op.kind == OpKind::Lut {
                        k.lut_inplace(buf, self.lut(&op));
                    } else {
                        k.requant_inplace(buf, op.shift);
                    }
                }
                OpKind::Resize => {
                    let g = ResizeGeom { in_h: x.h, in_w: x.w, c: x.c, sh: op.sh, sw: op.sw };
                    let (yo, xi, _) = carve(ar, y.range(), x.range(), 0..0);
                    k.resize_nearest(&g, xi, yo);
                }
                OpKind::Concat => {
                    // every input is disjoint from the output (checked by parse)
                    let (lo, rest) = ar.split_at_mut(y.offset);
                    let (yo, hi) = rest.split_at_mut(y.size);
                    let (lo, hi): (&[i8], &[i8]) = (lo, hi);
                    // NHWC with N = 1: rows = product of the dims before the axis
                    let rows = match op.concat_axis() {
                        1 => 1,
                        2 => y.h,
                        _ => y.h * y.w,
                    };
                    let stride = y.size / rows;
                    let mut offset = 0;
                    for i in 0..op.concat_n() {
                        let t = self.tensor(self.concat_input(&op, i) as usize);
                        let len = t.size / rows;
                        k.concat_copy(pick(lo, hi, &y.range(), t.range()), rows, len, yo, offset, stride);
                        offset += len;
                    }
                }
                OpKind::Add => {
                    let z = self.tensor(op.in1.expect("validated") as usize);
                    if y.range() == x.range() {
                        let (yo, zi, _) = carve(ar, y.range(), z.range(), 0..0);
                        k.add_inplace(yo, zi);
                    } else if y.range() == z.range() {
                        let (yo, xi, _) = carve(ar, y.range(), x.range(), 0..0);
                        k.add_inplace(yo, xi);
                    } else {
                        let (yo, xi, zi) = carve(ar, y.range(), x.range(), z.range());
                        yo.copy_from_slice(xi);
                        k.add_inplace(yo, zi);
                    }
                }
            }
            hook(i, &op, &ar[y.range()]);
        }
        Ok(&as_i8_mut(arena)[self.output().range()])
    }

    /// The logits of the last run inside `arena`.
    pub fn logits<'b>(&self, arena: &'b [u8]) -> &'b [i8] {
        &as_i8(arena)[self.output().range()]
    }

    /// Greedy CTC decode of the logits in `arena` into `ids` (returns the count written).
    /// A raw-output model has no CTC head: the result is always 0.
    pub fn decode(&self, arena: &[u8], ids: &mut [u16]) -> usize {
        if self.raw_output() {
            return 0;
        }
        let o = self.output();
        crate::ctc::ctc_greedy(self.logits(arena), o.h * o.w, o.c, self.num_classes(), self.blank(), ids)
    }
}
