//! `.hxhm` parsing and validation (layout documented in `rust/FORMAT.md`).
//!
//! [`Model::parse`] checks every offset, shape, exponent and arena range once, so that
//! inference afterwards cannot index out of bounds or alias tensors unexpectedly.

use crate::kernels::Bias;

pub const MAGIC: [u8; 4] = *b"HXHM";
/// Newest format version this engine reads (and the converter writes when a model needs it).
pub const VERSION: u16 = 2;
/// Oldest format version this engine reads. Version 1 files have neither the Resize / Concat
/// ops nor the raw-output flag; the converter still writes version 1 for such models.
pub const MIN_VERSION: u16 = 1;
pub const HEADER_SIZE: usize = 64;
pub const TENSOR_REC: usize = 16;
pub const OP_REC: usize = 40;
pub const NONE16: u16 = 0xFFFF;

/// Header flag: the arena plan reuses the input tensor's memory, so `run` overwrites the
/// input and every run needs a fresh `set_input_u8` / `input_mut`.
pub const HDR_INPUT_CLOBBERED: u32 = 1;
/// Header flag (version 2): the output is a raw tensor (e.g. detector maps), not CTC logits;
/// [`Model::decode`] returns 0 ids for such a model.
pub const HDR_RAW_OUTPUT: u32 = 2;

pub const FLAG_RELU: u8 = 1;
pub const FLAG_REDUCE_MEAN: u8 = 2;

/// Largest accepted |exponent| of a tensor or filter (keeps every shift well inside i32).
const MAX_EXP: i32 = 24;
/// Largest accepted |bias|: with MAX_TAPS below the int32 accumulator cannot overflow.
const MAX_BIAS: i32 = 1 << 30;
/// Largest kh*kw*cin of a conv: |sum x*w| <= 2^15 * 2^14 = 2^29.
const MAX_TAPS: usize = 1 << 15;
/// Most inputs of one Concat op.
pub const MAX_CONCAT_INPUTS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// File shorter than its tables or blobs claim.
    Truncated,
    BadMagic,
    BadVersion(u16),
    /// Inconsistent header field (header size, file size, tensor ids, LUT offset).
    BadHeader,
    /// Tensor record `n` is inconsistent (size, exponent, arena range).
    BadTensor(u16),
    /// Op record `n` is inconsistent (kind, shapes, exponents, data ranges, aliasing).
    BadOp(u16),
    /// The arena passed to `run` is smaller than [`Model::arena_size`].
    ArenaTooSmall { need: usize, have: usize },
    /// Input image / tensor has the wrong number of elements.
    InputSize { need: usize, have: usize },
    /// `set_input_u8` on a model without an input LUT.
    NoInputLut,
}

/// One activation tensor: NHWC with N = 1, stored contiguously in the arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub h: usize,
    pub w: usize,
    pub c: usize,
    /// Power-of-two exponent: real value = int8 * 2^exp.
    pub exp: i8,
    /// Byte offset in the arena.
    pub offset: usize,
    /// h * w * c bytes.
    pub size: usize,
}

impl TensorInfo {
    #[inline]
    pub fn range(&self) -> core::ops::Range<usize> {
        self.offset..self.offset + self.size
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpKind {
    /// Regular conv (group 1), weights (Cp/16)[kh][kw][cin][16].
    Conv,
    /// Depthwise conv (group = C), weights (Cp/16)[kh][kw][16].
    DwConv,
    /// Saturating int8 add, all three exponents equal.
    Add,
    /// 256-entry int8 table, `y = table[x + 128]` (e.g. ReLU after an Add).
    Lut,
    /// int8 -> int8 at another exponent: `shift = out_exp - in_exp`.
    Requant,
    /// Average pool without padding (also ReduceMean over H or H,W).
    AvgPool,
    /// Nearest-neighbour upsampling by integer factors `sh`, `sw` (version 2):
    /// `y[oy][ox][c] = x[oy / sh][ox / sw][c]`, input and output exponents equal.
    Resize,
    /// Concatenation of 2..=64 inputs along one NHWC axis (version 2): a pure copy, every
    /// input exponent equal to the output's. The record stores the input count in `kh`, the
    /// axis (1 = H, 2 = W, 3 = C) in `kw`, and the input tensor ids as a u16 blob; see
    /// [`Op::concat_n`], [`Op::concat_axis`] and [`Model::concat_input`].
    Concat,
}

impl OpKind {
    fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            1 => Self::Conv,
            2 => Self::DwConv,
            3 => Self::Add,
            4 => Self::Lut,
            5 => Self::Requant,
            6 => Self::AvgPool,
            7 => Self::Resize,
            8 => Self::Concat,
            _ => return None,
        })
    }
}

/// A decoded op record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Op {
    pub kind: OpKind,
    pub flags: u8,
    pub in0: u16,
    /// Second input (Add only).
    pub in1: Option<u16>,
    pub out: u16,
    pub kh: usize,
    pub kw: usize,
    pub sh: usize,
    pub sw: usize,
    /// top, left, bottom, right
    pub pad: [usize; 4],
    /// Conv: out_exp - in_exp - w_exp. Requant: out_exp - in_exp. Positive = right shift.
    pub shift: i8,
    pub w_exp: i8,
    /// Conv output channels rounded up to a multiple of 16 (weights and bias are padded).
    pub cout_padded: usize,
    pub data_off: usize,
    pub data_len: usize,
    pub bias_off: usize,
}

impl Op {
    #[inline]
    pub fn relu(&self) -> bool {
        self.flags & FLAG_RELU != 0
    }

    /// Number of inputs of a Concat op (stored in the `kh` field).
    #[inline]
    pub fn concat_n(&self) -> usize {
        self.kh
    }

    /// NHWC axis of a Concat op: 1 = H, 2 = W, 3 = C (stored in the `kw` field).
    #[inline]
    pub fn concat_axis(&self) -> usize {
        self.kw
    }
}

/// A parsed, validated `.hxhm` model borrowing the file bytes.
#[derive(Clone, Copy, Debug)]
pub struct Model<'a> {
    data: &'a [u8],
    n_tensors: usize,
    n_ops: usize,
    input: u16,
    output: u16,
    arena_size: usize,
    tensor_table: usize,
    op_table: usize,
    input_lut: usize,
    num_classes: usize,
    blank: usize,
    macs: u64,
    flags: u32,
    version: u16,
}

#[inline]
fn rd_u16(d: &[u8], o: usize) -> Result<u16, Error> {
    let b = d.get(o..o + 2).ok_or(Error::Truncated)?;
    Ok(u16::from_le_bytes([b[0], b[1]]))
}

#[inline]
fn rd_u32(d: &[u8], o: usize) -> Result<u32, Error> {
    let b = d.get(o..o + 4).ok_or(Error::Truncated)?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

#[inline]
fn rd_u64(d: &[u8], o: usize) -> Result<u64, Error> {
    Ok(rd_u32(d, o)? as u64 | (rd_u32(d, o + 4)? as u64) << 32)
}

#[inline]
fn rd_u8(d: &[u8], o: usize) -> Result<u8, Error> {
    d.get(o).copied().ok_or(Error::Truncated)
}

fn disjoint(a: &core::ops::Range<usize>, b: &core::ops::Range<usize>) -> bool {
    a.end <= b.start || b.end <= a.start
}

impl<'a> Model<'a> {
    /// Parse and fully validate a `.hxhm` file.
    pub fn parse(data: &'a [u8]) -> Result<Self, Error> {
        if data.len() < HEADER_SIZE {
            return Err(Error::Truncated);
        }
        if data[0..4] != MAGIC {
            return Err(Error::BadMagic);
        }
        let version = rd_u16(data, 4)?;
        if !(MIN_VERSION..=VERSION).contains(&version) {
            return Err(Error::BadVersion(version));
        }
        if rd_u16(data, 6)? as usize != HEADER_SIZE {
            return Err(Error::BadHeader);
        }
        let file_size = rd_u32(data, 8)? as usize;
        if file_size > data.len() {
            return Err(Error::Truncated);
        }
        // Trailing bytes after file_size are ignored (e.g. a padded buffer).
        let data = &data[..file_size];
        let m = Model {
            data,
            n_tensors: rd_u16(data, 12)? as usize,
            n_ops: rd_u16(data, 14)? as usize,
            input: rd_u16(data, 16)?,
            output: rd_u16(data, 18)?,
            arena_size: rd_u32(data, 20)? as usize,
            tensor_table: rd_u32(data, 24)? as usize,
            op_table: rd_u32(data, 28)? as usize,
            input_lut: rd_u32(data, 32)? as usize,
            num_classes: rd_u16(data, 36)? as usize,
            blank: rd_u16(data, 38)? as usize,
            macs: rd_u64(data, 40)?,
            flags: rd_u32(data, 48)?,
            version,
        };
        // unknown flag bits, or the raw-output flag in a version 1 file
        let known = if version >= 2 { HDR_INPUT_CLOBBERED | HDR_RAW_OUTPUT } else { HDR_INPUT_CLOBBERED };
        if m.flags & !known != 0 {
            return Err(Error::BadHeader);
        }
        let tt_end = m.tensor_table.checked_add(m.n_tensors * TENSOR_REC).ok_or(Error::BadHeader)?;
        let op_end = m.op_table.checked_add(m.n_ops * OP_REC).ok_or(Error::BadHeader)?;
        if m.tensor_table < HEADER_SIZE || m.op_table < HEADER_SIZE || tt_end > data.len() || op_end > data.len() {
            return Err(Error::Truncated);
        }
        if m.input as usize >= m.n_tensors || m.output as usize >= m.n_tensors || m.input == m.output {
            return Err(Error::BadHeader);
        }
        if m.input_lut != 0 && (m.input_lut < HEADER_SIZE || m.input_lut + 256 > data.len()) {
            return Err(Error::BadHeader);
        }
        for i in 0..m.n_tensors {
            m.check_tensor(i)?;
        }
        let out = m.read_tensor(m.output as usize)?;
        if m.num_classes == 0 || m.num_classes > out.c || m.blank >= m.num_classes {
            return Err(Error::BadHeader);
        }
        let input_range = m.read_tensor(m.input as usize)?.range();
        for k in 0..m.n_ops {
            m.check_op(k).map_err(|_| Error::BadOp(k as u16))?;
            if m.input_preserved() {
                let y = m.read_tensor(m.read_op(k)?.out as usize)?;
                if !disjoint(&y.range(), &input_range) {
                    return Err(Error::BadOp(k as u16));
                }
            }
        }
        Ok(m)
    }

    fn read_tensor(&self, i: usize) -> Result<TensorInfo, Error> {
        let o = self.tensor_table + i * TENSOR_REC;
        Ok(TensorInfo {
            h: rd_u16(self.data, o)? as usize,
            w: rd_u16(self.data, o + 2)? as usize,
            c: rd_u16(self.data, o + 4)? as usize,
            exp: rd_u8(self.data, o + 6)? as i8,
            offset: rd_u32(self.data, o + 8)? as usize,
            size: rd_u32(self.data, o + 12)? as usize,
        })
    }

    fn check_tensor(&self, i: usize) -> Result<(), Error> {
        let bad = Error::BadTensor(i as u16);
        let t = self.read_tensor(i)?;
        let n = t.h.checked_mul(t.w).and_then(|v| v.checked_mul(t.c)).ok_or(bad)?;
        if n == 0 || n != t.size || (t.exp as i32).abs() > MAX_EXP {
            return Err(bad);
        }
        if t.offset.checked_add(t.size).ok_or(bad)? > self.arena_size {
            return Err(bad);
        }
        Ok(())
    }

    fn read_op(&self, k: usize) -> Result<Op, Error> {
        let d = self.data;
        let o = self.op_table + k * OP_REC;
        let kind = OpKind::from_u8(rd_u8(d, o)?).ok_or(Error::BadOp(k as u16))?;
        let in1 = rd_u16(d, o + 4)?;
        Ok(Op {
            kind,
            flags: rd_u8(d, o + 1)?,
            in0: rd_u16(d, o + 2)?,
            in1: if in1 == NONE16 { None } else { Some(in1) },
            out: rd_u16(d, o + 6)?,
            kh: rd_u16(d, o + 8)? as usize,
            kw: rd_u16(d, o + 10)? as usize,
            sh: rd_u16(d, o + 12)? as usize,
            sw: rd_u16(d, o + 14)? as usize,
            pad: [
                rd_u8(d, o + 16)? as usize,
                rd_u8(d, o + 17)? as usize,
                rd_u8(d, o + 18)? as usize,
                rd_u8(d, o + 19)? as usize,
            ],
            shift: rd_u8(d, o + 20)? as i8,
            w_exp: rd_u8(d, o + 21)? as i8,
            cout_padded: rd_u16(d, o + 22)? as usize,
            data_off: rd_u32(d, o + 24)? as usize,
            data_len: rd_u32(d, o + 28)? as usize,
            bias_off: rd_u32(d, o + 32)? as usize,
        })
    }

    fn blob_ok(&self, off: usize, len: usize) -> bool {
        off >= HEADER_SIZE && off.checked_add(len).is_some_and(|e| e <= self.data.len())
    }

    fn check_op(&self, k: usize) -> Result<(), Error> {
        let bad = Error::BadOp(k as u16);
        let op = self.read_op(k)?;
        let tid_ok = |t: u16| (t as usize) < self.n_tensors;
        if !tid_ok(op.in0) || !tid_ok(op.out) || op.out == self.input {
            return Err(bad);
        }
        let x = self.read_tensor(op.in0 as usize)?;
        let y = self.read_tensor(op.out as usize)?;
        let same_shape = |a: &TensorInfo, b: &TensorInfo| a.h == b.h && a.w == b.w && a.c == b.c;
        // in-place is allowed only for elementwise ops and only as an exact alias
        let alias_ok = |a: &TensorInfo| a.range() == y.range() || disjoint(&a.range(), &y.range());
        match op.kind {
            OpKind::Conv | OpKind::DwConv | OpKind::AvgPool => {
                if op.in1.is_some() || !disjoint(&x.range(), &y.range()) {
                    return Err(bad);
                }
                if op.kh == 0 || op.kw == 0 || op.sh == 0 || op.sw == 0 {
                    return Err(bad);
                }
                let hp = x.h + op.pad[0] + op.pad[2];
                let wp = x.w + op.pad[1] + op.pad[3];
                if hp < op.kh || wp < op.kw || (hp - op.kh) / op.sh + 1 != y.h || (wp - op.kw) / op.sw + 1 != y.w {
                    return Err(bad);
                }
                // (a window entirely inside the padding is fine: it yields requant(bias))
            }
            OpKind::Add => {
                let b = op.in1.ok_or(bad)?;
                if !tid_ok(b) {
                    return Err(bad);
                }
                let z = self.read_tensor(b as usize)?;
                if !same_shape(&x, &y) || !same_shape(&z, &y) || x.exp != y.exp || z.exp != y.exp {
                    return Err(bad);
                }
                if !alias_ok(&x) || !alias_ok(&z) || (x.range() == y.range() && z.range() == y.range()) {
                    return Err(bad);
                }
            }
            OpKind::Lut | OpKind::Requant => {
                if op.in1.is_some() || !same_shape(&x, &y) || !alias_ok(&x) {
                    return Err(bad);
                }
            }
            OpKind::Resize => {
                // ESP-DL's int8 nearest Resize is a copy when the exponents are equal (ESP-PPQ
                // aligns its input to the output and inserts RequantizeLinear before it)
                if self.version < 2 || op.in1.is_some() || !disjoint(&x.range(), &y.range()) {
                    return Err(bad);
                }
                if op.kh != 0 || op.kw != 0 || op.sh == 0 || op.sw == 0 || op.pad != [0; 4] {
                    return Err(bad);
                }
                if x.c != y.c || x.exp != y.exp || x.h.checked_mul(op.sh) != Some(y.h) || x.w.checked_mul(op.sw) != Some(y.w) {
                    return Err(bad);
                }
            }
            OpKind::Concat => {
                // ESP-DL's int8 Concat copies bytes and ignores exponents: all must be equal
                let (n, axis) = (op.concat_n(), op.concat_axis());
                if self.version < 2 || op.in1.is_some() || !(2..=MAX_CONCAT_INPUTS).contains(&n) || !(1..=3).contains(&axis) {
                    return Err(bad);
                }
                if op.data_len != 2 * n || !self.blob_ok(op.data_off, op.data_len) || op.sh != 0 || op.sw != 0 {
                    return Err(bad);
                }
                let dims = |t: &TensorInfo| [t.h, t.w, t.c];
                let mut along = 0usize;
                for i in 0..n {
                    let id = rd_u16(self.data, op.data_off + 2 * i)?;
                    if !tid_ok(id) || id == op.out || (i == 0 && id != op.in0) {
                        return Err(bad);
                    }
                    let t = self.read_tensor(id as usize)?;
                    let (dt, dy) = (dims(&t), dims(&y));
                    if (0..3).any(|d| d != axis - 1 && dt[d] != dy[d]) || t.exp != y.exp || !disjoint(&t.range(), &y.range()) {
                        return Err(bad);
                    }
                    along += dt[axis - 1];
                }
                if along != dims(&y)[axis - 1] {
                    return Err(bad);
                }
            }
        }
        match op.kind {
            OpKind::Conv | OpKind::DwConv => {
                let dw = op.kind == OpKind::DwConv;
                let cin = if dw { 1 } else { x.c };
                if dw && x.c != y.c {
                    return Err(bad);
                }
                if op.cout_padded % 16 != 0 || op.cout_padded < y.c || op.cout_padded >= y.c + 16 {
                    return Err(bad);
                }
                // usize is 32 bits on wasm32 / Xtensa: keep every product checked
                let taps = op.kh.checked_mul(op.kw).and_then(|v| v.checked_mul(cin)).ok_or(bad)?;
                if taps > MAX_TAPS
                    || Some(op.data_len) != op.cout_padded.checked_mul(taps)
                    || !self.blob_ok(op.data_off, op.data_len)
                {
                    return Err(bad);
                }
                if !self.blob_ok(op.bias_off, op.cout_padded * 4) {
                    return Err(bad);
                }
                let bias = self.bias(&op);
                if (0..op.cout_padded).any(|i| bias.get(i).unsigned_abs() > MAX_BIAS as u32) {
                    return Err(bad);
                }
                let w_exp = op.w_exp as i32;
                if w_exp.abs() > MAX_EXP || op.shift as i32 != y.exp as i32 - x.exp as i32 - w_exp {
                    return Err(bad);
                }
                if !(-24..=31).contains(&(op.shift as i32)) {
                    return Err(bad);
                }
            }
            OpKind::AvgPool => {
                if x.c != y.c || op.pad != [0; 4] || op.kh.checked_mul(op.kw).is_none_or(|a| a > 1 << 23) {
                    return Err(bad);
                }
            }
            OpKind::Lut => {
                if op.data_len != 256 || !self.blob_ok(op.data_off, 256) {
                    return Err(bad);
                }
            }
            OpKind::Requant => {
                if op.shift as i32 != y.exp as i32 - x.exp as i32 {
                    return Err(bad);
                }
            }
            OpKind::Add | OpKind::Resize | OpKind::Concat => {}
        }
        Ok(())
    }

    // ---------------------------------------------------------------- accessors
    // Everything below reads records that `parse` has already validated.

    /// Number of ops.
    #[inline]
    pub fn n_ops(&self) -> usize {
        self.n_ops
    }

    /// Number of tensors.
    #[inline]
    pub fn n_tensors(&self) -> usize {
        self.n_tensors
    }

    /// Tensor record `i` (`i < n_tensors()`).
    #[inline]
    pub fn tensor(&self, i: usize) -> TensorInfo {
        self.read_tensor(i).expect("validated in parse")
    }

    /// Op record `k` (`k < n_ops()`).
    #[inline]
    pub fn op(&self, k: usize) -> Op {
        self.read_op(k).expect("validated in parse")
    }

    /// Bytes of arena the model needs.
    #[inline]
    pub fn arena_size(&self) -> usize {
        self.arena_size
    }

    /// The model input tensor (int8, NHWC).
    #[inline]
    pub fn input(&self) -> TensorInfo {
        self.tensor(self.input as usize)
    }

    /// The output tensor: CTC logits (int8, NHWC `[1, 1, T, C]`; `C` may include padding
    /// classes), or any NHWC tensor for a raw-output model (see [`Model::raw_output`]).
    #[inline]
    pub fn output(&self) -> TensorInfo {
        self.tensor(self.output as usize)
    }

    /// Classes the decoder looks at (the first `num_classes` output channels).
    #[inline]
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// CTC blank class id.
    #[inline]
    pub fn blank(&self) -> usize {
        self.blank
    }

    /// True when `run` leaves the input tensor intact (the default plan), so the same input
    /// can be run repeatedly. False when the converter let activations reuse its memory.
    #[inline]
    pub fn input_preserved(&self) -> bool {
        self.flags & HDR_INPUT_CLOBBERED == 0
    }

    /// Format version of the file (1 or 2).
    #[inline]
    pub fn version(&self) -> u16 {
        self.version
    }

    /// True for models whose output is a raw tensor (e.g. a text detector's maps) rather than
    /// CTC logits: [`Model::decode`] then returns 0.
    #[inline]
    pub fn raw_output(&self) -> bool {
        self.flags & HDR_RAW_OUTPUT != 0
    }

    /// Input tensor id `i` (`i < op.concat_n()`) of a Concat op.
    #[inline]
    pub fn concat_input(&self, op: &Op, i: usize) -> u16 {
        debug_assert!(op.kind == OpKind::Concat && i < op.concat_n());
        let o = op.data_off + 2 * i;
        u16::from_le_bytes([self.data[o], self.data[o + 1]])
    }

    /// Multiply-accumulates of one inference (convs only).
    #[inline]
    pub fn macs(&self) -> u64 {
        self.macs
    }

    /// Conv filter (hxhm packed layout) or LUT table bytes of `op`.
    #[inline]
    pub fn weights(&self, op: &Op) -> &'a [i8] {
        crate::as_i8(&self.data[op.data_off..op.data_off + op.data_len])
    }

    /// Conv bias (int32 LE, `cout_padded` entries).
    #[inline]
    pub fn bias(&self, op: &Op) -> Bias<'a> {
        Bias::new(&self.data[op.bias_off..op.bias_off + op.cout_padded * 4])
    }

    /// LUT of a `Lut` op.
    #[inline]
    pub fn lut(&self, op: &Op) -> &'a [i8; 256] {
        self.weights(op).try_into().expect("validated in parse")
    }

    /// u8 pixel -> int8 input table (normalisation + input quantisation folded in).
    #[inline]
    pub fn input_lut(&self) -> Option<&'a [i8; 256]> {
        if self.input_lut == 0 {
            return None;
        }
        Some(crate::as_i8(&self.data[self.input_lut..self.input_lut + 256]).try_into().expect("validated"))
    }

    /// The raw file bytes (as passed to `parse`, trimmed to the file size).
    #[inline]
    pub fn bytes(&self) -> &'a [u8] {
        self.data
    }
}
