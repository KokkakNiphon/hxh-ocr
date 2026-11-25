//! Cargo feature `profile`: per-op wall time of a run, with a millisecond clock imported from
//! JS (`env.hxh_now`, e.g. `performance.now`). Dev builds only: a release module must not
//! need imports.

use crate::{run_with_hook, st};

#[link(wasm_import_module = "env")]
unsafe extern "C" {
    fn hxh_now() -> f64;
}

/// Fields per op in the `hxh_profile` result.
const FIELDS: usize = 9;

/// Run `reps` times on the current image buffer (with the selected kernels) and return a
/// pointer to `n_ops * 9` f64 per op: kind (1 Conv, 2 DwConv, 3 Add, 4 Lut, 5 Requant,
/// 6 AvgPool), kh, kw, sh, sw, input channels, output pixels, output channels, and the
/// fastest of the `reps` times in ms (robust to other load on the machine).
/// Null if no model is loaded. Re-read `memory.buffer` afterwards (the first call allocates).
#[unsafe(no_mangle)]
pub extern "C" fn hxh_profile(reps: u32) -> *const f64 {
    static mut OUT: *mut f64 = core::ptr::null_mut();
    let s = st();
    let Some(m) = s.model else { return core::ptr::null() };
    let n = m.n_ops();
    // SAFETY: single-threaded; OUT is only touched here and sized for this model's ops
    // (one model per instance).
    let out = unsafe {
        if OUT.is_null() {
            OUT = crate::hxh_alloc(n * FIELDS * 8).cast::<f64>();
        }
        core::slice::from_raw_parts_mut(OUT, n * FIELDS)
    };
    out.fill(f64::INFINITY);
    // SAFETY: allocated by hxh_load with these sizes.
    let (arena, image) = unsafe {
        (core::slice::from_raw_parts_mut(s.arena, m.arena_size()), core::slice::from_raw_parts(s.image, m.input().size))
    };
    for _ in 0..reps {
        if m.set_input_u8(image, arena).is_err() {
            return core::ptr::null();
        }
        // SAFETY: imported pure function
        let mut last = unsafe { hxh_now() };
        let r = run_with_hook(&m, arena, s.simd, |i, _, _| {
            let now = unsafe { hxh_now() };
            let t = &mut out[i * FIELDS + 8];
            *t = t.min(now - last);
            last = now;
        });
        if r.is_err() {
            return core::ptr::null();
        }
    }
    for i in 0..n {
        let op = m.op(i);
        let (x, y) = (m.tensor(op.in0 as usize), m.tensor(op.out as usize));
        let kind = op.kind as u8 as f64 + 1.0;
        let f = [kind, op.kh as f64, op.kw as f64, op.sh as f64, op.sw as f64, x.c as f64, (y.h * y.w) as f64, y.c as f64];
        out[i * FIELDS..i * FIELDS + 8].copy_from_slice(&f);
    }
    out.as_ptr()
}

/// Number of ops of the loaded model (0 if none): the row count of `hxh_profile`.
#[unsafe(no_mangle)]
pub extern "C" fn hxh_op_count() -> u32 {
    st().model.map_or(0, |m| m.n_ops() as u32)
}
