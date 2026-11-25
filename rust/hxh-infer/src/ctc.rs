//! Greedy CTC decoding.

/// Greedy CTC over int8 logits laid out `[T][stride]`: per timestep take the argmax of the
/// first `num_classes` values (lowest index wins a tie, like `numpy.argmax`), collapse
/// repeats, drop `blank`. Writes at most `out.len()` ids and returns how many it wrote.
pub fn ctc_greedy(logits: &[i8], t: usize, stride: usize, num_classes: usize, blank: usize, out: &mut [u16]) -> usize {
    let mut n = 0;
    let mut prev = usize::MAX;
    for ti in 0..t {
        let row = &logits[ti * stride..ti * stride + num_classes];
        let mut best = 0;
        let mut best_v = row[0];
        for (i, &v) in row.iter().enumerate().skip(1) {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        if best != prev && best != blank {
            if n == out.len() {
                return n;
            }
            out[n] = best as u16;
            n += 1;
        }
        prev = best;
    }
    n
}
