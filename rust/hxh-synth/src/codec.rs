//! Grayscale JPEG encode / decode.

use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;
use zune_jpeg::JpegDecoder;

pub fn encode_jpeg(px: &[u8], w: usize, h: usize, quality: u8) -> Vec<u8> {
    let mut buf = Vec::with_capacity(w * h / 4);
    let enc = jpeg_encoder::Encoder::new(&mut buf, quality);
    enc.encode(px, w as u16, h as u16, jpeg_encoder::ColorType::Luma).expect("jpeg encode");
    buf
}

/// Decode to 8-bit grayscale: (pixels, width, height).
pub fn decode_jpeg_gray(data: &[u8]) -> Option<(Vec<u8>, usize, usize)> {
    let opts = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::Luma);
    let mut d = JpegDecoder::new_with_options(ZCursor::new(data), opts);
    let px = d.decode().ok()?;
    let info = d.info()?;
    Some((px, info.width as usize, info.height as usize))
}
