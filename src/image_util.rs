use image::{imageops::FilterType, DynamicImage, EncodableLayout};
use sha2::Digest;
use std::fmt::Write;

pub fn downsample_image(out_resolution: u32, img: DynamicImage) -> DynamicImage {
    let in_width = img.width();
    let in_height = img.height();
    let scale = (out_resolution as f64) / (in_width.min(in_height) as f64);
    if scale < 1.0 {
        img.resize(
            ((in_width as f64) * scale) as u32,
            ((in_height as f64) * scale) as u32,
            FilterType::Lanczos3,
        )
    } else {
        img
    }
}

pub fn hash_image(resolution: u32, img: &DynamicImage) -> String {
    let orig_width = img.width();
    let orig_height = img.height();
    let small_img = img
        .resize_exact(resolution, resolution, FilterType::Lanczos3)
        .into_rgb8();

    let mut bytes = Vec::new();
    for px in small_img.as_bytes() {
        // Quantize each color to allow some wiggle room.
        bytes.push(px >> 4);
    }

    // Bin the aspect ratio to make sure we don't match very
    // differently sized images.
    let log_aspect_ratio = ((((orig_width as f64) / (orig_height as f64)).log2())
        .clamp(-4.0, 4.0)
        .round()
        + 4.0) as u8;
    bytes.push(log_aspect_ratio);

    let mut hasher = sha2::Sha256::new();
    hasher.update(&bytes);
    let mut res = String::with_capacity(64);
    for ch in hasher.finalize() {
        write!(&mut res, "{:02x}", ch).unwrap();
    }
    res
}
