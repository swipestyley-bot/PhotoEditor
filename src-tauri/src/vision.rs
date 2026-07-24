//! Image decoding + single-image quality metrics for the culling backend.
//!
//! Pure-Rust, in-process:
//!   * RAW decoding    -> `rawloader` + `imagepipe`
//!   * standard decode -> `image`
//!   * blur detection  -> Laplacian variance via `imageproc`
//!
//! Duplicate detection (perceptual hashing, clustering, EXIF) lives in
//! `dedup.rs`; face / eyes-closed analysis in `face.rs`.

use std::path::Path;

use image::{imageops::FilterType, DynamicImage, GrayImage};
use serde::Serialize;

/// File extensions we route through the RAW decoder rather than the standard
/// `image` decoders. Lower-cased before comparison.
const RAW_EXTENSIONS: &[&str] = &[
    "cr2", "cr3", "crw", // Canon
    "nef", "nrw", // Nikon
    "arw", "srf", "sr2", // Sony
    "dng", // Adobe / generic
    "raf", // Fujifilm
    "rw2", // Panasonic
    "orf", // Olympus
    "pef", // Pentax
    "srw", // Samsung
    "raw", "mrw", // Misc / Minolta
];

/// Returns true if the path's extension is a known RAW format.
pub fn is_raw_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| RAW_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Decode any supported image (RAW or standard) into an in-memory `DynamicImage`.
///
/// RAW files are demosaiced to 8-bit sRGB; everything else goes through the
/// `image` crate's decoders (JPEG/PNG/TIFF/WebP/...).
pub fn decode_image(path: &Path) -> Result<DynamicImage, String> {
    if is_raw_path(path) {
        decode_raw(path)
    } else {
        image::open(path).map_err(|e| format!("failed to decode {}: {e}", path.display()))
    }
}

/// Decode a camera RAW file to a full-resolution 8-bit sRGB image.
///
/// `imagepipe::simple_decode_8bit(path, 0, 0)` decodes at native resolution
/// (0/0 = no downscale) and returns tightly packed RGB8 bytes, which we wrap
/// back into an `image` 0.25 `RgbImage` (imagepipe uses `image` 0.24 only
/// internally; the public boundary is a plain `Vec<u8>`, so the two versions
/// never meet in the type system).
fn decode_raw(path: &Path) -> Result<DynamicImage, String> {
    let decoded = imagepipe::simple_decode_8bit(path, 0, 0)
        .map_err(|e| format!("RAW decode failed for {}: {e}", path.display()))?;

    let expected = decoded.width * decoded.height * 3;
    if decoded.data.len() != expected {
        return Err(format!(
            "RAW decode produced {} bytes, expected {} for {}x{} RGB",
            decoded.data.len(),
            expected,
            decoded.width,
            decoded.height
        ));
    }

    let buffer = image::RgbImage::from_raw(decoded.width as u32, decoded.height as u32, decoded.data)
        .ok_or_else(|| "RAW buffer/dimension mismatch".to_string())?;
    Ok(DynamicImage::ImageRgb8(buffer))
}

/// Blur score = variance of the Laplacian (a.k.a. "variance of Laplacian" /
/// Pech-Pacheco focus measure). Higher = sharper, lower = blurrier.
///
/// This is the same metric OpenCV users compute with
/// `cv2.Laplacian(img, CV_64F).var()`. We run it on the luma channel.
pub fn blur_score(img: &DynamicImage) -> f64 {
    let gray: GrayImage = img.to_luma8();
    let laplacian = imageproc::filter::laplacian_filter(&gray); // Luma<i16>
    variance(laplacian.as_raw())
}

/// Default blur-score threshold below which an image is flagged blurry.
/// VALIDATE: the Laplacian-variance scale is strongly resolution- and
/// content-dependent (a sharp full-res frame scores far higher than a
/// downscaled one), so calibrate this against your real photo set.
pub const DEFAULT_BLUR_THRESHOLD: f64 = 100.0;

/// Blur verdict for an image: the raw focus measure plus a threshold decision.
/// Mirrors the score+threshold+verdict shape of `face::EyesState`.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct BlurAssessment {
    pub score: f64,
    pub is_blurry: bool,
}

/// Score an image's sharpness and classify it against `threshold`
/// (see [`DEFAULT_BLUR_THRESHOLD`]).
pub fn assess_blur(img: &DynamicImage, threshold: f64) -> BlurAssessment {
    let score = blur_score(img);
    BlurAssessment {
        score,
        is_blurry: score < threshold,
    }
}

/// Clamp a sub-rectangle to the image bounds and return that crop, or `None`
/// for a zero-sized image.
fn clamp_crop(img: &DynamicImage, x: u32, y: u32, w: u32, h: u32) -> Option<DynamicImage> {
    let (iw, ih) = (img.width(), img.height());
    if iw == 0 || ih == 0 {
        return None;
    }
    let x = x.min(iw - 1);
    let y = y.min(ih - 1);
    let w = w.min(iw - x).max(1);
    let h = h.min(ih - y).max(1);
    Some(img.crop_imm(x, y, w, h))
}

/// Canonical long-edge (px) that normalized blur scores resize to before
/// measuring. Variance-of-Laplacian is **scale-dependent**: a sharp close-up
/// face is mostly smooth-skin pixels that dilute its few edges, so it scores far
/// lower than an equally sharp *distant* face whose edges are packed densely.
/// Cross-photo comparison therefore has to normalize scale first. 400px was
/// calibrated on the test set — it collapses the close-up-vs-distant gap while
/// only ever downscaling real face crops (never upscaling, which would inject
/// interpolation blur).
pub const BLUR_NORM_LONG_EDGE: u32 = 400;

/// Downscale `img` so its long edge is at most [`BLUR_NORM_LONG_EDGE`]. Crops
/// already at/under that size pass through unchanged (no upscaling).
fn normalize_for_blur(img: &DynamicImage) -> DynamicImage {
    if img.width().max(img.height()) > BLUR_NORM_LONG_EDGE {
        img.resize(BLUR_NORM_LONG_EDGE, BLUR_NORM_LONG_EDGE, FilterType::Lanczos3)
    } else {
        img.clone()
    }
}

/// Raw variance-of-Laplacian over a sub-rectangle (clamped to bounds).
///
/// NOTE: this is **scale-dependent** (see [`BLUR_NORM_LONG_EDGE`]) — prefer
/// [`blur_score_region_normalized`] when comparing scores across photos. Kept as
/// a building block and for diagnostics.
pub fn blur_score_region(img: &DynamicImage, x: u32, y: u32, w: u32, h: u32) -> f64 {
    clamp_crop(img, x, y, w, h)
        .map(|c| blur_score(&c))
        .unwrap_or(0.0)
}

/// Scale-normalized blur score over a sub-rectangle: crop the region (e.g. a
/// detected face box), normalize its size ([`normalize_for_blur`]), then measure
/// variance-of-Laplacian. This makes a close-up and a distant face of equal true
/// sharpness score comparably, which is what fixes portrait-mode false positives.
pub fn blur_score_region_normalized(img: &DynamicImage, x: u32, y: u32, w: u32, h: u32) -> f64 {
    clamp_crop(img, x, y, w, h)
        .map(|c| blur_score(&normalize_for_blur(&c)))
        .unwrap_or(0.0)
}

/// [`assess_blur`] over a scale-normalized sub-rectangle (see
/// [`blur_score_region_normalized`]).
pub fn assess_blur_region_normalized(
    img: &DynamicImage,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    threshold: f64,
) -> BlurAssessment {
    let score = blur_score_region_normalized(img, x, y, w, h);
    BlurAssessment {
        score,
        is_blurry: score < threshold,
    }
}

fn variance(data: &[i16]) -> f64 {
    let n = data.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mean = data.iter().map(|&v| v as f64).sum::<f64>() / n;
    data.iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n
}

/// Result of analysing a single image, returned to the frontend.
#[derive(Debug, Clone, Serialize)]
pub struct ImageAnalysis {
    pub width: u32,
    pub height: u32,
    pub blur_score: f64,
    pub is_blurry: bool,
    pub perceptual_hash: String,
    pub is_raw: bool,
}

/// Tauri command: decode an image from disk and run the single-image metrics.
/// The perceptual hash comes from [`crate::dedup`] (pHash) for duplicate
/// detection.
#[tauri::command]
pub fn analyze_image(path: String) -> Result<ImageAnalysis, String> {
    let p = Path::new(&path);
    let img = decode_image(p)?;
    let blur = assess_blur(&img, DEFAULT_BLUR_THRESHOLD);
    Ok(ImageAnalysis {
        width: img.width(),
        height: img.height(),
        blur_score: blur.score,
        is_blurry: blur.is_blurry,
        perceptual_hash: crate::dedup::perceptual_hash(&img),
        is_raw: is_raw_path(p),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    /// A high-frequency checkerboard: maximum high-frequency content, so its
    /// Laplacian variance (blur score) is very high.
    fn sharp_checkerboard(size: u32) -> RgbImage {
        RgbImage::from_fn(size, size, |x, y| {
            if (x + y) % 2 == 0 {
                Rgb([255, 255, 255])
            } else {
                Rgb([0, 0, 0])
            }
        })
    }

    #[test]
    fn decode_from_disk_roundtrips() {
        // Write a PNG placeholder to a temp path and decode it back through the
        // real file-based path (`decode_image`), confirming decoding works.
        let dir = std::env::temp_dir();
        let path = dir.join("culling_pipeline_decode_test.png");
        let original = sharp_checkerboard(64);
        original
            .save(&path)
            .expect("failed to write test PNG");

        let decoded = decode_image(&path).expect("decode_image failed");
        assert_eq!(decoded.width(), 64);
        assert_eq!(decoded.height(), 64);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn blur_pipeline_ranks_sharp_above_blurred() {
        // End-to-end CV check: a sharp image must score higher than a blurred
        // copy of the same image. This validates the Laplacian-variance metric,
        // not just that the code runs.
        let sharp = DynamicImage::ImageRgb8(sharp_checkerboard(128));
        let blurred = DynamicImage::ImageRgb8(imageproc::filter::gaussian_blur_f32(
            sharp.as_rgb8().unwrap(),
            4.0,
        ));

        let sharp_score = blur_score(&sharp);
        let blurred_score = blur_score(&blurred);

        println!("sharp blur_score   = {sharp_score:.2}");
        println!("blurred blur_score = {blurred_score:.2}");

        assert!(sharp_score > 0.0, "sharp image should have signal");
        assert!(
            sharp_score > blurred_score * 5.0,
            "expected sharp ({sharp_score:.2}) >> blurred ({blurred_score:.2})"
        );
    }

    #[test]
    fn assess_blur_flags_blurred_not_sharp() {
        let sharp = DynamicImage::ImageRgb8(sharp_checkerboard(128));
        let blurred = DynamicImage::ImageRgb8(imageproc::filter::gaussian_blur_f32(
            sharp.as_rgb8().unwrap(),
            4.0,
        ));
        // A threshold between the two scores classifies each correctly.
        let sharp_a = assess_blur(&sharp, 100.0);
        let blurred_a = assess_blur(&blurred, 100.0);
        println!("sharp={sharp_a:?} blurred={blurred_a:?}");
        assert!(!sharp_a.is_blurry, "sharp should not be flagged blurry");
        assert!(blurred_a.is_blurry, "blurred should be flagged blurry");
    }

    #[test]
    fn raw_extension_detection() {
        assert!(is_raw_path(Path::new("DSC_0001.NEF")));
        assert!(is_raw_path(Path::new("IMG.cr3")));
        assert!(is_raw_path(Path::new("photo.dng")));
        assert!(!is_raw_path(Path::new("photo.jpg")));
        assert!(!is_raw_path(Path::new("photo.png")));
        assert!(!is_raw_path(Path::new("noext")));
    }

    /// Decodes a real RAW file and confirms it produces a *viewable* image:
    /// plausible dimensions, real content (non-trivial luma variance — a failed
    /// decode tends to be uniform/black), and writes a PNG next to the RAW so you
    /// can eyeball the result. Ignored until you supply a sample.
    ///
    /// NOTE: RAW decoding here is pure-Rust `rawloader` + `imagepipe`, NOT libraw.
    ///
    ///   $env:CULLING_TEST_RAW = "C:\path\to\photo.cr3"
    ///   cargo test -p tauri-app decode_real_raw -- --ignored --nocapture
    #[test]
    #[ignore = "requires a real RAW file; set CULLING_TEST_RAW"]
    fn decode_real_raw() {
        let path = std::env::var("CULLING_TEST_RAW").expect("set CULLING_TEST_RAW to a RAW file path");
        let raw_path = Path::new(&path);
        assert!(is_raw_path(raw_path), "CULLING_TEST_RAW should point to a RAW file");

        let img = decode_image(raw_path).expect("RAW decode failed");
        let (w, h) = (img.width(), img.height());
        assert!(w >= 64 && h >= 64, "implausible RAW dimensions {w}x{h}");

        // "Viewable" content check: real photos have luminance variation; a
        // broken decode is typically uniform/black (Laplacian variance ~0).
        let content = blur_score(&img);
        assert!(content > 0.0, "decoded RAW looks uniform/blank (content score {content})");

        // VALIDATE: eyeball this PNG — confirm correct colours (no R/B swap),
        // orientation, and reasonable exposure/white balance from imagepipe.
        let out = raw_path.with_extension("decoded.png");
        img.save(&out).expect("failed to save decoded PNG");
        println!("decoded RAW {w}x{h}, content score {content:.1}");
        println!("wrote viewable PNG: {}", out.display());
    }
}
