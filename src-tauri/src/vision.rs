//! Image decoding + basic computer-vision primitives for the culling backend.
//!
//! Everything here is pure-Rust and runs in-process for performance:
//!   * RAW decoding      -> `rawloader` + `imagepipe`
//!   * standard decoding -> `image`
//!   * blur detection    -> Laplacian variance via `imageproc`
//!   * duplicate hashing  -> perceptual hash via `image_hasher`
//!
//! Face landmark / eyes-closed detection is intentionally NOT implemented here
//! yet: pure-Rust options are weak. See the module-level notes in the project
//! README / the chat write-up for the recommended `ort` (ONNX Runtime) vs.
//! Python-sidecar path. `blur_score` and `perceptual_hash` are the two pieces
//! proven end-to-end by the tests below.

use std::path::Path;

use image::{DynamicImage, GrayImage};
use image_hasher::{HashAlg, HasherConfig};
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

/// Perceptual hash (gradient / "dHash" by default) as a base64 string, suitable
/// for near-duplicate detection: visually similar frames yield hashes with a
/// small Hamming distance. Compare two with [`hash_distance`].
pub fn perceptual_hash(img: &DynamicImage) -> String {
    let hasher = HasherConfig::new()
        .hash_alg(HashAlg::Gradient)
        .hash_size(16, 16)
        .to_hasher();
    hasher.hash_image(img).to_base64()
}

/// Hamming distance between two base64 perceptual hashes produced by
/// [`perceptual_hash`]. `0` = identical; larger = more different.
pub fn hash_distance(a: &str, b: &str) -> Result<u32, String> {
    use image_hasher::ImageHash;
    let ha = ImageHash::<Box<[u8]>>::from_base64(a).map_err(|e| format!("{e:?}"))?;
    let hb = ImageHash::<Box<[u8]>>::from_base64(b).map_err(|e| format!("{e:?}"))?;
    Ok(ha.dist(&hb))
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
    pub perceptual_hash: String,
    pub is_raw: bool,
}

/// Tauri command: decode an image from disk and run the CV pipeline on it.
#[tauri::command]
pub fn analyze_image(path: String) -> Result<ImageAnalysis, String> {
    let p = Path::new(&path);
    let img = decode_image(p)?;
    Ok(ImageAnalysis {
        width: img.width(),
        height: img.height(),
        blur_score: blur_score(&img),
        perceptual_hash: perceptual_hash(&img),
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
    fn perceptual_hash_is_stable_and_discriminating() {
        let sharp = DynamicImage::ImageRgb8(sharp_checkerboard(128));
        let blurred = DynamicImage::ImageRgb8(imageproc::filter::gaussian_blur_f32(
            sharp.as_rgb8().unwrap(),
            4.0,
        ));

        let h_sharp = perceptual_hash(&sharp);
        let h_sharp_again = perceptual_hash(&sharp);
        let h_blurred = perceptual_hash(&blurred);

        // Deterministic: same image -> same hash (distance 0).
        assert_eq!(hash_distance(&h_sharp, &h_sharp_again).unwrap(), 0);
        // Discriminating: a materially different image -> nonzero distance.
        assert!(hash_distance(&h_sharp, &h_blurred).unwrap() > 0);
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

    /// Decodes a real RAW file when one is available. Ignored by default because
    /// the test photo set doesn't exist yet. Run once you have RAW samples:
    ///
    ///   $env:CULLING_TEST_RAW = "C:\path\to\photo.cr3"
    ///   cargo test -p tauri-app decode_real_raw -- --ignored --nocapture
    #[test]
    #[ignore = "requires a RAW sample; set CULLING_TEST_RAW to enable"]
    fn decode_real_raw() {
        let path = std::env::var("CULLING_TEST_RAW")
            .expect("set CULLING_TEST_RAW to a RAW file path");
        let img = decode_image(Path::new(&path)).expect("RAW decode failed");
        assert!(img.width() > 0 && img.height() > 0);
        println!(
            "decoded RAW {}x{}, blur_score={:.2}",
            img.width(),
            img.height(),
            blur_score(&img)
        );
    }
}
