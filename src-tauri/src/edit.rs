//! Offline auto-editing: exposure, white balance, and tone/contrast, each with
//! an independent 0–100% strength. Pure functions over `image::DynamicImage` —
//! no network, no external services.
//!
//! Each correction computes a full-strength target for every pixel, then blends
//! it with the original by `strength` (so it's never all-or-nothing). All three
//! are per-image *adaptive*: they read the image's own histogram/statistics, so
//! one strength setting corrects each photo to its own content.

use image::DynamicImage;
use serde::Deserialize;

/// Rec.601 luma from 8-bit RGB, in `0..=255`.
fn luma(r: u8, g: u8, b: u8) -> f32 {
    0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32
}

/// Linear blend from `orig` toward `corrected` by `s` (0..1).
fn blend(orig: f32, corrected: f32, s: f32) -> f32 {
    orig + (corrected - orig) * s
}

fn to_u8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

/// strength% (0..100) -> factor (0..1), clamped.
fn strength01(strength: f32) -> f32 {
    (strength / 100.0).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// 1. Auto-exposure (histogram mean toward mid-gray)
// ---------------------------------------------------------------------------

/// Mean luma (0..1) that auto-exposure aims for. Mid-gray.
/// VALIDATE: 0.5 is a neutral target; tune against real shots — some prefer
/// ~0.45 for a slightly moodier baseline.
const EXPOSURE_TARGET: f32 = 0.5;
/// Clamp the exposure gain so a very dark/bright frame can't be pushed absurdly.
const EXPOSURE_GAIN_MIN: f32 = 0.33;
const EXPOSURE_GAIN_MAX: f32 = 3.0;

/// Brighten/darken toward a balanced mean exposure. Computes the mean luma from
/// the histogram and applies a (clamped) multiplicative gain to bring it toward
/// [`EXPOSURE_TARGET`], blended by `strength`.
pub fn auto_exposure(img: &DynamicImage, strength: f32) -> DynamicImage {
    let s = strength01(strength);
    let mut rgb = img.to_rgb8();
    let n = (rgb.width() as usize) * (rgb.height() as usize);
    if s == 0.0 || n == 0 {
        return DynamicImage::ImageRgb8(rgb);
    }

    let mut sum = 0.0f32;
    for p in rgb.pixels() {
        sum += luma(p[0], p[1], p[2]);
    }
    let mean = sum / n as f32 / 255.0; // 0..1
    if mean <= 0.0 {
        return DynamicImage::ImageRgb8(rgb);
    }

    let gain = (EXPOSURE_TARGET / mean).clamp(EXPOSURE_GAIN_MIN, EXPOSURE_GAIN_MAX);
    // Blending a multiply is itself a multiply: lerp(v, v*gain, s) = v*(1+s(gain-1)).
    let eff = 1.0 + s * (gain - 1.0);
    for p in rgb.pixels_mut() {
        p[0] = to_u8(p[0] as f32 * eff);
        p[1] = to_u8(p[1] as f32 * eff);
        p[2] = to_u8(p[2] as f32 * eff);
    }
    DynamicImage::ImageRgb8(rgb)
}

// ---------------------------------------------------------------------------
// 2. Auto white balance (gray-world)
// ---------------------------------------------------------------------------

/// Clamp per-channel white-balance gains so a strongly tinted frame can't invert.
const WB_GAIN_MIN: f32 = 0.5;
const WB_GAIN_MAX: f32 = 2.0;

/// Correct color casts via the gray-world assumption: the average of R, G, B
/// should be equal. Scales each channel so its mean moves toward the overall
/// gray mean, blended by `strength`.
pub fn auto_white_balance(img: &DynamicImage, strength: f32) -> DynamicImage {
    let s = strength01(strength);
    let mut rgb = img.to_rgb8();
    let n = (rgb.width() as usize) * (rgb.height() as usize);
    if s == 0.0 || n == 0 {
        return DynamicImage::ImageRgb8(rgb);
    }

    let (mut rs, mut gs, mut bs) = (0.0f32, 0.0f32, 0.0f32);
    for p in rgb.pixels() {
        rs += p[0] as f32;
        gs += p[1] as f32;
        bs += p[2] as f32;
    }
    let nf = n as f32;
    let (rm, gm, bm) = (rs / nf, gs / nf, bs / nf);
    if rm <= 0.0 || gm <= 0.0 || bm <= 0.0 {
        return DynamicImage::ImageRgb8(rgb);
    }
    let gray = (rm + gm + bm) / 3.0;
    let eff = [
        1.0 + s * ((gray / rm).clamp(WB_GAIN_MIN, WB_GAIN_MAX) - 1.0),
        1.0 + s * ((gray / gm).clamp(WB_GAIN_MIN, WB_GAIN_MAX) - 1.0),
        1.0 + s * ((gray / bm).clamp(WB_GAIN_MIN, WB_GAIN_MAX) - 1.0),
    ];
    for p in rgb.pixels_mut() {
        p[0] = to_u8(p[0] as f32 * eff[0]);
        p[1] = to_u8(p[1] as f32 * eff[1]);
        p[2] = to_u8(p[2] as f32 * eff[2]);
    }
    DynamicImage::ImageRgb8(rgb)
}

// ---------------------------------------------------------------------------
// 3. Tone / contrast normalization (percentile stretch)
// ---------------------------------------------------------------------------

/// Low/high luma percentiles mapped to black/white by the contrast stretch.
/// Using 1%/99% (rather than 0/100) ignores a few outlier pixels so a single
/// speck doesn't defeat the stretch.
const CONTRAST_LO_PCT: f32 = 0.01;
const CONTRAST_HI_PCT: f32 = 0.99;

/// Normalize tone/contrast by stretching the luma range: map the 1st/99th
/// luma percentiles to 0/255 and apply that same linear map to every channel
/// (which preserves color while expanding tonal range), blended by `strength`.
pub fn auto_contrast(img: &DynamicImage, strength: f32) -> DynamicImage {
    let s = strength01(strength);
    let mut rgb = img.to_rgb8();
    let n = (rgb.width() as usize) * (rgb.height() as usize);
    if s == 0.0 || n == 0 {
        return DynamicImage::ImageRgb8(rgb);
    }

    let mut hist = [0u32; 256];
    for p in rgb.pixels() {
        hist[to_u8(luma(p[0], p[1], p[2])) as usize] += 1;
    }
    let lo = percentile(&hist, n, CONTRAST_LO_PCT) as f32;
    let hi = percentile(&hist, n, CONTRAST_HI_PCT) as f32;
    if hi <= lo {
        return DynamicImage::ImageRgb8(rgb); // flat image: nothing to stretch
    }
    let scale = 255.0 / (hi - lo);
    for p in rgb.pixels_mut() {
        for c in 0..3 {
            let corrected = (p[c] as f32 - lo) * scale;
            p[c] = to_u8(blend(p[c] as f32, corrected, s));
        }
    }
    DynamicImage::ImageRgb8(rgb)
}

/// Smallest luma value whose cumulative histogram count reaches `pct` of `n`.
fn percentile(hist: &[u32; 256], n: usize, pct: f32) -> u8 {
    let target = (pct * n as f32) as u32;
    let mut cum = 0u32;
    for (value, &count) in hist.iter().enumerate() {
        cum += count;
        if cum >= target {
            return value as u8;
        }
    }
    255
}

// ---------------------------------------------------------------------------
// Combined pipeline
// ---------------------------------------------------------------------------

/// Per-correction strengths (each 0–100%). Received from the frontend.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EditParams {
    pub exposure: f32,
    pub white_balance: f32,
    pub contrast: f32,
}

impl Default for EditParams {
    fn default() -> Self {
        Self {
            exposure: 100.0,
            white_balance: 100.0,
            contrast: 100.0,
        }
    }
}

impl EditParams {
    /// True when every strength is zero — applying it would be a no-op copy.
    pub fn is_noop(&self) -> bool {
        self.exposure <= 0.0 && self.white_balance <= 0.0 && self.contrast <= 0.0
    }
}

/// Apply white balance, then exposure, then tone/contrast, each at its own
/// strength. WB first neutralizes color, exposure fixes brightness, contrast
/// spreads the tonal range last.
pub fn auto_edit(img: &DynamicImage, params: EditParams) -> DynamicImage {
    let wb = auto_white_balance(img, params.white_balance);
    let exposed = auto_exposure(&wb, params.exposure);
    auto_contrast(&exposed, params.contrast)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    fn solid(r: u8, g: u8, b: u8) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_pixel(64, 64, Rgb([r, g, b])))
    }

    fn mean_luma(img: &DynamicImage) -> f32 {
        let rgb = img.to_rgb8();
        let n = (rgb.width() * rgb.height()) as f32;
        let mut s = 0.0;
        for p in rgb.pixels() {
            s += luma(p[0], p[1], p[2]);
        }
        s / n / 255.0
    }

    fn channel_means(img: &DynamicImage) -> (f32, f32, f32) {
        let rgb = img.to_rgb8();
        let n = (rgb.width() * rgb.height()) as f32;
        let (mut r, mut g, mut b) = (0.0, 0.0, 0.0);
        for p in rgb.pixels() {
            r += p[0] as f32;
            g += p[1] as f32;
            b += p[2] as f32;
        }
        (r / n, g / n, b / n)
    }

    fn luma_range(img: &DynamicImage) -> (f32, f32) {
        let rgb = img.to_rgb8();
        let (mut lo, mut hi) = (255.0f32, 0.0f32);
        for p in rgb.pixels() {
            let l = luma(p[0], p[1], p[2]);
            lo = lo.min(l);
            hi = hi.max(l);
        }
        (lo, hi)
    }

    /// A low-contrast horizontal gray gradient confined to `base..base+span`.
    fn low_contrast_gradient(base: u8, span: u8) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_fn(64, 64, |x, _| {
            let v = base as u32 + (x as u32 * span as u32 / 63);
            let v = v.min(255) as u8;
            Rgb([v, v, v])
        }))
    }

    // ---- exposure ----

    #[test]
    fn exposure_brightens_underexposed() {
        let dark = solid(40, 40, 40);
        let before = mean_luma(&dark);
        let after = mean_luma(&auto_exposure(&dark, 100.0));
        assert!(after > before + 0.1, "expected brighten: {before} -> {after}");
        assert!((after - EXPOSURE_TARGET).abs() < 0.12, "should move toward target: {after}");
    }

    #[test]
    fn exposure_darkens_overexposed() {
        let bright = solid(220, 220, 220);
        let before = mean_luma(&bright);
        let after = mean_luma(&auto_exposure(&bright, 100.0));
        assert!(after < before - 0.05, "expected darken: {before} -> {after}");
    }

    #[test]
    fn exposure_strength_zero_is_identity() {
        let img = solid(40, 40, 40);
        assert_eq!(
            auto_exposure(&img, 0.0).to_rgb8().into_raw(),
            img.to_rgb8().into_raw()
        );
    }

    #[test]
    fn exposure_strength_scales_between_none_and_full() {
        let img = solid(40, 40, 40);
        let none = mean_luma(&img);
        let half = mean_luma(&auto_exposure(&img, 50.0));
        let full = mean_luma(&auto_exposure(&img, 100.0));
        assert!(none < half && half < full, "{none} < {half} < {full}");
    }

    // ---- white balance ----

    #[test]
    fn white_balance_neutralizes_color_cast() {
        let reddish = solid(200, 120, 120);
        let (rb, gb, _bb) = channel_means(&reddish);
        assert!(rb - gb > 40.0, "test image should have a red cast");
        let (r, g, b) = channel_means(&auto_white_balance(&reddish, 100.0));
        assert!((r - g).abs() < 6.0 && (g - b).abs() < 6.0, "cast not neutralized: {r},{g},{b}");
    }

    #[test]
    fn white_balance_leaves_neutral_gray_unchanged() {
        let gray = solid(128, 128, 128);
        let out = auto_white_balance(&gray, 100.0).to_rgb8().into_raw();
        for (a, b) in out.iter().zip(gray.to_rgb8().into_raw()) {
            assert!((*a as i32 - b as i32).abs() <= 1, "gray shifted: {a} vs {b}");
        }
    }

    #[test]
    fn white_balance_strength_zero_is_identity() {
        let img = solid(200, 120, 120);
        assert_eq!(
            auto_white_balance(&img, 0.0).to_rgb8().into_raw(),
            img.to_rgb8().into_raw()
        );
    }

    // ---- contrast ----

    #[test]
    fn contrast_expands_narrow_range() {
        let low = low_contrast_gradient(110, 30); // luma ~110..140
        let (lo0, hi0) = luma_range(&low);
        let (lo1, hi1) = luma_range(&auto_contrast(&low, 100.0));
        assert!(hi1 - lo1 > (hi0 - lo0) * 2.0, "range should expand: {}->{}", hi0 - lo0, hi1 - lo1);
        assert!(lo1 < 20.0 && hi1 > 235.0, "should reach near full range: {lo1}..{hi1}");
    }

    #[test]
    fn contrast_flat_image_is_safe_noop() {
        let flat = solid(128, 128, 128);
        assert_eq!(
            auto_contrast(&flat, 100.0).to_rgb8().into_raw(),
            flat.to_rgb8().into_raw()
        );
    }

    #[test]
    fn contrast_strength_zero_is_identity() {
        let low = low_contrast_gradient(110, 30);
        assert_eq!(
            auto_contrast(&low, 0.0).to_rgb8().into_raw(),
            low.to_rgb8().into_raw()
        );
    }

    // ---- combined ----

    #[test]
    fn auto_edit_noop_when_all_zero() {
        let img = solid(60, 50, 40);
        let params = EditParams {
            exposure: 0.0,
            white_balance: 0.0,
            contrast: 0.0,
        };
        assert!(params.is_noop());
        assert_eq!(
            auto_edit(&img, params).to_rgb8().into_raw(),
            img.to_rgb8().into_raw()
        );
    }

    #[test]
    fn auto_edit_improves_dark_cast_image() {
        // Dark and blue-tinted: exposure should brighten, WB should neutralize.
        let img = solid(30, 40, 70);
        let out = auto_edit(&img, EditParams::default());
        assert!(mean_luma(&out) > mean_luma(&img) + 0.1, "should brighten");
        let (r, g, b) = channel_means(&out);
        let spread_before = 70.0 - 30.0;
        let spread_after = r.max(g).max(b) - r.min(g).min(b);
        assert!(spread_after < spread_before, "color cast should shrink: {spread_before}->{spread_after}");
    }
}
