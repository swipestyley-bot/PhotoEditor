//! Offline photo-editing engine: auto corrections (exposure / white balance /
//! tone) plus a full set of manual adjustments (exposure, contrast, highlights,
//! shadows, whites, blacks, saturation, vibrance, temperature, tint,
//! sharpening) and a straighten (rotate + auto-crop). Pure functions over
//! `image::DynamicImage`; no network.
//!
//! All tonal/color work happens in a single f32 buffer (channels in 0..1,
//! sRGB/gamma space) so chained adjustments don't accumulate 8-bit rounding, and
//! everything is clamped to [0,1] exactly once at the end — combining many
//! strong sliders can't produce NaN or "broken" output, only (correctly)
//! clamped highlights/shadows. Geometry (straighten) runs first, sharpening
//! (needs neighbours) runs last on the 8-bit image.

use image::{DynamicImage, Rgb, RgbImage};
use serde::Deserialize;

/// Per-photo edit settings. Auto strengths are 0..100; manual sliders are
/// -100..100 (bipolar) except `sharpening` (0..100) and `straighten`
/// (-100..100 mapped to ±[`STRAIGHTEN_MAX_DEG`]°). All default to 0 = identity.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct EditParams {
    pub auto_exposure: f32,
    pub auto_white_balance: f32,
    pub auto_contrast: f32,
    pub exposure: f32,
    pub contrast: f32,
    pub highlights: f32,
    pub shadows: f32,
    pub whites: f32,
    pub blacks: f32,
    pub saturation: f32,
    pub vibrance: f32,
    pub temperature: f32,
    pub tint: f32,
    pub sharpening: f32,
    pub straighten: f32,
}

impl Default for EditParams {
    fn default() -> Self {
        Self {
            auto_exposure: 0.0,
            auto_white_balance: 0.0,
            auto_contrast: 0.0,
            exposure: 0.0,
            contrast: 0.0,
            highlights: 0.0,
            shadows: 0.0,
            whites: 0.0,
            blacks: 0.0,
            saturation: 0.0,
            vibrance: 0.0,
            temperature: 0.0,
            tint: 0.0,
            sharpening: 0.0,
            straighten: 0.0,
        }
    }
}

impl EditParams {
    /// True when nothing would change the image (all sliders at 0).
    pub fn is_noop(&self) -> bool {
        [
            self.auto_exposure, self.auto_white_balance, self.auto_contrast,
            self.exposure, self.contrast, self.highlights, self.shadows,
            self.whites, self.blacks, self.saturation, self.vibrance,
            self.temperature, self.tint, self.sharpening, self.straighten,
        ]
        .iter()
        .all(|v| v.abs() < 0.01)
    }
}

/// Max straighten angle at slider = ±100.
const STRAIGHTEN_MAX_DEG: f32 = 10.0;
const EXPOSURE_TARGET: f32 = 0.5;

fn strength01(s: f32) -> f32 {
    (s / 100.0).clamp(0.0, 1.0)
}
fn bipolar(s: f32) -> f32 {
    (s / 100.0).clamp(-1.0, 1.0)
}
fn lum(px: &[f32]) -> f32 {
    0.299 * px[0] + 0.587 * px[1] + 0.114 * px[2]
}
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn to_buf(rgb: &RgbImage) -> Vec<f32> {
    rgb.as_raw().iter().map(|&v| v as f32 / 255.0).collect()
}
fn from_buf(buf: &[f32], w: u32, h: u32) -> RgbImage {
    let bytes: Vec<u8> = buf.iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect();
    RgbImage::from_raw(w, h, bytes).expect("buffer size matches dimensions")
}

// --- global-stat auto corrections (buffer form) ---------------------------

fn auto_wb_buf(buf: &mut [f32], s: f32) {
    if s <= 0.0 {
        return;
    }
    let n = (buf.len() / 3) as f32;
    let (mut rs, mut gs, mut bs) = (0.0f32, 0.0, 0.0);
    for px in buf.chunks_exact(3) {
        rs += px[0];
        gs += px[1];
        bs += px[2];
    }
    let (rm, gm, bm) = (rs / n, gs / n, bs / n);
    if rm <= 0.0 || gm <= 0.0 || bm <= 0.0 {
        return;
    }
    let gray = (rm + gm + bm) / 3.0;
    let e = [
        1.0 + s * ((gray / rm).clamp(0.5, 2.0) - 1.0),
        1.0 + s * ((gray / gm).clamp(0.5, 2.0) - 1.0),
        1.0 + s * ((gray / bm).clamp(0.5, 2.0) - 1.0),
    ];
    for px in buf.chunks_exact_mut(3) {
        px[0] *= e[0];
        px[1] *= e[1];
        px[2] *= e[2];
    }
}

fn auto_exp_buf(buf: &mut [f32], s: f32) {
    if s <= 0.0 {
        return;
    }
    let n = (buf.len() / 3) as f32;
    let mut sum = 0.0;
    for px in buf.chunks_exact(3) {
        sum += lum(px);
    }
    let mean = sum / n;
    if mean <= 0.0 {
        return;
    }
    let gain = (EXPOSURE_TARGET / mean).clamp(0.33, 3.0);
    let eff = 1.0 + s * (gain - 1.0);
    for v in buf.iter_mut() {
        *v *= eff;
    }
}

fn auto_contrast_buf(buf: &mut [f32], s: f32) {
    if s <= 0.0 {
        return;
    }
    let n = buf.len() / 3;
    let mut hist = [0u32; 256];
    for px in buf.chunks_exact(3) {
        let idx = (lum(px).clamp(0.0, 1.0) * 255.0) as usize;
        hist[idx] += 1;
    }
    let lo = percentile(&hist, n, 0.01) as f32 / 255.0;
    let hi = percentile(&hist, n, 0.99) as f32 / 255.0;
    if hi <= lo {
        return;
    }
    let scale = 1.0 / (hi - lo);
    for v in buf.iter_mut() {
        let corrected = (*v - lo) * scale;
        *v += (corrected - *v) * s;
    }
}

fn percentile(hist: &[u32; 256], n: usize, pct: f32) -> u8 {
    let target = (pct * n as f32) as u32;
    let mut cum = 0u32;
    for (v, &c) in hist.iter().enumerate() {
        cum += c;
        if cum >= target {
            return v as u8;
        }
    }
    255
}

// --- manual per-pixel adjustments (buffer form) ---------------------------

fn manual_white_balance(buf: &mut [f32], temperature: f32, tint: f32) {
    if temperature == 0.0 && tint == 0.0 {
        return;
    }
    let er = 1.0 + 0.3 * temperature; // warm: red up
    let eb = 1.0 - 0.3 * temperature; // warm: blue down
    let eg = 1.0 - 0.2 * tint; // +tint = magenta (green down)
    for px in buf.chunks_exact_mut(3) {
        px[0] *= er;
        px[1] *= eg;
        px[2] *= eb;
    }
}

fn manual_exposure(buf: &mut [f32], exposure: f32) {
    if exposure == 0.0 {
        return;
    }
    let gain = 2f32.powf(2.0 * exposure); // ±2 stops
    for v in buf.iter_mut() {
        *v *= gain;
    }
}

fn manual_tone(buf: &mut [f32], contrast: f32, highlights: f32, shadows: f32, whites: f32, blacks: f32) {
    if contrast == 0.0 && highlights == 0.0 && shadows == 0.0 && whites == 0.0 && blacks == 0.0 {
        return;
    }
    for px in buf.chunks_exact_mut(3) {
        let l = lum(px);
        let m_hi = smoothstep(0.5, 1.0, l);
        let m_sh = smoothstep(0.5, 0.0, l);
        let m_wh = smoothstep(0.72, 1.0, l);
        let m_bk = smoothstep(0.28, 0.0, l);
        for v in px.iter_mut() {
            let mut x = *v;
            x = (x - 0.5) * (1.0 + contrast) + 0.5; // contrast around mid
            x *= 1.0 + highlights * 0.4 * m_hi;
            x *= 1.0 + shadows * 0.5 * m_sh;
            x += whites * 0.22 * m_wh;
            x += blacks * 0.22 * m_bk;
            *v = x;
        }
    }
}

fn manual_color(buf: &mut [f32], saturation: f32, vibrance: f32) {
    if saturation == 0.0 && vibrance == 0.0 {
        return;
    }
    for px in buf.chunks_exact_mut(3) {
        let l = lum(px);
        if vibrance != 0.0 {
            let mx = px[0].max(px[1]).max(px[2]);
            let mn = px[0].min(px[1]).min(px[2]);
            let sat_now = (mx - mn).clamp(0.0, 1.0);
            let factor = 1.0 + vibrance * (1.0 - sat_now); // boost muted colors more
            for v in px.iter_mut() {
                *v = l + (*v - l) * factor;
            }
        }
        if saturation != 0.0 {
            let factor = 1.0 + saturation;
            for v in px.iter_mut() {
                *v = l + (*v - l) * factor;
            }
        }
    }
}

// --- geometry + detail (image form) ---------------------------------------

/// Largest upright rectangle that fits inside `w`x`h` rotated by `angle` (rad).
fn inscribed_rect(w: f32, h: f32, angle: f32) -> (f32, f32) {
    if w <= 0.0 || h <= 0.0 {
        return (0.0, 0.0);
    }
    let width_longer = w >= h;
    let (long, short) = if width_longer { (w, h) } else { (h, w) };
    let sin_a = angle.sin().abs();
    let cos_a = angle.cos().abs();
    if short <= 2.0 * sin_a * cos_a * long || (sin_a - cos_a).abs() < 1e-10 {
        let x = 0.5 * short;
        if width_longer {
            (x / sin_a.max(1e-6), x / cos_a.max(1e-6))
        } else {
            (x / cos_a.max(1e-6), x / sin_a.max(1e-6))
        }
    } else {
        let cos_2a = cos_a * cos_a - sin_a * sin_a;
        ((w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a)
    }
}

fn straighten(img: &DynamicImage, slider: f32) -> DynamicImage {
    let deg = bipolar(slider) * STRAIGHTEN_MAX_DEG;
    if deg.abs() < 0.01 {
        return img.clone();
    }
    let theta = deg.to_radians();
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let rotated = imageproc::geometric_transformations::rotate_about_center(
        &rgb,
        theta,
        imageproc::geometric_transformations::Interpolation::Bilinear,
        imageproc::geometric_transformations::Border::Constant(Rgb([0, 0, 0])),
    );
    let (cw, ch) = inscribed_rect(w as f32, h as f32, theta.abs());
    let cw = (cw.floor() as u32).min(w).max(1);
    let ch = (ch.floor() as u32).min(h).max(1);
    let x0 = (w - cw) / 2;
    let y0 = (h - ch) / 2;
    DynamicImage::ImageRgb8(image::imageops::crop_imm(&rotated, x0, y0, cw, ch).to_image())
}

fn sharpen(rgb: &mut RgbImage, amount: f32) {
    if amount <= 0.0 {
        return;
    }
    let blurred = imageproc::filter::gaussian_blur_f32(rgb, 1.2);
    let k = amount * 0.9;
    for (o, b) in rgb.pixels_mut().zip(blurred.pixels()) {
        for c in 0..3 {
            let v = o[c] as f32 + k * (o[c] as f32 - b[c] as f32);
            o[c] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
}

// --- public entry points ---------------------------------------------------

/// Full edit pipeline: geometry → white balance → exposure → tone →
/// color → sharpening. Order mirrors a typical raw editor.
pub fn auto_edit(img: &DynamicImage, p: EditParams) -> DynamicImage {
    let geo = straighten(img, p.straighten);
    let rgb = geo.to_rgb8();
    let (w, h) = rgb.dimensions();
    let mut buf = to_buf(&rgb);

    auto_wb_buf(&mut buf, strength01(p.auto_white_balance));
    manual_white_balance(&mut buf, bipolar(p.temperature), bipolar(p.tint));

    auto_exp_buf(&mut buf, strength01(p.auto_exposure));
    manual_exposure(&mut buf, bipolar(p.exposure));

    auto_contrast_buf(&mut buf, strength01(p.auto_contrast));
    manual_tone(
        &mut buf,
        bipolar(p.contrast),
        bipolar(p.highlights),
        bipolar(p.shadows),
        bipolar(p.whites),
        bipolar(p.blacks),
    );

    manual_color(&mut buf, bipolar(p.saturation), bipolar(p.vibrance));

    let mut out = from_buf(&buf, w, h); // clamps to [0,1] here
    sharpen(&mut out, strength01(p.sharpening));
    DynamicImage::ImageRgb8(out)
}

/// Auto-exposure only (0..100 strength) — used by tests/diagnostics.
pub fn auto_exposure(img: &DynamicImage, strength: f32) -> DynamicImage {
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let mut buf = to_buf(&rgb);
    auto_exp_buf(&mut buf, strength01(strength));
    DynamicImage::ImageRgb8(from_buf(&buf, w, h))
}

/// Auto white balance only (0..100 strength).
pub fn auto_white_balance(img: &DynamicImage, strength: f32) -> DynamicImage {
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let mut buf = to_buf(&rgb);
    auto_wb_buf(&mut buf, strength01(strength));
    DynamicImage::ImageRgb8(from_buf(&buf, w, h))
}

/// Auto tone/contrast only (0..100 strength).
pub fn auto_contrast(img: &DynamicImage, strength: f32) -> DynamicImage {
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let mut buf = to_buf(&rgb);
    auto_contrast_buf(&mut buf, strength01(strength));
    DynamicImage::ImageRgb8(from_buf(&buf, w, h))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    fn solid(r: u8, g: u8, b: u8) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_pixel(64, 64, Rgb([r, g, b])))
    }

    fn gradient() -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_fn(96, 64, |x, _| {
            let v = (x * 255 / 95) as u8;
            Rgb([v, (v as u32 * 3 / 4) as u8, 255 - v])
        }))
    }

    fn mean_luma(img: &DynamicImage) -> f32 {
        let rgb = img.to_rgb8();
        let n = (rgb.width() * rgb.height()) as f32;
        let mut s = 0.0;
        for p in rgb.pixels() {
            s += 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32;
        }
        s / n / 255.0
    }

    fn params() -> EditParams {
        EditParams::default()
    }

    #[test]
    fn default_is_identity() {
        let img = gradient();
        assert!(params().is_noop());
        assert_eq!(auto_edit(&img, params()).to_rgb8().into_raw(), img.to_rgb8().into_raw());
    }

    #[test]
    fn manual_exposure_brightens_and_darkens() {
        let img = solid(120, 120, 120);
        let up = mean_luma(&auto_edit(&img, EditParams { exposure: 60.0, ..params() }));
        let down = mean_luma(&auto_edit(&img, EditParams { exposure: -60.0, ..params() }));
        assert!(up > mean_luma(&img) + 0.05 && down < mean_luma(&img) - 0.05, "{down} < {} < {up}", mean_luma(&img));
    }

    #[test]
    fn saturation_down_greys_out() {
        let img = solid(200, 80, 40);
        let out = auto_edit(&img, EditParams { saturation: -100.0, ..params() }).to_rgb8();
        let p = out.get_pixel(0, 0);
        assert!((p[0] as i32 - p[1] as i32).abs() < 8 && (p[1] as i32 - p[2] as i32).abs() < 8, "should be near-grey: {p:?}");
    }

    #[test]
    fn temperature_warms_and_cools() {
        let img = solid(128, 128, 128);
        let warm = auto_edit(&img, EditParams { temperature: 80.0, ..params() }).to_rgb8();
        let cool = auto_edit(&img, EditParams { temperature: -80.0, ..params() }).to_rgb8();
        assert!(warm.get_pixel(0, 0)[0] > warm.get_pixel(0, 0)[2], "warm: red>blue");
        assert!(cool.get_pixel(0, 0)[2] > cool.get_pixel(0, 0)[0], "cool: blue>red");
    }

    #[test]
    fn contrast_increases_spread() {
        let img = gradient();
        let out = auto_edit(&img, EditParams { contrast: 70.0, ..params() });
        let (a, b) = (img.to_rgb8(), out.to_rgb8());
        let spread = |im: &RgbImage| {
            let (mut lo, mut hi) = (255i32, 0i32);
            for p in im.pixels() {
                let l = p[0] as i32;
                lo = lo.min(l);
                hi = hi.max(l);
            }
            hi - lo
        };
        assert!(spread(&b) >= spread(&a), "contrast should not shrink range");
    }

    /// The key safety test: many strong sliders at once must stay finite and in
    /// range, and must not collapse the image to a single flat value.
    #[test]
    fn combined_extremes_stay_valid() {
        let img = gradient();
        let out = auto_edit(
            &img,
            EditParams {
                auto_exposure: 100.0,
                auto_white_balance: 100.0,
                exposure: 70.0,
                contrast: 80.0,
                highlights: -70.0,
                shadows: 80.0,
                whites: 60.0,
                blacks: -60.0,
                saturation: 90.0,
                vibrance: 80.0,
                temperature: 60.0,
                tint: -40.0,
                sharpening: 100.0,
                ..params()
            },
        )
        .to_rgb8();
        let raw = out.into_raw();
        // every value is a valid u8 (guaranteed by type) — check it isn't degenerate
        let min = *raw.iter().min().unwrap();
        let max = *raw.iter().max().unwrap();
        assert!(max > min, "combined edit collapsed the image to a flat value");
        assert!(max - min > 30, "combined edit crushed almost all contrast: {min}..{max}");
    }

    #[test]
    fn straighten_rotates_and_crops() {
        let img = gradient(); // 96x64
        let out = auto_edit(&img, EditParams { straighten: 100.0, ..params() });
        // rotate+inscribed-crop yields a smaller frame than the original
        assert!(out.width() < 96 && out.height() < 64, "straighten should crop in: {}x{}", out.width(), out.height());
        assert!(out.width() > 40 && out.height() > 25, "crop too aggressive: {}x{}", out.width(), out.height());
    }

    // --- auto-only behaviour (unchanged engine) ---
    #[test]
    fn auto_exposure_brightens_dark() {
        let dark = solid(40, 40, 40);
        assert!(mean_luma(&auto_exposure(&dark, 100.0)) > mean_luma(&dark) + 0.1);
    }
    #[test]
    fn auto_exposure_strength_scales() {
        let img = solid(40, 40, 40);
        let none = mean_luma(&img);
        let half = mean_luma(&auto_exposure(&img, 50.0));
        let full = mean_luma(&auto_exposure(&img, 100.0));
        assert!(none < half && half < full);
    }
    #[test]
    fn auto_wb_neutralizes_cast() {
        let out = auto_white_balance(&solid(200, 120, 120), 100.0).to_rgb8();
        let p = out.get_pixel(0, 0);
        assert!((p[0] as i32 - p[1] as i32).abs() < 8);
    }
    #[test]
    fn auto_contrast_expands_low_contrast() {
        let low = DynamicImage::ImageRgb8(RgbImage::from_fn(64, 64, |x, _| {
            let v = 110 + (x as u32 * 30 / 63) as u8;
            Rgb([v, v, v])
        }));
        let a = auto_contrast(&low, 100.0).to_rgb8();
        let (mut lo, mut hi) = (255i32, 0i32);
        for p in a.pixels() {
            lo = lo.min(p[0] as i32);
            hi = hi.max(p[0] as i32);
        }
        assert!(hi - lo > 60, "range should expand: {lo}..{hi}");
    }
}
