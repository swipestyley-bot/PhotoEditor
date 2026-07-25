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

use image::{imageops::FilterType, DynamicImage, Rgb, RgbImage};
use serde::{Deserialize, Serialize};

/// Per-photo edit settings. Auto strengths are 0..100; manual sliders are
/// -100..100 (bipolar) except `sharpening` (0..100) and `straighten`
/// (-100..100 mapped to ±[`STRAIGHTEN_MAX_DEG`]°). All default to 0 = identity.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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
    /// Crop rect, normalized 0..1 of the straightened frame. A full-frame rect
    /// (the default all-zero) means "no crop".
    pub crop_x: f32,
    pub crop_y: f32,
    pub crop_w: f32,
    pub crop_h: f32,
    /// Noise reduction strength 0..100 (median blend).
    pub noise_reduction: f32,
    /// Local-contrast clarity -100..100.
    pub clarity: f32,
    /// Vignette: negative darkens edges, positive lightens (-100..100).
    pub vignette_amount: f32,
    /// Vignette midpoint 0..100 (where the falloff starts).
    pub vignette_midpoint: f32,
    /// Split toning: shadow hue 0..360 / saturation 0..100.
    pub shadow_hue: f32,
    pub shadow_sat: f32,
    /// Split toning: highlight hue 0..360 / saturation 0..100.
    pub highlight_hue: f32,
    pub highlight_sat: f32,
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
            crop_x: 0.0,
            crop_y: 0.0,
            crop_w: 0.0,
            crop_h: 0.0,
            noise_reduction: 0.0,
            clarity: 0.0,
            vignette_amount: 0.0,
            vignette_midpoint: 0.0,
            shadow_hue: 0.0,
            shadow_sat: 0.0,
            highlight_hue: 0.0,
            highlight_sat: 0.0,
        }
    }
}

impl EditParams {
    /// True when nothing would change the image (all sliders at 0 and no crop).
    pub fn is_noop(&self) -> bool {
        !self.crop_active()
            && [
                self.auto_exposure, self.auto_white_balance, self.auto_contrast,
                self.exposure, self.contrast, self.highlights, self.shadows,
                self.whites, self.blacks, self.saturation, self.vibrance,
                self.temperature, self.tint, self.sharpening, self.straighten,
                self.noise_reduction, self.clarity, self.vignette_amount,
                self.vignette_midpoint, self.shadow_hue, self.shadow_sat,
                self.highlight_hue, self.highlight_sat,
            ]
            .iter()
            .all(|v| v.abs() < 0.01)
    }

    /// True when the crop rect is set to something other than the full frame.
    pub fn crop_active(&self) -> bool {
        self.crop_w > 0.001
            && self.crop_h > 0.001
            && (self.crop_x > 0.001
                || self.crop_y > 0.001
                || self.crop_w < 0.999
                || self.crop_h < 0.999)
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

/// Straighten: rotate by the slider's angle and crop to the largest upright
/// rectangle containing no rotated-out (blank) corners. Runs before the user
/// crop, so the crop can never include blank area.
pub fn straighten(img: &DynamicImage, p: EditParams) -> DynamicImage {
    let deg = bipolar(p.straighten) * STRAIGHTEN_MAX_DEG;
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
    // Inset ~1.5% so no thin black edge from the rotation boundary survives.
    let cw = ((cw * 0.985).floor() as u32).min(w).max(1);
    let ch = ((ch * 0.985).floor() as u32).min(h).max(1);
    let x0 = (w - cw) / 2;
    let y0 = (h - ch) / 2;
    DynamicImage::ImageRgb8(image::imageops::crop_imm(&rotated, x0, y0, cw, ch).to_image())
}

/// Apply the user crop rect (normalized 0..1 of the straightened frame). A rect
/// covering the full frame (the default) is a no-op.
pub fn crop(img: &DynamicImage, p: EditParams) -> DynamicImage {
    if !p.crop_active() {
        return img.clone();
    }
    let (w, h) = (img.width(), img.height());
    let x = (p.crop_x.clamp(0.0, 1.0) * w as f32) as u32;
    let y = (p.crop_y.clamp(0.0, 1.0) * h as f32) as u32;
    let cw = ((p.crop_w * w as f32).round() as u32).min(w.saturating_sub(x)).max(1);
    let ch = ((p.crop_h * h as f32).round() as u32).min(h.saturating_sub(y)).max(1);
    DynamicImage::ImageRgb8(image::imageops::crop_imm(&img.to_rgb8(), x, y, cw, ch).to_image())
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

// --- noise / clarity / vignette / split toning ------------------------------

/// Median-filter denoise, blended with the original by `strength` (0..100).
fn denoise(img: &DynamicImage, strength: f32) -> DynamicImage {
    let s = strength01(strength);
    if s <= 0.0 {
        return img.clone();
    }
    let mut rgb = img.to_rgb8();
    let radius = if strength >= 60.0 { 2 } else { 1 };
    let med = imageproc::filter::median_filter(&rgb, radius, radius);
    for (o, m) in rgb.pixels_mut().zip(med.pixels()) {
        for c in 0..3 {
            o[c] = (o[c] as f32 + (m[c] as f32 - o[c] as f32) * s).round().clamp(0.0, 255.0) as u8;
        }
    }
    DynamicImage::ImageRgb8(rgb)
}

/// Large-radius blur, approximated by down/up-sampling so a big sigma stays fast
/// even on full-resolution exports.
fn large_blur(rgb: &RgbImage, sigma: f32) -> RgbImage {
    if sigma <= 6.0 {
        return imageproc::filter::gaussian_blur_f32(rgb, sigma);
    }
    let scale = (sigma / 4.0).clamp(1.0, 8.0);
    let sw = (rgb.width() as f32 / scale).max(1.0) as u32;
    let sh = (rgb.height() as f32 / scale).max(1.0) as u32;
    let small = image::imageops::resize(rgb, sw, sh, FilterType::Triangle);
    let blurred = imageproc::filter::gaussian_blur_f32(&small, sigma / scale);
    image::imageops::resize(&blurred, rgb.width(), rgb.height(), FilterType::Triangle)
}

/// Clarity: local (midtone) contrast via a large-radius unsharp mask (-1..1).
fn clarity(rgb: &mut RgbImage, amount: f32) {
    if amount == 0.0 {
        return;
    }
    let (w, h) = rgb.dimensions();
    let sigma = ((w.min(h) as f32) / 60.0).clamp(4.0, 60.0);
    let blurred = large_blur(rgb, sigma);
    let k = amount * 0.6;
    for (o, b) in rgb.pixels_mut().zip(blurred.pixels()) {
        for c in 0..3 {
            let v = o[c] as f32 + k * (o[c] as f32 - b[c] as f32);
            o[c] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
}

/// Radial vignette in the working buffer. `amount` -1..1 (negative darkens the
/// edges), `midpoint` 0..1 where the falloff begins.
fn vignette_buf(buf: &mut [f32], w: u32, h: u32, amount: f32, midpoint: f32) {
    if amount == 0.0 {
        return;
    }
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let maxd = (cx * cx + cy * cy).sqrt().max(1.0);
    for (i, px) in buf.chunks_exact_mut(3).enumerate() {
        let x = (i as u32 % w) as f32;
        let y = (i as u32 / w) as f32;
        let d = (((x - cx).powi(2) + (y - cy).powi(2)).sqrt()) / maxd;
        let t = smoothstep(midpoint, 1.0, d);
        let factor = 1.0 + amount * 0.8 * t;
        px[0] *= factor;
        px[1] *= factor;
        px[2] *= factor;
    }
}

fn hue_to_rgb(hue_deg: f32) -> [f32; 3] {
    // Fully saturated colour at the given hue (HSL s=1, l=0.5).
    let h = (hue_deg.rem_euclid(360.0)) / 60.0;
    let x = 1.0 - (h % 2.0 - 1.0).abs();
    let (r, g, b) = match h as u32 {
        0 => (1.0, x, 0.0),
        1 => (x, 1.0, 0.0),
        2 => (0.0, 1.0, x),
        3 => (0.0, x, 1.0),
        4 => (x, 0.0, 1.0),
        _ => (1.0, 0.0, x),
    };
    [r, g, b]
}

/// Split toning: tint shadows and highlights toward separate hues, weighted by
/// luminance.
fn split_tone_buf(buf: &mut [f32], sh_hue: f32, sh_sat: f32, hi_hue: f32, hi_sat: f32) {
    let sh_amt = (sh_sat / 100.0).clamp(0.0, 1.0) * 0.5;
    let hi_amt = (hi_sat / 100.0).clamp(0.0, 1.0) * 0.5;
    if sh_amt <= 0.0 && hi_amt <= 0.0 {
        return;
    }
    let sh = hue_to_rgb(sh_hue);
    let hi = hue_to_rgb(hi_hue);
    for px in buf.chunks_exact_mut(3) {
        let l = lum(px).clamp(0.0, 1.0);
        let ws = (1.0 - l) * sh_amt;
        let wh = l * hi_amt;
        for c in 0..3 {
            px[c] += ws * (sh[c] - 0.5) + wh * (hi[c] - 0.5);
        }
    }
}

// --- retouch: healing / spot removal ---------------------------------------

/// A circular healing stamp, normalized: `x`,`y` in 0..1, `r` as a fraction of
/// image width.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HealStamp {
    pub x: f32,
    pub y: f32,
    pub r: f32,
}

/// One healing brush stroke (a click is a single-stamp stroke).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealStroke {
    pub stamps: Vec<HealStamp>,
}

/// Per-photo spatial retouch operations, kept separate from the scalar
/// `EditParams` (which stays simple/Copy). Liquify will join here later.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RetouchOps {
    #[serde(default)]
    pub heals: Vec<HealStroke>,
}

impl RetouchOps {
    pub fn is_empty(&self) -> bool {
        self.heals.iter().all(|s| s.stamps.is_empty())
    }
}

const POISSON_ITERS: usize = 180;

/// Apply every healing stamp to a copy of the image, in order.
pub fn apply_retouch(img: &DynamicImage, retouch: &RetouchOps) -> DynamicImage {
    if retouch.is_empty() {
        return img.clone();
    }
    let mut rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    for stroke in &retouch.heals {
        for s in &stroke.stamps {
            let r = (s.r * w as f32).round().clamp(2.0, (w.min(h) as f32) / 3.0) as i32;
            let cx = (s.x * w as f32).round() as i32;
            let cy = (s.y * h as f32).round() as i32;
            heal_spot(&mut rgb, cx, cy, r);
        }
    }
    DynamicImage::ImageRgb8(rgb)
}

/// Heal one circular spot: find the best-matching nearby source patch, then
/// Poisson (gradient-domain) seamless-clone it over the spot.
fn heal_spot(img: &mut RgbImage, cx: i32, cy: i32, r: i32) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    if r < 1 || cx - r - 1 < 0 || cy - r - 1 < 0 || cx + r + 1 >= w || cy + r + 1 >= h {
        return; // require the target disc (+1px) fully in bounds
    }
    let Some((sx, sy)) = best_source(img, cx, cy, r) else {
        return;
    };
    poisson_clone(img, cx, cy, sx, sy, r);
}

/// The nearby source disc whose surrounding ring best matches the target's ring
/// (so texture/lighting continues across the patch).
fn best_source(img: &RgbImage, cx: i32, cy: i32, r: i32) -> Option<(i32, i32)> {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let (r0, r1) = (r, (r as f32 * 1.5) as i32);
    let mut best: Option<(i32, i32)> = None;
    let mut best_score = f32::MAX;
    for &mult in &[3.0f32, 5.0, 8.0] {
        let d = r as f32 * mult;
        for a in 0..12 {
            let ang = a as f32 / 12.0 * std::f32::consts::TAU;
            let sx = cx + (d * ang.cos()) as i32;
            let sy = cy + (d * ang.sin()) as i32;
            if sx - r1 < 0 || sy - r1 < 0 || sx + r1 >= w || sy + r1 >= h {
                continue;
            }
            let (mut sum, mut n) = (0.0f32, 0.0f32);
            for dy in -r1..=r1 {
                for dx in -r1..=r1 {
                    let dd = dx * dx + dy * dy;
                    if dd < r0 * r0 || dd > r1 * r1 {
                        continue; // annulus around the disc only
                    }
                    let tp = img.get_pixel((cx + dx) as u32, (cy + dy) as u32);
                    let sp = img.get_pixel((sx + dx) as u32, (sy + dy) as u32);
                    for c in 0..3 {
                        let e = tp[c] as f32 - sp[c] as f32;
                        sum += e * e;
                    }
                    n += 3.0;
                }
            }
            let score = if n > 0.0 { sum / n } else { f32::MAX };
            if score < best_score {
                best_score = score;
                best = Some((sx, sy));
            }
        }
    }
    best
}

/// Poisson seamless clone of the source disc into the target disc: the result
/// takes the source's gradients (texture) but the target's boundary colours.
fn poisson_clone(img: &mut RgbImage, cx: i32, cy: i32, sx: i32, sy: i32, r: i32) {
    let r2 = r * r;
    let (x0, x1, y0, y1) = (cx - r - 1, cx + r + 1, cy - r - 1, cy + r + 1); // +1 for boundary
    let bw = (x1 - x0 + 1) as usize;
    let idx = |x: i32, y: i32| -> usize { ((y - y0) as usize) * bw + (x - x0) as usize };
    let inside = |x: i32, y: i32| -> bool {
        let (dx, dy) = (x - cx, y - cy);
        dx * dx + dy * dy <= r2
    };
    let count = (bw * (y1 - y0 + 1) as usize) as usize;
    let mut tv = vec![[0f32; 3]; count]; // target snapshot (boundary + init)
    let mut sv = vec![[0f32; 3]; count]; // source snapshot (gradient guidance)
    for y in y0..=y1 {
        for x in x0..=x1 {
            let tp = img.get_pixel(x as u32, y as u32);
            let sp = img.get_pixel((sx + (x - cx)) as u32, (sy + (y - cy)) as u32);
            let i = idx(x, y);
            tv[i] = [tp[0] as f32, tp[1] as f32, tp[2] as f32];
            sv[i] = [sp[0] as f32, sp[1] as f32, sp[2] as f32];
        }
    }
    let mut f = tv.clone();
    for _ in 0..POISSON_ITERS {
        for y in (y0 + 1)..y1 {
            for x in (x0 + 1)..x1 {
                if !inside(x, y) {
                    continue;
                }
                let i = idx(x, y);
                for c in 0..3 {
                    let mut sum = 0.0;
                    for (nx, ny) in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)] {
                        let ni = idx(nx, ny);
                        let nv = if inside(nx, ny) { f[ni][c] } else { tv[ni][c] };
                        sum += nv + (sv[i][c] - sv[ni][c]);
                    }
                    f[i][c] = sum / 4.0;
                }
            }
        }
    }
    for y in y0..=y1 {
        for x in x0..=x1 {
            if !inside(x, y) {
                continue;
            }
            let i = idx(x, y);
            let px = img.get_pixel_mut(x as u32, y as u32);
            for c in 0..3 {
                px[c] = f[i][c].round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

// --- public entry points ---------------------------------------------------

/// Color/tone/detail stage (no geometry): white balance → exposure → tone →
/// color → sharpening. Order mirrors a typical raw editor.
pub fn color(img: &DynamicImage, p: EditParams) -> DynamicImage {
    let base = denoise(img, p.noise_reduction);
    let rgb = base.to_rgb8();
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

    split_tone_buf(&mut buf, p.shadow_hue, p.shadow_sat, p.highlight_hue, p.highlight_sat);
    manual_color(&mut buf, bipolar(p.saturation), bipolar(p.vibrance));
    vignette_buf(&mut buf, w, h, bipolar(p.vignette_amount), (p.vignette_midpoint / 100.0).clamp(0.0, 1.0));

    let mut out = from_buf(&buf, w, h); // clamps to [0,1] here
    clarity(&mut out, bipolar(p.clarity));
    sharpen(&mut out, strength01(p.sharpening));
    DynamicImage::ImageRgb8(out)
}

/// Full pipeline: straighten → color → crop.
pub fn auto_edit(img: &DynamicImage, p: EditParams) -> DynamicImage {
    crop(&color(&straighten(img, p), p), p)
}

/// Full pipeline including spatial retouch: straighten → color → heal → crop.
pub fn render_full(img: &DynamicImage, p: EditParams, retouch: &RetouchOps) -> DynamicImage {
    let straightened = straighten(img, p);
    let colored = color(&straightened, p);
    let retouched = apply_retouch(&colored, retouch);
    crop(&retouched, p)
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
                noise_reduction: 80.0,
                clarity: 70.0,
                vignette_amount: -60.0,
                vignette_midpoint: 40.0,
                shadow_hue: 220.0,
                shadow_sat: 60.0,
                highlight_hue: 40.0,
                highlight_sat: 60.0,
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

    #[test]
    fn crop_reduces_to_rect() {
        let img = gradient(); // 96x64
        let out = auto_edit(&img, EditParams { crop_x: 0.25, crop_y: 0.25, crop_w: 0.5, crop_h: 0.5, ..params() });
        assert!((out.width() as i32 - 48).abs() <= 2, "w {}", out.width());
        assert!((out.height() as i32 - 32).abs() <= 2, "h {}", out.height());
    }

    #[test]
    fn crop_full_frame_is_noop() {
        let img = gradient();
        let out = auto_edit(&img, EditParams { crop_x: 0.0, crop_y: 0.0, crop_w: 1.0, crop_h: 1.0, ..params() });
        assert_eq!(out.width(), 96);
        assert_eq!(out.height(), 64);
    }

    /// Rotating then cropping must not leave rotated-out (black) corners in the
    /// output — straighten's inscribed auto-crop removes them before the crop.
    #[test]
    fn rotate_plus_crop_has_no_blank() {
        let img = gradient();
        let out = auto_edit(
            &img,
            EditParams { straighten: 100.0, crop_x: 0.1, crop_y: 0.1, crop_w: 0.8, crop_h: 0.8, ..params() },
        )
        .to_rgb8();
        assert!(out.width() > 10 && out.height() > 10, "empty result");
        let black = out.pixels().filter(|p| p[0] < 3 && p[1] < 3 && p[2] < 3).count();
        let frac = black as f32 / (out.width() * out.height()) as f32;
        assert!(frac < 0.02, "rotate+crop left blank pixels: {frac}");
    }

    #[test]
    fn vignette_darkens_corners() {
        let img = solid(140, 140, 140);
        let out = auto_edit(&img, EditParams { vignette_amount: -100.0, vignette_midpoint: 0.0, ..params() }).to_rgb8();
        let center = out.get_pixel(out.width() / 2, out.height() / 2)[0] as i32;
        let corner = out.get_pixel(0, 0)[0] as i32;
        assert!(center - corner > 15, "center {center} brighter than corner {corner}");
    }

    #[test]
    fn split_tone_tints_shadows() {
        let img = solid(60, 60, 60); // dark = shadows
        let p = auto_edit(&img, EditParams { shadow_hue: 0.0, shadow_sat: 100.0, ..params() }).to_rgb8();
        let px = p.get_pixel(0, 0);
        assert!(px[0] as i32 > px[2] as i32 + 8, "red shadow tint: {px:?}");
    }

    #[test]
    fn denoise_reduces_variance() {
        let noisy = DynamicImage::ImageRgb8(RgbImage::from_fn(48, 48, |x, y| {
            let n = (x * 31 + y * 17) % 11; // sparse salt-and-pepper on a flat field
            let v = if n == 0 { 255u8 } else if n == 1 { 0u8 } else { 128u8 };
            Rgb([v, v, v])
        }));
        let std = |im: &DynamicImage| {
            let r = im.to_rgb8();
            let n = (r.width() * r.height()) as f32;
            let (mut s, mut ss) = (0.0f32, 0.0f32);
            for p in r.pixels() { let l = p[0] as f32; s += l; ss += l * l; }
            let m = s / n;
            (ss / n - m * m).max(0.0).sqrt()
        };
        let before = std(&noisy);
        let after = std(&auto_edit(&noisy, EditParams { noise_reduction: 100.0, ..params() }));
        assert!(after < before * 0.85, "denoise should cut variance: {before} -> {after}");
    }

    #[test]
    fn clarity_runs_and_stays_valid() {
        let out = auto_edit(&gradient(), EditParams { clarity: 80.0, ..params() }).to_rgb8().into_raw();
        assert!(*out.iter().max().unwrap() > *out.iter().min().unwrap());
    }

    #[test]
    fn heal_removes_a_spot() {
        // Flat gray field with a dark blemish disc at the centre.
        let mut base = RgbImage::from_pixel(80, 80, Rgb([130, 130, 130]));
        for y in 34..47 {
            for x in 34..47 {
                if (x as i32 - 40).pow(2) + (y as i32 - 40).pow(2) <= 30 {
                    base.put_pixel(x, y, Rgb([40, 40, 40]));
                }
            }
        }
        let img = DynamicImage::ImageRgb8(base);
        let before = img.to_rgb8().get_pixel(40, 40)[0];
        let retouch = RetouchOps {
            heals: vec![HealStroke { stamps: vec![HealStamp { x: 0.5, y: 0.5, r: 6.0 / 80.0 }] }],
        };
        let after = render_full(&img, EditParams::default(), &retouch).to_rgb8().get_pixel(40, 40)[0];
        assert!(before < 60, "test spot should start dark: {before}");
        assert!(after > 100, "spot should heal toward the gray field: {before} -> {after}");
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
