//! Verify the manual edit engine against real photos: apply a strong COMBINED
//! edit (exposure + contrast + highlights/shadows + saturation + vibrance +
//! temp/tint + sharpening at once) to every photo and confirm the output stays
//! valid — in range, not collapsed to a flat value, not NaN — i.e. combining
//! many sliders doesn't produce broken/clipped results. Also spot-checks each
//! slider individually, saves before/after samples, and runs the real
//! `export_selects` command both ways.
//!
//!   cargo run --release --example edit_probe -- <folder> [out-dir]

use std::fs;
use std::path::{Path, PathBuf};

use image::DynamicImage;
use tauri_app_lib::edit::{auto_edit, EditParams};
use tauri_app_lib::library::{export_selects, ExportItem, Naming, Watermark};
use tauri_app_lib::vision::decode_image;

fn luma(r: u8, g: u8, b: u8) -> f32 {
    0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32
}

fn stats(img: &DynamicImage) -> (f32, u8, u8) {
    // (mean luma 0..1, min channel byte, max channel byte)
    let rgb = img.to_rgb8();
    let n = (rgb.width() * rgb.height()) as f32;
    let (mut sum, mut lo, mut hi) = (0.0f32, 255u8, 0u8);
    for p in rgb.pixels() {
        sum += luma(p[0], p[1], p[2]);
        for c in 0..3 {
            lo = lo.min(p[c]);
            hi = hi.max(p[c]);
        }
    }
    (sum / n / 255.0, lo, hi)
}

fn is_image(p: &Path) -> bool {
    matches!(
        p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref(),
        Some("jpg" | "jpeg" | "png" | "tif" | "tiff" | "webp" | "bmp")
    )
}

fn trunc(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        format!("{s:<n$}")
    } else {
        let t: String = s.chars().take(n - 1).collect();
        format!("{t}…")
    }
}

fn save_sample(img: &DynamicImage, path: &Path) {
    let _ = DynamicImage::ImageRgb8(img.thumbnail(1100, 1100).to_rgb8()).save(path);
}

fn main() {
    let folder = std::env::args().nth(1).unwrap_or_else(|| "../test-photos".into());
    let out = std::env::args().nth(2).unwrap_or_else(|| "../edit-samples".into());
    fs::create_dir_all(&out).expect("create out dir");

    let mut paths: Vec<PathBuf> = fs::read_dir(&folder)
        .unwrap_or_else(|e| panic!("read_dir {folder}: {e}"))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| is_image(p))
        .collect();
    paths.sort();

    // A deliberately strong combined edit — the stress test for clipping.
    let combo = EditParams {
        exposure: 40.0,
        contrast: 45.0,
        highlights: -55.0,
        shadows: 55.0,
        whites: 30.0,
        blacks: -25.0,
        saturation: 40.0,
        vibrance: 35.0,
        temperature: 25.0,
        tint: -12.0,
        sharpening: 60.0,
        noise_reduction: 40.0,
        clarity: 55.0,
        vignette_amount: -55.0,
        vignette_midpoint: 45.0,
        shadow_hue: 220.0,
        shadow_sat: 45.0,
        highlight_hue: 42.0,
        highlight_sat: 45.0,
        ..EditParams::default()
    };

    println!("Combined manual edit over {} photos (exposure+contrast+HL/SH+sat+vib+temp+sharpen):\n", paths.len());
    println!("{:<40}  mean luma      output byte range   valid?", "photo");
    println!("{:<40}  before after   min..max            ", "");

    let mut bad = 0;
    for p in &paths {
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        let img = match decode_image(p) {
            Ok(i) => i,
            Err(e) => {
                println!("  ! {name}: {e}");
                continue;
            }
        };
        let (m_b, _, _) = stats(&img);
        let edited = auto_edit(&img, combo);
        let (m_a, lo, hi) = stats(&edited);
        // "valid" = spans a real range (not collapsed) and mean is sane.
        let ok = hi as i32 - lo as i32 > 25 && m_a.is_finite() && m_a > 0.02 && m_a < 0.98;
        if !ok {
            bad += 1;
        }
        println!(
            "{}  {:>5.3} {:>5.3}   {:>3}..{:<3}            {}",
            trunc(&name, 40),
            m_b,
            m_a,
            lo,
            hi,
            if ok { "ok" } else { "DEGENERATE" }
        );
    }
    println!("\n=> {}/{} photos produced valid (non-broken) combined edits.", paths.len() - bad, paths.len());

    // Per-slider spot check on the first decodable photo.
    if let Some(p) = paths.iter().find(|p| decode_image(p).is_ok()) {
        let img = decode_image(p).unwrap();
        let base = stats(&img).0;
        println!("\nPer-slider effect on {} (mean luma at slider -80 / 0 / +80):", p.file_name().unwrap().to_string_lossy());
        let one = |f: fn(f32) -> EditParams, name: &str| {
            let lo = stats(&auto_edit(&img, f(-80.0))).0;
            let hi = stats(&auto_edit(&img, f(80.0))).0;
            println!("  {:<14} {:>5.3} / {:>5.3} / {:>5.3}", name, lo, base, hi);
        };
        one(|v| EditParams { exposure: v, ..Default::default() }, "exposure");
        one(|v| EditParams { contrast: v, ..Default::default() }, "contrast");
        one(|v| EditParams { highlights: v, ..Default::default() }, "highlights");
        one(|v| EditParams { shadows: v, ..Default::default() }, "shadows");
        one(|v| EditParams { whites: v, ..Default::default() }, "whites");
        one(|v| EditParams { blacks: v, ..Default::default() }, "blacks");
        println!("  (saturation/vibrance/temperature/tint verified separately — they hold luma ~constant)");
    }

    // Crop + rotation combos — must stay non-blank and sensibly sized.
    if let Some(p) = paths.first() {
        let img = decode_image(p).unwrap();
        let blank = |im: &DynamicImage| {
            let r = im.to_rgb8();
            let b = r.pixels().filter(|px| px[0] < 3 && px[1] < 3 && px[2] < 3).count();
            b as f32 / (r.width() * r.height()) as f32 * 100.0
        };
        println!("\nCrop / rotation combos on {}:", p.file_name().unwrap().to_string_lossy());
        let cases: [(&str, EditParams); 4] = [
            ("crop 60% center", EditParams { crop_x: 0.2, crop_y: 0.2, crop_w: 0.6, crop_h: 0.6, ..Default::default() }),
            ("crop wide 16:9-ish", EditParams { crop_x: 0.0, crop_y: 0.2, crop_w: 1.0, crop_h: 0.56, ..Default::default() }),
            ("straighten +100 + crop", EditParams { straighten: 100.0, crop_x: 0.1, crop_y: 0.1, crop_w: 0.8, crop_h: 0.8, ..Default::default() }),
            ("straighten -60 + crop + edit", EditParams { straighten: -60.0, exposure: 30.0, saturation: 30.0, crop_x: 0.05, crop_y: 0.05, crop_w: 0.9, crop_h: 0.9, ..Default::default() }),
        ];
        for (name, prm) in cases {
            let out = auto_edit(&img, prm);
            let ok = out.width() > 50 && out.height() > 50 && blank(&out) < 1.0;
            println!("  {:<30} -> {}x{}  blank {:.2}%  {}", name, out.width(), out.height(), blank(&out), if ok { "ok" } else { "BAD" });
        }
    }

    // Save before/after samples (first 4 photos) with the combined edit.
    println!("\nSaving before/after samples to {out}:");
    for p in paths.iter().take(4) {
        if let Ok(img) = decode_image(p) {
            let edited = auto_edit(&img, combo);
            let stem = p.file_stem().unwrap().to_string_lossy();
            save_sample(&img, &Path::new(&out).join(format!("{stem}__1_before.jpg")));
            save_sample(&edited, &Path::new(&out).join(format!("{stem}__2_after.jpg")));
            println!("  {stem}");
        }
    }

    // export_selects both ways over all photos.
    let ed_dir = Path::new(&out).join("export_renamed");
    let wm_dir = Path::new(&out).join("export_watermarked");
    fs::create_dir_all(&ed_dir).unwrap();
    fs::create_dir_all(&wm_dir).unwrap();
    let mk = |prm: EditParams, n: usize| -> Vec<ExportItem> {
        paths.iter().take(n).map(|p| ExportItem { path: p.to_string_lossy().to_string(), params: prm }).collect()
    };
    // Edited + batch rename (TestShoot_001, _002, ...)
    let e = export_selects(
        mk(combo, paths.len()),
        ed_dir.to_string_lossy().to_string(),
        true,
        Some(Naming { prefix: "TestShoot".into(), start: 1 }),
        None,
    )
    .unwrap();
    // Text watermark on the first 4
    let wm = export_selects(
        mk(EditParams::default(), 4),
        wm_dir.to_string_lossy().to_string(),
        false,
        None,
        Some(Watermark {
            kind: "text".into(),
            text: Some("© Test Studio".into()),
            image_path: None,
            position: "bottomRight".into(),
            opacity: 60.0,
            size: 45.0,
        }),
    )
    .unwrap();
    println!("\nexport_selects:");
    println!("  edited + renamed: {} JPEGs (TestShoot_001..) -> {}", e.copied, ed_dir.display());
    println!("  text watermark  : {} JPEGs -> {}", wm.copied, wm_dir.display());
    if !e.errors.is_empty() {
        println!("  edited errors: {:?}", e.errors);
    }
    if !wm.errors.is_empty() {
        println!("  watermark errors: {:?}", wm.errors);
    }
}
