//! One-off diagnostic: is variance-of-Laplacian scale-dependent? Re-score each
//! saved face crop both raw and after normalizing its long edge to 400px. If the
//! portrait close-ups' scores jump toward the distant faces' range, the low
//! portrait scores were a scale artifact (skin/edge pixel ratio), not softness.
//!
//!   cargo run --release --example blur_scale_probe -- ../face-crops

use std::path::PathBuf;

use image::imageops::FilterType;
use tauri_app_lib::vision::blur_score;

fn main() {
    let folder = std::env::args().nth(1).unwrap_or_else(|| "../face-crops".into());
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&folder)
        .unwrap_or_else(|e| panic!("read_dir {folder}: {e}"))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("png"))
        .collect();
    paths.sort();

    const LONG: u32 = 400;
    println!(
        "{:<58} {:>9}  {:>11}  {:>9}",
        "crop", "raw", "size", "norm400"
    );
    for p in &paths {
        let img = image::open(p).unwrap();
        let raw = blur_score(&img);
        let (w, h) = (img.width(), img.height());
        // Fit within LONGxLONG preserving aspect (all crops are taller than 400 -> downscale only).
        let norm = blur_score(&img.resize(LONG, LONG, FilterType::Lanczos3));
        let name = p.file_name().unwrap().to_string_lossy();
        let name = name.trim_end_matches(".png");
        println!("{name:<58} {raw:>9.1}  {w:>4}x{h:<6}  {norm:>9.1}");
    }
}
