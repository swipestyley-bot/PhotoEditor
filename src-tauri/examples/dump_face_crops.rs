//! Dump the expanded face-box crop (the exact region `subject blur` is measured
//! over) for each image, named with its subject-blur score, so you can eyeball
//! whether "low subject blur" faces are genuinely soft or the metric is wrong.
//!
//!   cargo run --release --example dump_face_crops -- <folder> <out-dir>
//!   (defaults: ../test-photos  ../face-crops)

use std::path::{Path, PathBuf};

use tauri_app_lib::face::FaceDetector;
use tauri_app_lib::vision::{blur_score_region, decode_image};

fn is_image(p: &Path) -> bool {
    matches!(
        p.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .as_deref(),
        Some("jpg" | "jpeg" | "png")
    )
}

fn main() {
    let folder = std::env::args().nth(1).unwrap_or_else(|| "../test-photos".into());
    let out = std::env::args().nth(2).unwrap_or_else(|| "../face-crops".into());
    std::fs::create_dir_all(&out).expect("create out dir");

    let mut paths: Vec<PathBuf> = std::fs::read_dir(&folder)
        .unwrap_or_else(|e| panic!("read_dir {folder}: {e}"))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| is_image(p))
        .collect();
    paths.sort();

    let mut det = FaceDetector::from_model_path("../models/face_detection_yunet_2023mar.onnx")
        .expect("load detector");

    println!("Dumping face crops for {} images -> {}\n", paths.len(), out);
    for p in &paths {
        let stem = p.file_stem().unwrap().to_string_lossy().replace([' ', '.'], "_");
        let img = match decode_image(p) {
            Ok(i) => i,
            Err(e) => {
                println!("  ! {stem}: decode failed: {e}");
                continue;
            }
        };
        let Some(face) = det.primary_face(&img).expect("detect") else {
            println!("  - {stem}: no face");
            continue;
        };
        // Same 10% expansion `FacePipeline::analyze` measures subject blur over.
        let m = 0.1f32;
        let x = (face.x - m * face.w).max(0.0) as u32;
        let y = (face.y - m * face.h).max(0.0) as u32;
        let w = (face.w * (1.0 + 2.0 * m)) as u32;
        let h = (face.h * (1.0 + 2.0 * m)) as u32;
        let score = blur_score_region(&img, x, y, w, h);

        let (iw, ih) = (img.width(), img.height());
        let cx = x.min(iw - 1);
        let cy = y.min(ih - 1);
        let cw = w.min(iw - cx).max(1);
        let ch = h.min(ih - cy).max(1);
        let crop = img.crop_imm(cx, cy, cw, ch);

        // Zero-pad the score so a lexical sort of the folder = ascending sharpness.
        let name = format!("crop_{:07.1}_{stem}.png", score);
        let dst = Path::new(&out).join(&name);
        crop.save(&dst).expect("save crop");
        println!("  {name}  ({cw}x{ch}, score {score:.1})");
    }
    println!("\nOpen {out} and sort by name: softest faces first.", out = out);
}
