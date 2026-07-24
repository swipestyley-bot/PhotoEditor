//! Run the full culling analysis (blur, duplicates/bursts, eyes-closed) over a
//! folder of images and print a report. Blur is measured both whole-frame and
//! on the detected face region ("subject blur"), the latter being the effective
//! verdict when a face is present.
//!
//!   cargo run --release --example analyze_folder -- <folder>   (default: ../test-photos)

use std::path::{Path, PathBuf};

use tauri_app_lib::dedup::{
    cluster_by_similarity, exif_timestamp, group_bursts, hash_distance, perceptual_hash,
    Fingerprint, DEFAULT_BURST_GAP_SECS, DEFAULT_DUP_DISTANCE,
};
use tauri_app_lib::face::{EyesState, FacePipeline, DEFAULT_EAR_THRESHOLD};
use tauri_app_lib::vision::{assess_blur, decode_image, BlurAssessment, DEFAULT_BLUR_THRESHOLD};

struct Row {
    name: String,
    w: u32,
    h: u32,
    frame_blur: f64,
    frame_blurry: bool,
    face_blur: Option<BlurAssessment>,
    phash: String,
    ts: Option<chrono::NaiveDateTime>,
    eyes: Option<EyesState>,
    face_err: Option<String>,
}

impl Row {
    /// Effective blur verdict: subject-region blur when a face was found, else
    /// whole-frame.
    fn effective_blurry(&self) -> bool {
        self.face_blur.map(|b| b.is_blurry).unwrap_or(self.frame_blurry)
    }
}

fn is_image(p: &Path) -> bool {
    matches!(
        p.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .as_deref(),
        Some("jpg" | "jpeg" | "png" | "tif" | "tiff" | "webp" | "bmp")
    )
}

fn main() {
    let folder = std::env::args().nth(1).unwrap_or_else(|| "../test-photos".into());
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&folder)
        .unwrap_or_else(|e| panic!("read_dir {folder}: {e}"))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| is_image(p))
        .collect();
    paths.sort();
    println!("Analyzing {} images in {}\n", paths.len(), folder);

    let mut pipe = FacePipeline::new(
        "../models/face_detection_yunet_2023mar.onnx",
        "../models/face_landmarker.onnx",
    )
    .expect("load face pipeline");

    let mut rows: Vec<Row> = Vec::new();
    for p in &paths {
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        let img = match decode_image(p) {
            Ok(i) => i,
            Err(e) => {
                println!("  ! {name}: decode failed: {e}");
                continue;
            }
        };
        let frame = assess_blur(&img, DEFAULT_BLUR_THRESHOLD);
        // Single detection pass: subject-region blur + eyes.
        let (eyes, face_blur, face_err) =
            match pipe.analyze(&img, DEFAULT_EAR_THRESHOLD, DEFAULT_BLUR_THRESHOLD) {
                Ok(a) => (a.eyes, a.face_blur, None),
                Err(e) => (None, None, Some(e)),
            };
        rows.push(Row {
            name,
            w: img.width(),
            h: img.height(),
            frame_blur: frame.score,
            frame_blurry: frame.is_blurry,
            face_blur,
            phash: perceptual_hash(&img),
            ts: exif_timestamp(p),
            eyes,
            face_err,
        });
    }

    let nw = rows.iter().map(|r| r.name.len()).max().unwrap_or(8).max(8);

    // ---- Blur: whole-frame vs subject region ----
    println!("== Blur: whole-frame vs face region (threshold {DEFAULT_BLUR_THRESHOLD}) ==");
    for r in &rows {
        let subj = match r.face_blur {
            Some(b) => format!("{:>8.1} ({})", b.score, if b.is_blurry { "blurry" } else { "sharp " }),
            None => "  no-face        ".to_string(),
        };
        println!(
            "  {:<nw$}  {:>4}x{:<4}  frame {:>8.1} ({})  subject {}  => {}",
            r.name,
            r.w,
            r.h,
            r.frame_blur,
            if r.frame_blurry { "blurry" } else { "sharp " },
            subj,
            if r.effective_blurry() { "BLURRY" } else { "sharp" },
            nw = nw
        );
    }
    let subj: Vec<f64> = rows.iter().filter_map(|r| r.face_blur.map(|b| b.score)).collect();
    if !subj.is_empty() {
        let mut s = subj.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "  -- subject-blur range (faces): min {:.1} | median {:.1} | max {:.1}",
            s[0], s[s.len() / 2], s[s.len() - 1]
        );
    }
    let blurry = rows.iter().filter(|r| r.effective_blurry()).count();
    println!("  -- effective verdict: {blurry}/{} blurry", rows.len());

    // ---- Nearest duplicate neighbour ----
    println!("\n== Nearest neighbour by pHash distance ==");
    for (i, r) in rows.iter().enumerate() {
        let best = rows
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(j, o)| (j, hash_distance(&r.phash, &o.phash).unwrap()))
            .min_by_key(|(_, d)| *d);
        if let Some((j, d)) = best {
            println!("  {:<nw$} -> {:<nw$}  dist {}", r.name, rows[j].name, d, nw = nw);
        }
    }

    // ---- Duplicate clusters ----
    let hashes: Vec<String> = rows.iter().map(|r| r.phash.clone()).collect();
    let clusters = cluster_by_similarity(&hashes, DEFAULT_DUP_DISTANCE).unwrap();
    println!("\n== Duplicate clusters (pHash distance <= {DEFAULT_DUP_DISTANCE}) ==");
    let mut any = false;
    for c in clusters.iter().filter(|c| c.len() > 1) {
        any = true;
        let names: Vec<&str> = c.iter().map(|&i| rows[i].name.as_str()).collect();
        println!("  {{ {} }}", names.join(", "));
    }
    if !any {
        println!("  (none within distance {DEFAULT_DUP_DISTANCE})");
    }

    // ---- Bursts ----
    let fps: Vec<Fingerprint> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| Fingerprint {
            id: i,
            phash: r.phash.clone(),
            timestamp: r.ts,
        })
        .collect();
    let bursts = group_bursts(&fps, DEFAULT_BURST_GAP_SECS);
    let with_ts = rows.iter().filter(|r| r.ts.is_some()).count();
    println!(
        "\n== Bursts (capture-time gap <= {DEFAULT_BURST_GAP_SECS}s; {with_ts}/{} timestamped) ==",
        rows.len()
    );
    if with_ts == 0 {
        println!("  (no EXIF timestamps — cannot group bursts)");
    } else {
        let mut any = false;
        for b in bursts.iter().filter(|b| b.len() > 1) {
            any = true;
            let names: Vec<&str> = b.iter().map(|&i| rows[i].name.as_str()).collect();
            println!("  {{ {} }}", names.join(", "));
        }
        if !any {
            println!("  (no multi-frame bursts)");
        }
    }

    // ---- Face / eyes-closed ----
    println!("\n== Face / eyes-closed (EAR threshold {DEFAULT_EAR_THRESHOLD}) ==");
    let mut faces = 0;
    for r in &rows {
        match (&r.eyes, &r.face_err) {
            (Some(s), _) => {
                faces += 1;
                let v = if s.both_closed {
                    "BOTH CLOSED"
                } else if s.any_closed {
                    "one closed"
                } else {
                    "open"
                };
                println!(
                    "  {:<nw$}  L-EAR {:.3}  R-EAR {:.3}  -> {}",
                    r.name, s.left_ear, s.right_ear, v, nw = nw
                );
            }
            (None, Some(e)) => println!("  {:<nw$}  face error: {}", r.name, e, nw = nw),
            (None, None) => println!("  {:<nw$}  no face detected", r.name, nw = nw),
        }
    }
    println!("  -- {faces}/{} photos had a detected face", rows.len());
}
