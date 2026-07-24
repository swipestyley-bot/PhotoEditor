//! Folder-level library analysis for the UI.
//!
//! A single pass over a folder that decodes each image once and returns
//! everything the thumbnail grid needs: a thumbnail (base64 data URI), the blur
//! verdict (scale-normalized subject blur when a face is found, else whole
//! frame), eyes-closed state, and duplicate / burst grouping. This is the
//! `analyze_folder` example's logic behind a Tauri command, plus thumbnails.

use std::ffi::OsStr;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use base64::Engine as _;
use image::{DynamicImage, ImageFormat};
use serde::Serialize;

use crate::dedup::{
    cluster_by_similarity, exif_timestamp, group_bursts, perceptual_hash, Fingerprint,
    DEFAULT_BURST_GAP_SECS, DEFAULT_DUP_DISTANCE,
};
use crate::face::{FacePipeline, DEFAULT_EAR_THRESHOLD};
use crate::vision::{assess_blur, decode_image, DEFAULT_BLUR_THRESHOLD};

/// Longest edge (px) of the grid thumbnails we encode.
const THUMB_MAX: u32 = 320;

/// Extensions we treat as images. RAW formats are decoded by
/// [`crate::vision::decode_image`]; the browser can't render them, which is
/// exactly why thumbnails are generated backend-side.
const IMAGE_EXTS: &[&str] = &[
    "jpg", "jpeg", "png", "tif", "tiff", "webp", "bmp", // standard
    "cr2", "cr3", "crw", "nef", "nrw", "arw", "srf", "sr2", "dng", "raf", "rw2", "orf", "pef",
    "srw", "raw", "mrw", // RAW
];

fn is_supported(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| IMAGE_EXTS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EyesInfo {
    pub left_ear: f32,
    pub right_ear: f32,
    pub both_closed: bool,
    pub any_closed: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PhotoReport {
    pub path: String,
    pub name: String,
    pub width: u32,
    pub height: u32,
    /// `data:image/jpeg;base64,...`, or empty on decode failure.
    pub thumbnail: String,
    pub frame_blur: f64,
    pub subject_blur: Option<f64>,
    /// Effective verdict: subject blur when a face was found, else frame blur.
    pub is_blurry: bool,
    pub has_face: bool,
    pub eyes: Option<EyesInfo>,
    pub phash: String,
    pub timestamp: Option<String>,
    /// 1-based duplicate-group id (set only when in a group of >1).
    pub cluster: Option<usize>,
    /// 1-based burst id (set only when in a burst of >1).
    pub burst: Option<usize>,
    pub error: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FolderReport {
    pub folder: String,
    pub photos: Vec<PhotoReport>,
    pub duplicate_groups: usize,
    pub burst_groups: usize,
}

/// Encode a fast downscaled thumbnail as a base64 JPEG data URI.
fn thumbnail_data_uri(img: &DynamicImage) -> Result<String, String> {
    let thumb = DynamicImage::ImageRgb8(img.thumbnail(THUMB_MAX, THUMB_MAX).to_rgb8());
    let mut buf = Vec::new();
    thumb
        .write_to(&mut Cursor::new(&mut buf), ImageFormat::Jpeg)
        .map_err(|e| format!("thumbnail encode failed: {e}"))?;
    Ok(format!(
        "data:image/jpeg;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(&buf)
    ))
}

/// Resolve the ONNX models directory. Dev-only: bakes the build machine's
/// `<repo>/models` path (`CARGO_MANIFEST_DIR` is `<repo>/src-tauri`).
/// TODO(bundle): ship the models as Tauri resources and resolve them via the
/// app's resource dir for a distributable build.
fn models_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("models")
}

/// Analyze every supported image in `folder` (Tauri command).
#[tauri::command]
pub fn analyze_library(folder: String) -> Result<FolderReport, String> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&folder)
        .map_err(|e| format!("read folder {folder}: {e}"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| is_supported(p))
        .collect();
    paths.sort();
    analyze_paths(paths, folder)
}

/// Analyze an explicit list of image files (Tauri command). Backs the "Import
/// files…" picker, so a filtered multi-select (JPG/PNG/RAW) works regardless of
/// how the photos are laid out on disk.
#[tauri::command]
pub fn analyze_files(paths: Vec<String>) -> Result<FolderReport, String> {
    let mut paths: Vec<PathBuf> = paths
        .into_iter()
        .map(PathBuf::from)
        .filter(|p| is_supported(p))
        .collect();
    paths.sort();
    let folder = paths
        .first()
        .and_then(|p| p.parent())
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();
    analyze_paths(paths, folder)
}

/// Shared core: decode + analyze each path once, then cluster duplicates and
/// group bursts. Runs synchronously — the caller shows a spinner — so the ONNX
/// pipeline is loaded once for the whole batch.
fn analyze_paths(paths: Vec<PathBuf>, folder: String) -> Result<FolderReport, String> {
    let md = models_dir();
    let mut pipe = FacePipeline::new(
        md.join("face_detection_yunet_2023mar.onnx"),
        md.join("face_landmarker.onnx"),
    )?;

    let mut photos: Vec<PhotoReport> = Vec::with_capacity(paths.len());
    // Hashes aligned to their photo index (skips decode failures, so track the map).
    let mut hashes: Vec<String> = Vec::new();
    let mut hash_photo: Vec<usize> = Vec::new();
    let mut fingerprints: Vec<Fingerprint> = Vec::new();

    for p in &paths {
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        let path = p.to_string_lossy().to_string();
        let idx = photos.len();

        let img = match decode_image(p) {
            Ok(img) => img,
            Err(e) => {
                photos.push(PhotoReport {
                    path,
                    name,
                    width: 0,
                    height: 0,
                    thumbnail: String::new(),
                    frame_blur: 0.0,
                    subject_blur: None,
                    is_blurry: false,
                    has_face: false,
                    eyes: None,
                    phash: String::new(),
                    timestamp: None,
                    cluster: None,
                    burst: None,
                    error: Some(e),
                });
                continue;
            }
        };

        let (width, height) = (img.width(), img.height());
        let thumbnail = thumbnail_data_uri(&img).unwrap_or_default();
        let frame = assess_blur(&img, DEFAULT_BLUR_THRESHOLD);

        // Single detection pass: scale-normalized subject blur + eyes.
        let analysis = pipe
            .analyze(&img, DEFAULT_EAR_THRESHOLD, DEFAULT_BLUR_THRESHOLD)
            .ok();
        let has_face = analysis.as_ref().map(|a| a.face.is_some()).unwrap_or(false);
        let subject_blur = analysis.as_ref().and_then(|a| a.face_blur).map(|b| b.score);
        let is_blurry = analysis
            .as_ref()
            .and_then(|a| a.face_blur)
            .map(|b| b.is_blurry)
            .unwrap_or(frame.is_blurry);
        let eyes = analysis.as_ref().and_then(|a| a.eyes).map(|e| EyesInfo {
            left_ear: e.left_ear,
            right_ear: e.right_ear,
            both_closed: e.both_closed,
            any_closed: e.any_closed,
        });

        let phash = perceptual_hash(&img);
        let timestamp = exif_timestamp(p);

        hashes.push(phash.clone());
        hash_photo.push(idx);
        fingerprints.push(Fingerprint {
            id: idx,
            phash: phash.clone(),
            timestamp,
        });

        photos.push(PhotoReport {
            path,
            name,
            width,
            height,
            thumbnail,
            frame_blur: frame.score,
            subject_blur,
            is_blurry,
            has_face,
            eyes,
            phash,
            timestamp: timestamp.map(|t| t.format("%Y-%m-%d %H:%M:%S").to_string()),
            cluster: None,
            burst: None,
            error: None,
        });
    }

    // Duplicate clusters: indices into `hashes` map back to photos via `hash_photo`.
    let mut duplicate_groups = 0;
    for cluster in cluster_by_similarity(&hashes, DEFAULT_DUP_DISTANCE)? {
        if cluster.len() > 1 {
            duplicate_groups += 1;
            for k in cluster {
                photos[hash_photo[k]].cluster = Some(duplicate_groups);
            }
        }
    }

    // Bursts: group_bursts returns lists of Fingerprint ids (== photo indices).
    let mut burst_groups = 0;
    for burst in group_bursts(&fingerprints, DEFAULT_BURST_GAP_SECS) {
        if burst.len() > 1 {
            burst_groups += 1;
            for idx in burst {
                photos[idx].burst = Some(burst_groups);
            }
        }
    }

    Ok(FolderReport {
        folder,
        photos,
        duplicate_groups,
        burst_groups,
    })
}

/// Result of an export: how many files were copied, plus any per-file errors.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportResult {
    pub copied: usize,
    pub dest: String,
    pub errors: Vec<String>,
}

/// Copy the given source files into `dest` (a folder), leaving originals in
/// place. Names that would collide in the destination get a ` (n)` suffix, so
/// nothing is overwritten. Backs the "export my keepers" action.
#[tauri::command]
pub fn export_kept(paths: Vec<String>, dest: String) -> Result<ExportResult, String> {
    let dest_dir = PathBuf::from(&dest);
    if !dest_dir.is_dir() {
        return Err(format!("destination is not a folder: {dest}"));
    }
    let mut copied = 0;
    let mut errors = Vec::new();
    for p in &paths {
        let src = Path::new(p);
        let Some(name) = src.file_name() else {
            errors.push(format!("skipped (no file name): {p}"));
            continue;
        };
        let target = unique_target(&dest_dir, name);
        match std::fs::copy(src, &target) {
            Ok(_) => copied += 1,
            Err(e) => errors.push(format!("{}: {e}", src.display())),
        }
    }
    Ok(ExportResult {
        copied,
        dest,
        errors,
    })
}

/// A path in `dir` for `name` that doesn't already exist, inserting ` (n)`
/// before the extension on collision so an export never clobbers a file.
fn unique_target(dir: &Path, name: &OsStr) -> PathBuf {
    let first = dir.join(name);
    if !first.exists() {
        return first;
    }
    let as_path = Path::new(name);
    let stem = as_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let ext = as_path.extension().map(|e| e.to_string_lossy().to_string());
    for i in 1.. {
        let fname = match &ext {
            Some(e) => format!("{stem} ({i}).{e}"),
            None => format!("{stem} ({i})"),
        };
        let candidate = dir.join(fname);
        if !candidate.exists() {
            return candidate;
        }
    }
    unreachable!()
}
