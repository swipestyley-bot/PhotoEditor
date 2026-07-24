//! Folder-level library analysis for the UI.
//!
//! A single pass over a folder that decodes each image once and returns
//! everything the thumbnail grid needs: a thumbnail (base64 data URI), the blur
//! verdict (scale-normalized subject blur when a face is found, else whole
//! frame), eyes-closed state, and duplicate / burst grouping. This is the
//! `analyze_folder` example's logic behind a Tauri command, plus thumbnails.

use std::collections::HashMap;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use base64::Engine as _;
use image::codecs::jpeg::JpegEncoder;
use image::{DynamicImage, ExtendedColorType, ImageEncoder, ImageFormat};
use serde::{Deserialize, Serialize};
use tauri::Manager;

use crate::dedup::{
    cluster_by_similarity, exif_timestamp, group_bursts, perceptual_hash, Fingerprint,
    DEFAULT_BURST_GAP_SECS, DEFAULT_DUP_DISTANCE,
};
use crate::face::{FacePipeline, DEFAULT_EAR_THRESHOLD};
use crate::vision::{assess_blur, decode_image, DEFAULT_BLUR_THRESHOLD};

/// Longest edge (px) of the grid thumbnails we encode.
const THUMB_MAX: u32 = 320;

/// Longest edge (px) of the larger before/after edit preview.
const PREVIEW_MAX: u32 = 900;

/// Longest edge (px) of the full-screen single-photo culling review image.
const REVIEW_MAX: u32 = 1400;

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
/// Encode an already-sized image as a base64 JPEG data URI.
fn encode_jpeg_data_uri(img: &DynamicImage) -> Result<String, String> {
    let mut buf = Vec::new();
    DynamicImage::ImageRgb8(img.to_rgb8())
        .write_to(&mut Cursor::new(&mut buf), ImageFormat::Jpeg)
        .map_err(|e| format!("jpeg encode failed: {e}"))?;
    Ok(format!(
        "data:image/jpeg;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(&buf)
    ))
}

fn thumbnail_data_uri(img: &DynamicImage) -> Result<String, String> {
    encode_jpeg_data_uri(&img.thumbnail(THUMB_MAX, THUMB_MAX))
}

/// Encode an already-sized image as a base64 JPEG data URI at a chosen quality.
fn encode_jpeg_quality(img: &DynamicImage, quality: u8) -> Result<String, String> {
    let rgb = img.to_rgb8();
    let mut buf = Vec::new();
    JpegEncoder::new_with_quality(&mut buf, quality)
        .write_image(rgb.as_raw(), rgb.width(), rgb.height(), ExtendedColorType::Rgb8)
        .map_err(|e| format!("jpeg encode failed: {e}"))?;
    Ok(format!(
        "data:image/jpeg;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(&buf)
    ))
}

/// Decode a photo and return a large JPEG data URI for the full-screen culling
/// review (loaded on demand, one photo at a time, so the initial grid payload
/// stays small).
#[tauri::command]
pub fn large_preview(path: String) -> Result<String, String> {
    let img = decode_image(Path::new(&path))?;
    encode_jpeg_quality(&img.thumbnail(REVIEW_MAX, REVIEW_MAX), 88)
}

/// Camera settings from a photo's EXIF, formatted for display.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExifInfo {
    pub camera: Option<String>,
    pub lens: Option<String>,
    pub iso: Option<String>,
    pub aperture: Option<String>,
    pub shutter: Option<String>,
    pub focal_length: Option<String>,
}

/// Read camera settings (ISO, aperture, shutter, focal length, model) from EXIF.
#[tauri::command]
pub fn read_exif(path: String) -> Result<ExifInfo, String> {
    let file = std::fs::File::open(&path).map_err(|e| format!("open {path}: {e}"))?;
    let mut reader = std::io::BufReader::new(file);
    let exif = exif::Reader::new()
        .read_from_container(&mut reader)
        .map_err(|e| format!("no readable EXIF: {e}"))?;
    let get = |tag: exif::Tag| {
        exif.get_field(tag, exif::In::PRIMARY)
            .map(|f| f.display_value().with_unit(&exif).to_string())
    };
    Ok(ExifInfo {
        camera: get(exif::Tag::Model),
        lens: get(exif::Tag::LensModel),
        iso: get(exif::Tag::PhotographicSensitivity),
        aperture: get(exif::Tag::FNumber),
        shutter: get(exif::Tag::ExposureTime),
        focal_length: get(exif::Tag::FocalLength),
    })
}

/// A named, saved edit-setting combination.
#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Preset {
    pub name: String,
    pub params: crate::edit::EditParams,
}

/// `<app config dir>/presets.json`, creating the dir if needed.
fn presets_path(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let dir = app.path().app_config_dir().map_err(|e| format!("config dir: {e}"))?;
    std::fs::create_dir_all(&dir).map_err(|e| format!("create config dir: {e}"))?;
    Ok(dir.join("presets.json"))
}

fn load_presets(app: &tauri::AppHandle) -> Vec<Preset> {
    presets_path(app)
        .ok()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn write_presets(app: &tauri::AppHandle, presets: &[Preset]) -> Result<(), String> {
    let path = presets_path(app)?;
    let json = serde_json::to_string_pretty(presets).map_err(|e| e.to_string())?;
    std::fs::write(&path, json).map_err(|e| format!("write presets: {e}"))
}

#[tauri::command]
pub fn list_presets(app: tauri::AppHandle) -> Vec<Preset> {
    load_presets(&app)
}

/// Save (or overwrite) a named preset; returns the updated list.
#[tauri::command]
pub fn save_preset(
    app: tauri::AppHandle,
    name: String,
    params: crate::edit::EditParams,
) -> Result<Vec<Preset>, String> {
    let name = name.trim().to_string();
    if name.is_empty() {
        return Err("preset name is empty".into());
    }
    let mut presets = load_presets(&app);
    if let Some(existing) = presets.iter_mut().find(|p| p.name == name) {
        existing.params = params;
    } else {
        presets.push(Preset { name, params });
    }
    write_presets(&app, &presets)?;
    Ok(presets)
}

#[tauri::command]
pub fn delete_preset(app: tauri::AppHandle, name: String) -> Result<Vec<Preset>, String> {
    let mut presets = load_presets(&app);
    presets.retain(|p| p.name != name);
    write_presets(&app, &presets)?;
    Ok(presets)
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

/// One select to export: its file path and the per-photo edit settings.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportItem {
    pub path: String,
    pub params: crate::edit::EditParams,
}

/// Batch-rename pattern: output files become `{prefix}_{NNN}` from `start`.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Naming {
    pub prefix: String,
    pub start: u32,
}

/// Optional export watermark — text or an image, placed at a corner/center.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Watermark {
    pub kind: String, // "text" | "image"
    pub text: Option<String>,
    pub image_path: Option<String>,
    pub position: String, // topLeft | topRight | bottomLeft | bottomRight | center
    pub opacity: f32,     // 0..100
    pub size: f32,        // 0..100
}

/// Export selects into `dest`, leaving originals in place. `corrected` applies
/// each photo's edits; `naming` renames output sequentially; `watermark` stamps
/// text or an image on each export. Anything that renders (edit or watermark) is
/// written as JPEG; otherwise the original is copied. Collisions get ` (n)`.
#[tauri::command]
pub fn export_selects(
    items: Vec<ExportItem>,
    dest: String,
    corrected: bool,
    naming: Option<Naming>,
    watermark: Option<Watermark>,
) -> Result<ExportResult, String> {
    let dest_dir = PathBuf::from(&dest);
    if !dest_dir.is_dir() {
        return Err(format!("destination is not a folder: {dest}"));
    }
    let mut copied = 0;
    let mut errors = Vec::new();
    for (i, item) in items.iter().enumerate() {
        let src = Path::new(&item.path);
        let Some(stem) = src.file_stem() else {
            errors.push(format!("skipped (no file name): {}", item.path));
            continue;
        };
        let edit = corrected && !item.params.is_noop();
        let render = edit || watermark.is_some();
        let ext = if render {
            "jpg".to_string()
        } else {
            src.extension().map(|e| e.to_string_lossy().to_string()).unwrap_or_else(|| "jpg".into())
        };
        let filename = match &naming {
            Some(n) => format!("{}_{:03}.{}", n.prefix, n.start as usize + i, ext),
            None if render => format!("{}.jpg", stem.to_string_lossy()),
            None => src.file_name().unwrap_or(stem).to_string_lossy().to_string(),
        };
        let target = unique_in(&dest_dir, &filename);

        let result = if render {
            decode_image(src).and_then(|img| {
                let img = if edit { crate::edit::auto_edit(&img, item.params) } else { img };
                let img = match &watermark {
                    Some(wm) => apply_watermark(img, wm)?,
                    None => img,
                };
                save_jpeg(&img, &target)
            })
        } else {
            std::fs::copy(src, &target)
                .map(|_| ())
                .map_err(|e| format!("{}: {e}", src.display()))
        };
        match result {
            Ok(()) => copied += 1,
            Err(e) => errors.push(e),
        }
    }
    Ok(ExportResult {
        copied,
        dest,
        errors,
    })
}

fn wm_position(pos: &str, w: i32, h: i32, ew: i32, eh: i32, margin: i32) -> (i32, i32) {
    match pos {
        "topLeft" => (margin, margin),
        "topRight" => (w - ew - margin, margin),
        "bottomLeft" => (margin, h - eh - margin),
        "center" => ((w - ew) / 2, (h - eh) / 2),
        _ => (w - ew - margin, h - eh - margin), // bottomRight
    }
}

/// Load a font for the text watermark. Dev: uses a Windows system font.
/// TODO(ship): bundle a permissively-licensed font for cross-platform builds.
fn load_font() -> Result<ab_glyph::FontVec, String> {
    let candidates = [
        r"C:\Windows\Fonts\segoeuib.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
    ];
    for c in candidates {
        if let Ok(bytes) = std::fs::read(c) {
            if let Ok(font) = ab_glyph::FontVec::try_from_vec(bytes) {
                return Ok(font);
            }
        }
    }
    Err("no system font found for the text watermark".into())
}

fn watermark_text(rgb: &mut image::RgbImage, text: &str, pos: &str, opacity: f32, size: f32) -> Result<(), String> {
    let font = load_font()?;
    let (w, h) = rgb.dimensions();
    let px = (h as f32 * (0.02 + (size / 100.0).clamp(0.0, 1.0) * 0.08)).max(9.0);
    let scale = ab_glyph::PxScale::from(px);
    let (tw, th) = imageproc::drawing::text_size(scale, &font, text);
    let margin = (h as f32 * 0.02).max(6.0) as i32;
    let (x, y) = wm_position(pos, w as i32, h as i32, tw as i32, th as i32, margin);
    let mut mask = image::GrayImage::new(w, h);
    imageproc::drawing::draw_text_mut(&mut mask, image::Luma([255u8]), x, y, scale, &font, text);
    let op = (opacity / 100.0).clamp(0.0, 1.0);
    for (out, m) in rgb.pixels_mut().zip(mask.pixels()) {
        let a = m[0] as f32 / 255.0 * op;
        if a > 0.0 {
            for c in 0..3 {
                out[c] = (out[c] as f32 * (1.0 - a) + 255.0 * a).round() as u8;
            }
        }
    }
    Ok(())
}

fn watermark_image(rgb: &mut image::RgbImage, path: &str, pos: &str, opacity: f32, size: f32) -> Result<(), String> {
    let overlay = image::open(path)
        .map_err(|e| format!("open watermark image {path}: {e}"))?
        .to_rgba8();
    let (w, h) = rgb.dimensions();
    let target_w = ((w as f32) * (0.1 + (size / 100.0).clamp(0.0, 1.0) * 0.4)).max(1.0) as u32;
    let scale = target_w as f32 / overlay.width().max(1) as f32;
    let target_h = ((overlay.height() as f32) * scale).max(1.0) as u32;
    let overlay = image::imageops::resize(&overlay, target_w, target_h, image::imageops::FilterType::Triangle);
    let margin = (w as f32 * 0.02).max(6.0) as i32;
    let (ox, oy) = wm_position(pos, w as i32, h as i32, target_w as i32, target_h as i32, margin);
    let op = (opacity / 100.0).clamp(0.0, 1.0);
    for (wx, wy, wp) in overlay.enumerate_pixels() {
        let (px, py) = (ox + wx as i32, oy + wy as i32);
        if px < 0 || py < 0 || px >= w as i32 || py >= h as i32 {
            continue;
        }
        let a = wp[3] as f32 / 255.0 * op;
        if a > 0.0 {
            let dst = rgb.get_pixel_mut(px as u32, py as u32);
            for c in 0..3 {
                dst[c] = (dst[c] as f32 * (1.0 - a) + wp[c] as f32 * a).round() as u8;
            }
        }
    }
    Ok(())
}

fn apply_watermark(img: DynamicImage, wm: &Watermark) -> Result<DynamicImage, String> {
    let mut rgb = img.to_rgb8();
    match wm.kind.as_str() {
        "text" => {
            let text = wm.text.as_deref().unwrap_or("").trim().to_string();
            if !text.is_empty() {
                watermark_text(&mut rgb, &text, &wm.position, wm.opacity, wm.size)?;
            }
        }
        "image" => {
            if let Some(p) = &wm.image_path {
                watermark_image(&mut rgb, p, &wm.position, wm.opacity, wm.size)?;
            }
        }
        _ => {}
    }
    Ok(DynamicImage::ImageRgb8(rgb))
}

/// Write `img` as a high-quality JPEG to `path`.
fn save_jpeg(img: &DynamicImage, path: &Path) -> Result<(), String> {
    let rgb = img.to_rgb8();
    let file = std::fs::File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    JpegEncoder::new_with_quality(&mut writer, 92)
        .write_image(rgb.as_raw(), rgb.width(), rgb.height(), ExtendedColorType::Rgb8)
        .map_err(|e| format!("encode {}: {e}", path.display()))
}

/// A path in `dir` for `filename` that doesn't already exist, inserting ` (n)`
/// before the extension on collision so an export never clobbers a file.
fn unique_in(dir: &Path, filename: &str) -> PathBuf {
    let first = dir.join(filename);
    if !first.exists() {
        return first;
    }
    let as_path = Path::new(filename);
    let stem = as_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let ext = as_path.extension().map(|e| e.to_string_lossy().to_string());
    for i in 1.. {
        let candidate = dir.join(match &ext {
            Some(e) => format!("{stem} ({i}).{e}"),
            None => format!("{stem} ({i})"),
        });
        if !candidate.exists() {
            return candidate;
        }
    }
    unreachable!()
}

/// Before/after edit preview for one photo, both at [`PREVIEW_MAX`] size.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PreviewPair {
    pub before: String,
    pub after: String,
}

/// Cache of decoded preview-size originals, keyed by path, so dragging a slider
/// re-renders without re-reading the file.
fn preview_cache() -> &'static Mutex<HashMap<String, Arc<DynamicImage>>> {
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<DynamicImage>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Render a before/after edit preview for a single photo at preview size.
/// `before` is geometry-only (straighten [+ crop]) and `after` is the full edit,
/// so hold-to-compare shows the color change at the same framing. When
/// `crop_preview` is true the crop is NOT applied and full straightened frames
/// are returned, so the UI can overlay the crop box. The decoded preview image
/// is cached per path so real-time drags only pay for edit + encode.
#[tauri::command]
pub fn preview_edit(
    path: String,
    params: crate::edit::EditParams,
    crop_preview: bool,
) -> Result<PreviewPair, String> {
    let img = {
        let mut cache = preview_cache().lock().map_err(|e| format!("cache lock: {e}"))?;
        if let Some(img) = cache.get(&path) {
            img.clone()
        } else {
            let small = Arc::new(decode_image(Path::new(&path))?.thumbnail(PREVIEW_MAX, PREVIEW_MAX));
            cache.insert(path.clone(), small.clone());
            small
        }
    };
    let straightened = crate::edit::straighten(&img, params);
    let colored = crate::edit::color(&straightened, params);
    let (before_img, after_img) = if crop_preview {
        (straightened, colored)
    } else {
        (crate::edit::crop(&straightened, params), crate::edit::crop(&colored, params))
    };
    Ok(PreviewPair {
        before: encode_jpeg_data_uri(&before_img)?,
        after: encode_jpeg_data_uri(&after_img)?,
    })
}
