//! Face detection + landmark-based eyes-closed detection.
//!
//! Two-stage ONNX pipeline (both models load via `ort`, dynamic-linked):
//!   1. **YuNet** (`models/face_detection_yunet_2023mar.onnx`, MIT) — detects
//!      face bounding boxes.
//!   2. **MediaPipe FaceMesh** (`models/face_landmarker.onnx`, Apache-2.0) —
//!      478/468-point landmark mesh on the cropped face.
//! Eyes-open/closed is derived from the eye landmarks via **Eye Aspect Ratio
//! (EAR)**, using MediaPipe mesh indices.
//!
//! See [`docs/MODELS.md`](../../docs/MODELS.md) for model sourcing/licensing.
//!
//! ## Validation status
//!
//! The pure EAR/classification logic is unit-tested. The ONNX inference halves
//! ([`FaceDetector`], [`LandmarkModel`]) are written against each model's real,
//! introspected I/O and compile + load, but the numeric pre/post-processing
//! (YuNet score/box decode; FaceMesh input normalization and output coordinate
//! space) is **not yet validated against a real face photo**. Assumptions are
//! marked `VALIDATE:` inline. Run the ignored `detect_real_face` test with a
//! photo to confirm/tune them.

use serde::Serialize;

/// A 2D point in image-pixel coordinates (origin top-left).
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    fn dist(self, other: Point) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Eyes-closed logic (pure, model-agnostic, unit-tested)
// ---------------------------------------------------------------------------

/// MediaPipe FaceMesh indices for the left eye (subject's left), in the p1..p6
/// order the EAR formula expects: [outer corner, top, top, inner corner, bottom,
/// bottom].
pub const LEFT_EYE_IDX: [usize; 6] = [33, 160, 158, 133, 153, 144];
/// MediaPipe FaceMesh indices for the right eye, in p1..p6 order.
pub const RIGHT_EYE_IDX: [usize; 6] = [362, 385, 387, 263, 373, 380];

/// Default EAR below which an eye is considered closed. 0.20–0.25 is typical for
/// MediaPipe-derived EAR; tune per-camera against your own data.
pub const DEFAULT_EAR_THRESHOLD: f32 = 0.22;

/// Eye Aspect Ratio for a single eye given its 6 contour points in p1..p6 order.
///
/// `EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)` — ~0.3–0.4 open, →0 closed.
pub fn eye_aspect_ratio(eye: &[Point; 6]) -> f32 {
    let horizontal = eye[0].dist(eye[3]);
    if horizontal <= f32::EPSILON {
        return 0.0;
    }
    let vertical = eye[1].dist(eye[5]) + eye[2].dist(eye[4]);
    vertical / (2.0 * horizontal)
}

/// Per-eye and overall open/closed classification for a face.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct EyesState {
    pub left_ear: f32,
    pub right_ear: f32,
    /// Both eyes at/under threshold (the eyes-closed frame you'd cull).
    pub both_closed: bool,
    /// Either eye at/under threshold.
    pub any_closed: bool,
}

/// Classify eyes from a full landmark set (MediaPipe mesh ordering).
pub fn classify_eyes(landmarks: &[Point], threshold: f32) -> Result<EyesState, String> {
    let max_idx = LEFT_EYE_IDX
        .iter()
        .chain(RIGHT_EYE_IDX.iter())
        .copied()
        .max()
        .unwrap();
    if landmarks.len() <= max_idx {
        return Err(format!(
            "need at least {} landmarks for eye indices, got {}",
            max_idx + 1,
            landmarks.len()
        ));
    }
    let pick = |idx: [usize; 6]| -> [Point; 6] { idx.map(|i| landmarks[i]) };
    let left_ear = eye_aspect_ratio(&pick(LEFT_EYE_IDX));
    let right_ear = eye_aspect_ratio(&pick(RIGHT_EYE_IDX));
    let left_closed = left_ear <= threshold;
    let right_closed = right_ear <= threshold;
    Ok(EyesState {
        left_ear,
        right_ear,
        both_closed: left_closed && right_closed,
        any_closed: left_closed || right_closed,
    })
}

/// A detected face box in original-image pixel coordinates.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct FaceBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub score: f32,
}

impl FaceBox {
    fn area(&self) -> f32 {
        self.w * self.h
    }

    fn iou(&self, o: &FaceBox) -> f32 {
        let x1 = self.x.max(o.x);
        let y1 = self.y.max(o.y);
        let x2 = (self.x + self.w).min(o.x + o.w);
        let y2 = (self.y + self.h).min(o.y + o.h);
        let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
        let union = self.area() + o.area() - inter;
        if union <= 0.0 {
            0.0
        } else {
            inter / union
        }
    }
}

// ---------------------------------------------------------------------------
// ONNX inference pipeline (YuNet detector + FaceMesh landmarker)
// ---------------------------------------------------------------------------

mod pipeline {
    use std::path::Path;

    use image::{imageops::FilterType, DynamicImage};
    use ndarray::Array4;
    use ort::session::Session;
    use ort::value::TensorRef;

    use super::{classify_eyes, EyesState, FaceBox, Point};

    /// YuNet's fixed square input side.
    const YUNET_INPUT: u32 = 640;
    /// YuNet output strides (feature-map downsampling factors).
    const YUNET_STRIDES: [u32; 3] = [8, 16, 32];
    /// FaceMesh's fixed square input side.
    const FACEMESH_INPUT: u32 = 192;
    /// FaceMesh landmark count (output is this * 3 for x,y,z).
    const FACEMESH_POINTS: usize = 468;

    fn load_session(path: impl AsRef<Path>) -> Result<Session, String> {
        Session::builder()
            .and_then(|mut b| b.commit_from_file(path))
            .map_err(|e| format!("failed to load ONNX model: {e}"))
    }

    /// Stage 1: YuNet face detector.
    pub struct FaceDetector {
        session: Session,
        /// Minimum combined score (sqrt(cls*obj)) to keep a detection.
        pub score_threshold: f32,
        /// IoU threshold for non-max suppression.
        pub nms_threshold: f32,
    }

    impl FaceDetector {
        pub fn from_model_path(path: impl AsRef<Path>) -> Result<Self, String> {
            Ok(Self {
                session: load_session(path)?,
                score_threshold: 0.6,
                nms_threshold: 0.3,
            })
        }

        pub fn detect(&mut self, img: &DynamicImage) -> Result<Vec<FaceBox>, String> {
            let (orig_w, orig_h) = (img.width() as f32, img.height() as f32);

            // Preprocess: resize to 640x640, feed BGR, raw 0-255, NCHW.
            // VALIDATE: YuNet (libfacedetection) trains on BGR with no
            // normalization; direct resize (not letterbox) distorts aspect ratio
            // but keeps faces detectable.
            let resized = img
                .resize_exact(YUNET_INPUT, YUNET_INPUT, FilterType::Triangle)
                .to_rgb8();
            let mut arr = Array4::<f32>::zeros((1, 3, YUNET_INPUT as usize, YUNET_INPUT as usize));
            for (x, y, px) in resized.enumerate_pixels() {
                let [r, g, b] = px.0;
                arr[[0, 0, y as usize, x as usize]] = b as f32;
                arr[[0, 1, y as usize, x as usize]] = g as f32;
                arr[[0, 2, y as usize, x as usize]] = r as f32;
            }

            let outputs = self
                .session
                .run(ort::inputs![TensorRef::from_array_view(&arr).map_err(|e| e.to_string())?])
                .map_err(|e| format!("YuNet inference failed: {e}"))?;

            let (sx, sy) = (orig_w / YUNET_INPUT as f32, orig_h / YUNET_INPUT as f32);
            let mut boxes = Vec::new();

            for &stride in &YUNET_STRIDES {
                let extract = |prefix: &str| -> Result<Vec<f32>, String> {
                    let name = format!("{prefix}_{stride}");
                    let (_shape, data) = outputs[name.as_str()]
                        .try_extract_tensor::<f32>()
                        .map_err(|e| format!("missing YuNet output {name}: {e}"))?;
                    Ok(data.to_vec())
                };
                let cls = extract("cls")?;
                let obj = extract("obj")?;
                let bbox = extract("bbox")?;

                let cols = YUNET_INPUT / stride;
                for i in 0..cls.len() {
                    // VALIDATE: score = sqrt(cls*obj), matching OpenCV FaceDetectorYN.
                    let score = (cls[i] * obj[i]).max(0.0).sqrt();
                    if score < self.score_threshold {
                        continue;
                    }
                    let c = (i as u32 % cols) as f32;
                    let r = (i as u32 / cols) as f32;
                    // VALIDATE: prior-cell decode (cx=(col+dx)*stride, w=exp(dw)*stride).
                    let cx = (c + bbox[i * 4]) * stride as f32;
                    let cy = (r + bbox[i * 4 + 1]) * stride as f32;
                    let w = bbox[i * 4 + 2].exp() * stride as f32;
                    let h = bbox[i * 4 + 3].exp() * stride as f32;
                    boxes.push(FaceBox {
                        x: (cx - w / 2.0) * sx,
                        y: (cy - h / 2.0) * sy,
                        w: w * sx,
                        h: h * sy,
                        score,
                    });
                }
            }

            Ok(nms(boxes, self.nms_threshold))
        }
    }

    /// Stage 2: MediaPipe FaceMesh landmark model.
    pub struct LandmarkModel {
        session: Session,
    }

    impl LandmarkModel {
        pub fn from_model_path(path: impl AsRef<Path>) -> Result<Self, String> {
            Ok(Self {
                session: load_session(path)?,
            })
        }

        /// Run FaceMesh on a cropped face. Returns landmarks in the model's
        /// 192x192 input-pixel space (x,y in 0..192) plus the face-presence score.
        pub fn landmarks(&mut self, crop: &DynamicImage) -> Result<(Vec<Point>, f32), String> {
            // VALIDATE: FaceMesh expects RGB, NCHW, normalized to [0,1].
            let resized = crop
                .resize_exact(FACEMESH_INPUT, FACEMESH_INPUT, FilterType::Triangle)
                .to_rgb8();
            let mut arr =
                Array4::<f32>::zeros((1, 3, FACEMESH_INPUT as usize, FACEMESH_INPUT as usize));
            for (x, y, px) in resized.enumerate_pixels() {
                let [r, g, b] = px.0;
                arr[[0, 0, y as usize, x as usize]] = r as f32 / 255.0;
                arr[[0, 1, y as usize, x as usize]] = g as f32 / 255.0;
                arr[[0, 2, y as usize, x as usize]] = b as f32 / 255.0;
            }

            let outputs = self
                .session
                .run(ort::inputs![TensorRef::from_array_view(&arr).map_err(|e| e.to_string())?])
                .map_err(|e| format!("FaceMesh inference failed: {e}"))?;

            let (_s, lm) = outputs["landmarks"]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("missing FaceMesh 'landmarks' output: {e}"))?;
            let score = outputs["score"]
                .try_extract_tensor::<f32>()
                .ok()
                .and_then(|(_, s)| s.first().copied())
                .unwrap_or(1.0);

            let need = FACEMESH_POINTS * 3;
            if lm.len() < need {
                return Err(format!(
                    "FaceMesh returned {} values, expected {} ({} pts x 3)",
                    lm.len(),
                    need,
                    FACEMESH_POINTS
                ));
            }
            // VALIDATE: FaceMesh outputs landmarks in input-pixel space (0..192).
            let pts = (0..FACEMESH_POINTS)
                .map(|i| Point {
                    x: lm[i * 3],
                    y: lm[i * 3 + 1],
                })
                .collect();
            Ok((pts, score))
        }
    }

    /// Both stages combined: detect the primary face, crop it, run landmarks,
    /// classify eyes.
    pub struct FacePipeline {
        detector: FaceDetector,
        landmarker: LandmarkModel,
    }

    impl FacePipeline {
        pub fn new(
            detector_model: impl AsRef<Path>,
            landmarker_model: impl AsRef<Path>,
        ) -> Result<Self, String> {
            Ok(Self {
                detector: FaceDetector::from_model_path(detector_model)?,
                landmarker: LandmarkModel::from_model_path(landmarker_model)?,
            })
        }

        /// Detect the largest face, run landmarks on it, and classify eyes.
        /// Returns `Ok(None)` when no face passes the detector.
        pub fn analyze_primary_face(
            &mut self,
            img: &DynamicImage,
            ear_threshold: f32,
        ) -> Result<Option<EyesState>, String> {
            let faces = self.detector.detect(img)?;
            let Some(face) = faces
                .into_iter()
                .max_by(|a, b| a.area().partial_cmp(&b.area()).unwrap_or(std::cmp::Ordering::Equal))
            else {
                return Ok(None);
            };

            let (crop, rect) = crop_face(img, &face, 0.25);
            let (lm192, _score) = self.landmarker.landmarks(&crop)?;

            // Map 192-space landmarks back to original-image coordinates.
            let (x0, y0, cw, ch) = rect;
            let landmarks: Vec<Point> = lm192
                .iter()
                .map(|p| Point {
                    x: x0 + p.x / FACEMESH_INPUT as f32 * cw,
                    y: y0 + p.y / FACEMESH_INPUT as f32 * ch,
                })
                .collect();

            Ok(Some(classify_eyes(&landmarks, ear_threshold)?))
        }
    }

    /// Crop a square region around a face box (with margin), clamped to the
    /// image. Returns the crop and its `(x0, y0, w, h)` rect in original pixels.
    fn crop_face(
        img: &DynamicImage,
        face: &FaceBox,
        margin: f32,
    ) -> (DynamicImage, (f32, f32, f32, f32)) {
        let cx = face.x + face.w / 2.0;
        let cy = face.y + face.h / 2.0;
        let side = face.w.max(face.h) * (1.0 + 2.0 * margin);
        let (iw, ih) = (img.width() as f32, img.height() as f32);
        let x0 = (cx - side / 2.0).clamp(0.0, iw - 1.0);
        let y0 = (cy - side / 2.0).clamp(0.0, ih - 1.0);
        let w = (x0 + side).min(iw) - x0;
        let h = (y0 + side).min(ih) - y0;
        let crop = img.crop_imm(x0 as u32, y0 as u32, w.max(1.0) as u32, h.max(1.0) as u32);
        (crop, (x0, y0, w.max(1.0), h.max(1.0)))
    }

    /// Greedy non-max suppression by descending score.
    fn nms(mut boxes: Vec<FaceBox>, iou_threshold: f32) -> Vec<FaceBox> {
        boxes.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut kept: Vec<FaceBox> = Vec::new();
        'candidate: for b in boxes {
            for k in &kept {
                if b.iou(k) > iou_threshold {
                    continue 'candidate;
                }
            }
            kept.push(b);
        }
        kept
    }
}

pub use pipeline::{FaceDetector, FacePipeline, LandmarkModel};

/// Tauri command: detect the primary face in an image and classify eyes-closed.
/// Loads both ONNX models per call. Returns `None` if no face is found.
#[tauri::command]
pub fn analyze_face_eyes(
    image_path: String,
    detector_model: String,
    landmarker_model: String,
    ear_threshold: Option<f32>,
) -> Result<Option<EyesState>, String> {
    let img = image::open(&image_path).map_err(|e| format!("open {image_path}: {e}"))?;
    let mut pipe = FacePipeline::new(&detector_model, &landmarker_model)?;
    pipe.analyze_primary_face(&img, ear_threshold.unwrap_or(DEFAULT_EAR_THRESHOLD))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn landmarks_with_eyes(left: [Point; 6], right: [Point; 6]) -> Vec<Point> {
        let mut pts = vec![Point { x: 0.0, y: 0.0 }; 468];
        for (slot, &p) in LEFT_EYE_IDX.iter().zip(left.iter()) {
            pts[*slot] = p;
        }
        for (slot, &p) in RIGHT_EYE_IDX.iter().zip(right.iter()) {
            pts[*slot] = p;
        }
        pts
    }

    fn open_eye(cx: f32, cy: f32) -> [Point; 6] {
        [
            Point { x: cx - 20.0, y: cy },
            Point { x: cx - 7.0, y: cy - 8.0 },
            Point { x: cx + 7.0, y: cy - 8.0 },
            Point { x: cx + 20.0, y: cy },
            Point { x: cx + 7.0, y: cy + 8.0 },
            Point { x: cx - 7.0, y: cy + 8.0 },
        ]
    }

    fn closed_eye(cx: f32, cy: f32) -> [Point; 6] {
        [
            Point { x: cx - 20.0, y: cy },
            Point { x: cx - 7.0, y: cy - 0.4 },
            Point { x: cx + 7.0, y: cy - 0.4 },
            Point { x: cx + 20.0, y: cy },
            Point { x: cx + 7.0, y: cy + 0.4 },
            Point { x: cx - 7.0, y: cy + 0.4 },
        ]
    }

    #[test]
    fn ear_separates_open_from_closed() {
        let open = eye_aspect_ratio(&open_eye(100.0, 100.0));
        let closed = eye_aspect_ratio(&closed_eye(100.0, 100.0));
        println!("EAR open={open:.3} closed={closed:.3}");
        assert!(open > 0.35 && closed < 0.1 && open > closed * 5.0);
    }

    #[test]
    fn ear_degenerate_eye_is_zero() {
        assert_eq!(eye_aspect_ratio(&[Point { x: 0.0, y: 0.0 }; 6]), 0.0);
    }

    #[test]
    fn classify_both_open() {
        let lm = landmarks_with_eyes(open_eye(80.0, 100.0), open_eye(160.0, 100.0));
        assert!(!classify_eyes(&lm, DEFAULT_EAR_THRESHOLD).unwrap().any_closed);
    }

    #[test]
    fn classify_both_closed() {
        let lm = landmarks_with_eyes(closed_eye(80.0, 100.0), closed_eye(160.0, 100.0));
        assert!(classify_eyes(&lm, DEFAULT_EAR_THRESHOLD).unwrap().both_closed);
    }

    #[test]
    fn classify_one_closed_is_any_not_both() {
        let lm = landmarks_with_eyes(open_eye(80.0, 100.0), closed_eye(160.0, 100.0));
        let s = classify_eyes(&lm, DEFAULT_EAR_THRESHOLD).unwrap();
        assert!(s.any_closed && !s.both_closed);
    }

    #[test]
    fn classify_rejects_too_few_landmarks() {
        assert!(classify_eyes(&[Point { x: 0.0, y: 0.0 }; 10], DEFAULT_EAR_THRESHOLD).is_err());
    }

    /// Confirms onnxruntime.dll loads via ORT_DYLIB_PATH (.cargo/config.toml).
    /// Forces `ort` to dlopen the runtime and read its version — no inference.
    #[test]
    fn onnxruntime_dylib_loads() {
        let dylib = std::env::var("ORT_DYLIB_PATH").unwrap_or_else(|_| "<unset>".into());
        println!("ORT_DYLIB_PATH = {dylib}");
        let info = ort::info();
        println!("ONNX Runtime build info: {info}");
        assert!(!info.is_empty());
    }

    /// Full pipeline against real models + a face photo. Ignored by default
    /// (no test photo yet). Both models are already in `models/`; supply a photo:
    ///
    ///   $env:CULLING_TEST_FACE_IMAGE = "C:\path\to\face.jpg"
    ///   cargo test -p tauri-app detect_real_face -- --ignored --nocapture
    #[test]
    #[ignore = "requires a face photo; set CULLING_TEST_FACE_IMAGE"]
    fn detect_real_face() {
        let image = std::env::var("CULLING_TEST_FACE_IMAGE").expect("set CULLING_TEST_FACE_IMAGE");
        let detector = std::env::var("CULLING_TEST_DETECTOR")
            .unwrap_or_else(|_| "../models/face_detection_yunet_2023mar.onnx".into());
        let landmarker = std::env::var("CULLING_TEST_LANDMARKER")
            .unwrap_or_else(|_| "../models/face_landmarker.onnx".into());

        let img = image::open(&image).expect("open face image");
        let mut pipe = FacePipeline::new(&detector, &landmarker).expect("load pipeline");
        match pipe
            .analyze_primary_face(&img, DEFAULT_EAR_THRESHOLD)
            .expect("analyze")
        {
            Some(state) => println!("eyes: {state:?}"),
            None => println!("no face detected"),
        }
    }
}
