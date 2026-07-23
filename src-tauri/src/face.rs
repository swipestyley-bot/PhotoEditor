//! Face landmark detection + eyes-closed classification via ONNX Runtime (`ort`).
//!
//! ## Design
//!
//! The *decision* logic — Eye Aspect Ratio (EAR) and the open/closed threshold —
//! is pure and fully unit-tested here (see `tests`). It does not depend on any
//! model and is the actual "eyes-closed" algorithm used by dlib/OpenCV tutorials.
//!
//! The *inference* half ([`LandmarkDetector`]) wraps an `ort` session that runs a
//! **68-point facial-landmark ONNX model** and maps its output onto that EAR
//! logic. It compiles and is structured against the real `ort` 2.0 API, but it
//! is **not yet validated end-to-end** because we don't have a model file or a
//! face photo in the test set. The exact input/output contract a given model
//! expects varies, so [`ModelConfig`] exposes the knobs (input size, channel
//! order, normalization, output coordinate space). Validate against your chosen
//! model via the ignored `detect_real_face` integration test.
//!
//! ## Getting a model
//!
//! You need a landmark model that emits the classic **dlib 68-point** layout
//! (eyes at indices 36–41 and 42–47). Options:
//!   * Convert dlib's `shape_predictor_68_face_landmarks` to ONNX, or
//!   * Use a pretrained 68-pt ONNX (e.g. from the PIPNet / FAN / 3DDFA families).
//!
//! These landmark models expect an already-cropped, roughly-centered face. In a
//! full pipeline you'd run a face *detector* first (e.g. YuNet/SCRFD ONNX) to get
//! the crop; here `detect_landmarks` assumes the passed image is that crop (or a
//! full frame with a single centered face). Wiring the detector stage is the
//! natural next step once this half is validated.

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

/// dlib 68-point indices for the left eye contour (subject's left), in the
/// canonical p1..p6 order used by the EAR formula.
pub const LEFT_EYE_IDX: [usize; 6] = [36, 37, 38, 39, 40, 41];
/// dlib 68-point indices for the right eye contour, in p1..p6 order.
pub const RIGHT_EYE_IDX: [usize; 6] = [42, 43, 44, 45, 46, 47];

/// Default EAR below which an eye is considered closed. 0.20–0.25 is the range
/// used by most references; tune per-camera against your own data.
pub const DEFAULT_EAR_THRESHOLD: f32 = 0.22;

/// Eye Aspect Ratio for a single eye given its 6 contour points in p1..p6 order
/// (p1 = outer corner, p4 = inner corner, p2/p3 top lid, p5/p6 bottom lid).
///
/// `EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)`
///
/// A wide-open eye yields ~0.3–0.4; a closed eye collapses toward 0.
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
    /// True when *both* eyes are at/under the threshold (a "blink" / eyes-closed
    /// frame you'd typically cull).
    pub both_closed: bool,
    /// True when *either* eye is at/under the threshold.
    pub any_closed: bool,
}

/// Classify eyes from a full set of 68 landmarks.
pub fn classify_eyes(landmarks: &[Point], threshold: f32) -> Result<EyesState, String> {
    if landmarks.len() < 48 {
        return Err(format!(
            "expected at least 48 landmarks for 68-point eye indices, got {}",
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

/// How a model's output coordinates are expressed.
#[derive(Debug, Clone, Copy)]
pub enum OutputSpace {
    /// Coordinates in [0, 1], relative to the model input crop.
    Normalized,
    /// Coordinates in input-pixel units (0..input_width / 0..input_height).
    InputPixels,
}

/// Preprocessing / postprocessing contract for a specific landmark model.
///
/// Defaults target a common convention (112×112 RGB input, scaled to [0,1],
/// output normalized to [0,1]); adjust to match your model's card.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub input_width: u32,
    pub input_height: u32,
    /// Feed channels as BGR instead of RGB.
    pub bgr: bool,
    /// Per-channel mean subtracted after scaling (in scaled units).
    pub mean: [f32; 3],
    /// Per-channel std divided after mean subtraction.
    pub std: [f32; 3],
    /// Multiplier applied to raw 0–255 bytes before mean/std (1/255 → [0,1]).
    pub scale: f32,
    /// Coordinate space of the model output.
    pub output_space: OutputSpace,
    /// Number of landmark points the model emits (68 for dlib layout).
    pub num_points: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            input_width: 112,
            input_height: 112,
            bgr: false,
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
            scale: 1.0 / 255.0,
            output_space: OutputSpace::Normalized,
            num_points: 68,
        }
    }
}

#[cfg(not(test))]
mod detector {
    use std::path::Path;

    use image::{imageops::FilterType, DynamicImage};
    use ort::session::Session;
    use ort::value::TensorRef;

    use super::{classify_eyes, EyesState, ModelConfig, OutputSpace, Point};

    /// A landmark detector backed by an `ort` ONNX Runtime session.
    pub struct LandmarkDetector {
        session: Session,
        config: ModelConfig,
    }

    impl LandmarkDetector {
        /// Load a landmark model from an `.onnx` file.
        pub fn from_model_path<P: AsRef<Path>>(
            model_path: P,
            config: ModelConfig,
        ) -> Result<Self, String> {
            let session = Session::builder()
                .and_then(|mut b| b.commit_from_file(model_path))
                .map_err(|e| format!("failed to load ONNX model: {e}"))?;
            Ok(Self { session, config })
        }

        /// Run the model on an image (assumed to be a cropped/centered face) and
        /// return landmark points in the *original image's* pixel coordinates.
        pub fn detect_landmarks(&mut self, img: &DynamicImage) -> Result<Vec<Point>, String> {
            let (orig_w, orig_h) = (img.width() as f32, img.height() as f32);
            let input = self.preprocess(img);

            let outputs = self
                .session
                .run(ort::inputs![
                    TensorRef::from_array_view(&input).map_err(|e| e.to_string())?
                ])
                .map_err(|e| format!("inference failed: {e}"))?;

            let (_shape, data) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("failed to read model output: {e}"))?;

            let need = self.config.num_points * 2;
            if data.len() < need {
                return Err(format!(
                    "model produced {} values, expected at least {} ({} points × 2)",
                    data.len(),
                    need,
                    self.config.num_points
                ));
            }

            let (sx, sy) = match self.config.output_space {
                OutputSpace::Normalized => (orig_w, orig_h),
                OutputSpace::InputPixels => (
                    orig_w / self.config.input_width as f32,
                    orig_h / self.config.input_height as f32,
                ),
            };

            Ok((0..self.config.num_points)
                .map(|i| Point {
                    x: data[2 * i] * sx,
                    y: data[2 * i + 1] * sy,
                })
                .collect())
        }

        /// Convenience: detect landmarks then classify eyes-open/closed.
        pub fn analyze_eyes(
            &mut self,
            img: &DynamicImage,
            threshold: f32,
        ) -> Result<EyesState, String> {
            let landmarks = self.detect_landmarks(img)?;
            classify_eyes(&landmarks, threshold)
        }

        /// Resize + normalize into an NCHW `[1, 3, H, W]` f32 tensor.
        fn preprocess(&self, img: &DynamicImage) -> ndarray::Array4<f32> {
            let cfg = &self.config;
            let resized = img
                .resize_exact(cfg.input_width, cfg.input_height, FilterType::Triangle)
                .to_rgb8();
            let (w, h) = (cfg.input_width as usize, cfg.input_height as usize);
            let mut arr = ndarray::Array4::<f32>::zeros((1, 3, h, w));
            for (x, y, px) in resized.enumerate_pixels() {
                let [r, g, b] = px.0;
                let channels = if cfg.bgr { [b, g, r] } else { [r, g, b] };
                for (c, &val) in channels.iter().enumerate() {
                    let scaled = val as f32 * cfg.scale;
                    arr[[0, c, y as usize, x as usize]] = (scaled - cfg.mean[c]) / cfg.std[c];
                }
            }
            arr
        }
    }
}

#[cfg(not(test))]
pub use detector::LandmarkDetector;

#[cfg(test)]
mod tests {
    use super::*;
    use image::imageops::FilterType;

    /// Build a 68-point set where every point is at the origin except the two
    /// eyes, which we set explicitly. Enough to exercise `classify_eyes`.
    fn landmarks_with_eyes(left: [Point; 6], right: [Point; 6]) -> Vec<Point> {
        let mut pts = vec![Point { x: 0.0, y: 0.0 }; 68];
        for (slot, &p) in LEFT_EYE_IDX.iter().zip(left.iter()) {
            pts[*slot] = p;
        }
        for (slot, &p) in RIGHT_EYE_IDX.iter().zip(right.iter()) {
            pts[*slot] = p;
        }
        pts
    }

    /// A synthetic open eye: 40px wide, ~16px tall lid separation -> EAR ~0.4.
    fn open_eye(cx: f32, cy: f32) -> [Point; 6] {
        [
            Point { x: cx - 20.0, y: cy },        // p1 outer corner
            Point { x: cx - 7.0, y: cy - 8.0 },   // p2 top
            Point { x: cx + 7.0, y: cy - 8.0 },   // p3 top
            Point { x: cx + 20.0, y: cy },        // p4 inner corner
            Point { x: cx + 7.0, y: cy + 8.0 },   // p5 bottom
            Point { x: cx - 7.0, y: cy + 8.0 },   // p6 bottom
        ]
    }

    /// A synthetic closed eye: same width, lids nearly touching -> EAR ~0.02.
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
        assert!(open > 0.35, "open eye EAR should be high, got {open:.3}");
        assert!(closed < 0.1, "closed eye EAR should be near zero, got {closed:.3}");
        assert!(open > closed * 5.0);
    }

    #[test]
    fn ear_degenerate_eye_is_zero() {
        let zero = eye_aspect_ratio(&[Point { x: 0.0, y: 0.0 }; 6]);
        assert_eq!(zero, 0.0);
    }

    #[test]
    fn classify_both_open() {
        let lm = landmarks_with_eyes(open_eye(80.0, 100.0), open_eye(160.0, 100.0));
        let s = classify_eyes(&lm, DEFAULT_EAR_THRESHOLD).unwrap();
        assert!(!s.any_closed, "both eyes open -> not closed: {s:?}");
    }

    #[test]
    fn classify_both_closed() {
        let lm = landmarks_with_eyes(closed_eye(80.0, 100.0), closed_eye(160.0, 100.0));
        let s = classify_eyes(&lm, DEFAULT_EAR_THRESHOLD).unwrap();
        assert!(s.both_closed, "both eyes closed -> both_closed: {s:?}");
        assert!(s.any_closed);
    }

    #[test]
    fn classify_one_closed_is_any_not_both() {
        let lm = landmarks_with_eyes(open_eye(80.0, 100.0), closed_eye(160.0, 100.0));
        let s = classify_eyes(&lm, DEFAULT_EAR_THRESHOLD).unwrap();
        assert!(s.any_closed && !s.both_closed, "one closed -> any but not both: {s:?}");
    }

    #[test]
    fn classify_rejects_too_few_landmarks() {
        assert!(classify_eyes(&[Point { x: 0.0, y: 0.0 }; 10], DEFAULT_EAR_THRESHOLD).is_err());
    }

    /// Confirms the ONNX Runtime dylib (onnxruntime.dll) loads via the
    /// ORT_DYLIB_PATH wired up in `.cargo/config.toml`. This does NOT run
    /// inference — it forces `ort` to dlopen the runtime and read its version
    /// string, proving the dynamic-loading setup works end-to-end. If the DLL
    /// is missing or the wrong version, `ort::info()` panics with the reason.
    #[test]
    fn onnxruntime_dylib_loads() {
        let dylib = std::env::var("ORT_DYLIB_PATH").unwrap_or_else(|_| "<unset>".into());
        println!("ORT_DYLIB_PATH = {dylib}");
        let info = ort::info();
        println!("ONNX Runtime build info: {info}");
        assert!(
            info.contains("ORT") || !info.is_empty(),
            "ort::info() returned nothing; the dylib did not load"
        );
    }

    /// End-to-end ONNX inference against a real model + face image. Ignored by
    /// default (no model/photo in the test set yet). To enable:
    ///
    ///   $env:CULLING_TEST_FACE_MODEL = "C:\path\to\landmarks_68.onnx"
    ///   $env:CULLING_TEST_FACE_IMAGE = "C:\path\to\face.jpg"
    ///   cargo test -p tauri-app detect_real_face -- --ignored --nocapture
    ///
    /// Note: this test builds its own session so it can run under `cfg(test)`,
    /// where the production `LandmarkDetector` is compiled out to keep the pure
    /// EAR logic testable without linking against a model.
    #[test]
    #[ignore = "requires an ONNX landmark model + face image"]
    fn detect_real_face() {
        let model = std::env::var("CULLING_TEST_FACE_MODEL")
            .expect("set CULLING_TEST_FACE_MODEL");
        let image = std::env::var("CULLING_TEST_FACE_IMAGE")
            .expect("set CULLING_TEST_FACE_IMAGE");

        let cfg = ModelConfig::default();
        let img = image::open(&image).expect("open face image");
        let resized = img
            .resize_exact(cfg.input_width, cfg.input_height, FilterType::Triangle)
            .to_rgb8();
        let (w, h) = (cfg.input_width as usize, cfg.input_height as usize);
        let mut arr = ndarray::Array4::<f32>::zeros((1, 3, h, w));
        for (x, y, px) in resized.enumerate_pixels() {
            let [r, g, b] = px.0;
            for (c, &val) in [r, g, b].iter().enumerate() {
                arr[[0, c, y as usize, x as usize]] = (val as f32 * cfg.scale - cfg.mean[c]) / cfg.std[c];
            }
        }

        let mut session = ort::session::Session::builder()
            .and_then(|mut b| b.commit_from_file(&model))
            .expect("load model");
        let outputs = session
            .run(ort::inputs![
                ort::value::TensorRef::from_array_view(&arr).unwrap()
            ])
            .expect("inference");
        let (shape, data) = outputs[0].try_extract_tensor::<f32>().expect("extract");
        println!("output shape {shape:?}, {} values", data.len());

        let pts: Vec<Point> = (0..cfg.num_points)
            .map(|i| Point {
                x: data[2 * i] * img.width() as f32,
                y: data[2 * i + 1] * img.height() as f32,
            })
            .collect();
        let state = classify_eyes(&pts, DEFAULT_EAR_THRESHOLD).expect("classify");
        println!("eyes: {state:?}");
    }
}
