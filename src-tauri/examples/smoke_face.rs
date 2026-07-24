//! Smoke-test the face pipeline's ONNX inference plumbing on a synthetic image
//! (no real face). Confirms both models load and `run()` with correct tensor
//! I/O — it does NOT validate detection/landmark accuracy (that needs a photo).
//!
//!   cargo run --example smoke_face

use image::{DynamicImage, Rgb, RgbImage};
use tauri_app_lib::face::{FaceDetector, LandmarkModel};

fn main() {
    let img = DynamicImage::ImageRgb8(RgbImage::from_fn(480, 640, |x, y| {
        Rgb([(x % 256) as u8, (y % 256) as u8, 128])
    }));

    let mut detector =
        FaceDetector::from_model_path("../models/face_detection_yunet_2023mar.onnx").expect("load YuNet");
    let faces = detector.detect(&img).expect("YuNet inference");
    println!("YuNet ran OK — {} face(s) on synthetic image (0 expected)", faces.len());

    let mut landmarker =
        LandmarkModel::from_model_path("../models/face_landmarker.onnx").expect("load FaceMesh");
    let (pts, score) = landmarker.landmarks(&img).expect("FaceMesh inference");
    println!("FaceMesh ran OK — {} landmarks, presence score {score:.4}", pts.len());
    println!("Inference plumbing verified.");
}
