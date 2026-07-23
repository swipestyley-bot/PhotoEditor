# Face detection & landmark models

The culling backend runs face analysis as a **two-stage ONNX pipeline**:

1. **Face detector** — finds face bounding boxes in the full image.
2. **Landmark model** — runs on each cropped face and returns landmark points;
   we derive **eyes-open/closed** from the eye landmarks (Eye Aspect Ratio) or,
   with MediaPipe, directly from the eye-blink blendshapes.

Both stages load through ONNX Runtime (`ort`, load-dynamic). ONNX Runtime itself
is fetched separately — see [`scripts/fetch-onnxruntime.ps1`](../scripts/fetch-onnxruntime.ps1).

> ⚠️ **Commercial licensing.** This app is sold, so every bundled model must
> permit commercial use. Most popular face-landmark models **do not** — they are
> trained on research-only datasets (iBUG 300-W, WFLW) or ship under
> non-commercial pretrained-weight terms (InsightFace). The recommendations
> below are limited to weights that are genuinely commercial-safe. **Always
> re-read each model's license before shipping; licenses change.**

---

## Where models go

Place downloaded `.onnx` files in the git-ignored `models/` directory at the
project root:

```
tauri-app/
  models/
    face_detection_yunet_2023mar.onnx     # detector (Stage 1)
    face_landmarker.onnx                   # landmarks (Stage 2)  ← see notes
  runtime/
    onnxruntime.dll                        # fetched by scripts/fetch-onnxruntime.ps1
```

`models/` and `runtime/` are in `.gitignore` (large binaries are fetched, not
committed).

---

## Stage 1 — Face detector: **YuNet**  ✅ commercial OK

- **License: MIT** — commercial use permitted. (Verified against the model's
  `LICENSE` in the OpenCV Zoo.)
- Small (~230 KB), fast, CPU-friendly, outputs a bounding box + 5 landmarks
  (eye centers, nose, mouth corners) per face — the box is what we use to crop
  for Stage 2.
- Source: [OpenCV Zoo — face_detection_yunet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)

**Download (note the `media.` host — OpenCV Zoo uses git-LFS, so the normal
`raw.githubusercontent.com` URL returns a 131-byte pointer, not the model):**

```powershell
Invoke-WebRequest `
  -Uri "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" `
  -OutFile "models/face_detection_yunet_2023mar.onnx"
```

`2023mar` has a fixed input shape (simplest to wire up); a newer `2026may`
variant with dynamic input shape also exists in the same folder.

---

## Stage 2 — Landmarks / eyes-closed: **MediaPipe Face Landmarker**  ✅ commercial OK

**There is no freely-available, commercially-licensed _68-point_ model.** The
classic 68-point layout (dlib `shape_predictor_68_face_landmarks`) is trained on
iBUG **300-W, which is research/non-commercial only**. So for a product you sell,
the recommended landmark model is **MediaPipe Face Landmarker**:

- **License: Apache-2.0** — model weights are Apache-2.0; commercial use
  permitted. (MediaPipe framework and model card both state Apache-2.0.)
- Outputs **478 3D landmarks** (dense face mesh, including full eye contours) and
  optional **52 blendshapes** — which include `eyeBlinkLeft` / `eyeBlinkRight`,
  a **direct eyes-closed signal** (often more robust than EAR).
- Official model bundle (`.task`):
  `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`
- Model card / license:
  [Face landmark detection guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)

**ONNX caveat.** MediaPipe ships `.task`/`.tflite`, not ONNX. Two options:
1. **Pre-converted ONNX** — community conversions exist (e.g. the PINTO model
   zoo hosts MediaPipe face-mesh in ONNX). The *weights* remain Google's
   Apache-2.0; verify the specific repo's terms before use.
2. **Convert yourself** — extract the `.tflite` from the `.task` bundle and run
   `tf2onnx` (`python -m tf2onnx.convert --tflite face_landmark.tflite
   --output face_landmarker.onnx`).

### Adapting the code to MediaPipe indices

`src/face.rs` currently computes EAR using **dlib 68-point** eye indices
(`LEFT_EYE_IDX` = 36–41, `RIGHT_EYE_IDX` = 42–47) as a reference implementation.
MediaPipe's mesh uses different indices — the common EAR point sets are:

- Left eye:  `33, 160, 158, 133, 153, 144`
- Right eye: `362, 385, 387, 263, 373, 380`

Swap those into `LEFT_EYE_IDX` / `RIGHT_EYE_IDX` (and set
`ModelConfig::num_points = 478`) when you wire up MediaPipe — or skip EAR
entirely and read the `eyeBlink*` blendshapes if you use the blendshape output.

---

## Models to AVOID for a commercial product ❌

| Model / weights | Why it's not commercial-safe |
|---|---|
| dlib `shape_predictor_68_face_landmarks` | Trained on iBUG **300-W** — non-commercial; commercial use requires licensing from Imperial College London |
| InsightFace **SCRFD**, `2d106det`, ArcFace pretrained | InsightFace *code* is MIT, but **pretrained weights are "non-commercial research only"** |
| PFLD / models trained on **WFLW** | WFLW dataset is academic-research only |
| Any "68-point ONNX" of unclear provenance | Almost always a 300-W/WFLW derivative — assume non-commercial unless proven otherwise |

If you specifically need the 68-point layout commercially, the honest options
are: **license 300-W commercially** from Imperial College, or **train your own**
landmark model on a commercially-licensed dataset.

---

## License summary

| Component | License | Commercial | Verified |
|---|---|---|---|
| ONNX Runtime (`onnxruntime.dll`) | MIT | ✅ | Microsoft |
| `ort` crate | MIT/Apache-2.0 | ✅ | crates.io |
| YuNet detector | MIT | ✅ | OpenCV Zoo `LICENSE` |
| MediaPipe Face Landmarker | Apache-2.0 | ✅ | Google model card |
| dlib 68-point | 300-W non-commercial | ❌ | — |
| InsightFace pretrained | non-commercial | ❌ | — |
