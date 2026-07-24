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

**One-command setup:** `pwsh scripts/fetch-models.ps1` downloads both models
into `models/`. (`pwsh scripts/fetch-onnxruntime.ps1` fetches the runtime DLL.)

Both land in the git-ignored `models/` directory at the project root:

```
tauri-app/
  models/
    face_detection_yunet_2023mar.onnx     # detector (Stage 1) — YuNet, MIT
    face_landmarker.onnx                   # landmarks (Stage 2) — FaceMesh, Apache-2.0
  runtime/
    onnxruntime.dll                        # fetched by scripts/fetch-onnxruntime.ps1
```

`models/` and `runtime/` are in `.gitignore` (large binaries are fetched, not
committed). The Stage-by-stage sections below document each model's source and
license; the manual download commands are kept for reference.

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

**ONNX build (already wired up).** MediaPipe ships `.task`/`.tflite`, not ONNX,
so this project uses a pre-converted ONNX build of Google's FaceMesh from the
**[PINTO model zoo](https://github.com/PINTO0309/PINTO_model_zoo) (repo: MIT;
weights: Google Apache-2.0)**. `scripts/fetch-models.ps1` downloads it and
installs it as `models/face_landmarker.onnx`. Its introspected I/O:

- **Input** `input`: `[1, 3, 192, 192]` f32 (NCHW, RGB, normalized to `[0,1]`)
- **Output** `landmarks`: `[1, 1, 1, 1404]` = **468 landmarks × (x, y, z)** in
  192×192 input-pixel space
- **Output** `score`: `[1, 1, 1, 1]` face-presence logit

If you prefer, the official `.task` bundle can be converted yourself
(`tf2onnx` on the extracted `.tflite`) — the weights are the same Apache-2.0.

### Eye indices (implemented in `src/face.rs`)

`src/face.rs` computes EAR from the MediaPipe mesh eye indices:

- Left eye:  `LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]`
- Right eye: `RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]`

Alternatively, skip EAR and read `eyeBlinkLeft`/`eyeBlinkRight` from the
blendshape model if you later switch to the blendshape output.

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
| MediaPipe FaceMesh (weights) | Apache-2.0 | ✅ | Google model card |
| PINTO model zoo (ONNX conversion) | MIT | ✅ | repo `LICENSE` |
| dlib 68-point | 300-W non-commercial | ❌ | — |
| InsightFace pretrained | non-commercial | ❌ | — |
