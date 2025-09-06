# --- Drowsiness (Open/Closed) – TFLite + MediaPipe Eye Crops + Alarm ---

# ===== Tunables =====
PREPROC = "mobilenet_v2"        # "mobilenet_v2" (you trained with this) or "zero_one"
BINARY_POSITIVE_CLASS = "closed"  # for sigmoid models: does 1.0 mean "closed" or "open"?
CLOSED_THRESH = 0.70            # confidence to consider closed
FRAMES_FOR_ALARM = 12           # consecutive closed frames to beep (~0.4s @30fps)
EYE_MARGIN = 0.40               # expand crop around eye bbox (0.35–0.5 helps)
KEEP_LAST_ROI_FRAMES = 8        # reuse last ROI when detection misses
USE_EAR_BACKUP = True           # geometric backup using EAR (Eye Aspect Ratio)
EAR_THRESH = 0.22               # <~0.20–0.25 typically indicates closed
# =====================

import json, time
from collections import deque
import cv2, numpy as np, tensorflow as tf
import mediapipe as mp

# ---------- labels ----------
with open("labels.json", "r") as f:
    raw = json.load(f)
labels = {int(k): v for k, v in (raw.items() if isinstance(raw, dict) else enumerate(raw))}
closed_idx = next((i for i, v in labels.items() if "close" in v.lower()), 0)
open_idx   = next((i for i, v in labels.items() if "open"  in v.lower()), 1)

# ---------- model ----------
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
in_det = interpreter.get_input_details()
out_det = interpreter.get_output_details()
H, W = in_det[0]["shape"][1], in_det[0]["shape"][2]
out_size = out_det[0]["shape"][-1]  # 1 for sigmoid, 2 for softmax

if PREPROC == "mobilenet_v2":
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mv2_preproc

def preprocess(bgr):
    img = cv2.resize(bgr, (W, H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = mv2_preproc(img) if PREPROC == "mobilenet_v2" else (img / 255.0)
    return np.expand_dims(img, 0)

def run_tflite(bgr):
    inp = preprocess(bgr)
    interpreter.set_tensor(in_det[0]["index"], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(out_det[0]["index"])[0]

    if out_size == 1:
        p1 = float(out[0])  # sigmoid
        if BINARY_POSITIVE_CLASS.lower().startswith("closed"):
            p_closed, p_open = p1, 1.0 - p1
        else:
            p_open, p_closed = p1, 1.0 - p1
        idx  = closed_idx if p_closed >= 0.5 else open_idx
        conf = p_closed if idx == closed_idx else p_open
        return idx, conf, np.array([p_open, p_closed], dtype=np.float32)
    else:
        idx = int(np.argmax(out))
        conf = float(np.max(out))
        p_open = float(out[open_idx]) if open_idx < len(out) else (1.0 - conf if idx != open_idx else conf)
        p_closed = float(out[closed_idx]) if closed_idx < len(out) else (1.0 - conf if idx != closed_idx else conf)
        return idx, conf, np.array([p_open, p_closed], dtype=np.float32)

# ---------- mediapipe face mesh ----------
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# selected landmarks for eyes (6 points each) for EAR + tight bbox
LEFT_EYE  = [33, 160, 158, 133, 153, 144]   # [p1,p2,p3,p4,p5,p6]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

def _bbox_from_landmarks(lms, img_w, img_h, margin=0.40):
    xs = np.array([p.x * img_w for p in lms], dtype=np.float32)
    ys = np.array([p.y * img_h for p in lms], dtype=np.float32)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    w, h = x2 - x1, y2 - y1
    x1 -= margin * w; y1 -= margin * h; x2 += margin * w; y2 += margin * h
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(img_w, int(x2)), min(img_h, int(y2))
    return x1, y1, x2 - x1, y2 - y1

def _ear_from_points(pts):
    # EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
    def d(a, b): return np.linalg.norm(pts[a] - pts[b])
    return (d(1, 5) + d(2, 4)) / (2.0 * d(0, 3) + 1e-6)

def eye_rois(frame):
    """Return [((x,y,w,h), eye_img), ...], (ear_left, ear_right). Uses MediaPipe Face Mesh."""
    h, w = frame.shape[:2]
    res = mp_face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return [], (None, None)

    lm = res.multi_face_landmarks[0].landmark

    # Left eye crop + EAR
    lx, ly, lw, lh = _bbox_from_landmarks([lm[i] for i in LEFT_EYE],  w, h, margin=EYE_MARGIN)
    left_crop = frame[ly:ly+lh, lx:lx+lw]
    Lpts = np.array([[lm[i].x * w, lm[i].y * h] for i in LEFT_EYE], dtype=np.float32)
    ear_left = _ear_from_points(Lpts)

    # Right eye crop + EAR
    rx, ry, rw, rh = _bbox_from_landmarks([lm[i] for i in RIGHT_EYE], w, h, margin=EYE_MARGIN)
    right_crop = frame[ry:ry+rh, rx:rx+rw]
    Rpts = np.array([[lm[i].x * w, lm[i].y * h] for i in RIGHT_EYE], dtype=np.float32)
    ear_right = _ear_from_points(Rpts)

    rois = [((lx, ly, lw, lh), left_crop), ((rx, ry, rw, rh), right_crop)]
    return rois, (ear_left, ear_right)

# ---------- alarm ----------
try:
    import winsound
    def beep(): winsound.Beep(2500, 500)
except Exception:
    def beep(): pass

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("❌ Webcam not available"); return

    last_eye_rois = deque(maxlen=KEEP_LAST_ROI_FRAMES)
    closed_streak, last_beep = 0, 0.0
    smooth_closed = deque(maxlen=15)

    while True:
        ok, frame = cap.read()
        if not ok: break

        rois, (earL, earR) = eye_rois(frame)

        # Reuse last known eye crops if detection misses
        if not rois and len(last_eye_rois) > 0:
            rois = [last_eye_rois[-1][0], last_eye_rois[-1][1]]
            earL, earR = last_eye_rois[-1][2], last_eye_rois[-1][3]

        probs_display = None
        preds = []
        if rois:
            for (x, y, w, h), eye in rois:
                idx, conf, probs = run_tflite(eye)
                preds.append((idx, conf))
                probs_display = probs
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 1)
            last_eye_rois.append((rois[0], rois[1], earL, earR))

            # model says closed?
            model_closed = preds and all(i == closed_idx and c >= CLOSED_THRESH for i, c in preds)

            # EAR backup (average of both eyes if available)
            if USE_EAR_BACKUP and (earL is not None and earR is not None):
                ear_avg = (earL + earR) / 2.0
                ear_closed = ear_avg < EAR_THRESH
            else:
                ear_closed = False

            is_closed = model_closed or ear_closed
            label = labels[closed_idx] if is_closed else labels[open_idx]
            conf_to_show = float(np.mean([c for _, c in preds])) if preds else 0.0
        else:
            # final fallback (rare): whole frame
            idx, conf_to_show, probs = run_tflite(frame)
            is_closed = (idx == closed_idx and conf_to_show >= CLOSED_THRESH)
            label = labels[idx]
            probs_display = probs

        smooth_closed.append(1 if is_closed else 0)
        closed_streak = closed_streak + 1 if is_closed else 0

        now = time.time()
        if closed_streak >= FRAMES_FOR_ALARM and (now - last_beep) > 2.0:
            last_beep = now; beep()

        # ---- UI ----
        color = (0, 0, 255) if sum(smooth_closed) > len(smooth_closed)//2 else (0, 200, 0)
        cv2.putText(frame, f"{label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        if probs_display is not None and len(probs_display) >= 2:
            p_open, p_closed = probs_display[0], probs_display[1]
            cv2.putText(frame, f"Open:{p_open*100:.0f}%  Closed:{p_closed*100:.0f}%",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        if USE_EAR_BACKUP and (earL is not None and earR is not None):
            cv2.putText(frame, f"EAR:{((earL+earR)/2.0):.3f}",
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,255,200), 2)

        cv2.imshow("Drowsiness Detection (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
