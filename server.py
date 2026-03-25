import csv, cv2, time, threading, numpy as np, os, uuid, shutil
from urllib.parse import quote
from PIL import Image as PILImage
from collections import defaultdict
from datetime import datetime
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import supervision as sv
from supervision import ByteTrack
from ultralytics.models.sam import SAM3SemanticPredictor

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "/home/paras/sam3/sam3.pt"
UPLOAD_DIR  = "/home/paras/sam3/video_processor/uploads"
OUTPUT_DIR  = "/home/paras/sam3/video_processor/outputs"
MAX_FILE_MB = 500

# ── Job store ─────────────────────────────────────────────────────────────────
# jobs[job_id] = {status, progress, total_frames, error, output_path}
jobs      = {}
jobs_lock = threading.Lock()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Load SAM3 once at startup ─────────────────────────────────────────────────
predictor = None

def load_model():
    global predictor
    print("Loading SAM3...")
    overrides = dict(
        conf=0.30, task="segment", mode="predict",
        model=MODEL_PATH, half=True, imgsz=644, verbose=False
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)
    print("SAM3 ready")

# ── Processing worker ─────────────────────────────────────────────────────────
def process_video(job_id: str, input_path: str, labels: list, confidence: float, every_n: int):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def update(status=None, progress=None, error=None, output_path=None):
        with jobs_lock:
            if status:      jobs[job_id]["status"]      = status
            if progress is not None:
                            jobs[job_id]["progress"]    = progress
            if error:       jobs[job_id]["error"]       = error
            if output_path: jobs[job_id]["output_path"] = output_path

    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            update(status="failed", error="Could not open video file")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        with jobs_lock:
            jobs[job_id]["total_frames"] = total_frames
            jobs[job_id]["fps"]          = round(fps, 1)
            jobs[job_id]["resolution"]   = f"{width}x{height}"

        print(f"[{job_id}] {width}x{height} @ {fps}fps — {total_frames} frames — labels: {labels}")

        output_path = f"{OUTPUT_DIR}/{job_id}_output.mp4"
        csv_path    = f"{OUTPUT_DIR}/{job_id}_detections.csv"
        writer      = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (width, height)
        )
        csv_file   = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame", "track_id", "class", "confidence", "x1", "y1", "x2", "y2"])

        # ByteTrack + annotators (fresh per job)
        tracker   = ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30,
                              minimum_matching_threshold=0.8, frame_rate=int(fps))
        box_ann   = sv.BoxAnnotator(thickness=2)
        lbl_ann   = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        trace_ann = sv.TraceAnnotator(thickness=2, trace_length=40)

        palette         = [(0,200,100),(255,165,0),(100,149,237),(200,50,200),
                           (0,200,200),(200,200,0),(255,80,80),(80,255,80)]
        last_sv_dets    = sv.Detections.empty()
        frame_idx       = 0
        t_start         = time.time()
        detection_counts = defaultdict(int)

        # Override confidence
        # Build a fresh predictor with the correct confidence for this job
        overrides = dict(
            conf=confidence, task="segment", mode="predict",
            model=MODEL_PATH, half=True, imgsz=644, verbose=False
        )
        job_predictor = SAM3SemanticPredictor(overrides=overrides)

        update(status="processing", progress=0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # SAM3 every N frames
            if frame_idx % every_n == 0:
                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil  = PILImage.fromarray(rgb)
                job_predictor.set_image(pil)
                results = job_predictor(text=labels)
                if results and results[0].boxes is not None:
                    r = results[0]
                    last_sv_dets = sv.Detections(
                        xyxy=r.boxes.xyxy.cpu().numpy(),
                        confidence=r.boxes.conf.cpu().numpy(),
                        class_id=r.boxes.cls.cpu().numpy().astype(int))
                else:
                    last_sv_dets = sv.Detections.empty()

            # ByteTrack every frame
            tracked     = tracker.update_with_detections(last_sv_dets)
            label_texts = []
            tids   = tracked.tracker_id if tracked.tracker_id is not None else []
            cids   = tracked.class_id   if tracked.class_id   is not None else []
            confs  = tracked.confidence if tracked.confidence is not None else []
            bboxes = tracked.xyxy       if len(tids) > 0 else []
            for i, (tid, cls_id, conf) in enumerate(zip(tids, cids, confs)):
                name = labels[cls_id] if cls_id < len(labels) else f"cls_{cls_id}"
                label_texts.append(f"#{tid} {name} {conf:.2f}")
                detection_counts[name] += 1
                if i < len(bboxes):
                    x1, y1, x2, y2 = bboxes[i]
                    csv_writer.writerow([frame_idx, tid, name, f"{conf:.4f}",
                                         f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}"])

            # Annotate
            annotated = frame.copy()
            annotated = trace_ann.annotate(annotated, tracked)
            annotated = box_ann.annotate(annotated, tracked)
            if label_texts:
                annotated = lbl_ann.annotate(annotated, tracked, labels=label_texts)

            writer.write(annotated)

            frame_idx += 1
            progress   = round((frame_idx / max(total_frames, 1)) * 100, 1)
            elapsed    = time.time() - t_start
            frames_left = total_frames - frame_idx
            fps_so_far  = frame_idx / max(elapsed, 0.001)
            eta_seconds = int(frames_left / max(fps_so_far, 0.001))

            with jobs_lock:
                jobs[job_id]["progress"]     = progress
                jobs[job_id]["eta_seconds"]  = eta_seconds
                jobs[job_id]["proc_fps"]     = round(fps_so_far, 1)
                jobs[job_id]["frames_done"]  = frame_idx

            if frame_idx % 50 == 0:
                print(f"[{job_id}] {frame_idx}/{total_frames} ({progress}%) — {fps_so_far:.1f}fps — ETA {eta_seconds}s")

        cap.release()
        writer.release()
        csv_file.close()

        # Re-encode with ffmpeg for browser-compatible H.264
        fixed_path = f"{OUTPUT_DIR}/{job_id}_final.mp4"
        ret = os.system(
            f'ffmpeg -y -i "{output_path}" -c:v libx264 -preset fast '
            f'-crf 23 "{fixed_path}" -loglevel quiet'
        )
        if ret == 0 and os.path.exists(fixed_path):
            os.remove(output_path)
            final_path = fixed_path
        else:
            final_path = output_path   # fallback if ffmpeg not available

        size_mb = round(os.path.getsize(final_path) / 1e6, 1)
        update(status="done", progress=100, output_path=final_path)
        with jobs_lock:
            jobs[job_id]["size_mb"]          = size_mb
            jobs[job_id]["eta_seconds"]      = 0
            jobs[job_id]["csv_path"]         = csv_path
            jobs[job_id]["detection_counts"] = dict(detection_counts)

        print(f"[{job_id}] Done — {size_mb}MB — {final_path}")

    except Exception as e:
        update(status="failed", error=str(e))
        print(f"[{job_id}] ERROR: {e}")
    finally:
        # Clean up upload
        if os.path.exists(input_path):
            os.remove(input_path)

# ── API endpoints ─────────────────────────────────────────────────────────────
@app.post("/process")
async def process(
    file:       UploadFile = File(...),
    labels:     str        = Form(...),   # comma-separated
    confidence: float      = Form(0.30),
    every_n:    int        = Form(5),
):
    # Validate
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        raise HTTPException(status_code=400, detail="Unsupported file type. Use MP4, AVI, MOV, MKV or WEBM.")

    label_list = [l.strip().lower() for l in labels.split(",") if l.strip()]
    if not label_list:
        raise HTTPException(status_code=400, detail="Provide at least one label.")

    # Check file size
    contents = await file.read()
    size_mb  = len(contents) / 1e6
    if size_mb > MAX_FILE_MB:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_FILE_MB}MB.")

    # Save upload
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    job_id     = str(uuid.uuid4())[:8]
    input_path = f"{UPLOAD_DIR}/{job_id}_{file.filename}"
    with open(input_path, "wb") as f:
        f.write(contents)

    # Register job
    with jobs_lock:
        jobs[job_id] = {
            "status":       "queued",
            "progress":     0,
            "total_frames": 0,
            "frames_done":  0,
            "fps":          0,
            "proc_fps":     0,
            "resolution":   "",
            "eta_seconds":  0,
            "size_mb":      0,
            "error":             None,
            "output_path":       None,
            "csv_path":          None,
            "detection_counts":  {},
            "filename":          file.filename,
            "labels":            label_list,
        }

    # Start processing thread
    t = threading.Thread(
        target=process_video,
        args=(job_id, input_path, label_list, confidence, every_n),
        daemon=True
    )
    t.start()

    return JSONResponse({"job_id": job_id})


@app.get("/job/{job_id}")
def job_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(job)


@app.get("/download/{job_id}")
def download(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail="Job not complete yet")
    path = job["output_path"]
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Output file not found")

    original_name = os.path.splitext(job["filename"])[0]
    download_name = f"{original_name}_detected.mp4"
    encoded_name = quote(download_name)
    return FileResponse(path, media_type="video/mp4",
                        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded_name}"})


@app.get("/download_csv/{job_id}")
def download_csv(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail="Job not complete yet")
    csv_path = job.get("csv_path")
    if not csv_path or not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="CSV file not found")
    original_name = os.path.splitext(job["filename"])[0]
    encoded_name  = quote(f"{original_name}_detections.csv")
    return FileResponse(csv_path, media_type="text/csv",
                        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded_name}"})


@app.delete("/job/{job_id}")
def delete_job(job_id: str):
    with jobs_lock:
        job = jobs.pop(job_id, None)
    if job:
        for key in ("output_path", "csv_path"):
            p = job.get(key)
            if p and os.path.exists(p):
                os.remove(p)
    return {"status": "deleted"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "gpu": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(os.path.dirname(__file__), "frontend.html")) as f:
        return f.read()


@app.on_event("startup")
def startup():
    threading.Thread(target=load_model, daemon=False).start()