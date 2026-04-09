import csv, cv2, gc, io, json, queue, time, threading, zipfile, numpy as np, os, uuid, shutil, subprocess
from fractions import Fraction
from urllib.parse import quote
from PIL import Image as PILImage
from collections import defaultdict
from datetime import datetime
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:512")
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
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
# Ensures only one job runs inference at a time (GPU memory is not shared)
infer_lock = threading.Lock()

# ── Batch store ────────────────────────────────────────────────────────────────
batches      = {}
batches_lock = threading.Lock()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Load SAM3 once at startup ─────────────────────────────────────────────────
predictor = None

def _purge_predictor(p):
    """Clear all cached image state from a predictor to free GPU memory."""
    try:
        for attr in ("img", "features", "orig_img", "results", "input_size"):
            if hasattr(p, attr):
                setattr(p, attr, None)
        # SAM3SemanticPredictor wraps an inner predictor object
        inner = getattr(p, "predictor", None)
        if inner is not None:
            for attr in ("img", "features", "orig_img", "results", "input_size"):
                if hasattr(inner, attr):
                    setattr(inner, attr, None)
    except Exception:
        pass
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()


def load_model():
    """Load the single shared SAM3 predictor.
    Holds infer_lock while loading so no job can start before the model is ready."""
    global predictor
    infer_lock.acquire()
    try:
        print("Loading SAM3...")
        torch.backends.cudnn.benchmark = True
        overrides = dict(
            conf=0.30, task="segment", mode="predict",
            model=MODEL_PATH, half=True, imgsz=1024, verbose=False
        )
        predictor = SAM3SemanticPredictor(overrides=overrides)
        print("SAM3 ready")
    finally:
        infer_lock.release()

# ── Tiled (SAHI-style) inference ──────────────────────────────────────────────
def tiled_infer(frame, predictor, labels, tile_size=640, overlap=0.25):
    """Slice frame into overlapping tiles, run predictor one tile at a time
    (never holds >1 tile result in GPU memory), merge to full-frame coords + NMS."""
    h, w   = frame.shape[:2]
    stride = int(tile_size * (1 - overlap))

    all_xyxy, all_conf, all_cls, all_masks = [], [], [], []
    has_masks = False

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x2, y2 = min(x + tile_size, w), min(y + tile_size, h)
            x1, y1 = max(0, x2 - tile_size),  max(0, y2 - tile_size)
            crop = frame[y1:y2, x1:x2]
            pil  = PILImage.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            predictor.set_image(pil)
            try:
                with torch.amp.autocast("cuda"):
                    r = predictor(text=labels)
            except torch.cuda.OutOfMemoryError:
                _purge_predictor(predictor)
                continue  # skip this tile rather than crash the whole job

            # Convert to numpy immediately so GPU tensors are freed
            if r and r[0].boxes is not None and len(r[0].boxes):
                dets  = sv.Detections.from_ultralytics(r[0])
                boxes = dets.xyxy.copy()
                boxes[:, [0, 2]] += x1
                boxes[:, [1, 3]] += y1
                all_xyxy.append(boxes)
                all_conf.append(dets.confidence)
                all_cls.append(dets.class_id)
                if dets.mask is not None:
                    has_masks = True
                    full = np.zeros((len(dets.mask), h, w), dtype=bool)
                    for i, m in enumerate(dets.mask):
                        ph = min(m.shape[0], h - y1)
                        pw = min(m.shape[1], w - x1)
                        full[i, y1:y1 + ph, x1:x1 + pw] = m[:ph, :pw]
                    all_masks.append(full)

            del r, pil, crop
            torch.cuda.empty_cache()

    if not all_xyxy:
        return sv.Detections.empty()

    merged = sv.Detections(
        xyxy       = np.concatenate(all_xyxy),
        confidence = np.concatenate(all_conf),
        class_id   = np.concatenate(all_cls),
        mask       = np.concatenate(all_masks) if has_masks and all_masks else None,
    )
    return merged.with_nms(threshold=0.5)


# ── Processing worker ─────────────────────────────────────────────────────────
def process_video(job_id: str, input_path: str, labels: list, confidence: float, every_n: int, imgsz: int = 1024, batch_size: int = 4, tile_size: int = 0, label_colors: dict = None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def update(status=None, progress=None, error=None, output_path=None):
        with jobs_lock:
            if status:      jobs[job_id]["status"]      = status
            if progress is not None:
                            jobs[job_id]["progress"]    = progress
            if error:       jobs[job_id]["error"]       = error
            if output_path: jobs[job_id]["output_path"] = output_path

    try:
        # Probe metadata with ffprobe (handles any codec including AV1)
        probe = subprocess.run([
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", input_path
        ], capture_output=True, text=True)
        if probe.returncode != 0:
            update(status="failed", error="Could not probe video file")
            return
        video_stream = next(
            (s for s in json.loads(probe.stdout).get("streams", []) if s["codec_type"] == "video"),
            None
        )
        if not video_stream:
            update(status="failed", error="No video stream found")
            return

        width        = int(video_stream["width"])
        height       = int(video_stream["height"])
        fps          = float(Fraction(video_stream.get("r_frame_rate", "25/1")))
        total_frames = int(video_stream.get("nb_frames") or 0)
        if total_frames == 0 and "duration" in video_stream:
            total_frames = int(float(video_stream["duration"]) * fps)

        with jobs_lock:
            jobs[job_id]["total_frames"] = total_frames
            jobs[job_id]["fps"]          = round(fps, 1)
            jobs[job_id]["resolution"]   = f"{width}x{height}"

        print(f"[{job_id}] {width}x{height} @ {fps}fps — {total_frames} frames — labels: {labels}")

        # Open ffmpeg decoder pipe (decodes any codec ffmpeg supports)
        ffmpeg_read = subprocess.Popen([
            "ffmpeg", "-i", input_path,
            "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"
        ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        final_path = f"{OUTPUT_DIR}/{job_id}_final.mp4"
        csv_path   = f"{OUTPUT_DIR}/{job_id}_detections.csv"
        ffmpeg_proc = subprocess.Popen([
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "pipe:0",
            "-c:v", "libx264", "-preset", "slow",
            "-crf", "18", "-pix_fmt", "yuv420p",
            final_path
        ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        csv_file   = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame", "track_id", "class", "color", "confidence", "x1", "y1", "x2", "y2"])

        # ByteTrack + annotators (fresh per job)
        tracker   = ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30,
                              minimum_matching_threshold=0.8, frame_rate=int(fps))
        color_palette = sv.ColorPalette(colors=[
            sv.Color.from_hex((label_colors or {}).get(lbl, "#00c864"))
            for lbl in labels
        ])
        mask_ann  = sv.MaskAnnotator(opacity=0.45, color=color_palette)
        box_ann   = sv.BoxAnnotator(thickness=2, color=color_palette)
        lbl_ann   = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, color=color_palette, text_color=sv.Color.WHITE)
        trace_ann = sv.TraceAnnotator(thickness=2, trace_length=40, color=color_palette)
        last_sv_dets     = sv.Detections.empty()
        frame_idx        = 0
        t_start          = time.time()
        detection_counts = defaultdict(int)

        # Acquire the lock then reuse the single shared predictor.
        # infer_lock also guarantees the model is loaded (load_model holds it during startup).
        infer_lock.acquire()
        job_predictor = predictor
        job_predictor.args.conf  = confidence
        job_predictor.args.imgsz = imgsz
        update(status="processing", progress=0)

        frame_bytes  = width * height * 3
        last_sv_dets = sv.Detections.empty()

        while True:
            raw = ffmpeg_read.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()

            # Run SAM3 every N frames
            if frame_idx % every_n == 0:
                if tile_size > 0:
                    last_sv_dets = tiled_infer(frame, job_predictor, labels, tile_size)
                else:
                    pil = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    job_predictor.set_image(pil)
                    try:
                        with torch.amp.autocast("cuda"):
                            results = job_predictor(text=labels)
                    except torch.cuda.OutOfMemoryError:
                        _purge_predictor(job_predictor)
                        # Retry at half resolution
                        job_predictor.args.imgsz = max(640, imgsz // 2)
                        job_predictor.set_image(pil)
                        with torch.amp.autocast("cuda"):
                            results = job_predictor(text=labels)
                        job_predictor.args.imgsz = imgsz
                        print(f"[{job_id}] OOM — retried at imgsz={max(640, imgsz // 2)}")
                    if results and results[0].boxes is not None and len(results[0].boxes):
                        last_sv_dets = sv.Detections.from_ultralytics(results[0])
                    else:
                        last_sv_dets = sv.Detections.empty()
                    del results
                _purge_predictor(job_predictor)

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
                    color = (label_colors or {}).get(name, "")
                    csv_writer.writerow([frame_idx, tid, name, color, f"{conf:.4f}",
                                         f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}"])

            annotated = frame.copy()
            annotated = mask_ann.annotate(annotated, tracked)
            annotated = trace_ann.annotate(annotated, tracked)
            annotated = box_ann.annotate(annotated, tracked)
            if label_texts:
                annotated = lbl_ann.annotate(annotated, tracked, labels=label_texts)

            ffmpeg_proc.stdin.write(annotated.tobytes())

            frame_idx += 1
            progress    = round((frame_idx / max(total_frames, 1)) * 100, 1)
            elapsed     = time.time() - t_start
            fps_so_far  = frame_idx / max(elapsed, 0.001)
            eta_seconds = int((total_frames - frame_idx) / max(fps_so_far, 0.001))

            with jobs_lock:
                jobs[job_id]["progress"]    = progress
                jobs[job_id]["eta_seconds"] = eta_seconds
                jobs[job_id]["proc_fps"]    = round(fps_so_far, 1)
                jobs[job_id]["frames_done"] = frame_idx

            if frame_idx % 50 == 0:
                print(f"[{job_id}] {frame_idx}/{total_frames} ({progress}%) — {fps_so_far:.1f}fps — ETA {eta_seconds}s")

        ffmpeg_read.stdout.close()
        ffmpeg_read.wait()
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
        csv_file.close()

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
        if 'job_predictor' in dir() and job_predictor is not None:
            _purge_predictor(job_predictor)   # clear cached image state; don't delete (shared)
        if infer_lock.locked():
            infer_lock.release()
        if os.path.exists(input_path):
            os.remove(input_path)

# ── Image processing worker ───────────────────────────────────────────────────
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

def process_image(job_id: str, input_path: str, labels: list, confidence: float, imgsz: int, tile_size: int, label_colors: dict = None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def update(status=None, progress=None, error=None, output_path=None):
        with jobs_lock:
            if status:      jobs[job_id]["status"]      = status
            if progress is not None: jobs[job_id]["progress"] = progress
            if error:       jobs[job_id]["error"]       = error
            if output_path: jobs[job_id]["output_path"] = output_path

    try:
        frame = cv2.imread(input_path)
        if frame is None:
            update(status="failed", error="Could not read image file")
            return

        h, w = frame.shape[:2]
        with jobs_lock:
            jobs[job_id]["resolution"]   = f"{w}x{h}"
            jobs[job_id]["total_frames"] = 1

        infer_lock.acquire()
        job_predictor = predictor
        job_predictor.args.conf  = confidence
        job_predictor.args.imgsz = imgsz
        update(status="processing", progress=0)

        if tile_size > 0:
            dets = tiled_infer(frame, job_predictor, labels, tile_size)
        else:
            pil = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            job_predictor.set_image(pil)
            try:
                with torch.amp.autocast("cuda"):
                    results = job_predictor(text=labels)
            except torch.cuda.OutOfMemoryError:
                _purge_predictor(job_predictor)
                job_predictor.args.imgsz = max(640, imgsz // 2)
                job_predictor.set_image(pil)
                with torch.amp.autocast("cuda"):
                    results = job_predictor(text=labels)
                job_predictor.args.imgsz = imgsz
                print(f"[{job_id}] OOM — retried at imgsz={max(640, imgsz // 2)}")
            if results and results[0].boxes is not None and len(results[0].boxes):
                dets = sv.Detections.from_ultralytics(results[0])
            else:
                dets = sv.Detections.empty()
            del results
        _purge_predictor(job_predictor)

        color_palette = sv.ColorPalette(colors=[
            sv.Color.from_hex((label_colors or {}).get(lbl, "#00c864"))
            for lbl in labels
        ])
        mask_ann  = sv.MaskAnnotator(opacity=0.45, color=color_palette)
        box_ann   = sv.BoxAnnotator(thickness=2, color=color_palette)
        lbl_ann   = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, color=color_palette, text_color=sv.Color.WHITE)
        label_texts = []
        detection_counts = defaultdict(int)

        tids  = dets.tracker_id if dets.tracker_id is not None else [None] * len(dets)
        cids  = dets.class_id   if dets.class_id   is not None else []
        confs = dets.confidence if dets.confidence is not None else []
        for tid, cls_id, conf in zip(tids, cids, confs):
            name = labels[cls_id] if cls_id < len(labels) else f"cls_{cls_id}"
            label_texts.append(f"{name} {conf:.2f}")
            detection_counts[name] += 1

        annotated = frame.copy()
        annotated = mask_ann.annotate(annotated, dets)
        annotated = box_ann.annotate(annotated, dets)
        if label_texts:
            annotated = lbl_ann.annotate(annotated, dets, labels=label_texts)

        out_path = f"{OUTPUT_DIR}/{job_id}_detected.jpg"
        cv2.imwrite(out_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])

        csv_path = f"{OUTPUT_DIR}/{job_id}_detections.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            w_ = csv.writer(cf)
            w_.writerow(["class", "color", "confidence", "x1", "y1", "x2", "y2"])
            for cls_id, conf, (x1, y1, x2, y2) in zip(
                dets.class_id   if dets.class_id   is not None else [],
                dets.confidence if dets.confidence is not None else [],
                dets.xyxy       if len(dets) > 0   else []):
                name = labels[cls_id] if cls_id < len(labels) else f"cls_{cls_id}"
                color = (label_colors or {}).get(name, "")
                w_.writerow([name, color, f"{conf:.4f}", f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}"])

        size_mb = round(os.path.getsize(out_path) / 1e6, 1)
        update(status="done", progress=100, output_path=out_path)
        with jobs_lock:
            jobs[job_id]["size_mb"]         = size_mb
            jobs[job_id]["frames_done"]     = 1
            jobs[job_id]["csv_path"]        = csv_path
            jobs[job_id]["detection_counts"]= dict(detection_counts)
            jobs[job_id]["output_type"]     = "image"

        print(f"[{job_id}] Image done — {len(dets)} detections — {out_path}")

    except Exception as e:
        update(status="failed", error=str(e))
        print(f"[{job_id}] ERROR: {e}")
    finally:
        if 'job_predictor' in dir() and job_predictor is not None:
            _purge_predictor(job_predictor)
        if infer_lock.locked():
            infer_lock.release()
        if os.path.exists(input_path):
            os.remove(input_path)


# ── API endpoints ─────────────────────────────────────────────────────────────
@app.post("/process")
async def process(
    file:       UploadFile = File(...),
    labels:       str   = Form(...),   # comma-separated
    confidence:   float = Form(0.30),
    every_n:      int   = Form(5),
    imgsz:        int   = Form(1024),
    batch_size:   int   = Form(4),
    tile_size:    int   = Form(0),
    label_colors: str   = Form("{}"),  # JSON: {"label": "#hexcolor"}
):
    fname = file.filename.lower()
    # Validate
    if not (any(fname.endswith(e) for e in VIDEO_EXTS) or any(fname.endswith(e) for e in IMAGE_EXTS)):
        raise HTTPException(status_code=400, detail="Unsupported file type. Use MP4/AVI/MOV/MKV/WEBM or JPG/PNG/BMP/WEBP.")

    label_list = [l.strip().lower() for l in labels.split(",") if l.strip()]
    if not label_list:
        raise HTTPException(status_code=400, detail="Provide at least one label.")
    try:
        colors_map = json.loads(label_colors)
    except Exception:
        colors_map = {}

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
            "output_type":       "image" if any(fname.endswith(e) for e in IMAGE_EXTS) else "video",
            "filename":          file.filename,
            "labels":            label_list,
        }

    is_image = any(fname.endswith(e) for e in IMAGE_EXTS)
    if is_image:
        t = threading.Thread(
            target=process_image,
            args=(job_id, input_path, label_list, confidence, imgsz, tile_size, colors_map),
            daemon=True
        )
    else:
        t = threading.Thread(
            target=process_video,
            args=(job_id, input_path, label_list, confidence, every_n, imgsz, batch_size, tile_size, colors_map),
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
    if job.get("output_type") == "image":
        download_name = f"{original_name}_detected.jpg"
        media_type    = "image/jpeg"
    else:
        download_name = f"{original_name}_detected.mp4"
        media_type    = "video/mp4"
    encoded_name = quote(download_name)
    return FileResponse(path, media_type=media_type,
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


@app.post("/process_batch")
async def process_batch(
    files:        List[UploadFile] = File(...),
    labels:       str   = Form(...),
    confidence:   float = Form(0.30),
    every_n:      int   = Form(5),
    imgsz:        int   = Form(1024),
    batch_size:   int   = Form(4),
    tile_size:    int   = Form(0),
    label_colors: str   = Form("{}"),
    file_paths:   str   = Form("[]"),   # JSON array of relative paths (e.g. folder/sub/file.mp4)
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    label_list = [l.strip().lower() for l in labels.split(",") if l.strip()]
    if not label_list:
        raise HTTPException(status_code=400, detail="Provide at least one label.")
    try:
        colors_map = json.loads(label_colors)
    except Exception:
        colors_map = {}
    try:
        rel_paths = json.loads(file_paths)
    except Exception:
        rel_paths = []

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    batch_id = str(uuid.uuid4())[:8]
    job_ids  = []

    for i, file in enumerate(files):
        fname    = file.filename.lower()
        rel_path = rel_paths[i] if i < len(rel_paths) else file.filename
        if not (any(fname.endswith(e) for e in VIDEO_EXTS) or any(fname.endswith(e) for e in IMAGE_EXTS)):
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")

        contents = await file.read()
        size_mb  = len(contents) / 1e6
        if size_mb > MAX_FILE_MB:
            raise HTTPException(status_code=413, detail=f"{file.filename} too large. Max {MAX_FILE_MB}MB.")

        job_id     = str(uuid.uuid4())[:8]
        input_path = f"{UPLOAD_DIR}/{job_id}_{file.filename}"
        with open(input_path, "wb") as fh:
            fh.write(contents)

        is_image = any(fname.endswith(e) for e in IMAGE_EXTS)
        with jobs_lock:
            jobs[job_id] = {
                "status":           "queued",
                "progress":         0,
                "total_frames":     0,
                "frames_done":      0,
                "fps":              0,
                "proc_fps":         0,
                "resolution":       "",
                "eta_seconds":      0,
                "size_mb":          0,
                "error":            None,
                "output_path":      None,
                "csv_path":         None,
                "detection_counts": {},
                "output_type":      "image" if is_image else "video",
                "filename":         file.filename,
                "relative_path":    rel_path,   # preserves folder structure
                "labels":           label_list,
                "batch_id":         batch_id,
            }

        if is_image:
            t = threading.Thread(
                target=process_image,
                args=(job_id, input_path, label_list, confidence, imgsz, tile_size, colors_map),
                daemon=True
            )
        else:
            t = threading.Thread(
                target=process_video,
                args=(job_id, input_path, label_list, confidence, every_n, imgsz, batch_size, tile_size, colors_map),
                daemon=True
            )
        t.start()
        job_ids.append(job_id)

    with batches_lock:
        batches[batch_id] = {"job_ids": job_ids}

    return JSONResponse({"batch_id": batch_id, "job_ids": job_ids})


@app.get("/batch/{batch_id}")
def batch_status_endpoint(batch_id: str):
    with batches_lock:
        batch = batches.get(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    with jobs_lock:
        jobs_data = [{**jobs[jid], "job_id": jid} for jid in batch["job_ids"] if jid in jobs]
    return JSONResponse({"batch_id": batch_id, "jobs": jobs_data})


@app.get("/download_batch/{batch_id}")
def download_batch(batch_id: str):
    with batches_lock:
        batch = batches.get(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    with jobs_lock:
        batch_jobs = [{**jobs[jid], "job_id": jid} for jid in batch["job_ids"] if jid in jobs]

    done_jobs = [j for j in batch_jobs if j["status"] == "done"]
    if not done_jobs:
        raise HTTPException(status_code=400, detail="No completed jobs to download.")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for job in done_jobs:
            # Use relative_path to preserve folder structure; fall back to bare filename
            rel      = job.get("relative_path") or job["filename"]
            base     = os.path.splitext(rel)[0]
            out_path = job.get("output_path")
            csv_path = job.get("csv_path")
            ext      = ".jpg" if job.get("output_type") == "image" else ".mp4"
            if out_path and os.path.exists(out_path):
                zf.write(out_path, f"{base}_detected{ext}")
            if csv_path and os.path.exists(csv_path):
                zf.write(csv_path, f"{base}_detections.csv")
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=batch_{batch_id}_results.zip"},
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "gpu": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


@app.get("/gpu")
def gpu_stats():
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
            "--format=csv,noheader,nounits"
        ], text=True, stderr=subprocess.DEVNULL)
        gpus = []
        for line in out.strip().splitlines():
            name, util, mem_used, mem_total, temp, power, power_limit = [v.strip() for v in line.split(",")]
            gpus.append({
                "name":        name,
                "util_pct":    int(util),
                "mem_used_mb": int(mem_used),
                "mem_total_mb":int(mem_total),
                "temp_c":      int(temp),
                "power_w":     round(float(power)),
                "power_limit_w": round(float(power_limit)),
            })
        return {"gpus": gpus}
    except Exception as e:
        return {"gpus": [], "error": str(e)}


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(os.path.dirname(__file__), "frontend.html")) as f:
        return f.read()


@app.on_event("startup")
def startup():
    threading.Thread(target=load_model, daemon=False).start()