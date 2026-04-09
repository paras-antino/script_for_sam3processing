import json, os, threading, time, uuid
import torch, torchaudio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sam_audio import SAMAudio, SAMAudioProcessor

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID    = "facebook/sam-audio-large"
UPLOAD_DIR  = "/home/paras/sam_audio/uploads"
OUTPUT_DIR  = "/home/paras/sam_audio/outputs"
MAX_FILE_MB = 200

AUDIO_EXTS = (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aiff", ".aif")

# ── Job store ─────────────────────────────────────────────────────────────────
jobs      = {}
jobs_lock = threading.Lock()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Load model once at startup ────────────────────────────────────────────────
model     = None
processor = None

def load_model():
    global model, processor
    print("Loading SAM-Audio...")
    model     = SAMAudio.from_pretrained(MODEL_ID).eval().cuda()
    processor = SAMAudioProcessor.from_pretrained(MODEL_ID)
    print("SAM-Audio ready")

# ── Processing worker ─────────────────────────────────────────────────────────
def process_audio(job_id: str, input_path: str, description: str,
                  reranking_candidates: int, predict_spans: bool):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def update(status=None, progress=None, error=None):
        with jobs_lock:
            if status   is not None: jobs[job_id]["status"]   = status
            if progress is not None: jobs[job_id]["progress"] = progress
            if error    is not None: jobs[job_id]["error"]    = error

    try:
        update(status="processing", progress=10)

        batch = processor(audios=[input_path], descriptions=[description]).to("cuda")
        update(progress=40)

        with torch.inference_mode():
            result = model.separate(
                batch,
                predict_spans=predict_spans,
                reranking_candidates=reranking_candidates,
            )
        update(progress=85)

        sr = processor.audio_sampling_rate

        target_path   = f"{OUTPUT_DIR}/{job_id}_target.wav"
        residual_path = f"{OUTPUT_DIR}/{job_id}_residual.wav"

        torchaudio.save(target_path,   result.target.cpu(),   sr)
        torchaudio.save(residual_path, result.residual.cpu(), sr)

        spans = None
        if predict_spans and hasattr(result, "spans") and result.spans is not None:
            spans = result.spans.tolist()

        with jobs_lock:
            jobs[job_id]["status"]        = "done"
            jobs[job_id]["progress"]      = 100
            jobs[job_id]["target_path"]   = target_path
            jobs[job_id]["residual_path"] = residual_path
            jobs[job_id]["spans"]         = spans
            jobs[job_id]["sample_rate"]   = sr

        print(f"[{job_id}] Done — target: {target_path}")

    except Exception as e:
        update(status="failed", error=str(e))
        print(f"[{job_id}] ERROR: {e}")
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

# ── API endpoints ─────────────────────────────────────────────────────────────
@app.post("/process")
async def process(
    file:                 UploadFile = File(...),
    description:          str        = Form(...),
    reranking_candidates: int        = Form(1),
    predict_spans:        bool       = Form(False),
):
    fname = file.filename.lower()
    if not any(fname.endswith(e) for e in AUDIO_EXTS):
        raise HTTPException(status_code=400,
            detail=f"Unsupported file type. Use: {', '.join(AUDIO_EXTS).upper()}")

    if not description.strip():
        raise HTTPException(status_code=400, detail="Description is required.")

    contents = await file.read()
    size_mb  = len(contents) / 1e6
    if size_mb > MAX_FILE_MB:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_FILE_MB}MB.")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    job_id     = str(uuid.uuid4())[:8]
    input_path = f"{UPLOAD_DIR}/{job_id}_{file.filename}"
    with open(input_path, "wb") as f:
        f.write(contents)

    with jobs_lock:
        jobs[job_id] = {
            "status":        "queued",
            "progress":      0,
            "error":         None,
            "target_path":   None,
            "residual_path": None,
            "spans":         None,
            "sample_rate":   None,
            "filename":      file.filename,
            "description":   description,
        }

    threading.Thread(
        target=process_audio,
        args=(job_id, input_path, description, reranking_candidates, predict_spans),
        daemon=True,
    ).start()

    return JSONResponse({"job_id": job_id})


@app.get("/job/{job_id}")
def job_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(job)


@app.get("/download/{job_id}/target")
def download_target(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(status_code=400, detail="Job not complete")
    path = job["target_path"]
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    stem = os.path.splitext(job["filename"])[0]
    return FileResponse(path, media_type="audio/wav",
                        headers={"Content-Disposition": f'attachment; filename="{stem}_target.wav"'})


@app.get("/download/{job_id}/residual")
def download_residual(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(status_code=400, detail="Job not complete")
    path = job["residual_path"]
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    stem = os.path.splitext(job["filename"])[0]
    return FileResponse(path, media_type="audio/wav",
                        headers={"Content-Disposition": f'attachment; filename="{stem}_residual.wav"'})


@app.delete("/job/{job_id}")
def delete_job(job_id: str):
    with jobs_lock:
        job = jobs.pop(job_id, None)
    if job:
        for key in ("target_path", "residual_path"):
            p = job.get(key)
            if p and os.path.exists(p):
                os.remove(p)
    return {"status": "deleted"}


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": model is not None,
        "gpu":          torch.cuda.is_available(),
        "device":       torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(os.path.dirname(__file__), "audio_frontend.html")) as f:
        return f.read()


@app.on_event("startup")
def startup():
    threading.Thread(target=load_model, daemon=False).start()
