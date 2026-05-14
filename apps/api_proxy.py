"""
VieNeu-TTS REST API

Exposes simple HTTP endpoints on top of the VieNeu SDK.
Mode is controlled by the MODE env var:
  - 'fast'     : GPU, LMDeploy in-process (default when CUDA available)
  - 'standard' : CPU, GGUF quantized model
  - 'remote'   : Connects to a running LMDeploy server (LMDEPLOY_URL required)

Endpoints:
  GET  /health       -> {"status":"ok","mode":"...","model":"..."}
  GET  /voices       -> [{id, name}, ...]
  POST /synthesize   -> {audioBase64, filename, sampleRate}
  POST /clone        -> {audioBase64, filename, sampleRate}
"""

import base64
import io
import logging
import os
import tempfile
import time
from typing import Optional

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("VieNeu.API")

MODE = os.getenv("VIENEU_MODE", "fast")
MODEL_NAME = os.getenv("MODEL_NAME", "pnnbao-ump/VieNeu-TTS-v2")
LMDEPLOY_URL = os.getenv("LMDEPLOY_URL", "http://localhost:23333/v1")
MEMORY_UTIL = float(os.getenv("MEMORY_UTIL", "0.5"))
GPU_DEVICE = os.getenv("GPU_DEVICE", "cuda:0")
HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", "8080"))

app = FastAPI(title="VieNeu-TTS API", version="1.0.0")
tts = None

_NOT_LOADED = "Model not loaded yet"


# ── Startup ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global tts
    gpu_info = GPU_DEVICE if MODE == "fast" else "cpu"
    logger.info(f"Starting VieNeu-TTS API — mode={MODE} model={MODEL_NAME} device={gpu_info}")

    from vieneu import Vieneu

    if MODE == "fast":
        tts = Vieneu(
            mode="fast",
            backbone_repo=MODEL_NAME,
            backbone_device=GPU_DEVICE,
            codec_repo="neuphonic/distill-neucodec",
            codec_device=GPU_DEVICE,
            memory_util=MEMORY_UTIL,
            tp=1,
        )
    elif MODE == "standard":
        tts = Vieneu(
            mode="standard",
            backbone_repo=MODEL_NAME,
            backbone_device="cpu",
            codec_repo="neuphonic/neucodec-onnx-decoder-int8",
            codec_device="cpu",
        )
    elif MODE == "remote":
        tts = Vieneu(
            mode="remote",
            api_base=LMDEPLOY_URL,
            model_name=MODEL_NAME,
            emotion="natural",
        )
    else:
        raise ValueError(f"Unknown VIENEU_MODE: {MODE}. Use 'fast', 'standard', or 'remote'.")

    voices = tts.list_preset_voices()
    logger.info(f"Ready — {len(voices)} preset voice(s) available.")


# ── Error format ─────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_exception_handler(_req: Request, exc: Exception):
    status = getattr(exc, "status_code", 500)
    msg = getattr(exc, "detail", str(exc))
    return JSONResponse(status_code=status, content={"error": msg})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _wav_to_base64(wav: np.ndarray, sample_rate: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, wav, sample_rate, format="WAV")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok" if tts is not None else "loading",
        "mode": MODE,
        "model": MODEL_NAME,
    }


@app.get("/voices")
async def list_voices():
    if tts is None:
        return JSONResponse(status_code=503, content={"error": _NOT_LOADED})
    return [{"id": vid, "name": desc} for desc, vid in tts.list_preset_voices()]


class SynthesizeRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None
    response_format: Optional[str] = "base64"


class CloneRequest(BaseModel):
    text: str
    ref_audio_base64: str
    ref_text: Optional[str] = None
    response_format: Optional[str] = "base64"


def _run_synthesize(text: str, voice_id: Optional[str]):
    voice_data = tts.get_preset_voice(voice_id) if voice_id else None
    return tts.infer(text=text, voice=voice_data)


def _run_clone(text: str, ref_audio_bytes: bytes, ref_text: str):
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    try:
        os.write(fd, ref_audio_bytes)
        os.close(fd)
        return tts.infer(text=text, ref_audio=tmp_path, ref_text=ref_text)
    finally:
        os.unlink(tmp_path)


@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    import asyncio
    if tts is None:
        return JSONResponse(status_code=503, content={"error": _NOT_LOADED})
    try:
        wav = await asyncio.to_thread(_run_synthesize, req.text, req.voice_id)
        filename = f"voice_{req.voice_id or 'default'}_{int(time.time())}.wav"
        return {
            "audioBase64": _wav_to_base64(wav, tts.sample_rate),
            "filename": filename,
            "sampleRate": tts.sample_rate,
        }
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        logger.exception("Synthesize error")
        return JSONResponse(status_code=500, content={"error": f"Synthesis failed: {e}"})


@app.post("/clone")
async def clone(req: CloneRequest):
    import asyncio
    if tts is None:
        return JSONResponse(status_code=503, content={"error": _NOT_LOADED})
    try:
        audio_bytes = base64.b64decode(req.ref_audio_base64)
        wav = await asyncio.to_thread(_run_clone, req.text, audio_bytes, req.ref_text or "")
        filename = f"voice_cloned_{int(time.time())}.wav"
        return {
            "audioBase64": _wav_to_base64(wav, tts.sample_rate),
            "filename": filename,
            "sampleRate": tts.sample_rate,
        }
    except Exception as e:
        logger.exception("Clone error")
        return JSONResponse(status_code=500, content={"error": f"Clone failed: {e}"})


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
