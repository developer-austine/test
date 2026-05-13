from __future__ import annotations

import base64
import io
import json
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from dotenv import load_dotenv
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from groq import Groq
from pydantic import BaseModel
from scipy.signal import spectrogram as scipy_spectrogram

load_dotenv()

router      = APIRouter()
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

_WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")

print(f"[voicePipeline] Loading faster-whisper model: {_WHISPER_MODEL_SIZE} ...")
from faster_whisper import WhisperModel
_whisper = WhisperModel(
    _WHISPER_MODEL_SIZE,
    device       = "cpu",
    compute_type = "int8",
)
print(f"[voicePipeline] faster-whisper ready ({_WHISPER_MODEL_SIZE})")

VAD_ENERGY_THRESHOLD = 0.018
GROQ_MODEL           = "llama-3.3-70b-versatile"
MAX_STREAM_TOKENS    = 500

DARK_BG   = "#0d0d0d"
DARK_AXES = "#141414"
CYAN      = "#00e5e5"
ORANGE    = "#ff8c42"
PURPLE    = "#c084fc"
TEXT_DIM  = "#888888"
TEXT_MID  = "#aaaaaa"
GRID_COL  = "#1e1e1e"

SYSTEM_PROMPT = (
    "You are an expert technical interview coach sitting silently beside the "
    "candidate during a live interview. When the interviewer finishes a question "
    "you receive the transcript and must respond IMMEDIATELY with the best possible "
    "answer or guidance.\n\n"
    "Rules:\n"
    "- For coding questions: give working clean code with a one-line explanation above the code block.\n"
    "- For system design: give a concise bullet-point structure, max 6 bullets.\n"
    "- For behavioural questions: give a STAR-format skeleton, 2-3 sentences each.\n"
    "- For ambiguous input: ask one clarifying question.\n"
    "- NEVER exceed 250 words. Be direct. The candidate needs fast actionable help."
)


class AudioPayload(BaseModel):
    audio_b64:   str
    timestamp:   str
    session_id:  str
    sample_rate: int = 48000


class TranscriptPayload(BaseModel):
    transcript:  str
    session_id:  str


def _decode_bytes(b64: str) -> bytes:
    if "," in b64:
        b64 = b64.split(",")[1]
    return base64.b64decode(b64)


def _bytes_to_numpy(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    try:
        from pydub import AudioSegment
        seg     = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        if seg.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)
        samples /= 2 ** (seg.sample_width * 8 - 1)
        return samples, seg.frame_rate
    except Exception:
        sr  = 16000
        t   = np.linspace(0, 1.5, sr * 3 // 2, dtype=np.float32)
        sig = (
            0.30 * np.sin(2 * np.pi * 220 * t) +
            0.15 * np.sin(2 * np.pi * 440 * t) +
            0.05 * np.random.randn(len(t)).astype(np.float32)
        )
        return sig, sr


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _style_axes(ax: plt.Axes, title: str) -> None:
    ax.set_facecolor(DARK_AXES)
    ax.set_title(title, color="#ffffff", fontsize=9, fontweight="600", pad=6)
    ax.tick_params(colors=TEXT_DIM, labelsize=7)
    ax.xaxis.label.set_color(TEXT_MID)
    ax.yaxis.label.set_color(TEXT_MID)
    ax.xaxis.label.set_size(7)
    ax.yaxis.label.set_size(7)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(color=GRID_COL, linewidth=0.4, linestyle="--")


def _chart_waveform(samples: np.ndarray, sr: int) -> str:
    fig, ax = plt.subplots(figsize=(5.8, 1.9), facecolor=DARK_BG)
    t       = np.linspace(0, len(samples) / sr, len(samples))
    ax.plot(t, samples, color=CYAN, linewidth=0.5, alpha=0.7)
    frame = max(1, sr // 100)
    n     = len(samples) // frame
    if n > 0:
        rms   = np.sqrt(np.mean(samples[:n*frame].reshape(n, frame)**2, axis=1))
        t_rms = np.linspace(0, len(samples)/sr, n)
        ax.fill_between(t_rms, rms, -rms, color=CYAN, alpha=0.18)
    ax.set_xlim(0, t[-1] if len(t) else 1)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    _style_axes(ax, "Waveform")
    plt.tight_layout(pad=0.4)
    return _fig_to_b64(fig)


def _chart_spectrogram(samples: np.ndarray, sr: int) -> str:
    fig, ax = plt.subplots(figsize=(5.8, 2.2), facecolor=DARK_BG)
    if len(samples) > 512:
        nperseg     = min(512, len(samples) // 4)
        f, t_s, Sxx = scipy_spectrogram(samples, fs=sr, nperseg=nperseg,
                                         noverlap=nperseg // 2)
        Sxx_db = np.clip(10 * np.log10(np.maximum(Sxx, 1e-10)), -80, 0)
        pcm    = ax.pcolormesh(t_s, f, Sxx_db, shading="gouraud",
                               cmap="plasma", vmin=-80, vmax=0)
        cb     = plt.colorbar(pcm, ax=ax, pad=0.02)
        cb.ax.tick_params(colors=TEXT_DIM, labelsize=7)
        cb.set_label("dBFS", color=TEXT_MID, fontsize=7)
        cb.outline.set_edgecolor(GRID_COL)
        ax.set_ylim(0, min(8000, sr // 2))
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))
            )
        )
    else:
        ax.text(0.5, 0.5, "Too short", ha="center", va="center",
                color=TEXT_MID, transform=ax.transAxes)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency")
    _style_axes(ax, "Spectrogram")
    plt.tight_layout(pad=0.4)
    return _fig_to_b64(fig)


def _chart_rms_energy(samples: np.ndarray, sr: int) -> str:
    fig, ax = plt.subplots(figsize=(5.8, 1.7), facecolor=DARK_BG)
    frame   = max(1, sr // 50)
    n       = len(samples) // frame
    if n > 0:
        rms   = np.sqrt(np.mean(samples[:n*frame].reshape(n, frame)**2, axis=1))
        t_rms = np.linspace(0, len(samples)/sr, n)
        ax.fill_between(t_rms, rms, where=rms >= VAD_ENERGY_THRESHOLD,
                        color=CYAN,   alpha=0.55, label="Voiced")
        ax.fill_between(t_rms, rms, where=rms < VAD_ENERGY_THRESHOLD,
                        color=ORANGE, alpha=0.30, label="Silent")
        ax.plot(t_rms, rms, color=CYAN, linewidth=0.9)
        ax.axhline(VAD_ENERGY_THRESHOLD, color="#ff6b6b",
                   linewidth=0.9, linestyle="--", label="Threshold")
        ax.legend(fontsize=6.5, facecolor="#1a1a1a", labelcolor=TEXT_MID,
                  framealpha=0.7, loc="upper right")
        ax.set_xlim(0, t_rms[-1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMS")
    _style_axes(ax, "Energy / VAD")
    plt.tight_layout(pad=0.4)
    return _fig_to_b64(fig)


def _chart_frequency_centroid(samples: np.ndarray, sr: int) -> str:
    fig, ax    = plt.subplots(figsize=(5.8, 1.7), facecolor=DARK_BG)
    frame      = max(1, sr // 50)
    n          = len(samples) // frame
    if n > 0:
        centroids  = []
        frames_arr = samples[:n*frame].reshape(n, frame)
        for frm in frames_arr:
            spectrum = np.abs(np.fft.rfft(frm))
            freqs    = np.fft.rfftfreq(len(frm), d=1.0/sr)
            denom    = spectrum.sum()
            centroids.append(float(np.dot(freqs, spectrum)/denom) if denom > 0 else 0.0)
        t_c = np.linspace(0, len(samples)/sr, n)
        ax.plot(t_c, centroids, color=PURPLE, linewidth=1.0)
        ax.fill_between(t_c, centroids, alpha=0.25, color=PURPLE)
        ax.set_xlim(0, t_c[-1])
        ax.set_ylim(0, min(max(centroids)*1.2 + 10, sr//2))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hz")
    _style_axes(ax, "Spectral Centroid (Pitch Proxy)")
    plt.tight_layout(pad=0.4)
    return _fig_to_b64(fig)


def generate_all_charts(audio_bytes: bytes) -> dict[str, str]:
    samples, sr = _bytes_to_numpy(audio_bytes)
    return {
        "waveform":          _chart_waveform(samples, sr),
        "spectrogram":       _chart_spectrogram(samples, sr),
        "energy":            _chart_rms_energy(samples, sr),
        "spectral_centroid": _chart_frequency_centroid(samples, sr),
    }


def _transcribe_local(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        segments, _info = _whisper.transcribe(
            tmp_path,
            language       = "en",
            beam_size      = 5,
            vad_filter     = True,
            vad_parameters = {"min_silence_duration_ms": 300},
        )
        transcript = " ".join(seg.text.strip() for seg in segments)
    finally:
        os.unlink(tmp_path)
    return transcript.strip()


@router.get("/voice-health")
async def voice_health():
    groq_key_set = bool(os.environ.get("GROQ_API_KEY", ""))
    return {
        "status":       "ok",
        "groq_api_key": "set" if groq_key_set else "MISSING — set GROQ_API_KEY in .env",
        "model_stt":    f"faster-whisper/{_WHISPER_MODEL_SIZE} — local free",
        "model_llm":    f"groq/{GROQ_MODEL} — free tier",
    }


@router.post("/transcribe")
async def transcribe_audio(payload: AudioPayload):
    audio_bytes = _decode_bytes(payload.audio_b64)
    transcript  = _transcribe_local(audio_bytes)
    charts      = generate_all_charts(audio_bytes)
    return {
        "transcript": transcript,
        "charts":     charts,
        "session_id": payload.session_id,
    }


@router.post("/voice-stream")
async def voice_stream(payload: TranscriptPayload):
    if not payload.transcript.strip():
        async def empty():
            yield 'data: {"token": "(empty transcript — try speaking louder)"}\n\n'
            yield "data: [DONE]\n\n"
        return StreamingResponse(empty(), media_type="text/event-stream")

    def generate():
        stream = groq_client.chat.completions.create(
            model    = GROQ_MODEL,
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": payload.transcript},
            ],
            stream      = True,
            max_tokens  = MAX_STREAM_TOKENS,
            temperature = 0.25,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield f"data: {json.dumps({'token': delta.content})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type = "text/event-stream",
        headers    = {
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )