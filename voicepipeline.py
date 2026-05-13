"""
voicePipeline.py
================
Microphone → VAD → Whisper STT → GPT-4o streaming answer pipeline.

Mounted as an APIRouter into detectorPipeline.py via:
    from voicePipeline import router as voice_router
    app.include_router(voice_router)

Endpoints
---------
POST /transcribe      — Receives base64 webm/opus audio blob → Whisper transcript
                        + three matplotlib charts (waveform, spectrogram, RMS energy)
                        returned as base64 PNGs.

POST /voice-stream    — Receives a transcript string → streams GPT-4o tokens
                        via Server-Sent Events (SSE).  First token ≈ 200-400 ms.

GET  /voice-health    — Quick ping to confirm the router is mounted and the
                        OpenAI API key is present.

Architecture notes
------------------
· Everything is stateless; each request is self-contained.
· pydub + ffmpeg decode the browser's webm/opus blob into a numpy float32
  array so scipy / matplotlib can work with it.
· If pydub / ffmpeg are unavailable the chart functions degrade gracefully
  to a synthetic noise waveform so the rest of the pipeline keeps running.
· GPT-4o is used for streaming answers (fastest reasoning model for this
  workload).  Temperature 0.25 keeps answers focused and consistent.
· The SSE generator is a plain synchronous generator wrapped in
  StreamingResponse — FastAPI handles the async boundary automatically.
"""

from __future__ import annotations

import base64
import io
import json
import os

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — must be before pyplot
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel
from scipy.signal import spectrogram as scipy_spectrogram

router = APIRouter()
client = OpenAI()

VAD_ENERGY_THRESHOLD = 0.018
SILENCE_DURATION_MS  = 800
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
    "• For coding questions — give working, clean code with a one-line explanation "
    "of the approach ABOVE the code block.\n"
    "• For system design — give a concise bullet-point structure (max 6 bullets).\n"
    "• For behavioural questions — give a STAR-format skeleton (2-3 sentences each).\n"
    "• For ambiguous input — ask one clarifying question.\n"
    "• NEVER exceed 250 words. Be direct. The candidate needs fast, actionable help."
)

# Pydantic models

class AudioPayload(BaseModel):
    audio_b64:   str           # base64 webm/opus from browser MediaRecorder
    timestamp:   str
    session_id:  str
    sample_rate: int = 48000   # browser default for MediaRecorder

class TranscriptPayload(BaseModel):
    transcript:  str
    session_id:  str

#  Audio decoding

def _decode_bytes(b64: str) -> bytes:
    """Strip the data-URI header if present, then base64-decode."""
    if "," in b64:
        b64 = b64.split(",")[1]
    return base64.b64decode(b64)


def _bytes_to_numpy(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Decode webm/opus → numpy float32 array using pydub + ffmpeg.
    Falls back to synthetic noise so chart generation never crashes.
    """
    try:
        from pydub import AudioSegment          # type: ignore
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        if seg.channels == 2:                   # stereo → mono
            samples = samples.reshape((-1, 2)).mean(axis=1)
        samples /= 2 ** (seg.sample_width * 8 - 1)
        return samples, seg.frame_rate
    except Exception:
        # Graceful degradation — fake signal so charts are still generated
        sr = 16000
        t  = np.linspace(0, 1.5, sr * 3 // 2, dtype=np.float32)
        sig = (
            0.30 * np.sin(2 * np.pi * 220 * t) +
            0.15 * np.sin(2 * np.pi * 440 * t) +
            0.05 * np.random.randn(len(t)).astype(np.float32)
        )
        return sig, sr

#  Matplotlib chart helpers

def _fig_to_b64(fig: plt.Figure) -> str:
    """Render a matplotlib figure to a PNG and return it as base64."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _style_axes(ax: plt.Axes, title: str) -> None:
    """Apply the shared dark-theme style to a single Axes."""
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
    """Time-domain waveform with RMS envelope overlay."""
    fig, ax = plt.subplots(figsize=(5.8, 1.9), facecolor=DARK_BG)
    t = np.linspace(0, len(samples) / sr, len(samples))

    # Thin waveform line
    ax.plot(t, samples, color=CYAN, linewidth=0.5, alpha=0.7)

    # RMS envelope (smooth)
    frame = max(1, sr // 100)
    n     = len(samples) // frame
    if n > 0:
        rms = np.sqrt(np.mean(
            samples[:n * frame].reshape(n, frame) ** 2, axis=1
        ))
        t_rms = np.linspace(0, len(samples) / sr, n)
        ax.fill_between(t_rms,  rms, -rms, color=CYAN, alpha=0.18)

    ax.set_xlim(0, t[-1] if len(t) else 1)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    _style_axes(ax, "Waveform")
    plt.tight_layout(pad=0.4)
    return _fig_to_b64(fig)


def _chart_spectrogram(samples: np.ndarray, sr: int) -> str:
    """
    Power spectrogram (dB) via scipy.  Uses a plasma colormap on the
    dark background for a vivid frequency picture.
    """
    fig, ax = plt.subplots(figsize=(5.8, 2.2), facecolor=DARK_BG)

    if len(samples) > 512:
        nperseg = min(512, len(samples) // 4)
        f, t_s, Sxx = scipy_spectrogram(
            samples, fs=sr, nperseg=nperseg, noverlap=nperseg // 2
        )
        Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-10))
        Sxx_db = np.clip(Sxx_db, -80, 0)
        pcm = ax.pcolormesh(t_s, f, Sxx_db, shading="gouraud",
                            cmap="plasma", vmin=-80, vmax=0)
        cb = plt.colorbar(pcm, ax=ax, pad=0.02)
        cb.ax.tick_params(colors=TEXT_DIM, labelsize=7)
        cb.set_label("dBFS", color=TEXT_MID, fontsize=7)
        cb.outline.set_edgecolor(GRID_COL)
        # Cap frequency axis to 8 kHz — most speech energy is here
        ax.set_ylim(0, min(8000, sr // 2))
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x)))
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
    """
    Frame-level RMS energy over time.  Draws a VAD threshold line and
    colour-fills above / below threshold to make voiced vs. silent
    regions immediately obvious.
    """
    fig, ax = plt.subplots(figsize=(5.8, 1.7), facecolor=DARK_BG)

    frame = max(1, sr // 50)   # 20 ms frames
    n     = len(samples) // frame
    if n > 0:
        rms   = np.sqrt(np.mean(
            samples[:n * frame].reshape(n, frame) ** 2, axis=1
        ))
        t_rms = np.linspace(0, len(samples) / sr, n)

        # Voiced (above threshold) in cyan, silent in dim orange
        ax.fill_between(t_rms, rms,
                        where=rms >= VAD_ENERGY_THRESHOLD,
                        color=CYAN, alpha=0.55, label="Voiced")
        ax.fill_between(t_rms, rms,
                        where=rms < VAD_ENERGY_THRESHOLD,
                        color=ORANGE, alpha=0.30, label="Silent")
        ax.plot(t_rms, rms, color=CYAN, linewidth=0.9)

        # VAD threshold marker
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
    """
    Spectral centroid over time — a proxy for 'brightness' / pitch
    of the voice.  Useful for distinguishing question vs. statement
    intonation patterns.
    """
    fig, ax = plt.subplots(figsize=(5.8, 1.7), facecolor=DARK_BG)

    frame = max(1, sr // 50)
    n     = len(samples) // frame

    if n > 0:
        centroids = []
        frames_arr = samples[:n * frame].reshape(n, frame)
        for frm in frames_arr:
            spectrum  = np.abs(np.fft.rfft(frm))
            freqs     = np.fft.rfftfreq(len(frm), d=1.0 / sr)
            denom     = spectrum.sum()
            centroid  = float(np.dot(freqs, spectrum) / denom) if denom > 0 else 0.0
            centroids.append(centroid)
        t_c = np.linspace(0, len(samples) / sr, n)
        ax.plot(t_c, centroids, color=PURPLE, linewidth=1.0)
        ax.fill_between(t_c, centroids, alpha=0.25, color=PURPLE)
        ax.set_xlim(0, t_c[-1])
        ax.set_ylim(0, min(max(centroids) * 1.2 + 10, sr // 2))

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hz")
    _style_axes(ax, "Spectral Centroid (Pitch Proxy)")
    plt.tight_layout(pad=0.4)
    return _fig_to_b64(fig)


def generate_all_charts(audio_bytes: bytes) -> dict[str, str]:
    """
    Decode audio and generate all four matplotlib charts.
    Returns { chart_key: base64_png_string }.
    """
    samples, sr = _bytes_to_numpy(audio_bytes)
    return {
        "waveform":          _chart_waveform(samples, sr),
        "spectrogram":       _chart_spectrogram(samples, sr),
        "energy":            _chart_rms_energy(samples, sr),
        "spectral_centroid": _chart_frequency_centroid(samples, sr),
    }

# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/voice-health")
async def voice_health():
    """Quick liveness check — confirms router is mounted and API key is set."""
    key_present = bool(os.environ.get("OPENAI_API_KEY", ""))
    return {
        "status":    "ok",
        "api_key":   "set" if key_present else "MISSING — set OPENAI_API_KEY",
        "model_stt": "whisper-1",
        "model_llm": "gpt-4o",
    }


@router.post("/transcribe")
async def transcribe_audio(payload: AudioPayload):
    """
    Pipeline
    --------
    1. base64 decode → raw webm bytes
    2. Send to Whisper-1 for transcription  (~250-400 ms)
    3. Decode audio to numpy for chart generation  (parallel-ish)
    4. Return transcript + four base64 PNG charts

    The frontend immediately fires /voice-stream with the transcript
    while displaying the charts — so total latency to first AI token
    is  Whisper_time + SSE_TTFT ≈ 400-700 ms end-to-end.
    """
    audio_bytes = _decode_bytes(payload.audio_b64)

    #  Whisper transcription
    audio_file      = io.BytesIO(audio_bytes)
    audio_file.name = "capture.webm"   # Whisper needs a filename extension hint
    transcript_obj  = client.audio.transcriptions.create(
        model    = "whisper-1",
        file     = audio_file,
        language = "en",
    )
    transcript = transcript_obj.text.strip()

    charts = generate_all_charts(audio_bytes)

    return {
        "transcript": transcript,
        "charts":     charts,
        "session_id": payload.session_id,
    }


@router.post("/voice-stream")
async def voice_stream(payload: TranscriptPayload):
    """
    Receive a transcript → stream GPT-4o tokens via Server-Sent Events.

    SSE format:  data: {"token": "..."}\n\n
    Terminator:  data: [DONE]\n\n

    TTFT (time to first token) ≈ 200–400 ms from when this endpoint
    receives the request.  The frontend should open this SSE stream
    immediately after /transcribe returns.
    """
    if not payload.transcript.strip():
        async def empty():
            yield 'data: {"token": "(empty transcript — try speaking louder)"}\n\n'
            yield "data: [DONE]\n\n"
        return StreamingResponse(empty(), media_type="text/event-stream")

    def generate():
        stream = client.chat.completions.create(
            model    = "gpt-4o",
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