import base64, io, re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List

import pytesseract
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from pydantic import BaseModel
from xml.etree import ElementTree as ET

# ── Tesseract path ────────────────────────────────────────────────────────────
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Screen Reader API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path("captures")
DEBUG_DIR  = Path("captures/debug")   # saved images so you can see what Tesseract receives
OUTPUT_DIR.mkdir(exist_ok=True)
DEBUG_DIR.mkdir(exist_ok=True)


# ── Request model ─────────────────────────────────────────────────────────────
class FramePayload(BaseModel):
    image_b64: str
    timestamp: str
    session_id: str


# ── Module A: Frame decode ────────────────────────────────────────────────────
def decode_frame(b64_string: str) -> Image.Image:
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    raw_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")


# ── Module B: OCR ─────────────────────────────────────────────────────────────
@dataclass
class OCRResult:
    full_text: str
    words: List[dict]
    confidence: float


def is_dark_image(img: Image.Image) -> bool:
    """Returns True if the image has a dark background (like VS Code, terminals)."""
    grayscale      = img.convert("L")
    pixels         = list(grayscale.getdata())
    avg_brightness = sum(pixels) / len(pixels)
    return avg_brightness < 128   # 0=black, 255=white


def preprocess(img: Image.Image) -> Image.Image:
    # Scale up — Tesseract needs large text to read accurately
    w, h = img.size
    if w < 1600:
        scale = 2
        img   = img.resize((w * scale, h * scale), Image.LANCZOS)

    img = img.convert("L")   # grayscale

    # KEY FIX: invert dark-background screens
    # Tesseract is trained on BLACK text on WHITE background.
    # VS Code dark theme is the opposite — so we flip it.
    if is_dark_image(img):
        img = ImageOps.invert(img)

    img = img.filter(ImageFilter.SHARPEN)
    img = ImageEnhance.Contrast(img).enhance(2.0)
    return img


def run_ocr(img: Image.Image, session_id: str, capture_num: int) -> OCRResult:
    processed = preprocess(img)

    # Save the processed image so you can inspect it in captures/debug/
    debug_path = DEBUG_DIR / f"{session_id}_cap{capture_num}.png"
    processed.save(debug_path)

    best_text  = ""
    best_conf  = 0.0
    best_words = []

    for psm in ["--psm 6", "--psm 3", "--psm 11"]:
        data = pytesseract.image_to_data(
            processed,
            output_type=pytesseract.Output.DICT,
            config=psm
        )
        words = [
            {
                "text": data["text"][i],
                "conf": int(data["conf"][i]),
                "x":    data["left"][i],
                "y":    data["top"][i],
            }
            for i in range(len(data["text"]))
            if str(data["conf"][i]).lstrip("-").isdigit()
            and int(data["conf"][i]) > 40
            and data["text"][i].strip()
        ]
        if not words:
            continue
        avg = sum(w["conf"] for w in words) / len(words)
        if avg > best_conf:
            best_conf  = avg
            best_words = words
            best_text  = " ".join(w["text"] for w in words)

    # Fallback to plain string if data approach got nothing
    if not best_text.strip():
        best_text = pytesseract.image_to_string(processed, config="--psm 6").strip()

    return OCRResult(
        full_text=best_text,
        words=best_words,
        confidence=round(best_conf, 2),
    )


# ── Module C: Content parser ──────────────────────────────────────────────────
@dataclass
class ContentBlock:
    block_type: str
    lines: List[str] = field(default_factory=list)


def classify_line(line: str) -> str:
    if re.match(r'^[A-Z][^a-z]{3,}$', line.strip()):
        return "heading"
    if re.match(r'^\s{4,}|^\t', line):
        return "code"
    if re.match(r'^[-•*]\s|^\d+\.\s', line.strip()):
        return "list"
    return "body"


def parse_content(raw_text: str) -> List[ContentBlock]:
    blocks, current = [], None
    for line in raw_text.splitlines():
        if not line.strip():
            continue
        kind = classify_line(line)
        if current and current.block_type == kind:
            current.lines.append(line.strip())
        else:
            current = ContentBlock(block_type=kind, lines=[line.strip()])
            blocks.append(current)
    return blocks


# ── Module D: XML writer — one file per session, append each capture ──────────
session_capture_counts: dict = {}

last_text_per_session: dict = {}

def write_xml(
    blocks: List[ContentBlock],
    session_id: str,
    confidence: float,
    raw_text: str,
) -> str:
    filename = OUTPUT_DIR / f"session_{session_id}.xml"
    ts       = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    # Load existing file or create a new root
    if filename.exists():
        tree = ET.parse(filename)
        root = tree.getroot()
    else:
        root = ET.Element("session", id=session_id, started=ts)
        tree = ET.ElementTree(root)

    capture_num = session_capture_counts.get(session_id, 0) + 1
    session_capture_counts[session_id] = capture_num

    capture_el = ET.SubElement(
        root, "capture",
        number=str(capture_num),
        timestamp=ts,
        ocr_confidence=str(confidence),
    )

    if blocks:
        for block in blocks:
            el = ET.SubElement(capture_el, "block", type=block.block_type)
            for line in block.lines:
                ln      = ET.SubElement(el, "line")
                ln.text = line
    else:
        capture_el.set("status", "no_structure_detected")
        if raw_text:
            raw_el      = ET.SubElement(capture_el, "raw")
            raw_el.text = raw_text
        else:
            capture_el.set("status", "no_text_detected")

    ET.indent(tree, space="  ")
    tree.write(filename, encoding="utf-8", xml_declaration=True)
    return str(filename)


# ── Endpoint ──────────────────────────────────────────────────────────────────
@app.post("/capture")
async def capture_frame(payload: FramePayload):
    try:
        capture_num = session_capture_counts.get(payload.session_id, 0)

        image = decode_frame(payload.image_b64)
        ocr   = run_ocr(image, payload.session_id, capture_num)

        # ── Deduplication check ──────────────────────────────────
        last = last_text_per_session.get(payload.session_id, "")
        if ocr.full_text.strip() and ocr.full_text.strip() == last.strip():
            return {
                "status":     "skipped",
                "text":       ocr.full_text,
                "xml_path":   None,
                "confidence": ocr.confidence,
                "word_count": 0,
            }
        last_text_per_session[payload.session_id] = ocr.full_text
        # ────────────────────────────────────────────────────────

        blocks   = parse_content(ocr.full_text)
        xml_path = write_xml(blocks, payload.session_id, ocr.confidence, ocr.full_text)

        print(f"[cap#{capture_num+1}] conf={ocr.confidence} | "
              f"words={len(ocr.full_text.split())} | "
              f"preview={ocr.full_text[:80]!r}")

        return {
            "status":     "ok",
            "text":       ocr.full_text,
            "xml_path":   xml_path,
            "confidence": ocr.confidence,
            "word_count": len(ocr.full_text.split()),
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("detectorPipeline:app", host="0.0.0.0", port=8000, reload=True)