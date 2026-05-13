"""
xml_extractor.py  (v2 — clean accumulator)
===========================================
A robust, context-agnostic XML extractor that reads screen-capture session files,
cleans OCR output, and either builds a prompt for the AI or maintains a single
accumulated, de-duplicated content block in key_points.json.

WHAT'S DIFFERENT IN v2
-----------------------
1.  OCR ARTIFACT CORRECTION
    '@' misread as '0' (e.g. "1@px" → "10px", "@.@8em" → "0.08em"),
    spurious Ml / Mi / Mr / [i prefixes on CSS colour values removed,
    and other common Tesseract mistakes fixed before any other processing.

2.  VS CODE CHROME STRIPPING
    File-tab bars, sidebar panel names (VARIABLES, WATCH, CALL STACK, etc.),
    terminal prompts, Uvicorn INFO lines, and stray UI glyphs are all removed
    before a line is evaluated.

3.  STRONGER FINGERPRINTING
    _content_fingerprint() now extracts only stable alphabetic words (≥ 4 chars),
    sorts them, and joins — making minor OCR variance between identical frames
    produce the exact same fingerprint.  This stops duplicate captures reaching
    the XML at all.

4.  SENTENCE-LEVEL DEDUPLICATION IN JSON
    save_key_points() loads the existing key_points.json, fingerprints every
    sentence already stored, and only appends sentences that are genuinely new.
    The result is a single growing `content` string, not a list of repeated blobs.

5.  SHORT CODE LINES PRESERVED
    MIN_LINE_LENGTH was 28 which silently dropped CSS properties like
    "color: #111;".  v2 keeps short lines that contain code punctuation.

PIPELINE
--------
    XML file
        → Layer 1 : Structural extraction   (pull raw data out of XML)
        → Layer 2 : OCR correction + quality filtering
        → Layer 3 : Semantic chunking        (group lines into meaningful blocks)
        → Layer 4 : Rolling window + dedup   (keep N most-recent unique captures)
        → Layer 5 : Prompt assembly / JSON accumulation
"""

from __future__ import annotations

import json
import re

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set
from xml.etree import ElementTree as ET


MIN_OCR_CONFIDENCE = 50.0
MIN_LINE_LENGTH    = 8 
ROLLING_WINDOW     = 5
MAX_PROMPT_CHARS   = 4000 

def _fix_ocr_artifacts(text: str) -> str:
    """
    Correct the most common Tesseract misreads that appear in code/CSS contexts.

    Fixes applied (in order):
      · '@' read instead of '0' in numeric contexts  (1@px → 10px)
      · 'Ml#' / 'Mi' / 'Mr' / '[i' prefixes before CSS colour values
      · Double-zero '@.@' → '0.0'
      · '@vh' / '@em' / '@px' suffixes
      · Stray single capital M before a lowercase identifier ('Mwhite' → 'white')
    """
    # Digit-zero substitutions
    text = re.sub(r'(\d)@(?=\d)',    r'\g<1>0',  text)
    text = re.sub(r'(?<=\d)@(?=px|em|vh|rem|%|;|\))', '0', text)
    text = re.sub(r'@(?=\d)',        '0',         text)
    text = re.sub(r'@\.',            '0.',         text)
    text = re.sub(r'\.@',            '.0',         text)
    text = re.sub(r'@px\b',          '0px',        text)
    text = re.sub(r'@em\b',          '0em',        text)
    text = re.sub(r'@vh\b',          '0vh',        text)
    text = re.sub(r'@;',             '0;',         text)
    text = re.sub(r'@\)',            '0)',          text)

    text = re.sub(r'\bMl#',          '#',          text)
    text = re.sub(r'\bMi(?=[a-zA-Z(#])', '',       text)
    text = re.sub(r'\bMr(?=[a-zA-Z(#])', '',       text)
    text = re.sub(r'\[i(?=rgba)',     '',           text)
    text = re.sub(r'\bM(?=[a-z]{3,})', '',         text)

    text = re.sub(r'\b(Ce|Ct|Cb|Db|Gh|Gy|HU|Vv|ov|gp|a~|Ov)\b', '', text)
    text = re.sub(r'(?<!\w)[€¢©®°]{1,3}(?!\w)', '', text)

    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


# ──────────────────────────────────────────────────────────
# LAYER 0b — BROWSER & IDE CHROME STRIPPING
# ──────────────────────────────────────────────────────────

# Patterns that identify browser UI text (applied to full raw strings)
_BROWSER_CHROME_PATTERNS: List[re.Pattern] = [
    re.compile(r'\bask\s+google\b',            re.IGNORECASE),
    re.compile(r'\bproblem\s+list\b',          re.IGNORECASE),
    re.compile(r'\bpremium\b',                 re.IGNORECASE),
    re.compile(r'@\s*submit',                  re.IGNORECASE),
    re.compile(r'\bverify\s+your\s+email\b',   re.IGNORECASE),
    re.compile(r'\bunlock\s+all\s+features\b', re.IGNORECASE),
    re.compile(r'\d+\s+online\b',              re.IGNORECASE),
    re.compile(r'\bcopyright\s+©\s*\d{4}\b',  re.IGNORECASE),
    re.compile(r'\ball\s+rights\s+reserved\b', re.IGNORECASE),
    re.compile(r'leetcode\.com/problems/\S+'),
    re.compile(r'©\s*\d[\d,.]+\s*(online|k)\b', re.IGNORECASE),
]

# Patterns that identify VS Code / IDE chrome (sidebar panels, tab bars, etc.)
_IDE_CHROME_PATTERNS: List[re.Pattern] = [
    # VS Code debug sidebar panel headings
    re.compile(
        r'\b(VARIABLES|WATCH|CALL STACK|LOADED SCRIPTS|BREAKPOINTS'
        r'|BROWSER OPTIONS|NETWORK|PROBLEMS|OUTPUT|DEBUG CONSOLE'
        r'|TERMINAL|PORTS|GITLENS)\b'
    ),
    # "index... RUNNING" or "index... [RUNNING"
    re.compile(r'index\.\.\.\s*\[?RUNNING\]?', re.IGNORECASE),
    # "> LOADED SCRIPTS", "> BREAKPOINTS" etc.
    re.compile(r'>\s+(LOADED|BREAKPOINTS|BROWSER|NETWORK)\s+\S*', re.IGNORECASE),
    # Top menu bar fragment: "File Edit Selection View Go Run"
    re.compile(r'\b(File|Edit)\s+(?:Edit\s+)?Selection\s+View\s+Go\b', re.IGNORECASE),
    # File-tab noise: "index.html X detectorPipeline.py M"
    re.compile(r'\b\w[\w.]+\.(html?|py|js|ts|css|xml)\s*[XM]\b'),
    # VS Code breadcrumb arrow noise: "€> test" or "<€ > | test"
    re.compile(r'[<€>|]\s*test\b'),
    re.compile(r'\bNoCon\.{2,3}\s*v\b'),
    re.compile(r'\bCt\s+v\s*[-–]\s*Db\b'),
    re.compile(r'\bCb\s+NoCon\b'),
    # Git terminal prompts
    re.compile(r'Max@DEVELOPER-ALEX\s+MINGW64\s+\S+'),
    re.compile(r'MINGW64\s+~/\S+\s+\(main\)'),
    # Uvicorn server log lines
    re.compile(r'INFO:\s+(?:Uvicorn|Started|Waiting|Application)\s+\S+', re.IGNORECASE),
    re.compile(r'Will watch for changes in these directories'),
    re.compile(r'Press CTRL\+C to quit'),
    # OCR window-decoration remnants
    re.compile(r'\bJ\s+File\b'),          # "J File Edit..."
    re.compile(r'\bx\s+File\b'),          # "x File Edit..."
    re.compile(r'\bRun\s+[€<>]\s*[>|]'), # "Run €> test"
    # Stray short token sequences that are purely VS Code UI
    re.compile(r'\b(Ce|NoCon|RUNNING|LOADED)\b'),
]


def _strip_ui_chrome(text: str) -> str:
    """
    Remove both browser and IDE/VS-Code UI artefacts from a raw text string.
    Operates on the full text before line-splitting for maximum coverage.
    """
    for pattern in _BROWSER_CHROME_PATTERNS:
        text = pattern.sub(' ', text)
    for pattern in _IDE_CHROME_PATTERNS:
        text = pattern.sub(' ', text)
    text = _fix_ocr_artifacts(text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


# ──────────────────────────────────────────────────────────
# FINGERPRINTING
# ──────────────────────────────────────────────────────────

def _content_fingerprint(text: str) -> str:
    """
    Return a stable fingerprint of a text block for near-duplicate detection.

    v2 algorithm:
      1. Apply OCR and chrome cleaning first (removes the variance sources).
      2. Lowercase.
      3. Extract only alphabetic words of length ≥ 4  (code identifiers,
         keywords — the bits that actually identify *what* is on screen).
      4. Deduplicate, sort, take up to 60 words, join.

    Two captures that show the same screen content will produce the same
    fingerprint even if minor OCR noise differs between frames.
    """
    text = _strip_ui_chrome(text)
    text = text.lower()
    words = re.findall(r'[a-z]{4,}', text)
    significant = sorted(set(words))[:60]
    return ' '.join(significant)


def _sentence_fingerprint(sentence: str) -> str:
    """
    Lightweight fingerprint for a single sentence / line used during
    JSON accumulation to avoid appending near-duplicate sentences.
    """
    s = sentence.lower()
    s = re.sub(r'\d+', '', s)
    s = re.sub(r'[^a-z\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# ──────────────────────────────────────────────────────────
# NOISE PATTERNS  (Layer 2 line-level filter)
# ──────────────────────────────────────────────────────────

_NOISE_PATTERNS = [
    re.compile(r'^https?://\S+$'),           # bare URL
    re.compile(r'^[\W\d\s]{1,15}$'),         # only symbols / digits
    re.compile(r'(\bx\b\s*){3,}'),           # "x x x" tab-close buttons
    re.compile(r'[=<>&¢°]{3,}'),             # runs of stray OCR symbols
    re.compile(r'^\s*[\|\-\+]{3,}\s*$'),     # ASCII table borders
]


# ──────────────────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────────────────

@dataclass
class RawCapture:
    number:     int
    timestamp:  str
    confidence: float
    lines:      List[str] = field(default_factory=list)
    raw_text:   Optional[str] = None


@dataclass
class SemanticChunk:
    kind:  str                            # "heading" | "code" | "list" | "body"
    lines: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        if self.kind == "heading":
            return "## " + " ".join(self.lines)
        if self.kind == "code":
            return "\n".join("  " + l for l in self.lines)
        return " ".join(self.lines)


# ──────────────────────────────────────────────────────────
# LAYER 1 — STRUCTURAL EXTRACTION
# ──────────────────────────────────────────────────────────

def extract_captures(xml_path: str) -> List[RawCapture]:
    """Parse the XML session file into RawCapture objects."""
    path = Path(xml_path)
    if not path.exists():
        return []
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ET.ParseError:
        return []

    for element in root.iter():
        if "}" in element.tag:
            element.tag = element.tag.split("}")[1]

    captures: List[RawCapture] = []
    for cap_el in root.findall("capture"):
        number     = int(cap_el.get("number", 0))
        timestamp  = cap_el.get("timestamp", "")
        confidence = float(cap_el.get("ocr_confidence", 0.0))

        lines: List[str] = []
        for block_el in cap_el.findall("block"):
            for line_el in block_el.findall("line"):
                if line_el.text:
                    lines.append(line_el.text)

        raw_el   = cap_el.find("raw")
        raw_text = raw_el.text if raw_el is not None and raw_el.text else None

        captures.append(RawCapture(
            number=number, timestamp=timestamp,
            confidence=confidence, lines=lines, raw_text=raw_text,
        ))
    return captures


# ──────────────────────────────────────────────────────────
# LAYER 2 — QUALITY FILTERING & CLEANING
# ──────────────────────────────────────────────────────────

def _clean_line(line: str) -> str:
    """
    Full cleaning pipeline for a single line:
      1. Strip browser + IDE chrome.
      2. Fix OCR artifacts.
      3. Replace HTML entities.
      4. Collapse multiple spaces.
    """
    line = _strip_ui_chrome(line)
    line = line.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    line = re.sub(r'[=<>&¢°©®]{2,}', ' ', line)
    line = re.sub(r'\s{2,}', ' ', line)
    return line.strip()


def _is_noise(line: str) -> bool:
    """
    Return True if the line should be discarded.

    v2 change: short lines (< MIN_LINE_LENGTH) are kept when they contain
    code punctuation — otherwise CSS properties like "color: #111;" would
    be silently dropped.
    """
    stripped = line.strip()
    if len(stripped) < 4:
        return True

    # Short lines: keep only if they look like code
    if len(stripped) < MIN_LINE_LENGTH:
        has_code_punct = bool(re.search(r'[{};:()\[\]=\'"#@.]', stripped))
        if not has_code_punct:
            return True

    for pattern in _NOISE_PATTERNS:
        if pattern.search(stripped):
            return True
    return False


def filter_captures(captures: List[RawCapture]) -> List[RawCapture]:
    """
    Layer 2: Drop low-confidence captures, clean lines, discard noise.
    """
    filtered: List[RawCapture] = []

    for cap in captures:
        if cap.confidence < MIN_OCR_CONFIDENCE:
            continue

        # Clean the whole joined text first (catches multi-word chrome artefacts)
        joined = _strip_ui_chrome(" ".join(cap.lines))

        # Re-split on sentence-ending punctuation for finer-grained lines
        candidate_lines = re.split(r'(?<=[.?!;{}])\s+', joined)

        clean_lines = []
        for raw_l in candidate_lines:
            cl = _clean_line(raw_l)
            if cl and not _is_noise(cl):
                clean_lines.append(cl)

        # Fall back to raw_text if block lines produced nothing
        if not clean_lines and cap.raw_text:
            raw = _clean_line(_strip_ui_chrome(cap.raw_text))
            if raw and not _is_noise(raw):
                clean_lines = [raw]

        if not clean_lines:
            continue

        filtered.append(RawCapture(
            number=cap.number, timestamp=cap.timestamp,
            confidence=cap.confidence, lines=clean_lines,
        ))

    return filtered


# ──────────────────────────────────────────────────────────
# LAYER 3 — SEMANTIC CHUNKING
# ──────────────────────────────────────────────────────────

def _classify_line(line: str) -> str:
    stripped = line.strip()
    if re.match(r'^[A-Z][^a-z]{3,}$', stripped):
        return "heading"
    if re.match(r'^\s{4,}|^\t', line):
        return "code"
    if re.match(r'^[-•*]\s|^\d+\.\s', stripped):
        return "list"
    return "body"


def chunk_capture(cap: RawCapture) -> List[SemanticChunk]:
    """Group consecutive same-type lines into SemanticChunks."""
    chunks:  List[SemanticChunk]     = []
    current: Optional[SemanticChunk] = None

    for line in cap.lines:
        kind = _classify_line(line)
        if current is not None and current.kind == kind:
            current.lines.append(line)
        else:
            current = SemanticChunk(kind=kind, lines=[line])
            chunks.append(current)

    return chunks


# ──────────────────────────────────────────────────────────
# LAYER 4 — ROLLING WINDOW & DEDUPLICATION
# ──────────────────────────────────────────────────────────

def apply_rolling_window(captures: List[RawCapture]) -> List[RawCapture]:
    """
    Keep the ROLLING_WINDOW most-recent captures; skip near-duplicates.

    v2: uses the stronger word-set fingerprint so minor OCR variance
    between otherwise-identical frames is correctly ignored.
    """
    window = captures[-ROLLING_WINDOW:]

    deduplicated:    List[RawCapture] = []
    seen_fingerprint: Optional[str]   = None

    for cap in window:
        full_text   = " ".join(cap.lines)
        fingerprint = _content_fingerprint(full_text)

        if fingerprint == seen_fingerprint:
            continue

        seen_fingerprint = fingerprint
        deduplicated.append(cap)

    return deduplicated


# ──────────────────────────────────────────────────────────
# LAYER 5 — PROMPT ASSEMBLY
# ──────────────────────────────────────────────────────────

def assemble_prompt(captures: List[RawCapture]) -> str:
    sections: List[str] = []

    for cap in captures:
        chunks      = chunk_capture(cap)
        chunk_texts = [c.to_text() for c in chunks if c.lines]
        if chunk_texts:
            sections.append("\n".join(chunk_texts))

    combined = "\n\n".join(sections)

    if len(combined) > MAX_PROMPT_CHARS:
        combined = "[… earlier context omitted …]\n\n" + combined[-MAX_PROMPT_CHARS:]

    return combined


# ──────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────

def build_ai_prompt(xml_path: str, user_question: str = "") -> str:
    """
    Main entry point for sending screen context to an AI.

    Pipeline: extract → filter → rolling window → assemble.
    """
    raw_captures = extract_captures(xml_path)
    filtered     = filter_captures(raw_captures)
    windowed     = apply_rolling_window(filtered)
    content      = assemble_prompt(windowed)

    if not content.strip():
        return "No readable screen content was captured yet."

    question_block = (
        f"\nUser question: {user_question.strip()}\n"
        if user_question.strip()
        else "\nProvide helpful assistance based on what the user is viewing.\n"
    )

    return (
        "The following text was captured from the user's screen:\n\n"
        + content
        + "\n\n"
        + question_block.strip()
    )


# ──────────────────────────────────────────────────────────
# ACCUMULATOR — single growing JSON content block
# ──────────────────────────────────────────────────────────

def _split_into_sentences(text: str) -> List[str]:
    """
    Split accumulated content back into individual sentences / lines so we
    can fingerprint what is already stored.
    """
    # Split on newlines and sentence-ending punctuation
    parts = re.split(r'\n+|(?<=[.;{}])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def save_key_points(
    xml_path:    str,
    output_path: str = "captures/key_points.json",
) -> dict:
    """
    Extract key content from the latest captures and APPEND only genuinely
    new sentences to the single `content` string in key_points.json.

    JSON shape
    ----------
    {
        "session":      "1778590189023",
        "last_updated": "2026-05-12T12:53:14Z",
        "capture_count": 15,
        "content": "... full accumulated clean text ..."
    }

    Rules
    -----
    · If the file does not exist it is created from scratch.
    · Sentences already in `content` (by fingerprint) are never appended again.
    · The result is a single, readable, non-repeating text block that grows as
      new screen content is detected.
    """
    # ── Load existing accumulated content ────────────────────
    existing_content: str          = ""
    existing_fps:     Set[str]     = set()
    existing_count:   int          = 0

    output_file = Path(output_path)
    if output_file.exists():
        try:
            with open(output_file, encoding="utf-8") as f:
                stored = json.load(f)
            existing_content = stored.get("content", "")
            existing_count   = stored.get("capture_count", 0)
            for sent in _split_into_sentences(existing_content):
                existing_fps.add(_sentence_fingerprint(sent))
        except (json.JSONDecodeError, KeyError):
            pass   # corrupt file — start fresh

    # ── Extract + clean current captures ─────────────────────
    raw_captures = extract_captures(xml_path)
    filtered     = filter_captures(raw_captures)
    windowed     = apply_rolling_window(filtered)

    new_sentences: List[str] = []
    for cap in windowed:
        for chunk in chunk_capture(cap):
            rendered = chunk.to_text()
            # Break into individual lines for finer deduplication
            for line in re.split(r'\n+|(?<=[.;{}])\s+', rendered):
                line = line.strip()
                if not line:
                    continue
                fp = _sentence_fingerprint(line)
                if fp not in existing_fps:
                    existing_fps.add(fp)
                    new_sentences.append(line)

    # ── Merge into single accumulated string ─────────────────
    if new_sentences:
        new_block = " ".join(new_sentences)
        # Join with a space if existing content ends mid-sentence, else newline
        if existing_content and not existing_content.rstrip().endswith("\n"):
            accumulated = existing_content.rstrip() + "\n\n" + new_block
        else:
            accumulated = (existing_content + new_block).strip()
    else:
        accumulated = existing_content

    # ── Session name: strip leading "session_" if present ────
    stem         = Path(xml_path).stem
    session_name = re.sub(r'^session_', '', stem)

    payload = {
        "session":       session_name,
        "last_updated":  datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "capture_count": max(existing_count, len(raw_captures)),
        "content":       accumulated,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # Return a legacy-compatible dict so callers that do
    # `result["key_points"]` don't crash
    payload["key_points"] = [{"type": "body", "content": accumulated}]
    return payload


# ──────────────────────────────────────────────────────────
# CLI TEST ENTRY POINT
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    path     = sys.argv[1] if len(sys.argv) > 1 else "captures/session_test.xml"
    question = sys.argv[2] if len(sys.argv) > 2 else ""

    prompt = build_ai_prompt(path, question)
    print("=" * 60)
    print(prompt)
    print("=" * 60)
    print(f"\nTotal characters sent to AI: {len(prompt)}")