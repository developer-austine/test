"""
xml_extractor.py
================
A robust, context-agnostic XML extractor that reads screen-capture session files,
cleans the OCR output, and assembles a concise prompt ready to be sent to an AI.

It does NOT know or care what is on screen — whether it is a code editor, a form,
a meeting transcript, a job assessment, a dashboard, or anything else.
It only knows HOW to find signal vs noise and how to structure it for the AI.

USAGE
-----
    from xml_extractor import build_ai_prompt

    prompt = build_ai_prompt("captures/session_abc123.xml")
    # send `prompt` to your AI endpoint

PIPELINE
--------
    XML file
        → Layer 1 : Structural extraction   (pull raw data out of XML)
        → Layer 2 : Quality filtering        (drop garbage lines)
        → Layer 3 : Semantic chunking        (group lines into meaningful blocks)
        → Layer 4 : Rolling window           (keep only the N most recent captures)
        → Layer 5 : Prompt assembly          (format everything for the AI)
"""

from __future__ import annotations

import re
import json

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from xml.etree import ElementTree as ET

# ----------------------------------------------
# CONFIGURATION
# ----------------------------------------------

MIN_OCR_CONFIDENCE = 50.0
MIN_LINE_LENGTH    = 28
ROLLING_WINDOW     = 5
MAX_PROMPT_CHARS   = 3000

# Noise Detection Layer (Layer 2).
_NOISE_PATTERNS = [
    re.compile(r'^https?://\S+$'),           # bare URL line
    re.compile(r'^[\W\d\s]{1,15}$'),         # line made entirely of symbols/digits
    re.compile(r'(\bx\b\s*){3,}'),           # browser tab close buttons "x x x"
    re.compile(r'[=<>&¢°]{3,}'),             # runs of stray OCR symbols
    re.compile(r'^\s*[\|\-\+]{3,}\s*$'),     # ASCII table borders
    re.compile(r'^[v€]\s+electron-|^v\s+\w+-course'),  # ignored url pattern
]

#   FIX 1 — NEW: _BROWSER_CHROME_PATTERNS                      
#   Problem : Every capture was polluted with browser UI text  
#             ("Ask Google", "Premium", "1679 Online", etc.)   
#             because nothing stripped those strings before the
#             text reached the AI.                             
#   Fix     : Added this pattern list + _strip_browser_chrome()
#            below. Called at the top of _clean_line() so    
#             chrome is removed before any other processing.    

_BROWSER_CHROME_PATTERNS: List[re.Pattern] = [
    # Tab bar / navigation chrome
    re.compile(r'\bask\s+google\b', re.IGNORECASE),
    re.compile(r'\bproblem\s+list\b', re.IGNORECASE),
    re.compile(r'\bpremium\b', re.IGNORECASE),
    re.compile(r'@\s*submit', re.IGNORECASE),
    re.compile(r'\bverify\s+your\s+email\b', re.IGNORECASE),
    re.compile(r'\bplease\s+verify\b', re.IGNORECASE),
    re.compile(r'\bunlock\s+all\s+features\b', re.IGNORECASE),
    re.compile(r'\bservices\s+on\s+leetcode\b', re.IGNORECASE),
    re.compile(r'\d+\s+online\b', re.IGNORECASE),           # "1679 Online" counter
    re.compile(r'\bcopyright\s+©\s+\d{4}\b', re.IGNORECASE),
    re.compile(r'\ball\s+rights\s+reserved\b', re.IGNORECASE),
    # OCR garbage that leaks through from icons and tab UI
    re.compile(r'\b(a0|go|oo|fo|xs|xe|au|ar|ne)\b'),
    re.compile(r'[€v]\s+\w+\s*-\s*leetcode'),              # "v Two Sum - LeetCode"
    re.compile(r'leetcode\.com/problems/\S+'),              # full URL fragment
    # Online user count varies every frame — normalise it away
    re.compile(r'©\s*\d[\d,\.]+\s*(online|k)\b', re.IGNORECASE),
    re.compile(r'\d[\d,]+\s*/\s*\d[\d,\.]+[km]?\b', re.IGNORECASE),
]


def _strip_browser_chrome(text: str) -> str:
    """
    Remove browser UI artefacts from a raw OCR text string.
    Operates on the full text (before line-splitting) for maximum coverage.
    """
    for pattern in _BROWSER_CHROME_PATTERNS:
        text = pattern.sub(' ', text)
    # Collapse multiple spaces left behind by removals
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


# ╔══════════════════════════════════════════════════════════════╗
# ║  FIX 2 — NEW: _content_fingerprint()                        ║
# ║  Problem : Deduplication used an exact string comparison.    ║
# ║            The live "1679 Online" counter changed every      ║
# ║            frame, so identical pages were never skipped.     ║
# ║  Fix     : This function strips all digits + punctuation     ║
# ║            before comparing, making near-identical frames    ║
# ║            collapse to the same fingerprint. Used in         ║
# ║            apply_rolling_window() and in detectorPipeline.py ║
# ╚══════════════════════════════════════════════════════════════╝
def _content_fingerprint(text: str) -> str:
    """
    Return a normalised fingerprint of a text block used for near-duplicate
    detection.  We:
      1. Lowercase everything.
      2. Remove all digits  (catches the "1679 Online" / "1684 Online" variance
         that made the exact-string check fail).
      3. Remove non-alphanumeric characters.
      4. Collapse whitespace.
    Two captures whose fingerprints match are treated as duplicates.
    """
    text = text.lower()
    text = re.sub(r'\d+', '', text)          # strip numbers
    text = re.sub(r'[^a-z\s]', '', text)     # strip non-alpha
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# -------------------------------------------------
# DATA STRUCTURES
# -------------------------------------------------

@dataclass
class RawCapture:
    """
    Holds a capture of a single <capture> element straight out of XML,
    before any filtering or chunking.
    """
    number:     int
    timestamp:  str
    confidence: float
    lines:      List[str] = field(default_factory=list)
    raw_text:   Optional[str] = None


@dataclass
class SemanticChunk:
    """
    A group of related lines that belong together — a paragraph, a list,
    a heading, or a code block.  Produced by Layer 3.

    Attributes
    ----------
    kind  : One of "heading", "code", "list", "body".
    lines : The cleaned lines that belong to this chunk.
    """
    kind:  str
    lines: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        """
        Render the chunk as a plain-text string suitable for the prompt.
        Headings get a markdown prefix so the AI can see hierarchy.
        Code blocks are indented.  Lists and body text are joined normally.
        """
        if self.kind == "heading":
            return "## " + " ".join(self.lines)
        if self.kind == "code":
            return "\n".join("  " + l for l in self.lines)
        return " ".join(self.lines)


# -----------------------------------------------
# LAYER 1: STRUCTURAL EXTRACTION
# -----------------------------------------------

def extract_captures(xml_path: str) -> List[RawCapture]:
    """
    Layer 1: Parse the XML file and return one RawCapture per <capture> element.
    """
    path = Path(xml_path)
    if not path.exists():
        return []

    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ET.ParseError:
        return []

    # Strip namespaces so we can use plain tag names
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
            number=number,
            timestamp=timestamp,
            confidence=confidence,
            lines=lines,
            raw_text=raw_text,
        ))

    return captures


# -----------------------------------------------------
# LAYER 2: QUALITY FILTERING
# -----------------------------------------------------

def _clean_line(line: str) -> str:
    """
    Apply lightweight text cleaning to a single line:
      1. Strip browser chrome artefacts.
      2. Replace HTML/XML entities (&lt; &gt; &amp;).
      3. Collapse runs of 3+ special symbols (OCR artefacts).
      4. Collapse multiple spaces.
      5. Strip leading/trailing whitespace.
    """
    # FIX 1 — CHANGED: added this call. Original had no chrome stripping here;
    # the line went straight to entity replacement, so browser UI text survived.
    line = _strip_browser_chrome(line)
    line = line.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    line = re.sub(r'[=<>&¢°©®]{2,}', ' ', line)
    line = re.sub(r'\s{2,}', ' ', line)
    return line.strip()


def _is_noise(line: str) -> bool:
    """
    Return True if the line should be discarded.
    """
    if len(line) < MIN_LINE_LENGTH:
        return True
    for pattern in _NOISE_PATTERNS:
        if pattern.search(line):
            return True
    return False


def filter_captures(captures: List[RawCapture]) -> List[RawCapture]:
    """
    Layer 2: Filter and clean a list of RawCapture objects.

    FOR EACH CAPTURE:
      1. Drop the whole capture if its OCR confidence is below MIN_OCR_CONFIDENCE.
      2. Strip browser chrome from the full joined text first (catches
         artefacts that span the single long OCR line).
      3. For each line: clean it, then discard if it is noise.
      4. If no <block> lines survive but <raw> text exists, use that.

    RETURNS a new list with garbage removed.
    """
    filtered: List[RawCapture] = []

    for cap in captures:
        if cap.confidence < MIN_OCR_CONFIDENCE:
            continue

        # BUG FIX: strip chrome from the whole joined text before splitting
        joined = " ".join(cap.lines)
        joined = _strip_browser_chrome(joined)

        # Re-split on sentence-ending punctuation or long gaps to get sub-lines
        # that the chunker can classify individually.
        candidate_lines = re.split(r'(?<=[.?!])\s+', joined)

        clean_lines = [
            _clean_line(l)
            for l in candidate_lines
            if not _is_noise(_clean_line(l))
        ]

        if not clean_lines and cap.raw_text:
            raw = _strip_browser_chrome(cap.raw_text)
            raw = _clean_line(raw)
            if not _is_noise(raw):
                clean_lines = [raw]

        if not clean_lines:
            continue

        filtered.append(RawCapture(
            number=cap.number,
            timestamp=cap.timestamp,
            confidence=cap.confidence,
            lines=clean_lines,
        ))

    return filtered


# ---------------------------------------------------------
# LAYER 3: SEMANTIC CHUNKING
# ---------------------------------------------------------

def _classify_line(line: str) -> str:
    """
    Guess the semantic type of a single line.

    heading : ALL-CAPS line, at least 4 chars, no lowercase.
    code    : Starts with 4+ spaces or a tab.
    list    : Starts with a bullet or number+dot.
    body    : Everything else.
    """
    stripped = line.strip()
    if re.match(r'^[A-Z][^a-z]{3,}$', stripped):
        return "heading"
    if re.match(r'^\s{4,}|^\t', line):
        return "code"
    if re.match(r'^[-•*]\s|^\d+\.\s', stripped):
        return "list"
    return "body"


def chunk_capture(cap: RawCapture) -> List[SemanticChunk]:
    """
    Layer 3: Convert the cleaned lines of a single capture into SemanticChunks.
    Consecutive lines of the same type are grouped together.
    """
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


# --------------------------------------------------------
# LAYER 4: ROLLING WINDOW AND DEDUPLICATION
# --------------------------------------------------------

def apply_rolling_window(captures: List[RawCapture]) -> List[RawCapture]:
    """
    Keep only the ROLLING_WINDOW most-recent captures and drop near-duplicates.

    BUG FIX: The original code compared only the first line with exact equality.
    This missed near-duplicates where the only difference was a live counter
    like "1679 Online" vs "1684 Online".

    We now use _content_fingerprint() on the full joined text, which strips
    digits and punctuation before comparing — making the check robust to
    minor OCR and counter variance between frames.
    """
    window = captures[-ROLLING_WINDOW:]

    deduplicated:    List[RawCapture] = []
    seen_fingerprint: Optional[str]   = None

    for cap in window:
        full_text   = " ".join(cap.lines)
        fingerprint = _content_fingerprint(full_text)

        if fingerprint == seen_fingerprint:
            continue  # near-duplicate — skip

        seen_fingerprint = fingerprint
        deduplicated.append(cap)

    return deduplicated


# ----------------------------------------------------------------
# LAYER 5: PROMPT ASSEMBLY
# ----------------------------------------------------------------

def assemble_prompt(captures: List[RawCapture]) -> str:
    """
    Layer 5: Turn filtered, windowed captures into a single string for the AI.

    Each capture becomes a section separated by a blank line.
    If the assembled text exceeds MAX_PROMPT_CHARS we truncate from the start
    (dropping oldest content) and add a note.
    """
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


# --------------------------------------------------
# PUBLIC API
# --------------------------------------------------

def build_ai_prompt(xml_path: str, user_question: str = "") -> str:
    """
    Main entry point.  Call this from your FastAPI endpoint or anywhere else
    that needs to send screen context to the AI.

    FULL PIPELINE
    -------------
    1. extract_captures      — parse XML into RawCapture objects
    2. filter_captures       — drop low-confidence captures and noisy lines
    3. apply_rolling_window  — keep only the N most-recent, de-duped captures
    4. assemble_prompt       — render to a clean string

    PARAMETERS
    ----------
    xml_path      : Path to the session XML file.
    user_question : Optional question from the user.

    RETURNS
    -------
    A complete prompt string ready to be sent to the AI.
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


# -------------------------------------------------
# SAVE KEY POINTS (called from detectorPipeline.py)
# -------------------------------------------------

def save_key_points(xml_path: str, output_path: str = "captures/key_points.json") -> dict:
    """
    Extract, filter, chunk, and save the key points from a session XML to JSON.

    BUG FIX: The original code stored Path(xml_path).stem as the session name.
    For a file named "session_1778588192958.xml" that stem is already
    "session_1778588192958", so the JSON ended up with "session_session_...".
    We now strip a leading "session_" prefix if present.
    """
    raw_captures = extract_captures(xml_path)
    filtered     = filter_captures(raw_captures)
    windowed     = apply_rolling_window(filtered)

    key_points: List[dict] = []
    for cap in windowed:
        chunks = chunk_capture(cap)
        for chunk in chunks:
            if chunk.lines:
                key_points.append({
                    "type":    chunk.kind,
                    "content": chunk.to_text(),
                })

    # BUG FIX: strip the redundant "session_" prefix from the stem
    stem         = Path(xml_path).stem          # e.g. "session_1778588192958"
    session_name = re.sub(r'^session_', '', stem)  # → "1778588192958"

    payload = {
        "session":     session_name,
        "captured_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "key_points":  key_points,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return payload


# -------------------------------------------------
# CLI TEST ENTRY POINT
# -------------------------------------------------

if __name__ == "__main__":
    import sys

    path     = sys.argv[1] if len(sys.argv) > 1 else "captures/session_test.xml"
    question = sys.argv[2] if len(sys.argv) > 2 else ""

    prompt = build_ai_prompt(path, question)
    print("=" * 60)
    print(prompt)
    print("=" * 60)
    print(f"\nTotal characters sent to AI: {len(prompt)}")