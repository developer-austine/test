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

#----------------------------------------------
# CONFIGURATION
#----------------------------------------------
# the constants here will control the behaviour of every layer

MIN_OCR_CONFIDENCE = 50.0
MIN_LINE_LENGTH = 28
ROLLING_WINDOW = 5
MAX_PROMPT_CHARS = 3000


# Noise Detection Layer (Layer 2).

_NOISE_PATTERNS = [
    re.compile(r'^https?://\S+$'),          # bare URL line
    re.compile(r'^[\W\d\s]{1,15}$'),        # line made entirely of symbols/digits
    re.compile(r'(\bx\b\s*){3,}'),          # browser tab close buttons "x x x"
    re.compile(r'[=<>&¢°]{3,}'),            # runs of stray OCR symbols
    re.compile(r'^\s*[\|\-\+]{3,}\s*$'),    # ASCII table borders
    re.compile(r'^[v€]\s+electron-|^v\s+\w+-course'), # Ignored url pattern fix after discovering the error.
]

#-------------------------------------------------
# DATA STRUCTURES
#-------------------------------------------------

@dataclass
class RawCapture:
    """
    Holds a capture of a single <capture> element straight out of xml, before any filtering or capturing.
    """

    number: int
    timestamp: str
    confidence: float
    lines: List[str] = field(default_factory=list)
    raw_text: Optional[str] = None

@dataclass
class SemanticChunk:
    """
    A group of related lines that belong together — a paragraph, a list,
    a heading, or a code block.  Produced by Layer 3.
 
    Attributes
    ----------
    kind  : One of "heading", "code", "list", "body".
            Mirrors the block_type values written by your backend.
    lines : The cleaned lines that belong to this chunk.
    """

    kind: str
    lines: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        """
        Render the chunk as a plain-text string suitable for the prompt.
        Headings get a markdown-style prefix so the AI can see hierarchy.
        Code blocks are indented.  Lists and body text are joined normally.
        """

        if self.kind == "heading":
            return "## " + " ".join(self.lines)
        if self.kind == "code":
            # Indent each line so that the AI treats each line as code
            return "\n".join("  " + l for l in self.lines)
        # list and body - plain join
        return " ".join(self.lines)
    

#-----------------------------------------------
# LAYER 1: STRUCTURAL EXTRACTION
#-----------------------------------------------

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
        # malformed XML - return nothing rather than crashing the whole pipeline
        return []
    
    # strip namespaces so we can use plain tag names like "capture", "block"
    for element in root.iter():
        if "}" in element.tag:
            element.tag = element.tag.split("}")[1]

    captures: List[RawCapture] = []

    for cap_el in root.findall("capture"):
        number = int(cap_el.get("number", 0))
        timestamp = cap_el.get("timestamp", "")
        confidence = float(cap_el.get("ocr_confidence", 0.0))

        # collect all <line> text values from every <block> inside this capture
        lines: List[str] = []
        for block_el in cap_el.findall("block"):
            for line_el in block_el.findall("line"):
                if line_el.text:
                    lines.append(line_el.text)

        # collect the <raw> fallback text if it exists
        raw_el = cap_el.find("raw")
        raw_text = raw_el.text if raw_el is not None and raw_el.text else None

        captures.append(RawCapture(
            number=number,
            timestamp=timestamp,
            confidence=confidence,
            lines=lines,
            raw_text=raw_text,
        ))

    return captures


#-----------------------------------------------------
# LAYER 2: QUALITY FILTERING
#-----------------------------------------------------

def _clean_line(line: str) -> str:
    """
    Apply lightweight text cleaning to a single line:
      1. Replace HTML/XML entities (&lt; &gt; &amp;) with their characters.
      2. Collapse runs of 3+ special symbols (OCR artefacts like "= = ¢ =").
      3. Collapse multiple spaces into one.
      4. Strip leading/trailing whitespace.
 
    This does NOT remove content — it only tidies noise within a line
    that would otherwise confuse the AI.
    """

    line = line.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    line = re.sub(r'[=<>&¢°©®]{2,}', ' ', line)
    line = re.sub(r'\s{2,}', ' ', line)
    return line.strip()


def _is_noise(line: str) -> bool:
    """
    Return True if line should be discarded.

    A line is noise if:
        - it is shorter than MIN_LINE_LENGTH after cleaning.
        - It matches any pattern in _NOISE_PATTERNS.

    Both checks are deliberately conservative: when in doubt, keep the line.
    The AI is good at ignoring irrelevant context; it is bad at hallucinating
    missing context.
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
      2. For each line: clean it, then discard it if it is noise.
      3. If the capture had no <block> lines but has <raw> text, clean and
         use the raw text as a single synthetic line (so we do not lose captures
         that fell back to raw mode).
 
    RETURNS
    -------
    A new list of RawCapture objects with garbage removed.
    Captures that become completely empty after filtering are also dropped.
    """
    filtered: List[RawCapture] = []

    for cap in captures:
        # check the confidence gate
        if cap.confidence < MIN_OCR_CONFIDENCE:
            continue

        # clean and filter individual lines
        clean_lines = [
            _clean_line(line)
            for line in cap.lines
            if not _is_noise(_clean_line(line))
        ]

        # fall back to raw text if we ended up with nothing
        if not clean_lines and cap.raw_text:
            raw = _clean_line(cap.raw_text)
            if not _is_noise(raw):
                clean_lines = [raw]

        # Drop the whole capture if it's empty after filtering
        if not clean_lines:
            continue

        filtered.append(RawCapture(
            number=cap.number,
            timestamp=cap.timestamp,
            confidence=cap.confidence,
            lines=clean_lines,
        ))

    return filtered

#---------------------------------------------------------
# LAYER 3 - SEMANTIC CHUNKING
#---------------------------------------------------------

def _classify_line(line: str) -> str:
    """
    Guess the semantic type of a single line based on simple heuristics.
    This mirrors the classify_line logic in the backend so the two sides
    stay consistent.
 
    RULES (in priority order)
    -------------------------
    heading : ALL-CAPS line, at least 4 characters, no lowercase letters.
              Examples: "INTRODUCTION", "SECTION 2"
    code    : Starts with 4+ spaces or a tab character.
              Examples: "    def foo():", "\tif x > 0:"
    list    : Starts with a bullet symbol or a number+dot.
              Examples: "- item", "• point", "1. First step"
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
 
    ALGORITHM
    ---------
    Walk through the lines one at a time.
    If the current line has the same type as the previous chunk, we append it.
    If the type changes, we start a new chunk.
 
    This means consecutive body lines form one paragraph, consecutive list
    items stay together, and so on — which gives the AI natural groupings
    rather than a flat soup of sentences.
 
    RETURNS
    -------
    A list of SemanticChunk objects for this capture.
    """

    chunks: List[SemanticChunk] = []
    current: Optional[SemanticChunk] = None

    for line in cap.lines:
        kind = _classify_line(line)
        if current is not None and current.kind == kind:
            current.lines.append(line)
        else:
            current = SemanticChunk(kind=kind, lines=[line])
            chunks.append(current)

    return chunks

#--------------------------------------------------------
# LAYER 4: ROLLING WINDOW AND DEDUPLICATION
#--------------------------------------------------------

def apply_rolling_window(captures: List[RawCapture]) -> List[RawCapture]:
    """
    Keep only the ROLLING_WINDOW most-recent captures.
    """

    window = captures[-ROLLING_WINDOW:]

    deduplicated: List[RawCapture] = []
    seen_first_line: Optional[str] = None

    for cap in window:
        first_line = cap.lines[0] if cap.lines else ""
        if first_line and first_line == seen_first_line:
            continue
        seen_first_line = first_line
        deduplicated.append(cap)

    return deduplicated

#----------------------------------------------------------------
# LAYER 5: PROMPT ASSEMBLY
#----------------------------------------------------------------

def assemble_prompt(captures: List[RawCapture]) -> str:
    """
    Layer 5: Turn a list of filtered, windowed captures into a single string
    that can be embedded into an AI prompt.
 
    STRUCTURE OF THE OUTPUT
    -----------------------
    Each capture becomes a section separated by a blank line.
    Within each capture, SemanticChunks are rendered according to their kind
    (see SemanticChunk.to_text).
 
    TRUNCATION
    ----------
    If the assembled text exceeds MAX_PROMPT_CHARS we truncate from the start
    (dropping the oldest content) and add a note so the AI knows the context
    is partial.  We always preserve the most recent content.
 
    RETURNS
    -------
    A plain-text string ready to be inserted into a prompt template.
    """

    sections: List[str] = []

    for cap in captures:
        chunks = chunk_capture(cap)
        chunk_texts = [c.to_text() for c in chunks if c.lines]
        if chunk_texts:
            sections.append("\n".join(chunk_texts))

    combined = "\n\n".join(sections)

    # truncate from the top if too long
    if len(combined) > MAX_PROMPT_CHARS:
        combined = "[… earlier context omitted …]\n\n" + combined[-MAX_PROMPT_CHARS:]

    return combined


#--------------------------------------------------
# PUBLIC API
#--------------------------------------------------

def build_ai_prompt(xml_path: str, user_question: str = "") -> str:
    """
    Main entry point. Call this from your FastAPI endpoint or anywhere else
    that needs to send screen context to the AI.
 
    FULL PIPELINE
    -------------
    1. extract_captures  — parse XML into RawCapture objects
    2. filter_captures   — drop low-confidence captures and noisy lines
    3. apply_rolling_window — keep only the N most recent captures
    4. assemble_prompt   — render to a clean string
 
    PARAMETERS
    ----------
    xml_path      : Path to the session XML file.
    user_question : Optional question from the user.  If provided it is
                    appended to the prompt so the AI can answer it directly.
                    If omitted the AI is asked for general assistance.
 
    RETURNS
    -------
    A complete prompt string ready to be sent to the AI.
 
    EXAMPLE
    -------
        prompt = build_ai_prompt(
            "captures/session_1778530318262.xml",
            user_question="What is the deadline for the video responses?"
        )
        # → send prompt to Claude / GPT / etc.
    """

    # Layer 1
    raw_captures = extract_captures(xml_path)

    # Layer 2
    filtered = filter_captures(raw_captures)

    # Layer 3 (chunking happens inside assemble_prompt per capture)

    # Layer 4
    windowed = apply_rolling_window(filtered)

    # Layer 5
    content = assemble_prompt(windowed)

    if not content.strip():
        return "No readable screen content was captured yet."
    
    # wrap in a neutral prompt template
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

#-------------------------------------------------
# FINAL TEST PHASE (BY DEV ALEX)
#-------------------------------------------------

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "captures/session_test.xml"
    question = sys.argv[2] if len(sys.argv) > 2 else ""
 
    prompt = build_ai_prompt(path, question)
    print("=" * 60)
    print(prompt)
    print("=" * 60)
    print(f"\nTotal characters sent to AI: {len(prompt)}")

    

def save_key_points(xml_path: str, output_path: str = "captures/key_points.json") -> dict:
    raw_captures = extract_captures(xml_path)
    filtered     = filter_captures(raw_captures)
    windowed     = apply_rolling_window(filtered)

    key_points = []
    for cap in windowed:
        chunks = chunk_capture(cap)
        for chunk in chunks:
            if chunk.lines:
                key_points.append({
                    "type":    chunk.kind,
                    "content": chunk.to_text()
                })

    payload = {
        "session":     Path(xml_path).stem,
        "captured_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "key_points":  key_points
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return payload