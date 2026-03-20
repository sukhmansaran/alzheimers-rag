import os
import re


# ── Regex and cleaning ────────────────────────────────────────

_CITATION_NOISE = re.compile(r"\[\d+(?:[,\-–]\s*\d+)*\]")
_EXTRA_WHITESPACE = re.compile(r"\s+")

_HEADING_PATTERN = re.compile(
    r"\n\s*(\d{0,2}\.?\s*(?:Introduction|Background|Methods?|Methodology|"
    r"Materials?\s*(?:and|&)\s*Methods?|Results?|Discussion|"
    r"Conclusion|Conclusions|Abstract|Limitations|"
    r"Future\s*Directions?|Literature\s*Review|Review|"
    r"References|Acknowledgments?))\s*\n",
    re.IGNORECASE,
)

_FALLBACK_HEADING = re.compile(r"\n\s*([A-Z][A-Za-z ]{3,40})\s*\n")


def normalize_section(name: str) -> str:
    """Normalize section names to consistent labels."""
    name = name.lower().strip()
    name = re.sub(r"^\d+\.?\s*", "", name)

    if "method" in name or "material" in name:
        return "Methods"
    if "result" in name:
        return "Results"
    if "discussion" in name:
        return "Discussion"
    if "intro" in name or "background" in name:
        return "Introduction"
    if "conclusion" in name:
        return "Conclusion"
    if "abstract" in name:
        return "Abstract"
    if "reference" in name:
        return "References"
    if "acknowledg" in name:
        return "Acknowledgments"
    if "limitation" in name:
        return "Limitations"
    if "review" in name:
        return "Review"
    return name.title()


def clean_text(text: str) -> str:
    """Normalize whitespace, strip citations, fix broken PDF text."""
    text = _CITATION_NOISE.sub("", text)
    text = text.replace("-\n", "")
    text = text.replace("\n", " ")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = _EXTRA_WHITESPACE.sub(" ", text)
    return text.strip()


# ── Section detection ─────────────────────────────────────────

def detect_sections(text: str) -> list[dict]:
    """Split text into sections based on academic headings with fallback."""
    matches = list(_HEADING_PATTERN.finditer(text))

    if not matches:
        matches = list(_FALLBACK_HEADING.finditer(text))

    if not matches:
        return [{"section": "Full Text", "text": text.strip()}]

    sections = []

    pre = text[: matches[0].start()].strip()
    if pre and len(pre.split()) > 50:
        sections.append({"section": "Preamble", "text": pre})

    for i, m in enumerate(matches):
        heading = normalize_section(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        if body and len(body.split()) > 50:
            sections.append({"section": heading, "text": body})

    return sections


# ── Chunking with section weighting ───────────────────────────

SECTION_WEIGHTS = {
    "Abstract": 1.5,
    "Introduction": 1.2,
    "Methods": 1.0,
    "Results": 2.0,
    "Discussion": 1.5,
    "Conclusion": 1.3,
    "Review": 1.2,
    "Preamble": 0.8,
}


def chunk_text_weighted(
    text: str, section: str, base_size: int = 400, overlap: int = 80,
) -> list[dict]:
    """Chunk text by word count with overlap and section weight."""
    weight = SECTION_WEIGHTS.get(section, 1.0)
    words = text.split()

    if len(words) <= base_size:
        return [{"section": section, "weight": weight, "text": text.strip()}]

    # Bigger chunks for dense sections
    chunk_size = base_size
    if section in ("Results", "Discussion"):
        chunk_size = int(base_size * 1.5)

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])

        if end < len(words):
            for j in range(end, min(end + 30, len(words))):
                if words[j].endswith((".", "?", "!")):
                    chunk = " ".join(words[start : j + 1])
                    end = j + 1
                    break

        chunks.append({
            "section": section,
            "weight": weight,
            "text": chunk.strip(),
        })

        start = end - overlap if end < len(words) else end

    return chunks


# ── Content quality filters ───────────────────────────────────

REFERENCE_PATTERNS = [
    r"\bdoi\b", r"\bet al\b", r"\bpubmed\b", r"\bjournal\b",
    r"\bvolume\b", r"\bissue\b", r"\bpages?\b", r"\bhttps?://", r"\bwww\.",
]


def is_reference_noise(text: str) -> bool:
    """Detect reference/bibliography garbage."""
    text_lower = text.lower()
    if sum(1 for p in REFERENCE_PATTERNS if re.search(p, text_lower)) >= 2:
        return True
    digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
    return digit_ratio > 0.25


def is_bad_chunk(text: str) -> bool:
    """Kill weak / useless chunks."""
    words = text.split()
    if len(words) < 30:  # lowered — keep small but valuable chunks
        return True
    if len(words) > 800:
        return True
    if text.count(";") > 10:
        return True
    return False


def is_answerable(text: str) -> bool:
    """Keep only chunks with concrete, answerable content."""
    text_lower = text.lower()

    WEAK_PATTERNS = [
        "is a progressive disease", "is a major cause",
        "characterized by memory loss", "growing rapidly",
        "is the most common",
    ]
    if any(w in text_lower for w in WEAK_PATTERNS):
        return False

    STRONG_SIGNALS = [
        "biomarker", "risk factor", "associated with",
        "increased risk", "decreased risk",
        "diagnosis", "diagnostic", "measured",
        "predict", "indicator", "linked to",
        "caused by", "results in", "defined as",
        "treatment", "therapy", "clinical trial",
        "amyloid", "tau", "cognitive", "neurodegeneration",
        "imaging", "csf", "plasma", "apoe",
        "demonstrated", "showed", "found that", "reported",
        "significant", "compared", "levels of",
    ]
    return any(s in text_lower for s in STRONG_SIGNALS)


# ── Helpers ───────────────────────────────────────────────────

def get_pdf_paths(data_dir: str = "data") -> list[str]:
    """Return sorted list of PDF file paths."""
    return sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".pdf")
    )
