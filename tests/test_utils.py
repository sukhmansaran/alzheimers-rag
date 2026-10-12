"""Unit tests for pure text-processing helpers in app.utils."""
from app.utils import (
    chunk_text_weighted, clean_text, detect_sections, is_reference_noise,
    normalize_section,
)


def _words(n: int, word: str = "word") -> str:
    return " ".join(f"{word}{i}" for i in range(n))


def test_normalize_section_variants():
    assert normalize_section("2. Materials and Methods") == "Methods"
    assert normalize_section("RESULTS") == "Results"
    assert normalize_section("Background") == "Introduction"
    assert normalize_section("Acknowledgement") == "Acknowledgments"


def test_clean_text_strips_citations_and_whitespace():
    assert clean_text("Amyloid [1,2] plaques\n  accumulate [3-5].") == "Amyloid plaques accumulate ."
    assert clean_text("neuro-\ndegeneration") == "neurodegeneration"


def test_detect_sections_splits_on_headings():
    text = f"\nAbstract\n{_words(60)}\nResults\n{_words(60)}\nReferences\n{_words(60)}\n"
    names = [s["section"] for s in detect_sections(text)]
    assert names == ["Abstract", "Results", "References"]


def test_detect_sections_drops_short_bodies():
    text = f"\nAbstract\n{_words(10)}\nResults\n{_words(60)}\n"
    assert [s["section"] for s in detect_sections(text)] == ["Results"]


def test_chunk_short_text_is_single_chunk():
    chunks = chunk_text_weighted(_words(100), "Abstract")
    assert len(chunks) == 1
    assert chunks[0]["weight"] == 1.5


def test_chunk_long_text_overlaps_and_covers_everything():
    words = _words(1000).split()
    chunks = chunk_text_weighted(" ".join(words), "Methods", base_size=400, overlap=80)
    assert len(chunks) > 1
    first, second = chunks[0]["text"].split(), chunks[1]["text"].split()
    assert first[-80:] == second[:80]
    assert chunks[-1]["text"].split()[-1] == words[-1]


def test_reference_noise_detection():
    assert is_reference_noise("Smith et al. J Neurol. doi:10.1000/xyz, volume 12")
    assert not is_reference_noise("Plasma p-tau217 predicted amyloid positivity in this cohort.")
