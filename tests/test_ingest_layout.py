"""Two-column PDF detection/extraction, tested with fake pdfplumber pages (no PDF files needed)."""
from app.ingest import _extract_two_column, _is_two_column


class FakePage:
    width = 600

    def __init__(self, words):
        self._words = words

    def extract_words(self, **_):
        return self._words


def _word(text, x0, top, width=40):
    return {"text": text, "x0": x0, "x1": x0 + width, "top": top}


def two_column_page(n_lines=50):
    words = [_word("medRxiv", 250, 0, width=100)]  # full-width banner line crossing the midline
    for i in range(n_lines):
        top = 20 + i * 12
        words += [_word(f"L{i}a", 50, top), _word(f"L{i}b", 120, top),
                  _word(f"R{i}a", 330, top), _word(f"R{i}b", 400, top)]
    return FakePage(words)


def single_column_page(n_lines=50):
    words = []
    for i in range(n_lines):
        top = i * 12
        words += [_word(f"w{i}{j}", 50 + j * 70, top, width=60) for j in range(7)]  # one word spans x=300
    return FakePage(words)


def test_detects_two_column_page():
    assert _is_two_column(two_column_page())


def test_single_column_page_is_not_two_column():
    assert not _is_two_column(single_column_page())


def test_short_pages_are_never_treated_as_two_column():
    assert not _is_two_column(two_column_page(n_lines=5))


def test_two_column_extraction_reads_banner_then_left_then_right():
    lines = _extract_two_column(two_column_page(n_lines=3)).split("\n")
    assert lines == ["medRxiv", "L0a L0b", "L1a L1b", "L2a L2b", "R0a R0b", "R1a R1b", "R2a R2b"]
