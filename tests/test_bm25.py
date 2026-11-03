"""Unit tests for the BM25 keyword scorer (pure Python, no models)."""
from app.bm25 import BM25, tokenize


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("What is the role of APOE e4 in p-tau217?") == ["role", "apoe", "e4", "p", "tau217"]


def test_exact_term_ranks_its_document_first():
    docs = [
        "Lecanemab infusion caused ARIA in some patients.",
        "Amyloid plaques accumulate in the cortex.",
        "Tau tangles spread through limbic regions.",
    ]
    scores = BM25(docs).scores("lecanemab side effects")
    assert scores.index(max(scores)) == 0
    assert scores[1] == scores[2] == 0


def test_rare_terms_outweigh_common_terms():
    docs = ["alzheimer amyloid", "alzheimer tau", "alzheimer apoe", "alzheimer lecanemab"]
    scores = BM25(docs).scores("alzheimer lecanemab")
    assert scores.index(max(scores)) == 3


def test_query_with_no_known_terms_scores_zero():
    assert BM25(["amyloid plaques", "tau tangles"]).scores("bake sourdough bread") == [0, 0]
