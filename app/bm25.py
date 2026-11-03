"""
Okapi BM25 keyword scoring for hybrid retrieval.

Embeddings miss exact technical terms (APOE ε4, p-tau217, lecanemab); BM25 catches them.
Small corpus (hundreds of chunks), so a plain in-memory implementation is fast enough.
"""
import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set(
    "a an the of in on for to and or is are was were be by with as at from that this which what how do does "
    "can why who whom whose than into about over under between their there these those it its not no".split()
)


def tokenize(text: str) -> list[str]:
    return [w for w in _TOKEN.findall(text.lower()) if w not in _STOP]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.doc_len = [sum(tf.values()) for tf in self.tf]
        self.avg_len = sum(self.doc_len) / max(len(self.doc_len), 1)
        df = Counter(term for tf in self.tf for term in tf)
        n = len(docs)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = [t for t in tokenize(query) if t in self.idf]
        out = []
        for tf, length in zip(self.tf, self.doc_len, strict=True):
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_len)
            out.append(sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in terms if t in tf))
        return out
