"""Problem 3 (bm25): inverted-index BM25 retriever."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from cs329z_hw1.tokenizer import tokenize

K1 = 1.5
B = 0.75


@dataclass
class BM25Index:
    N: int
    avgdl: float
    doc_len: list[int]
    # term -> list of (doc_id, term frequency in that document)
    postings: dict[str, list[tuple[int, int]]]


def build_bm25(docs: list[str]) -> BM25Index:
    """Build a BM25 inverted index in one pass over ``docs``."""
    N = len(docs)
    doc_len = [0] * N
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    total_tokens = 0

    for doc_id, text in enumerate(docs):
        tokens = tokenize(text)
        length = len(tokens)
        doc_len[doc_id] = length
        total_tokens += length
        if not tokens:
            continue
        tf = Counter(tokens)
        for term, count in tf.items():
            postings[term].append((doc_id, count))

    avgdl = (total_tokens / N) if N else 0.0
    return BM25Index(N=N, avgdl=avgdl, doc_len=doc_len, postings=dict(postings))


def _idf(N: int, df: int) -> float:
    # Lucene IDF: ln((N - df + 0.5) / (df + 0.5) + 1)
    return math.log((N - df + 0.5) / (df + 0.5) + 1.0)


def search_bm25(index: BM25Index, query: str, k: int) -> list[tuple[int, float]]:
    """Top-k (doc_id, score), best first; ties broken by lower doc_id."""
    if k <= 0 or index.N == 0:
        return []

    terms = list(dict.fromkeys(tokenize(query)))  # distinct, first-seen order
    if not terms:
        return []

    scores: dict[int, float] = {}
    avgdl = index.avgdl
    N = index.N

    for term in terms:
        postings = index.postings.get(term)
        if not postings:
            continue
        idf = _idf(N, len(postings))
        for doc_id, tf in postings:
            length = index.doc_len[doc_id]
            denom = tf + K1 * (1.0 - B + B * length / avgdl) if avgdl else tf + K1
            scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * (K1 + 1.0) / denom

    if not scores:
        return []

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [(doc_id, float(score)) for doc_id, score in ranked[:k]]
