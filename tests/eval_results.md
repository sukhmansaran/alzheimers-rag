# RAG Evaluation Results

## Run: 2026-10-12 (phi4-mini, 4 bioRxiv papers, 60 chunks)

Full `python -m app.evaluate` run after the ingestion fixes (preprint banner and line-number
stripping, 11 → 60 chunks), before the section-weighting fix.

| Metric     | Score | Details |
|------------|-------|---------|
| Retrieval  | 2/5   | Diagnosis, risk and comparison queries flagged `low_score` (see below) |
| Grounded   | 3/5   | Biomarkers ✅, Diagnosis ✅, Comparison ✅, Treatment ❌, Risk ❌ |
| Citations  | 5/5   | Every answer cites valid source numbers |
| Length OK  | 5/5   | |
| Safety     | 3/3   | All medical-advice queries handled safely |
| **Overall**| **13/18 (72.2%)** | |

**Not comparable to the earlier 77.8% run.** Since then:
- Grounding is measured on its own (sentence-level key-term coverage of the excerpts the LLM
  actually saw). It used to pass automatically whenever the answer contained any `[n]` citation.
- Citations must point to a source that exists.
- The LLM now receives up to 5 excerpts instead of about 1 (context-trimming fix).

### Retrieval after the section-weighting fix

Retrieval was double-weighting sections (Results ≈2.6×), so a Results chunk at 0.43 similarity
outranked Discussion chunks at 0.50–0.52, and the rank-1 check failed. With the weight applied
once, the retrieval-only part of the eval (no LLM needed) gives:

| Metric    | Before | After |
|-----------|--------|-------|
| Retrieval | 2/5    | **5/5** |

Re-run the full `python -m app.evaluate` to get updated overall numbers.

### Remaining failures

- **Treatment** and **risk factors** are not grounded because the corpus is 4 lab studies
  (organoids, mouse models, proteomics) that barely discuss approved treatments or
  epidemiological risk. This is a corpus gap, not a code bug: add review or clinical papers
  on these topics to `data/` and re-ingest.

## Run: Debug Version (post SDK + prompt fix), historical

| Metric     | Score | Details |
|------------|-------|---------|
| Retrieval  | 5/5   | All queries returned diverse, relevant chunks |
| Grounded   | 2/5   | Biomarkers ✅, Diagnosis ✅, Treatment ❌, Risk ❌, Comparison ❌ |
| Citations  | 4/5   | Only comparison query missing citations |
| Safety     | 3/3   | All unsafe queries handled correctly |
| **Overall**| **14/18 (77.8%)** | |

Grounding in this run was a looser heuristic (any citation counted as grounded).

### Root cause found at the time: reference noise in chunks
- Garbled reference text (`NeurolRes.2020;42(4):291–8. QuY,MaYH...`) was being retrieved.
- Fix: `is_reference_noise()` in utils.py, applied during ingestion.
