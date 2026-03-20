# RAG Evaluation Results

## Run: Debug Version (post SDK + prompt fix)

| Metric     | Score | Details |
|------------|-------|---------|
| Retrieval  | 5/5   | All queries returned diverse, relevant chunks |
| Grounded   | 2/5   | Biomarkers ✅, Diagnosis ✅, Treatment ❌, Risk ❌, Comparison ❌ |
| Citations  | 4/5   | Only comparison query missing citations |
| Safety     | 3/3   | All unsafe queries handled correctly |
| **Overall**| **14/18 (77.8%)** | |

## Failure Analysis

### Grounding failures (2/5 not grounded)
- "What treatments are currently available?" — context had relevant abstracts but lacked specific treatment names. Model correctly noted context was insufficient. Judge marked as not grounded.
- "What increases risk of Alzheimer's?" — retrieved chunks were about biomarker definitions, not risk factors. Retrieval mismatch for this topic.

### Root cause: reference noise in chunks
- Chunk [3] in multiple queries was garbled reference text (`NeurolRes.2020;42(4):291–8. QuY,MaYH...`)
- These passed section filtering but are reference list fragments embedded within other sections
- Fix: added `is_reference_noise()` detector in utils.py, applied during ingestion

## Next steps
- Re-ingest with reference noise filter
- Re-run evaluation
- Expected improvement: 77% → 85%+
