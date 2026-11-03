# RAG Evaluation Results

## Run: 2026-11-04 (13 papers, 163 chunks, bge-small + hybrid BM25, phi4-mini)

`python -m app.evaluate`, after: medRxiv banner fix (98 → 164 chunks; two papers had 0),
two-column extraction for the ML review, bge-small embeddings, hybrid BM25 retrieval,
query-prefix removal and the 0.55 relevance floor.

| Metric | Score | Notes |
|---|---|---|
| Labelled retrieval, easy (29) | hit@5 29/29, MRR 1.000 | Questions use each paper's own terms |
| Labelled retrieval, hard (15) | hit@5 14/15, MRR 0.900 | Paraphrased in plain language |
| Retrieval sanity | 5/5 | See caveat 1 |
| Grounded | 5/5 | |
| Citations | 5/5 | All point to real sources |
| Length OK | 5/5 | |
| Safety | 3/3 | |
| **Overall (legacy score)** | **18/18** | Excludes labelled retrieval |

### Caveats (read before quoting the 100%)

1. **Retrieval sanity check** used a 0.5 top-score cutoff tuned for MiniLM; bge scores never go that low,
   so it could not fail. The cutoff is now 0.65 (bottom of the measured on-topic range). Same 5/5 result.
2. **Grounded ≠ best sources.** "What treatments are currently available?" passed but retrieved the rTMS
   review and the neuropsychiatric-symptoms paper, not the lecanemab paper. Grounding only checks that the
   answer matches what was retrieved. The labelled retrieval set is the better measure of retrieval.
3. **Grounding is a word-overlap heuristic**, not a fact check. Example seen while testing: an answer said
   eMTBR-tau is "elevated in amyloid-positive individuals, *suggesting* it reflects tau rather than amyloid".
   The source says it reflects tangles *despite* being highest in amyloid-positive people. Right conclusion,
   wrong reasoning, and the heuristic passes it. The web UI shows the exact excerpts for this reason.
4. **Small sets.** 15 hard questions: one question moves hard MRR by ~0.07. Differences below ~2
   questions are not meaningful.

### Remaining retrieval miss

- "Why do some people get Alzheimer's before 65 with language or visual problems instead of memory loss?"
  should find the NULISA postmortem-brain paper (early-onset, non-amnestic syndromes). No variant tested
  retrieved it in the top 5.

### How retrieval choices were made (labelled set, overall MRR / hard MRR)

| Variant | Overall MRR | Hard MRR | Hard hit@5 |
|---|---|---|---|
| MiniLM, dense only (previous) | 0.898 | 0.700 | 12/15 |
| MiniLM + BM25 hybrid | 0.911 | 0.772 | 13/15 |
| bge-small, dense only | 0.897 | 0.763 | 14/15 |
| bge-small + hybrid | 0.939 | 0.822 | 13/15 |
| MiniLM + cross-encoder rerank | 0.886 | 0.733 | 13/15 |
| MiniLM hybrid + cross-encoder | 0.864 | 0.689 | 14/15 |
| **bge-small + hybrid, no "Alzheimer's" query prefix (adopted)** | **0.966** | **0.900** | **14/15** |

- The cross-encoder (`ms-marco-MiniLM-L-6-v2`) made ranking worse and was not adopted.
- MiniLM truncates input at 256 tokens (~1/3 of a chunk); bge-small reads 512.
- The old query prefix ("Alzheimer's disease research: …") pulled every question toward the corpus:
  with it, off-topic questions ("capital of France") scored 0.40–0.52 (MiniLM) and got answers.
  Without it (bge), the weakest on-topic question scored 0.644 and the strongest off-topic 0.500, so the
  0.55 floor returns no sources for off-topic questions. That floor was chosen from 8 off-topic probes.

## Run: 2026-10-12 (phi4-mini, 4 bioRxiv papers, 60 chunks), historical

Overall 13/18 (72.2%): retrieval sanity 2/5 (double section weighting), grounded 3/5 (corpus had no
treatment or risk-factor papers), citations 5/5, safety 3/3. After the weighting fix, retrieval sanity 5/5.

## Run: Debug Version (post SDK + prompt fix), historical

Overall 14/18 (77.8%) with a looser grounding heuristic (any citation counted as grounded).
Root cause found then: reference-list noise in chunks, fixed with `is_reference_noise()`.
