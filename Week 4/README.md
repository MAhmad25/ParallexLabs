# Week 4 — NLP Analysis: Topics and Sentiment

**October 26–November 1, 2026 · Python 3.12 · uv · scikit-learn · VADER · ChromaDB**

I extended the Week 3 Wikipedia retrieval workflow with unsupervised topic
modeling and sentiment metadata. I fit a reproducible Latent Dirichlet Allocation
(LDA) model to the Week 3 chunks, scored chunk tone with VADER, reviewed topic
terms and representative passages, and compared sentiment predictions with a
small manually labeled sample. I store the enriched chunks and their existing
Gemini vectors in a separate local ChromaDB collection. Search can filter by
topic ID, reviewed topic label, and sentiment before ranking the matching
vectors.

The default scope remains the first 10 cleaned Week 1 articles: 353 chunks. The
cleaned corpus, chunk files, Chroma databases, and API keys stay local. No
GitHub upload is needed to run this project. I am keeping the Week 4 files local
for now and have not committed or pushed this work.

## What I built

| Task | Implementation |
| --- | --- |
| Topics | scikit-learn count-vectorized LDA; fixed seed, eight requested topics, interpretable top terms |
| Topic review | High-confidence excerpts and top terms in the report; human-readable names in `config/topic_labels.json` |
| Sentiment | VADER compound score with positive/neutral/negative thresholds; short text is marked unknown |
| Manual evaluation | Reviewed chunk IDs and rationales in `queries/sentiment_gold.jsonl`; accuracy, macro-F1, and confusion matrix in the report |
| Vector metadata | Topic ID/label/confidence/distribution plus sentiment label, compound score, and status |
| Filtered retrieval | Gemini query embeddings with optional topic ID, topic label, and sentiment filters |
| Vector reuse | Reuses Week 2 vectors only after checking the chunk file hash, model, dimensions, IDs, text, and vector normalization |
| Local verification | Independently compares every stored text, metadata record, and vector, then exercises sentiment and combined filters |

## Run the workflow

From PowerShell:

```powershell
Set-Location 'D:\ParallexLabs\Week 4'
uv sync --locked
```

If this is a fresh checkout and no local Week 4 key is configured, create the
ignored environment file and add my Google AI Studio key:

```powershell
if (!(Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

Week 4 can read the already-configured local `Week 2/.env` key as a fallback. It
does not write the key into a report. Query embedding uses Gemini. For indexing,
the project first checks whether the exact Week 2 Gemini vectors can be reused;
if they are unavailable or do not match, it embeds the chunks with Gemini and
uses paced, resumable batches.

Create Week 3's default chunk file if it is missing:

```powershell
Set-Location 'D:\ParallexLabs\Week 3'
uv sync --locked
uv run --locked python -m scripts.chunk_corpus
Set-Location 'D:\ParallexLabs\Week 4'
```

Enrich and review the corpus:

```powershell
uv run --locked python -m scripts.enrich_corpus
uv run --locked python -m scripts.evaluate_sentiment
uv run --locked python -m scripts.validate_edge_cases
```

`reports/nlp_analysis.json` lists the topic terms, their strongest representative
chunks, sentiment counts, model settings, and corpus hash. I reviewed every topic
against its top terms and three highest-confidence chunks. Four topics were
coherent or mostly coherent, with some spillover in two; four had mixed examples.
The mixed topics are marked in `config/topic_labels.json` and in
`reports/topic_review.json`. The labels are saved to each chunk when I rerun
`enrich_corpus`. The manual sentiment labels are held out from VADER and are
used only by the evaluation command. `validate_edge_cases` records the short
text policy and the known climate-jargon false positive.

Build and verify the filtered vector index:

```powershell
uv run --locked python -m scripts.index_corpus
uv run --locked python -m scripts.verify_index
```

Search all indexed chunks or combine filters. Use the topic IDs in the current
`reports/nlp_analysis.json`; the topic labels are the manually reviewed
names in `config/topic_labels.json`.

```powershell
uv run --locked python -m scripts.ask "What does the corpus say about climate?"
uv run --locked python -m scripts.ask "What does the corpus say about climate?" --topic-id 4
uv run --locked python -m scripts.ask "What does the corpus say about awards?" --sentiment positive
uv run --locked python -m scripts.ask "What does the corpus say about Alabama?" --topic-id 1 --sentiment positive
uv run --locked python -m scripts.ask "What does the corpus say about war?" --topic-id 0 --sentiment negative
uv run --locked python -m scripts.ask "What does the corpus say about history?" --topic-label "reviewed topic name" --k 8
```

The filters use exact Chroma metadata equality. Multiple filters are combined
with AND. Chroma then ranks only matching chunks by cosine similarity to the
Gemini query embedding. The command returns excerpts and metadata; it does not
generate a new answer.

## Evaluation and interpretation

The sentiment sample is a set of manually reviewed chunk IDs, not a random
estimate of the whole Wikipedia corpus. In my 15-chunk sample, VADER reached
**53.3% accuracy** and **0.448 macro-F1**. It correctly identified all four
manually labeled negative chunks, but it predicted no neutral chunks; neutral
recall was **0%**. The confusion matrix, per-class results, and every reviewed
decision are in `reports/sentiment_evaluation.json`. This is a small-sample
result, not evidence of performance on other writing styles. VADER is
lexicon-based: it can mistake factual words, historical violence, negation, and
technical vocabulary for sentiment. Short or empty-of-signal text receives
`unknown` instead of an unsupported neutral prediction.

LDA is unsupervised, so its topic IDs are clusters rather than ground-truth
categories. I review top terms and high-confidence example chunks, then name
the topics manually. Four of the eight clusters were coherent or mostly
coherent; four were mixed on this small subset. The mixed topic labels make
that visible to retrieval users. Topic coherence is assessed by inspection rather than a
classification-accuracy score. The report includes corpus perplexity as a model
diagnostic; it does not prove that the topics are useful.

The sample is limited to 10 articles chosen by Week 1's existing corpus order.
The results and topic names should be rechecked when the corpus, chunking,
vectorizer, or topic count changes. Reproducibility inputs include the input
hash, vectorizer settings, LDA random seed, embedding model, and vector
dimensions.

## Current run results

| Check | Result |
| --- | --- |
| Week 1 articles and Week 3 chunks | 10 articles; 353 chunks |
| LDA topics | 8; all chunks assigned; 4 coherent or mostly coherent and 4 mixed after manual review |
| VADER corpus labels | 222 positive, 124 negative, 7 neutral |
| Manually labeled sentiment sample | 15 chunks; 8/15 correct; **53.3% accuracy**, **0.448 macro-F1** |
| Neutral sentiment recall | **0%** on the manual sample; VADER predicted no neutral examples in it |
| Enriched Chroma records | 353 records with 768-dimensional Gemini embeddings and NLP metadata |
| Embedding calls to build this index | **0**; all 353 vectors were reused from the matching Week 2 snapshot |
| Independent index verification | 353/353 texts, metadata, vector dimensions, and norms checked; sentiment, topic, and combined filters returned exact matches |
| Filtered retrieval example | A topic-label plus positive-sentiment query returned 2 Alabama chunks; cosine similarities were 0.742 and 0.727 |
| Edge cases | Short text is unknown/unassigned; climate-science “positive feedback” is a known VADER false positive |

The sentiment numbers describe this small reviewed sample only. The neutral
class failure and mixed LDA topics mean I treat this metadata as retrieval hints
that need review. Details are in [the topic review](reports/topic_review.json),
[the sentiment evaluation](reports/sentiment_evaluation.json),
[the edge-case review](reports/edge_case_review.json),
[the NLP report](reports/nlp_analysis.json), [the index report](reports/index_report.json),
and [the index verification](reports/index_verification.json).

## Layout

```text
Week 4/
├── .env.example
├── .python-version
├── config/topic_labels.json
├── data/processed/          # Local enriched JSONL; ignored by Git
├── chroma_db/               # Local filtered retrieval index; ignored by Git
├── queries/sentiment_gold.jsonl
├── reports/                 # Tracked run reports
├── scripts/
│   ├── ask.py
│   ├── enrich_corpus.py
│   ├── evaluate_sentiment.py
│   ├── index_corpus.py
│   ├── validate_edge_cases.py
│   └── verify_index.py
├── src/
│   ├── gemini.py
│   ├── io_utils.py
│   ├── nlp.py
│   ├── paths.py
│   └── vector_store.py
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

## References

- [scikit-learn Latent Dirichlet Allocation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment#about-the-scoring)
- [Chroma metadata filtering](https://docs.trychroma.com/docs/querying-collections/metadata-filtering)
- [Gemini embedding models](https://ai.google.dev/gemini-api/docs/embeddings)
