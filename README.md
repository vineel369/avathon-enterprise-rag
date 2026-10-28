# Enterprise Technical Knowledge Assistant

## Avathon AI/ML Challenge

This project is built for the **Avathon Senior AI Engineer hiring challenge** using:

- **Scenario S2:** Gen AI for Enterprise Documents
- **Track D:** RAG / LLM Knowledge Systems
- **Dataset:** NVIDIA `TechQA-RAG-Eval`
- **Modality:** Text only
- **UI:** Streamlit conversational chatbot
- **Orchestration:** LangChain + LangGraph
- **Vector DB:** Qdrant
- **Retrieval:** Dense + BM25 + Reciprocal Rank Fusion (RRF) + reranking
- **LLM gateway:** LiteLLM Proxy
- **Generation:** pinned stable Gemini Flash primary, pinned Groq-hosted fallback for retryable interactive failures
- **Semantic cache:** Valkey, with authorization and version scoping
- **Conversation storage:** PostgreSQL
- **Deployment:** Docker / Docker Compose

The goal is to build a simple, robust, measurable, and production-minded RAG system for enterprise technical support documentation.

### Current implementation status

Documentation synchronized on **2026-10-27**. Phase-3 chunk generation and the structure-v2 repair are complete; Phase 4 has not started. Full-corpus size/overlap/model selection experiments remain pending.

Phase 1 provides the runtime scaffold. Phase 2 adds a reproducible TechQA download, corpus inspection, separate QA manifest, and frozen development/test question IDs. Phase 3 implements versioned source-only chunk generation and an unscored staged experiment plan. The app still does not answer questions; embeddings, indexing, retrieval-based model/chunking selection, generation, caching, and Langfuse instrumentation follow in later phases.

To inspect the scaffold locally, install Docker Engine with Compose, copy `.env.example` to `.env`, and replace the `CHANGE_ME` values with private credentials. The Gemini key is required for actual gateway requests. Then run:

```bash
docker compose config --quiet
docker compose up --build
```

Open `http://localhost:8501` for the placeholder app. Qdrant, PostgreSQL, LiteLLM, and Valkey are wired in Compose, but the Phase 1 app does not call them. Langfuse Cloud settings are accepted through environment variables; tracing begins when the application workflow is implemented. No credentials should be committed.

The pinned Python version is 3.11. `requirements.txt` now pins Streamlit and the RAGAS 0.4.3 evaluation stack, including the compatible pre-1.0 LangChain and Pydantic versions required by that RAGAS release. Later phases will pin their added retrieval and application packages. For a local run without Docker, install that manifest and run `python -m streamlit run app/streamlit_app.py`.

Prepare the dataset from the pinned NVIDIA revision with either the local Python environment or the application image:

```bash
python -m scripts.data_preparation.prepare_dataset
# or: docker compose run --rm rag-app python -m scripts.data_preparation.prepare_dataset
python -m unittest discover -s tests
```

The first command downloads `corpus.zip` and `train.json` into ignored `data/raw/techqa/`, safely extracts the archive into `data/raw/techqa/extracted/<corpus-version>/`, writes canonical UTF-8 documents into `data/processed/techqa/documents/<corpus-version>/`, and prints actual audit counts. It needs network access to Hugging Face on the first run. To inspect already downloaded files offline, pass `--skip-download`.

Phase 2 creates:

- `corpus_manifest.jsonl`: source paths, checksums, encoding, title/section metadata, size, version, duplicate status, and canonical paths
- `qa_manifest.jsonl`: evaluation questions and labels with resolved source IDs, provenance hashes, blocking issues, and non-blocking audit flags
- `splits.json`: frozen development and held-out IDs
- `evaluation/*.jsonl`: materialized answerable and unanswerable development/test records
- `inspection_report.json`: source hashes/sizes, schema profile, data-quality counts, corpus coverage, collisions, and leakage checks

The split targets are 100 answerable development questions, 100 held-out answerable questions, and 20 unanswerable questions in each group; shared source documents and near-duplicate questions stay together, so actual counts may differ. The script rejects a held-out answerable target below the challenge minimum of 20 and fails if development/test IDs, evidence documents, or question fingerprints overlap. Inspect `inspection_report.json` and resolve missing or disputed sources before treating the benchmark as frozen. Only the versioned canonical source documents are intended for indexing. QA answers and gold context text stay outside that directory.

Run `python -m scripts.data_preparation.show_qa_examples` after preparation to preview records `TRAIN_Q000`, `TRAIN_Q001`, and `TRAIN_Q002` with their answerability labels and a short gold-context excerpt. These previews are for human inspection only. If raw files already exist, the preparation script reuses them and records their SHA-256 hashes; verify their provenance before claiming a pinned dataset run.

The planned batch ingestion entry point is `docker compose run --rm rag-app python -m scripts.indexing.ingest`; this module and Qdrant/BM25 indexing are not implemented yet. Phase-3 chunk generation is available below. Initial TechQA preparation remains a separate batch command without a task broker.

---

## Why This Dataset?

The project uses the NVIDIA TechQA-RAG-Eval dataset because it provides the core ingredients needed for rigorous RAG evaluation:

- a separate `corpus.zip` archive containing raw technical-note files
- technical support questions
- ground-truth answers
- supporting text contexts
- `is_impossible` labels for unanswerable questions

This allows us to evaluate both retrieval quality and answer grounding / faithfulness.

The pinned raw dataset download totals approximately 49.8 MB: about 45.8 MB for `corpus.zip` and 4.02 MB for `train.json`. This excludes extracted documents, generated chunk artifacts, and the local Python environment. The pinned audit found 910 QA records, of which 908 pass blocking checks, and 28,477 unique ready documents plus four content duplicates. The upstream split is named `train`; our frozen development and held-out splits are separate project artifacts.

### Dataset links

- NVIDIA TechQA-RAG-Eval:  
  https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval
- Original IBM TechQA repository:  
  https://github.com/IBM/techqa
- IBM TechQA publication:  
  https://research.ibm.com/publications/the-techqa-dataset

---

## Why RAG?

TechQA is well suited for RAG because answers depend on retrieving specific technical documentation at query time.

RAG keeps enterprise knowledge outside the model weights and provides:

- grounded answers
- source citations
- easier knowledge updates
- better hallucination control
- measurable retrieval quality

### Why not prompting alone?

Prompting can control model behavior, but it does not provide missing product-specific technical knowledge.

### Why not fine-tuning?

Fine-tuning is useful for adapting model behavior or task capability, but it is not ideal for frequently changing factual documentation because:

- knowledge updates require retraining
- source attribution is weak
- hallucinations can still occur
- it adds training and deployment complexity

### Why not long-context prompting?

Passing many documents directly to an LLM:

- increases cost
- increases latency
- does not scale well
- may reduce attention to the most relevant evidence

### Why hybrid RAG?

Technical support queries often include exact error codes, commands, product names, and version numbers.

Dense retrieval is strong for semantic similarity, while BM25 is strong for exact lexical matching.

Therefore, this project uses:

```text
Dense Retrieval
      +
BM25 Sparse Retrieval
      |
     RRF
      |
   Reranker
      |
  Top-k Context
      |
      LLM
```

---

## Dataset Usage and Ingestion

The dataset source and the ingestion pipeline are separate concerns. The project uses NVIDIA's raw corpus archive as the knowledge source and implements the required loading, chunking, embedding, and indexing stages itself.

| Artifact or field | Role |
|---|---|
| `corpus.zip` | Raw `.txt` technical documents loaded by the ingestion pipeline |
| `contexts[].filename` | Gold source-document relevance and corpus-coverage validation |
| `contexts[].text` | Evidence audit and provenance check; not the primary ingestion source |
| `question` | User and evaluation query |
| `answer` | Ground-truth answer used only for evaluation |
| `is_impossible` | Ground-truth unanswerable flag used only for evaluation |
| `id` | Stable evaluation record identifier |

The ingestion path is:

```text
Pinned NVIDIA dataset revision
          |
     corpus.zip
          |
extract raw technical-note files
          |
load -> validate -> normalize -> deduplicate
          |
canonical versioned document manifest
          |
chunk -> embed -> index in Qdrant and BM25
```

Deduplication uses normalized content hashes while preserving every source filename mapping. The loader rejects path traversal, records UTF-8 or Windows-1252 decoding, extracts title and section-heading metadata, and writes immutable version-scoped artifacts so an older run cannot leave stale documents in a newer corpus. Gold context filenames are checked against the extracted corpus; missing, ambiguous, and content-different evidence is reported without placing gold context text in the indexable corpus.

The following are **not** indexed:

- questions
- answers
- `is_impossible`
- QA pairs
- QA-specific `contexts[].text`

The same frozen corpus is available to development and held-out queries. Questions and answers remain isolated from indexing and held-out records are not used for chunking, retrieval, prompt, or threshold tuning.

The original IBM collection of roughly 802,000 Technotes is not required for the initial 48-hour implementation. It is considered only if inspection finds a material gap in NVIDIA's reduced corpus or as a separately reported scale extension.

---

## Architecture

```text
                     NVIDIA corpus.zip
                             |
                  Load / Validate / Deduplicate
                             |
                 Canonical Document Corpus
                             |
             Controlled Chunking Experiment
                             |
                +------------+------------+
                |                         |
         Dense Embeddings               BM25
                |                         |
             Qdrant                       |
                |                         |
                +------------+------------+
                             |
                            RRF
                             |
                         Reranker
                             |
                       Top-k Context
                             |
                         LangGraph
                             |
                      Evidence Gate
                       /       \
                  enough   insufficient
                     |          |
             Semantic Cache   Abstain
                /      \
              hit      miss
               |        |
               |   LiteLLM Proxy
               |      /      \
               |  Gemini    Groq fallback
               |        |
               +-- Validated Grounded Answer
                             |
                         Citations
                             |
          PostgreSQL-backed Streamlit Chat UI
```

---

## Why Qdrant?

Qdrant is preferred because it provides:

- simple Docker deployment
- persistence
- metadata filtering
- strong dense-vector retrieval
- easy Python integration
- production-oriented service architecture

### Alternatives considered

**Weaviate** is a strong alternative with native hybrid search. Qdrant is preferred because this project wants dense retrieval, BM25, fusion, and reranking to remain explicit for easier experimentation and evaluation.

**FAISS** is excellent for local ANN search, but it is a library rather than a full vector database. Qdrant better demonstrates service deployment, persistence, and production-oriented architecture.

---

## Why LangGraph?

LangGraph is used because the chatbot needs explicit state and conditional flow:

```text
Conversation
    |
Query Contextualization
    |
Retrieval
    |
Fusion
    |
Reranking
    |
Evidence Gate
   /       \
Answer   Abstain
```

LangChain is used for common RAG components, while LangGraph manages state, branching, and conversational control flow.

---

## Phase-3 chunk generation

Install `requirements.txt` (or use the rebuilt application image), then run from the repository root:

```bash
python -m scripts.chunking.prepare_tokenizer
python -m scripts.chunking.build_chunks --tokenizer data/tokenizers/bge-base-en-v1.5/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/tokenizer.json --tokenizer-revision a5beb1e3e68b9ab74eb54cfd186867f64f240e1a
python -m scripts.chunking.compare_chunking_strategies
python -m scripts.chunking.inspect_fragmentation
python -m unittest discover -s tests
```

The tokenizer downloader fetches only the pinned tokenizer JSON, verifies its SHA-256, and needs network access once. Chunk generation then runs offline against all `ready` sources in `corpus_manifest.jsonl`; it never loads QA records. A QA manifest is rejected. Duplicate/excluded source counts are reported. The model limit includes special tokens; titles and sections are retained as metadata rather than prepended to embedding text.

Each configuration writes `data/processed/techqa/chunks/<configuration-id>/{chunks.jsonl,config.json,report.json}`. Chunks retain exact source-text slices, character/token offsets, source filename/ID, title, section label, index, content hash, and strategy/corpus/configuration versions. Tokenizer model/revision/file checksum and package version are recorded in the configuration. Offsets are zero-based, half-open ranges in the decoded processed source with Python newline normalization. Token offsets refer to the whole-source tokenization; `token_count` is measured again on the actual standalone chunk, and `embedding_input_token_count` also includes special tokens. Section detection is heuristic for the plain-text corpus; `section_path` contains the recognized section at the chunk start, not an invented hierarchy; structure v2 also records `section_labels` for every section covered by a packed chunk. Structure v2 packs adjacent complete small sections and paragraphs toward the token target. It suppresses heading-only paragraph boundaries so headings stay with their first body text where possible. Overlap applies to oversized block splits; ordinary packed blocks advance without overlap. Heading detection rejects code-like uppercase values and requires stronger context for unfamiliar labels.

Runs publish only after every ready source passes hash and input-limit validation. Interrupted runs leave a `.partial` directory and restart that configuration; they do not append duplicate chunks. Existing completed manifests are checksum-verified and reused. Concurrent writers to the same configuration are unsupported; distinct configurations use separate directories. New diagnostic reports record chunk count, average/median/p95/max tokens, counts below 16/64 tokens, duplicated-source-token ratio, elapsed build time, output size, and truncation count. This size is the JSONL artifact size, not a Qdrant index-size measurement.

Use `--strategy fixed|recursive|structure|all`, `--target-tokens`, `--overlap-tokens`, and `--max-input-tokens` to build a configuration. For example, the bounded BGE size check can build `--target-tokens 256 --overlap-tokens 64`; a 480-token target fits BGE including its two special tokens. Longer configurations require a compatible tokenizer/model and explicit input limit, and are gated on development selection rather than built speculatively.

`config/chunking_experiments.json` records the free-only embedding shortlist, 256/384/480 size checks, 32/64/96 overlap checks, conditional 768/1,024-token checks, and Gemini 768/1,536/3,072 dimension comparison. The comparison command exports chunk diagnostics and this unscored schedule to `results/phase3/comparison.json`. It separates superseded strategy versions from active candidates. The fragmentation inspector exports before/after size distributions and actual source snippets to `results/phase3/fragmentation_audit.json`. Neither command computes retrieval scores, reads held-out questions, or chooses a winner. Dense screening requires Phase 4; hybrid/reranker confirmation follows the later retrieval phases. All such runs must search the full corpus with frozen development questions and comparable context budgets.

Baseline generation was validated against all 28,477 ready documents (four content duplicates excluded):

| Strategy | Chunks | Mean content tokens | Truncations |
|---|---:|---:|---:|
| Fixed | 98,241 | 337.74 | 0 |
| Recursive | 108,312 | 312.31 | 0 |
| Structure-aware v2 | 103,462 | 285.23 | 0 |

Structure v1 is superseded: it produced 530,553 chunks averaging 55.28 tokens, with 312,829 chunks under 16 tokens (58.96%). Its broad uppercase rule treated codes such as `IC62341` and `MQRC_OBJECT_NOT_FOUND.` as headings, and mandatory section barriers prevented short sections from being packed together. Version 2 uses conservative heading detection, packs complete adjacent short sections/paragraphs, and keeps headings with following body text. It produces 103,462 chunks, with a median of 327 tokens and only 347 chunks under 16 tokens (0.34%). All 28,477 sources remain included, and truncation is zero. No content was filtered to obtain this reduction.

All 15 unit tests pass. A v2 BGE smoke check covered 100 sampled source documents (359 chunks) with no token-coverage gaps or truncation. The old v1 artifact remains available for audit, while `comparison.json` lists it under `superseded_chunk_runs`; Phase 4 should use `chunk_runs`, which includes structure v2. The fragmentation audit retains size distributions and actual source examples for both versions. These are construction diagnostics; retrieval scores and final model/chunking selection remain pending.


### Chunk storage and configuration IDs

The current workspace stores chunks under `D:/Codex_Projects/Avathon_Enterprise_RAG_project/data/processed/techqa/chunks/`. The repository-relative location is `data/processed/techqa/chunks/` (including `techqa`).

| Configuration ID | Strategy / version | Chunks | Use |
|---|---|---:|---|
| `d116966770a2af165b87` | Fixed v1 | 98,241 | Active baseline candidate |
| `e15fb672b91720078683` | Recursive v1 | 108,312 | Active candidate |
| `210a70b67f13de2fd363` | Structure-aware v2 | 103,462 | Active candidate |
| `a95f195a2d1c5b45538e` | Structure-aware v1 | 530,553 | Superseded; audit only |

Each 20-character hexadecimal directory name is the first 20 characters of the SHA-256 hash of the serialized full configuration and provenance. This includes strategy/version, target/overlap/input limits, corpus and input-manifest identity, tokenizer model/revision/file checksum/package version, and text/offset policies. Identical settings reuse the same artifact; changed settings get a different directory. The directory name is a configuration ID, not a document ID.

Each published directory contains:

- `chunks.jsonl`: actual chunk text in each record's `text` field, plus chunk/source IDs, source filename/path, title, source offsets, token counts, hashes, and strategy/corpus/configuration metadata. Each line is one JSON object. Structure v2 also includes all covered `section_labels`.
- `config.json`: complete generation settings and provenance; this explains the configuration ID.
- `report.json`: generation counts, token statistics, truncation count, elapsed build time, artifact checksum, and pending retrieval status.

These artifacts are versioned on disk and ignored by Git; fresh checkouts regenerate them with the commands above. They are currently text and metadata artifacts; embedding vectors and Qdrant/BM25 indexes have not been built. `results/phase3/comparison.json` has three active `chunk_runs` and one `superseded_chunk_runs` entry. Use the active entries for subsequent indexing.

### Generated baselines versus pending experiments

Only the **384/64 full-corpus configuration** has been generated for the three active strategies. The 384-token size is a target/upper bound, not an exact size for every chunk; document endings and preserved boundaries create shorter chunks. Structure v2 uses the overlap budget for oversized-block splits rather than repeating text between every packed section.

| Experiment | Configuration | Current state |
|---|---|---|
| Boundary baseline | Fixed v1, recursive v1, structure v2; 384-token target / 64-token overlap budget | Full-corpus chunk artifacts generated; retrieval unscored |
| Size comparison | 256, 384, 480 tokens; overlap held at 64 | 384 baseline generated; 256/480 full-corpus artifacts not generated |
| Overlap comparison | 32, 64, 96 tokens at the strongest development-selected size | 64 baseline generated; alternate-overlap full-corpus artifacts not generated |
| Longer context | 768 / 1,024-token targets with 64 overlap, for surviving compatible models | Planned; requires model screening and input-limit validation |
| Gemini dimensions | 768 / 1,536 / 3,072 dimensions on identical chunks and queries | Planned; no embeddings generated |

The builder's configurable parameters and small smoke tests do not mean that full-corpus sweeps have already run. Each future size/overlap configuration gets its own configuration ID and leaves the current artifacts intact. Select the final combination on development questions against the full corpus, then freeze it before held-out evaluation. Source preservation, zero truncation, and construction diagnostics do not establish retrieval quality; Phase 4 must measure source Recall@5, Precision@5, MRR, and actual chunk evidence coverage.

### Planned free embedding comparison

The free-only embedding shortlist is `BAAI/bge-base-en-v1.5` (baseline), `Qwen/Qwen3-Embedding-0.6B`, `jinaai/jina-embeddings-v2-base-en`, and `gemini-embedding-2` using its standard free tier if available and feasible for the account. BGE is not the final selected model. Gemini screening starts at 768 dimensions, with a bounded 1,536/3,072 comparison before elimination or finalization. Verify hosted availability, quotas, and data-use terms before submitting corpus text. No paid endpoints, credits, rented compute, or automatic paid fallback are authorized. See PLAN.md Phase 4 for candidate properties and selection controls.

## Chunking Experiment

Chunking is selected through a controlled development-set experiment because chunk boundaries determine what can be retrieved and what evidence reaches the generator.

| Strategy | Controlled configuration | Rationale |
|---|---|---|
| Fixed token window | 384 tokens, 64-token overlap | reproducible baseline with predictable size |
| Recursive separator-aware | target 384 tokens, 64-token overlap | preserves paragraph and sentence boundaries where possible |
| Structure-aware section packing | pack titles, sections, and paragraphs toward 384 tokens; split oversized blocks safely | preserves procedural and product/version context in technical notes |

Within each boundary comparison, strategies use the same tokenizer, corpus, embedding model, queries, `k`, and retrieved-token budget. BGE's baseline input limit is 512 tokens including special tokens; longer-context candidates use their own validated limits. Across models, shared text/boundaries isolate the embedding comparison and a fixed evaluation tokenizer controls the context budget.

The three strategies are screened with dense retrieval on development data. The top two advance to hybrid retrieval and reranking. The winner is selected using Precision@k, Recall@k, faithfulness/answer support, MRR, and answer correctness, then frozen before the held-out run. Latency, index size, duplicate-token ratio, and truncation are supporting diagnostics.

---

## Retrieval Experiments

### 1. Dense-only baseline

```text
Query -> Embedding -> Qdrant -> Top-k
```

### 2. Hybrid retrieval

```text
             Query
           /       \
       Dense       BM25
          \         /
             RRF
              |
            Top-k
```

### 3. Hybrid + reranker

```text
Dense + BM25
     |
    RRF
     |
 Candidate Top-N
     |
 Cross-Encoder Reranker
     |
 Final Top-k
```

The final results compare retrieval and grounding quality. Latency is reported as a supporting operational diagnostic.

---

## Evaluation

The primary evaluation uses a frozen held-out set containing at least 20 answerable QA pairs.

### Required challenge metrics

- Precision@k through RAGAS `IDBasedContextPrecision`, using the top-k unique source-document IDs
- Recall@k through RAGAS `IDBasedContextRecall`, using TechQA gold source-document IDs
- Faithfulness through RAGAS `Faithfulness`, measuring whether generated claims are supported by retrieved context

### Additional quality metrics

- MRR through a project-defined RAGAS `SingleTurnMetric` over source-document IDs
- Answer correctness through RAGAS `AnswerCorrectness`

RAGAS is pinned to 0.4.3. Scored generation evaluation uses the fixed `evaluation-judge` LiteLLM alias with fallback disabled and temperature zero. The initial answer-correctness configuration uses RAGAS factual correctness only (`weights=[1.0, 0.0]`); semantic-similarity weighting remains disabled until the project freezes one evaluation embedding model. Judge prompts, model, RAGAS version, and correctness weights must be recorded with each scored run, and a sample must be manually audited. Latency, token use, index size, and semantic-cache behavior are reported separately from the headline quality metrics.

### Unanswerable-question evaluation

The dataset's `is_impossible` field is used only as evaluation ground truth.

The running system must decide whether enough evidence exists without reading this label.

Reported separately from the held-out answerable metrics:

- correct abstention count/rate
- false-answer count/rate
- false-abstention count/rate

---

## Conversational Chatbot

The final application is exposed through Streamlit.

Planned features:

- multi-turn chat
- conversational query rewriting
- PostgreSQL-backed conversation list, reopen, and delete operations scoped to the current principal
- source citations
- retrieved-context inspection
- abstention when evidence is insufficient
- optional debug view for retrieval scores
- clear chat history

---

## LLM Gateway and Provider Fallback

All generative-model calls pass through LiteLLM Proxy. The interactive application uses a pinned stable Gemini Flash model as primary and a pinned compatible Groq-hosted model only for retryable failures such as rate limiting, timeouts, connection failures, or provider `5xx` responses.

Fallback is not used for invalid requests, authentication failures, guardrail blocks, or evidence-based abstention. Scored evaluation fixes one provider and disables fallback so results remain attributable and reproducible. Every request records the requested alias, actual provider/model, fallback reason, latency, and token usage.

Provider API keys remain in the gateway environment; the application receives a restricted gateway credential and has no direct provider path.

---

## Semantic Cache

Valkey stores disposable semantic-cache entries. It is separate from Qdrant knowledge vectors and PostgreSQL conversation records.

A cache entry is eligible only after authorization and retrieval/evidence checks. It is scoped by tenant/access scope, corpus version, prompt version, model configuration, guardrail-policy version, and retrieval configuration. Cached answers retain citation IDs, expire through TTL, and are returned only if their sources still exist and remain authorized.

Only validated grounded answers are cached. Provider errors, malformed output, policy blocks, and unchecked answers are excluded. Semantic caching is disabled during chunking, retrieval, faithfulness, and answer-correctness evaluation. A separate cache experiment reports hit rate, false-hit rate/cache precision, latency saved, and estimated calls or tokens saved.

---

## Docker

The project will be reproducible with Docker Compose.

Minimum services:

```text
qdrant
rag-app
postgres
llm-gateway
valkey
```

Target startup:

```bash
docker compose up --build
```

---

## Scope

### In scope

- text-only RAG
- LangChain
- LangGraph
- Qdrant
- NVIDIA `corpus.zip` document ingestion and versioned corpus manifests
- BM25
- RRF
- reranking
- evaluation
- hallucination / abstention handling
- Streamlit
- Docker
- observability
- LiteLLM-based Gemini-to-Groq runtime fallback
- PostgreSQL conversation persistence and lifecycle controls
- Valkey semantic caching with TTL and version/authorization scoping
- layered input, retrieval-access, citation, and grounded-output guardrails
- production design discussion

### Out of scope

- multimodal RAG
- OCR
- image embeddings
- vision-language models
- fine-tuning
- multi-agent systems
- Kubernetes

The intent is to keep the project focused on **retrieval quality, evaluation rigor, failure analysis, and production-minded design**.

---

## Guiding Principle

Every major design choice should answer:

1. What did we choose?
2. What alternatives did we consider?
3. Why did we choose it?
4. What trade-off did we accept?

The project optimizes for:

> **Simple architecture + measurable experiments + explicit trade-offs + honest error analysis + reproducibility**
