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

Phase 1 provides the runtime scaffold. Phase 2 adds a reproducible TechQA download, corpus inspection, separate QA manifest, and frozen development/test question IDs. The app still does not answer questions; chunking, embeddings, indexing, generation, caching, and Langfuse instrumentation follow in later phases.

To inspect the scaffold locally, install Docker Engine with Compose, copy `.env.example` to `.env`, and replace the `CHANGE_ME` values with private credentials. The Gemini key is required for actual gateway requests. Then run:

```bash
docker compose config --quiet
docker compose up --build
```

Open `http://localhost:8501` for the placeholder app. Qdrant, PostgreSQL, LiteLLM, and Valkey are wired in Compose, but the Phase 1 app does not call them. Langfuse Cloud settings are accepted through environment variables; tracing begins when the application workflow is implemented. No credentials should be committed.

The pinned Python version is 3.11. `requirements.txt` now pins Streamlit and the RAGAS 0.4.3 evaluation stack, including the compatible pre-1.0 LangChain and Pydantic versions required by that RAGAS release. Later phases will pin their added retrieval and application packages. For a local run without Docker, install that manifest and run `python -m streamlit run app/streamlit_app.py`.

Prepare the dataset from the pinned NVIDIA revision with either the local Python environment or the application image:

```bash
python scripts/prepare_dataset.py
# or: docker compose run --rm rag-app python scripts/prepare_dataset.py
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

Run `python scripts/show_qa_examples.py` after preparation to preview records `TRAIN_Q000`, `TRAIN_Q001`, and `TRAIN_Q002` with their answerability labels and a short gold-context excerpt. These previews are for human inspection only. If raw files already exist, the preparation script reuses them and records their SHA-256 hashes; verify their provenance before claiming a pinned dataset run.

The later ingestion command will be `docker compose run --rm rag-app python scripts/ingest.py`; chunking and Qdrant/BM25 indexing are not implemented in Phase 2. Initial TechQA preparation remains a separate batch command without a task broker.

---

## Why This Dataset?

The project uses the NVIDIA TechQA-RAG-Eval dataset because it provides the core ingredients needed for rigorous RAG evaluation:

- a separate `corpus.zip` archive containing raw technical-note files
- technical support questions
- ground-truth answers
- supporting text contexts
- `is_impossible` labels for unanswerable questions

This allows us to evaluate both retrieval quality and answer grounding / faithfulness.

The current repository is approximately 49.8 MB: about 45.8 MB for `corpus.zip` and 4.02 MB for `train.json`. The dataset viewer currently exposes one `train` split with 910 rows, while its card says 908 QA pairs; the ingestion report records the exact counts from the pinned revision instead of assuming either value.

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

## Chunking Experiment

Chunking is selected through a controlled development-set experiment because chunk boundaries determine what can be retrieved and what evidence reaches the generator.

| Strategy | Controlled configuration | Rationale |
|---|---|---|
| Fixed token window | 384 tokens, 64-token overlap | reproducible baseline with predictable size |
| Recursive separator-aware | target 384 tokens, 64-token overlap | preserves paragraph and sentence boundaries where possible |
| Structure-aware section packing | pack titles, sections, and paragraphs toward 384 tokens; split oversized blocks safely | preserves procedural and product/version context in technical notes |

All strategies use the same embedding tokenizer, corpus, embedding model, queries, `k`, and retrieved-token budget. The embedding model's 512-token input limit is enforced and truncation is reported.

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
