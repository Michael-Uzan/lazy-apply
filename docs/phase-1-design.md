# Lazy Apply — Phase 1: Grounded RAG Fit Engine

## Context

Michael is building **Lazy Apply** as a portfolio project to learn RAG + vector
databases and to present as an AI Engineer / Full-Stack Engineer. The app lets a
user register their career evidence once, then check themselves against one or
many job roles: the system decides whether they genuinely fit, explains why with
citations, and (later phases) generates a tailored one-page CV.

Phases, per the user's own plan:

1. **AI + Vector DB only** — this plan. No backend, no frontend, no CV agent.
2. CV-generation agent (LangGraph — the user wants to learn it).
3. Backend API.
4. Frontend / UI.

Source documents: PRD in Notion (read; §4 "Technical Architecture & Data Model"
is empty — this plan fills it), system design in Whimsical (canvas would not
render; not yet incorporated).

Stated constraints that shape everything below:

- **Go slowly, understand every line.** Build is sliced into ten runnable steps.
- **Professional code, shareable on a resume.** Typed, tested where it counts,
  measured, documented.
- **Free / near-free.** All infra on free tiers; total LLM spend for Phase 1 is
  expected under ~$5.

Phase 1 is complete when `lazyapply fit <role.txt>` returns a typed, grounded
FitReport with a GENERATE/BORDERLINE/SKIP verdict, and `lazyapply eval` prints a
measured recall@3 for retrieval.

## Decisions already made (with the user, in brainstorming)

| Decision | Choice | Why |
|---|---|---|
| Language | Python 3.12 | Standard for AI roles; user wants the learning |
| Abstraction | Hand-written RAG core, no framework | "I need to understand everything"; LangGraph arrives in Phase 2 |
| Vector DB | Qdrant | Free (Docker + 1GB cloud tier); teaches HNSW, payload filters, aliases |
| Embeddings | Voyage `voyage-3-lite`, 512-dim | ~$0.02/1M tokens; tiny dependency, deploys on any free tier |
| Extraction model | `claude-opus-5` | Highest-leverage call; runs once per role, cached forever |
| Judgment model | `claude-haiku-4-5` | Repeated, well-structured task over pre-retrieved evidence |
| Chunking | Hand-authored semantic blocks | One complete claim per chunk; best retrieval quality |
| Rigor in scope | Golden eval set + recall@k, telemetry/cost log, citation verifier, unit tests on pure logic | No CI, no integration tests, no importers in Phase 1 |
| Repo | One public repo, plain directories, uv workspace — **not** Nx | Nx's Python support is a community plugin; its task-graph value needs many projects; a reviewer must be able to `cd core && uv sync && pytest` without learning our tooling |
| Code location | `core/` from day one | `api/` and `web/` arrive in Phases 3–4 without an import-rewriting migration |
| Real career data | Gitignored; `data/example/` committed | Impact stories carry client names and internal metrics; examples also let anyone clone and run it |

## Architecture

### Two load-bearing ideas

**1. Hard facts and soft evidence are stored and used differently.**

| | Hard facts | Soft evidence |
|---|---|---|
| What | Name, links, education, employment dates, titles | Impact stories, projects, what was actually done |
| Where | `profile.json`, hand-maintained | Chunked + embedded in Qdrant |
| Used | Injected verbatim into every prompt | Retrieved per requirement |

Dates, degrees and titles never pass through retrieval or model paraphrase, so
the most damaging class of CV hallucination is designed out rather than
prompted against.

**2. Retrieval decides candidacy; the model decides sufficiency; code owns the
verdict.**

```
role.txt
  └─ RequirementExtractor (Opus 5, cached by role hash)
       └─ [Requirement(text, MUST_HAVE|NICE_TO_HAVE, category)]
            └─ per requirement: Qdrant top-3 → scored Evidence
                 └─ EvidencePack
                      └─ Evaluator (Haiku 4.5, structured output)
                           └─ [RequirementMatch(STRONG|PARTIAL|NONE, evidence_ids)]
                                └─ citation verifier → verdict() → FitReport
```

The model never emits the verdict. A pure function aggregates:

```python
must_have_coverage = (n_STRONG + 0.5 * n_PARTIAL) / n_must_haves

SKIP        if must_have_coverage < 0.60
GENERATE    if must_have_coverage >= 0.80 and score >= 70
BORDERLINE  otherwise
```

Thresholds live in `config.py`. Consequence: reproducible, unit-testable with
zero API calls, tunable against the golden set, explainable to a user. This is
also the Phase 2 contract — the CV agent runs only on `GENERATE`; `SKIP` returns
the gap list instead.

### Module layout

All Phase 1 code lives under `core/src/lazyapply_core/`:

```
lazyapply_core/
  config.py          # settings, thresholds, model IDs (pydantic-settings)
  domain/            # Profile, Role, Requirement, Evidence, FitReport
                     #   pure types, zero I/O, no imports from sibling packages
  ingest/
    parser.py        # evidence/*.md → blocks
    chunker.py       # blocks → Evidence (+ readable IDs, content hashes)
    pipeline.py      # load → compose → hash → cache → embed misses → upsert
  embed/
    base.py          # Embedder protocol
    voyage.py        # VoyageEmbedder
    cache.py         # content-hash cache (on-disk)
  store/
    base.py          # VectorStore protocol
    qdrant.py        # QdrantStore (collection, alias, payload indexes)
  retrieve/
    requirements.py  # role → Requirement[] (Opus 5, cached by role hash)
    retriever.py     # per-requirement search → EvidencePack
  evaluate/
    prompts.py
    evaluator.py     # EvidencePack + Profile → RequirementMatch[]
    verifier.py      # every cited evidence_id must exist in the pack sent
    verdict.py       # pure aggregation → FitReport
  telemetry.py       # JSONL: model, tokens in/out, latency, computed cost
  cli.py             # typer: evidence / embed / ingest / search / profile /
                     #        requirements / pack / fit / eval / cost
```

`domain/` imports nothing from siblings and touches no network. `embed/` and
`store/` are protocols with one implementation each, so the pipeline is testable
with fakes (no API key, no running Qdrant) and a later swap to a local embedder
or pgvector is contained.

### Repository layout

One public GitHub repo, plain directories, no monorepo tool. Phases 3–4 slot in
as siblings without touching Phase 1 code.

```
lazy-apply/
├── README.md               the product: design, decisions, measured numbers
├── pyproject.toml          uv workspace root
├── docker-compose.yml      qdrant (later: api)
├── .env.example            VOYAGE_API_KEY, ANTHROPIC_API_KEY
├── .gitignore              data/ (except data/example/), .env, caches
├── core/                   Phase 1–2: RAG engine, later the CV agent
│   ├── pyproject.toml      ruff + mypy config, deps
│   ├── src/lazyapply_core/
│   ├── tests/              pure-logic unit tests with fakes
│   └── evals/              golden.yaml, run_eval.py
├── data/                   GITIGNORED — real evidence, profile, roles
│   └── example/            COMMITTED — fictional profile + evidence + roles
├── docs/                   PRD, system design, this plan
├── api/                    Phase 3 — does not exist yet
└── web/                    Phase 4 — does not exist yet
```

`api/` and `web/` are **not** scaffolded in Phase 1; they appear when their
phase does.

Python cross-dependencies use uv workspaces — when `api` arrives it declares
`core` as a dependency and uv resolves it from source, no path hacks:

```toml
# root pyproject.toml
[tool.uv.workspace]
members = ["core"]          # Phase 3 adds "api"
```

Deployment targets, later: `web` → Vercel (static, free), `api` → Fly.io or
Render (Docker, free), Qdrant → Qdrant Cloud (free 1GB tier). Three targets, one
repo, each reading its own subdirectory.

**Data privacy:** `data/` holds real career evidence and is gitignored.
`data/example/` holds a fictional profile, evidence files and sample roles, and
is committed — so the repo is clonable and runnable by anyone, and real client
names and internal metrics never reach a public repo. All Phase 1 commands take
a data directory, defaulting to `data/` and pointed at `data/example/` in the
README quickstart.

## Data model

**Evidence source format** — `data/evidence/*.md`, one file per context:

```markdown
---
company: Leverate
role: Full-Stack & Agentic AI Engineer
start: 2023-05
end: present
---

## Multi-agent LLM orchestration
tags: [python, llm, agents, claude, orchestration, evals]

Built the production multi-agent system that ... Cut X by Y%.
```

One `##` block = one Evidence = one chunk. Never split mid-story.

**Evidence IDs are readable, derived from file + title slug:**

```
ev_leverate__multi-agent-llm-orchestration
```

Readable IDs make FitReport citations legible and make `core/evals/golden.yaml`
hand-writable. A separate `content_hash` (sha1 of body) drives the embedding
cache — so rewording a block re-embeds it while keeping its ID and every
citation intact. ID stability and cache correctness are different jobs with
different fields. Collisions are caught at ingest.

**Embedded text is composed, not raw** — company and tags measurably help
matching on terse requirement queries:

```
Leverate — Multi-agent LLM orchestration
tags: python, llm, agents, claude, orchestration, evals
<body>
```

Raw body stored separately in the payload for display.

**Core types** (pydantic v2, in `domain/`):

```python
Requirement(id, text, kind: MUST_HAVE|NICE_TO_HAVE, category)
Evidence(evidence_id, text, composed_text, content_hash, company, role,
         start, end, tags, source_file)
RequirementMatch(requirement_id, status: STRONG|PARTIAL|NONE,
                 evidence_ids, best_retrieval_score, rationale)
FitReport(role_id, profile_version, created_at, matches,
          must_have_coverage, score, verdict, gaps)
```

`gaps` is derived from matches, never generated.

## Qdrant configuration

```
collection:      evidence_v1,  alias "evidence" → evidence_v1
vector:          512-dim, Cosine  (voyage-3-lite)
payload indexes: tags, company, source
```

The alias enables zero-downtime reindex later (build `evidence_v2` alongside,
repoint atomically) for three extra lines now.

Retrieval starts with `top_k=3` and **no score threshold** — scores are recorded
against the golden set first, then the threshold is chosen from that data in
slice 9.

## Build order

Ten slices, each ending in something runnable. Slices 1–5 cost $0; the first API
call is in slice 3 and costs fractions of a cent.

| # | Slice | Runnable result |
|---|---|---|
| 1 | GitHub repo, workspace skeleton, `domain/` types, `config.py`, docker-compose Qdrant, `data/example/` | `pytest` green, Qdrant UI on :6333 |
| 2 | `ingest/parser.py` + `chunker.py` + ID generation (pure) | `lazyapply evidence list` |
| 3 | `embed/` protocol + fake + Voyage + content-hash cache | `lazyapply embed --dry-run` shows hits/misses |
| 4 | `store/` protocol + QdrantStore + alias + ingest pipeline | `lazyapply ingest`, `lazyapply search "react"` |
| 5 | `profile.json` loader + validation | `lazyapply profile show` |
| 6 | `retrieve/requirements.py` (Opus 5, structured output, cached) | `lazyapply requirements role.txt` |
| 7 | `retrieve/retriever.py` → EvidencePack | `lazyapply pack role.txt` |
| 8 | `evaluate/` evaluator + verifier + verdict | `lazyapply fit role.txt` |
| 9 | `core/evals/golden.yaml` + `run_eval.py`, tune threshold | `lazyapply eval` |
| 10 | `telemetry.py` JSONL + cost report + README | `lazyapply cost` |

Pure logic gets unit tests as it is written (TDD where it fits): chunker, ID
generation, content hashing, `verdict.py`, `verifier.py`. Fakes for `Embedder`
and `VectorStore` keep the suite offline and sub-second.

## Verification

**Per slice** — the runnable command in the table above, plus `pytest`, `ruff
check`, `mypy src/lazyapply_core/domain` (run from `core/`).

**End to end, after slice 8:**

1. `docker compose up -d` — Qdrant healthy on :6333.
2. Author 2–3 `data/evidence/*.md` files and `data/profile.json`.
3. `lazyapply ingest` — reports chunks parsed, embeddings computed vs cached,
   points upserted. Re-run: zero new embeddings (idempotency check).
4. `lazyapply search "vector database experience"` — returns scored evidence,
   manually sane.
5. `lazyapply fit roles/strong-match.txt` — verdict `GENERATE`, matches cite
   real evidence IDs.
6. `lazyapply fit roles/bad-match.txt` — verdict `SKIP`, gap list names the
   genuinely missing must-haves. This is the gate working.
7. Corrupt a cited ID in a stubbed model response → citation verifier raises.

**Measured, after slices 9–10:**

- `lazyapply eval` prints recall@3 and MRR over `core/evals/golden.yaml`.
- `lazyapply cost` prints total spend and per-command breakdown from the
  telemetry JSONL.
- Both numbers go in the README. They are the Phase 1 deliverable that reads as
  AI engineering rather than API calling.

## Explicitly out of scope for Phase 1

PDF / LinkedIn / GitHub importers (they are drafting tools that need a review
UI); CV generation; hybrid dense+sparse search and reranking (earn these in a
second pass, against baseline numbers); any backend, frontend, auth, or
multi-user concern; CI; scaffolding `api/` or `web/` before their phase.
