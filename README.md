# Lazy Apply

A grounded RAG engine that decides whether a candidate genuinely fits a job
role, explains why with citations to real evidence, and refuses to generate a
CV when they don't.

> **Status: Phase 1 in progress.** Ingestion, retrieval and the fit engine.
> No backend, no frontend, no CV generation yet.

## Why this exists

Most AI resume tools generate confident prose and invent the details. This one
is built the other way around:

- **Hard facts are never generated.** Dates, degrees, titles and links live in
  `profile.json` and are injected verbatim. They never pass through retrieval
  or a model's paraphrase, so they cannot be hallucinated.
- **Every claim is cited.** The fit report cites evidence IDs, and a verifier
  hard-fails any citation to evidence that was not actually retrieved.
- **The decision is code, not a prompt.** The model judges each requirement
  STRONG / PARTIAL / NONE. A pure function computes coverage and the verdict.
  That makes the gate reproducible, testable without an API key, tunable
  against a labelled set, and explainable to the user.

## How it works

```
role.txt
  └─ extract requirements        (claude-opus-5, cached by role hash)
       └─ per requirement: retrieve top-3 evidence   (Qdrant + Voyage embeddings)
            └─ judge each requirement                (claude-haiku-4-5)
                 └─ verify citations → compute verdict
                      └─ FitReport: GENERATE | BORDERLINE | SKIP
```

Retrieval decides *candidacy* — what evidence exists at all. The model decides
*sufficiency* — whether that evidence satisfies the requirement. Code owns the
verdict.

## Stack

| Concern | Choice | Why |
|---|---|---|
| Vector DB | Qdrant | Free locally and hosted; HNSW tuning, payload filters, collection aliases |
| Embeddings | Voyage `voyage-3-lite` (512-dim) | ~$0.02/1M tokens; no heavyweight local dependency |
| Extraction | `claude-opus-5` | Highest-leverage call; runs once per role and is cached |
| Judgment | `claude-haiku-4-5` | Repeated, well-structured task over pre-retrieved evidence |
| Core | Hand-written, no RAG framework | The mechanics are the point |

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and a container runtime.

```bash
cp .env.example .env     # add ANTHROPIC_API_KEY and VOYAGE_API_KEY
docker compose up -d     # Qdrant on http://localhost:6333/dashboard
uv sync

cd core
uv run pytest            # offline: no API key or Qdrant needed
```

Run against the committed sample data (fictional profile, two job postings):

```bash
uv run lazyapply ingest --data-dir ../data/example
uv run lazyapply fit    --data-dir ../data/example ../data/example/roles/strong-match.txt
uv run lazyapply fit    --data-dir ../data/example ../data/example/roles/bad-match.txt
```

## Repository layout

```
core/     the RAG engine (Phase 1-2)
data/     your real evidence — gitignored; data/example/ is the committed sample
docs/     PRD and design
api/      Phase 3 — not yet
web/      Phase 4 — not yet
```

Your own career evidence lives in `data/` and never reaches this repository.
`data/example/` exists so anyone can clone this and run it.

## Evidence format

One `##` block is one retrievable, citable unit. Blocks are hand-authored and
never split mid-story — that is what makes retrieval work well here.

```markdown
---
company: Northwind Trading
role: Senior Full-Stack Engineer
start: 2022-03
end: present
---

## Real-time trading dashboard
tags: [react, typescript, websockets, performance]

Rebuilt the dashboard as a React + TypeScript application fed by a WebSocket
gateway. Time-to-interactive went from 4.2s to 900ms.
```

Becomes `ev_northwind__real-time-trading-dashboard` — readable, so citations in
a fit report can be checked by eye.

## Does retrieval actually work?

A smoke test, not an evaluation — six blocks, five queries. Six documents
cannot establish retrieval quality; that is measured properly below.

The sample candidate, **Dana Levi**, is fictional: a full-stack engineer with
React, TypeScript, GraphQL, Node.js and Python experience, and no hardware
background whatsoever. Her six evidence blocks live in `data/example/`.

Run against `data/example/` with `voyage-3-lite` (512-dim, cosine):

| Query | Top match | Score | Runner-up |
|---|---|---|---|
| "GraphQL API design across microservices" | `graphql-federation-across-four-services` | **0.662** | 0.366 |
| "experience tuning search relevance" | `internal-search-over-research-documents` | **0.584** | 0.217 |
| "leading and mentoring engineers" | `mentoring-and-code-review-culture` | **0.501** | 0.316 |
| "front-end performance optimisation" | `real-time-trading-dashboard` | **0.485** | 0.411 |
| "embedded C firmware for medical devices" | _(nothing relevant)_ | 0.229 | 0.164 |

```bash
uv run lazyapply search "GraphQL API design across microservices" --data-dir ../data/example
```

**Why these scores are good.** Absolute cosine values are model-specific, so
what matters is separation. Work Dana did: 0.49-0.66, correct block first.
Work she never did: 0.229. A 2-3x gap — and that gap *is* gap detection. No
model opinion needed to say evidence is missing.

Two details: "experience tuning search relevance" shares no words with the
block it found, so keyword search returns nothing there. And the narrow
0.485/0.411 row is retrieval being correctly uncertain — several blocks do
discuss performance. Hence thresholds from labelled data, not guesses.

## Measured results

Populated in slice 9-10, from `lazyapply eval` and `lazyapply cost`. Unlike
the smoke test above, these run against a hand-labelled golden set.

| Metric | Value |
|---|---|
| recall@3 | _pending_ |
| MRR | _pending_ |
| Cost per fit check | _pending_ |

## License

MIT
