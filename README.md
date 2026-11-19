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

## Profile: the hard facts

`data/profile.json` holds what must never be generated — dates, degrees,
titles, links. It is injected verbatim into every prompt and never passes
through retrieval, so it cannot be hallucinated. Validated strictly on load.

```json
{
  "name": "Dana Levi",
  "headline": "Full-Stack & AI Engineer",
  "location": "Tel Aviv, Israel",
  "email": "dana.levi@example.com",
  "links":     [{ "label": "GitHub", "url": "https://github.com/example-dana" }],
  "education": [{ "degree": "B.Sc. Computer Science", "institution": "Technion",
                  "graduated": "2018-06" }],
  "positions": [{ "company": "Northwind Trading", "title": "Senior Full-Stack Engineer",
                  "start": "2022-03", "end": "present" }],
  "stack": ["TypeScript", "React", "Node.js", "Python"]
}
```

`lazyapply profile show` prints it with a `version` — a hash of the facts that
invalidates cached fit reports when a fact changes, but not when the file is
merely reformatted.

## What lands in Qdrant

One evidence block becomes one point. Collection `evidence_v1`, 512-dim,
cosine, reached through the alias `evidence`.

```json
{
  "id": "23c802b4-1f8f-5d02-b37f-10b74e5ba500",
  "vector": [-0.017, 0.0026, 0.0607, 0.0104, "... 508 more"],
  "payload": {
    "evidence_id":   "ev_orchid__python-etl-for-clinical-data",
    "title":         "Python ETL for clinical data",
    "body":          "Built the ingestion pipeline that normalized ...",
    "composed_text": "Orchid Labs — Python ETL for clinical data\ntags: python, etl, ...",
    "content_hash":  "e595f2d120bb4f66",
    "company": "Orchid Labs", "role": "Software Engineer",
    "start": "2018-09", "end": "2022-02",
    "tags": ["python", "etl", "postgresql", "data-engineering"],
    "source_file": "orchid.md"
  }
}
```

Three choices worth noting:

- **The id is `uuid5(fixed_namespace, evidence_id)`.** Qdrant takes UUIDs or
  integers, not slugs. Deriving it deterministically means re-running ingest
  replaces the point instead of adding a duplicate.
- **The whole Evidence is the payload.** Search returns usable objects, not
  ids needing a second lookup against the markdown.
- **`body` and `composed_text` are both stored.** Only `composed_text` was
  embedded; `body` is what a human reads. Keeping both makes "why did this
  match?" answerable.

`tags`, `company` and `source_file` carry keyword indexes, so filtered search
(e.g. evidence from one employer) does not scan the collection.

## Reading a job posting

A posting is prose; the fit engine needs checkable claims. One `claude-opus-5`
call turns one into the other, cached forever by a hash of the posting text.

```
$ lazyapply requirements data/example/roles/strong-match.txt

Senior Full-Stack Engineer — Fintech Platform — Meridian Systems
role_id   207aace0a6157aff

must-have (8)
  react_production         [skill]       Built production web applications using React
  typescript               [skill]       Wrote production application code in TypeScript
  api_design               [skill]       Designed and evolved APIs over time, using GraphQL or REST
  nodejs_services          [skill]       Built and ran Node.js backend services in production
  postgres_data_modelling  [skill]       Worked with PostgreSQL and designed data models
  realtime_browser         [skill]       Delivered real-time data to the browser using WebSockets
  performance_work         [experience]  Improved application performance with measurable results
  architecture_ownership   [experience]  Made architectural decisions and owned a major part of a product

nice-to-have (3)
  fintech_domain       [domain]      Worked on fintech products or trading systems
  event_driven         [skill]       Worked with event-driven architectures and message queues
  mentoring_tech_lead  [experience]  Mentored other engineers or acted as a tech lead
```

Real output, not illustrative. Three things it got right: compound
requirements were split (the posting's "React and TypeScript" became two),
durations were stripped from every query, and `architecture_ownership` was
pulled from the posting's prose — "expected to make architectural calls, not
just implement tickets" — rather than only from the bullet list.

**Why two different models.** The pipeline makes two LLM calls, and they are
not the same kind of work.

| Step | Input | Output | Model | How often | Cost |
|---|---|---|---|---|---|
| **Extraction** (this step) | job posting | requirements | Opus 5 | once per posting, cached | ~$0.05 |
| **Matching** (slice 8) | requirement + retrieved evidence | STRONG / PARTIAL / NONE | Haiku 4.5 | every check | ~$0.01 |

Extraction reads prose and makes two calls nothing downstream can repair: the
must-have split (coverage counts must-haves only — reading "fintech a plus" as
required rejects a good candidate) and restating each requirement as a search
query ("5+ years with React" → "Built production web applications with React";
duration is checked against employment dates, not similarity).

Matching is bounded — evidence already found, schema fixed, three possible
labels — and repeats forever, so it gets the cheap model. Whether Haiku agrees
with a human is measured in slice 9, not assumed.

**Neither model produces the verdict.** Code turns those labels into coverage
and a GENERATE / BORDERLINE / SKIP decision.

Validation rejects an extraction with no requirements, duplicate ids, or no
must-have — coverage divides by that count.

## Retrieving evidence, per requirement

Each requirement gets its own search, not one search for the whole posting.
That is what makes a gap identifiable: a requirement with nothing near it has
no candidate evidence.

```
Role (11 requirements)
  └─ embed_queries([...])          1 batched request → 11 vectors
       └─ search(v, top_k=2)       11 searches → 22 hits
            └─ EvidencePack        11 items, each with its own hits
                 └─ unique_evidence()  6 blocks sent to the model
```

Batching matters: 11 separate requests on a 3 RPM free tier is four minutes.
Deduplication matters too — the same block is retrieved for several
requirements, so the prompt carries it once and refers to it by id.

### What the scores actually look like

Every requirement retrieves a best match with a score. Collecting those
across both example postings:

| Posting | Requirements | Weakest match | Average | Strongest match |
|---|---|---|---|---|
| Full-stack (Dana's field) | 11 | 0.353 | 0.465 | 0.561 |
| Embedded firmware (not her field) | 10 | 0.197 | 0.249 | 0.333 |

Read the first row as: of 11 requirements, even the worst-retrieving one found
evidence at 0.353. Read the second as: of 10 firmware requirements, even the
best one only reached 0.333.

So the two ranges do not overlap — **by 0.020**.

**Is 0.353 a good score?** On its own it means nothing. These are cosine
similarities, not percentages, and the scale is specific to `voyage-3-lite` —
another model would score the same pair differently and be equally right. Only
reading the pair tells you:

| Score | Requirement | Retrieved block | Correct? |
|---|---|---|---|
| 0.353 | "Made architectural decisions and owned a major part of a product" | GraphQL federation | yes — it describes designing a federated schema and owning the review process |
| 0.333 | "Followed a safety-critical lifecycle such as DO-178C" | Event-driven order pipeline | no — shares words like "pipeline", nothing else |

Twenty thousandths apart; one right, one nonsense. Which is why the numbers
below are labelled by hand before any threshold is chosen.

That thin margin is the most useful result here, and it argues for the
architecture rather than against it. A score threshold alone would be
fragile: 0.34 separates these two postings perfectly and would likely fail on
a third, since rewording one requirement moves a score by more than 0.02.

Retrieval always returns *something* — asked for firmware evidence it returns
the least-irrelevant block it has. So distance narrows the candidates; whether
they actually prove the requirement is the model's call. Distance was never
meant to render the verdict, and these numbers are why.

One requirement shows the limit of retrieval entirely:

```
ee_ce_degree  [must-have]  Earned a bachelor's degree in electrical engineering
    0.251  ev_orchid__mentoring-and-code-review-culture
```

No impact story says "I have a degree". Education and duration requirements
are facts to check against `profile.json`, not similarity questions — handled
by routing on `category`, not by retrieval.

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
