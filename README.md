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
uv run lazyapply eval   --data-dir ../data/example --sweep
```

## Repository layout

```
core/           the RAG engine (Phase 1-2)
  evals/        golden.yaml — the hand-labelled ground truth
data/           your real evidence — gitignored
  example/      the committed sample: profile, evidence, postings, extractions
docs/           PRD and design
api/            Phase 3 — not yet
web/            Phase 4 — not yet
```

Your own career evidence lives in `data/` and never reaches this repository.
`data/example/` exists so anyone can clone this and run it — including
`data/example/cache/`, which holds the two real Opus extractions so
`lazyapply eval` runs against the requirement ids the labels refer to without
an API key.

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

No impact story says "I have a degree". Searching for one returns the
least-irrelevant block in the collection, and the model then has to reason
over something unrelated.

So education requirements skip retrieval entirely and are judged against
`profile.json`. One property on the domain type does it:

```python
@property
def is_hard_fact(self) -> bool:
    return self.category is RequirementCategory.EDUCATION
```

The requirement still appears in the pack, is still judged, still counts
toward coverage — it just arrives with no hits, because none were asked for.
Its query is never embedded either, so a posting full of hard facts costs
less rather than more.

## The verdict: a model judges, code decides

The model answers one question per requirement. Code answers the only
question the user asked.

```
EvidencePack + profile.json
  └─ judge     claude-haiku-4-5, one request for the whole role
     │           → STRONG | PARTIAL | NONE, cited ids, one sentence each
     └─ verify  every cited id must be in the pack we sent
        │         → a bad citation is a hard failure, not a warning
        └─ decide    pure function: no model, no network
                     → coverage, score, verdict, gaps
```

The order is the point: an ungrounded judgment never reaches step 3, because
step 3's output is what the user acts on.

### Statuses become numbers

| Status | Points | Meaning |
|---|---|---|
| STRONG | 1.0 | The evidence directly shows it |
| PARTIAL | 0.5 | Adjacent, smaller, or older than asked |
| NONE | 0.0 | Nothing given supports it |

Half for PARTIAL because both alternatives are worse. Zero rejects people who
could defend the point in an interview; one sends them in unprepared.

```
coverage = points over must-haves ÷ number of must-haves in the role
score    = 100 × (0.8 × must-have ratio + 0.2 × nice-to-have ratio)
```

Two numbers because there are two questions. **Coverage: may I apply at all?**
Nice-to-haves are invisible to it — a "Kubernetes a plus" you lack must never
block an application. **Score: how good is this application?** Must-haves are
80% of it, so meeting every nice-to-have and no must-have scores 20 and
rescues nobody.

The denominator is every must-have *in the role*, not every one the model
answered. Otherwise a model that skipped four of eight requirements would
score 100%.

| coverage | score | Verdict | What happens in Phase 2 |
|---|---|---|---|
| < 0.60 | — | **SKIP** | No CV. The gap list is returned instead |
| ≥ 0.80 | ≥ 70 | **GENERATE** | The CV agent runs |
| anything else | | **BORDERLINE** | Shown to the user to decide |

The score check on GENERATE catches a coverage met entirely by PARTIALs —
technically covered, but thinly. All three numbers live in `config.py` and get
swept against the golden set in slice 9. A number inside a prompt cannot be
swept.

### It works on both example postings

```
$ lazyapply fit data/example/roles/strong-match.txt
verdict   GENERATE          coverage 100%    score 100/100
strong    api_design  [must-have]
          Designed and shipped a federated GraphQL layer unifying multiple services.
          → ev_northwind__graphql-federation-across-four-services

$ lazyapply fit data/example/roles/bad-match.txt
verdict   SKIP              coverage 0%      score 0/100
none      ee_ce_degree  [must-have]
          The candidate holds a B.Sc. in Computer Science, not a degree in
          electrical engineering or computer engineering.
```

Every STRONG and PARTIAL carries the ids it was based on, so any claim can be
looked up. `ee_ce_degree` cites nothing because nothing was retrieved for it —
that answer came from the facts block.

### Why the verdict is not the model's to make

Running the firmware posting twice, Haiku changed its mind on exactly one
requirement — whether clinical-trial ETL work counts as medical-device
software under IEC 62304. A genuinely arguable call.

| Run | `medical_regulatory` | coverage | Verdict |
|---|---|---|---|
| 1 | NONE | 0% | SKIP |
| 2 | PARTIAL | 6% | SKIP |

The judgment moved. The decision did not. A model flipping one call in ten
cannot flip the outcome, because the outcome is arithmetic over ten calls
rather than an eleventh opinion. Had the model been asked for the verdict
directly, there would be nothing to say about this.

## Measured results

`lazyapply eval` scores the system against `core/evals/golden.yaml` — 21
requirements across both example postings, labelled by hand. Two labels per
requirement: which evidence blocks *should* be retrieved, and what status a
person would give. The labels are the only thing here that is not derived.

```
$ lazyapply eval --sweep
```

### Retrieval

| k | recall@k | MRR |
|---|---|---|
| 1 | 0.818 | 1.000 |
| 2 | 0.955 | 1.000 |
| **3** | **0.955** | **1.000** |
| 4 | 1.000 | 1.000 |
| 5 | 1.000 | 1.000 |

Scored over the 11 requirements that have a correct answer. The firmware
posting's 10 are **excluded, not failed** — no block in the collection is
relevant to any of them, so recall is undefined. Counting them as misses
would blame retrieval for questions it was never asked.

**MRR is 1.000 at every cutoff**, which is the more interesting number: the
first relevant block is always ranked first. Recall climbs with k only
because four requirements have two relevant blocks each, and a narrow window
can only return one.

So **`top_k` stays at 3**. k=4 buys full recall, and the single block it
recovers is a second piece of evidence for a requirement already met STRONG
by the block at rank 1 — no decision changes, and every check pays the extra
tokens. That is a measured choice rather than the guess it was in slice 1.

The sweep costs one embedding request, not five: a retrieved vector does not
depend on k, so one pack is scored at every cutoff. On a 3 RPM free tier that
is the difference between working and waiting.

### Does the model agree with a person?

| Posting | Agreement | Verdict |
|---|---|---|
| Full-stack | 11 / 11 | generate — as labelled |
| Embedded firmware | 9 / 10 | skip — as labelled |

**20 of 21, and the single disagreement is in the safe direction.** The
posting asks for an electrical or computer engineering degree; Dana holds a
B.Sc. in Computer Science. I labelled that `partial` — neither degree exactly,
but adjacent enough that most firmware teams would accept it. Haiku said
`none`.

Direction matters more than the count. The model was **harsh**, not generous:
it claimed less than a person would. A harsh model costs opportunities; a
generous one costs credibility, and an inflated match is undetectable by
reading the report. Zero generous disagreements across 21 requirements is the
result worth having.

### The thresholds are still guesses

| Posting | coverage | verdict | distance to nearest boundary |
|---|---|---|---|
| Full-stack | 1.000 | generate | 0.200 |
| Embedded firmware | 0.062 | skip | 0.537 |

`lazyapply eval` reports this and says so itself: no posting lands within 0.1
of a boundary, so this set **cannot tell whether 0.60 / 0.80 / 70 are the
right cutoffs**. Both postings are easy cases. Tuning needs a genuinely
borderline posting — a role Dana half-fits — and the honest answer is that
two examples cannot do it.

Measuring that the measurement is under-powered beats publishing a swept
number from two data points.

| Still pending | Slice |
|---|---|
| Cost per fit check | 10 |

## License

MIT
