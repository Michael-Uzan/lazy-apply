---
company: Orchid Labs
role: Software Engineer
start: 2018-09
end: 2022-02
---

## Python ETL for clinical data
tags: [python, etl, postgresql, data-engineering]

Built the ingestion pipeline that normalized clinical trial exports from eleven
partner labs into a single PostgreSQL schema. Wrote the validation layer that
rejected malformed submissions at the boundary with actionable errors, which
cut the support load from partner onboarding by about half.

## Internal search over research documents
tags: [python, search, elasticsearch, relevance]

Implemented full-text search across roughly 40,000 internal research documents.
Tuned the analyzer chain and relevance scoring against a hand-labelled set of
60 queries, taking precision@5 from 0.41 to 0.78. Built the evaluation harness
that made those numbers repeatable rather than anecdotal.

## Mentoring and code review culture
tags: [mentoring, code-review, process]

Onboarded four junior engineers over two years and introduced the team's first
written code review guidelines. Review turnaround dropped from two days to
under four hours, measured over the following quarter.
