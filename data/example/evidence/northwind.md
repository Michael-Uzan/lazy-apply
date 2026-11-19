---
company: Northwind Trading
role: Senior Full-Stack Engineer
start: 2022-03
end: present
---

## Real-time trading dashboard
tags: [react, typescript, websockets, performance]

Rebuilt the firm's trading dashboard as a React + TypeScript application fed by
a WebSocket gateway. Introduced virtualized rendering and a normalized client
store, which brought time-to-interactive down from 4.2s to 900ms on the
desktop client and let the team drop a long-standing pagination workaround.
Owned the migration end to end across six weeks, including the rollout plan and
the rollback switch.

## GraphQL federation across four services
tags: [graphql, nodejs, microservices, api-design]

Designed and shipped a federated GraphQL layer unifying four Node.js services
that had previously been consumed through hand-written REST adapters. Defined
the schema ownership model and the review process that went with it. Removed
roughly 3,000 lines of adapter code and cut median page-load request count from
nine round trips to two.

## Event-driven order pipeline
tags: [nodejs, queues, postgresql, reliability]

Replaced a nightly batch reconciliation job with an event-driven pipeline on a
durable queue, with idempotent consumers keyed on order id. Reconciliation
latency went from up to 18 hours to under a minute, and the duplicate-write
incidents that had triggered three on-call pages that quarter stopped entirely.
