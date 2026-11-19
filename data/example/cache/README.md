# Committed extraction cache

The two `roles/*.json` files are real `claude-opus-5` extractions of the
postings in `../roles/`, keyed by a content hash of the posting text — the
same cache the app writes at runtime.

They are committed, unlike the rest of `.cache/`, for one reason:
`core/evals/golden.yaml` labels requirements **by id**, and those ids come
from the extraction. Committing it means a reviewer can clone the repo and
run `lazyapply eval` against the real requirement ids without an Anthropic
key and without paying for an extraction that has already been done.

Seed a working cache from them:

```bash
mkdir -p .cache && cp -r data/example/cache/roles .cache/
```

Delete a file to force a re-extraction. If a posting is reworded its hash
changes, the cache misses, and the labels in `golden.yaml` may need updating
— `lazyapply eval` reports that as label drift rather than quietly scoring a
smaller set.
