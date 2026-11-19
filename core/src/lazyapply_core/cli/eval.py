"""`lazyapply eval` — measure retrieval and the model's judgment.

Three numbers, against the hand-labelled set in ``core/evals/golden.yaml``:

- recall@k and MRR: did retrieval find the right evidence, and how far down?
- agreement: did the model's status match a person's, and in which direction?

Cost: one batched query-embedding request per posting, plus one Haiku call per
posting unless ``--retrieval-only``. Both are fractions of a cent, but the
flag exists because repeated sweeps add up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from lazyapply_core.cli._eval_print import print_margins, print_matching, print_retrieval
from lazyapply_core.cli._shared import (
    CacheDir,
    DataDir,
    Fake,
    InMemory,
    abort,
    cache_root,
    role_cache,
    settings_for,
)
from lazyapply_core.config import Settings
from lazyapply_core.domain import EvidencePack, Profile, Role
from lazyapply_core.embed import build_embedder
from lazyapply_core.embed.base import EmbeddingError
from lazyapply_core.evals.golden import DEFAULT_GOLDEN_PATH, GoldenCase, GoldenError, load_golden
from lazyapply_core.evals.runner import (
    LabelDriftError,
    MatchingScore,
    RetrievalScore,
    ThresholdMargin,
    score_matching,
    score_retrieval,
    score_thresholds,
)
from lazyapply_core.evaluate import CitationError, EvaluationError, build_evaluator, evaluate_fit
from lazyapply_core.ingest.parser import ParseError
from lazyapply_core.ingest.pipeline import ingest as run_ingest
from lazyapply_core.ingest.profile import load_profile, profile_version
from lazyapply_core.retrieve import ExtractionError, build_extractor, extract_role
from lazyapply_core.retrieve.retriever import build_pack
from lazyapply_core.store import build_store

_MAX_SWEEP_K = 5
"""Widest cutoff a sweep reports. Beyond this the prompt carries more
irrelevant evidence than a judgment benefits from."""

Golden = Annotated[
    Path | None,
    typer.Option("--golden", help="Golden set. Defaults to core/evals/golden.yaml."),
]
TopK = Annotated[int, typer.Option("--top-k", help="Candidates retrieved per requirement.")]
RetrievalOnly = Annotated[
    bool, typer.Option("--retrieval-only", help="Skip the model call; measure retrieval only.")
]
Sweep = Annotated[
    bool,
    typer.Option("--sweep", help=f"Score every cutoff from 1 to {_MAX_SWEEP_K} to choose --top-k."),
]


def eval_(
    golden: Golden = None,
    top_k: TopK = 3,
    sweep: Sweep = False,
    retrieval_only: RetrievalOnly = False,
    data_dir: DataDir = None,
    cache_dir: CacheDir = None,
    fake: Fake = False,
    in_memory: InMemory = False,
) -> None:
    """Score retrieval and judgment against the hand-labelled golden set."""
    settings = settings_for(data_dir)

    try:
        golden_set = load_golden(golden or DEFAULT_GOLDEN_PATH)
        # One pack, built at the widest cutoff we will report. A retrieved
        # vector does not depend on k, so sweeping costs no extra embedding
        # requests -- which matters on a 3 RPM free tier.
        cutoffs = tuple(range(1, _MAX_SWEEP_K + 1)) if sweep else (top_k,)
        retrieval, matching, margins = _measure(
            golden_set.cases,
            settings,
            cache_dir=cache_dir,
            cutoffs=cutoffs,
            retrieval_only=retrieval_only,
            fake=fake,
            in_memory=in_memory,
        )
    except LabelDriftError as exc:
        raise abort(f"{exc}{_drift_hint(fake)}") from exc
    except (
        GoldenError,
        ParseError,
        ExtractionError,
        EmbeddingError,
        EvaluationError,
        CitationError,
        ValueError,
    ) as exc:
        raise abort(str(exc)) from exc

    print_retrieval(retrieval)
    if matching:
        print_matching(matching, settings.judgment_model)
    print_margins(margins, settings)


def _drift_hint(fake: bool) -> str:
    """With ``--fake`` and a cold cache the ids come from the canned extractor
    rather than from the posting, so the drift is the flag's doing, not the
    labels'. Say so, with the fix."""
    if not fake:
        return ""
    return (
        "\n\n--fake uses a canned extractor whose requirement ids differ from the "
        "labelled\nones. Seed the real extractions instead:\n"
        "  cp -r data/example/cache/roles <your --cache-dir>/"
    )


def _measure(
    cases: tuple[GoldenCase, ...],
    settings: Settings,
    *,
    cache_dir: Path | None,
    cutoffs: tuple[int, ...],
    retrieval_only: bool,
    fake: bool,
    in_memory: bool,
) -> tuple[list[RetrievalScore], list[MatchingScore], list[ThresholdMargin]]:
    cache = cache_root(settings, cache_dir)
    embedder = build_embedder(settings, fake=fake, cache_dir=cache)
    store = build_store(settings, in_memory=in_memory)
    if in_memory and store.count() == 0:
        run_ingest(settings, embedder=embedder, store=store)

    # Loaded once: the hard facts do not change between postings, and a
    # malformed profile.json should fail before any model call is paid for.
    profile = None if retrieval_only else load_profile(settings.profile_path)

    retrieval: list[RetrievalScore] = []
    matching: list[MatchingScore] = []
    margins: list[ThresholdMargin] = []

    for case in cases:
        role = extract_role(
            (settings.roles_dir / case.role).read_text(),
            extractor=build_extractor(settings, fake=fake),
            cache_dir=role_cache(settings, cache_dir),
        )
        pack = build_pack(role, embedder=embedder, store=store, top_k=max(cutoffs))
        retrieval += [score_retrieval(case, pack, k=k) for k in cutoffs]
        margins.append(score_thresholds(case, role, settings.thresholds))

        if profile is not None:
            matching.append(_judge(case, role, pack, profile, settings, fake=fake))

    return retrieval, matching, margins


def _judge(
    case: GoldenCase,
    role: Role,
    pack: EvidencePack,
    profile: Profile,
    settings: Settings,
    *,
    fake: bool,
) -> MatchingScore:
    report = evaluate_fit(
        role,
        pack,
        profile,
        evaluator=build_evaluator(settings, fake=fake),
        thresholds=settings.thresholds,
        profile_version=profile_version(profile),
    )
    return score_matching(case, report)
