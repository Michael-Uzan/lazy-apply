"""Retrieval and agreement metrics.

Pure arithmetic over labelled data. Worth testing carefully because a metric
that is quietly wrong is worse than no metric: it reads as evidence.
"""

import pytest

from lazyapply_core.domain import MatchStatus
from lazyapply_core.evals.metrics import (
    agreement,
    mean,
    recall_at_k,
    reciprocal_rank,
)

STRONG = MatchStatus.STRONG
PARTIAL = MatchStatus.PARTIAL
NONE = MatchStatus.NONE


class TestRecallAtK:
    def test_the_only_relevant_block_ranked_first(self) -> None:
        assert recall_at_k(["a", "b", "c"], relevant={"a"}, k=3) == 1.0

    def test_the_only_relevant_block_ranked_last_still_counts(self) -> None:
        """Recall asks whether it was found at all, not where."""
        assert recall_at_k(["b", "c", "a"], relevant={"a"}, k=3) == 1.0

    def test_missing_entirely_is_zero(self) -> None:
        assert recall_at_k(["b", "c", "d"], relevant={"a"}, k=3) == 0.0

    def test_found_but_below_the_cutoff_does_not_count(self) -> None:
        """The model only ever sees the top k, so a hit at rank 4 is lost."""
        assert recall_at_k(["b", "c", "d", "a"], relevant={"a"}, k=3) == 0.0

    def test_partial_recall_when_several_blocks_are_relevant(self) -> None:
        assert recall_at_k(["a", "x", "y"], relevant={"a", "b"}, k=3) == 0.5

    def test_full_recall_when_all_relevant_blocks_are_retrieved(self) -> None:
        assert recall_at_k(["a", "b", "x"], relevant={"a", "b"}, k=3) == 1.0

    def test_an_empty_relevant_set_is_undefined(self) -> None:
        """A requirement with nothing to find cannot be scored for recall --
        it tests whether the system declines to match, which is a different
        measurement."""
        with pytest.raises(ValueError, match="no relevant"):
            recall_at_k(["a"], relevant=set(), k=3)

    def test_retrieving_nothing_is_zero(self) -> None:
        assert recall_at_k([], relevant={"a"}, k=3) == 0.0


class TestReciprocalRank:
    def test_first_place_scores_one(self) -> None:
        assert reciprocal_rank(["a", "b"], relevant={"a"}) == 1.0

    def test_second_place_scores_a_half(self) -> None:
        assert reciprocal_rank(["b", "a"], relevant={"a"}) == 0.5

    def test_third_place_scores_a_third(self) -> None:
        assert reciprocal_rank(["b", "c", "a"], relevant={"a"}) == pytest.approx(1 / 3)

    def test_absent_scores_zero(self) -> None:
        assert reciprocal_rank(["b", "c"], relevant={"a"}) == 0.0

    def test_only_the_first_relevant_hit_counts(self) -> None:
        """MRR measures how far a reader has to look, so the second hit adds
        nothing."""
        assert reciprocal_rank(["b", "a", "c"], relevant={"a", "c"}) == 0.5

    def test_this_is_why_it_differs_from_recall(self) -> None:
        """Both orderings have recall 1.0; only MRR separates them."""
        good, bad = ["a", "x", "y"], ["x", "y", "a"]
        assert recall_at_k(good, relevant={"a"}, k=3) == recall_at_k(bad, relevant={"a"}, k=3)
        assert reciprocal_rank(good, relevant={"a"}) > reciprocal_rank(bad, relevant={"a"})


class TestAgreement:
    def test_identical_labels_agree_completely(self) -> None:
        result = agreement([(STRONG, STRONG), (NONE, NONE)])
        assert result.rate == 1.0
        assert result.disagreements == ()

    def test_total_disagreement(self) -> None:
        assert agreement([(STRONG, NONE)]).rate == 0.0

    def test_counts_the_mismatches(self) -> None:
        result = agreement([(STRONG, STRONG), (STRONG, PARTIAL), (NONE, NONE), (NONE, STRONG)])
        assert result.rate == 0.5
        assert result.total == 4

    def test_records_what_disagreed_and_in_which_direction(self) -> None:
        """Direction matters: the model being generous is a different problem
        from the model being harsh."""
        result = agreement([(NONE, STRONG)])
        assert result.disagreements == ((NONE, STRONG),)

    def test_counts_generous_and_harsh_separately(self) -> None:
        result = agreement([(NONE, STRONG), (PARTIAL, STRONG), (STRONG, NONE)])
        assert result.generous == 2
        assert result.harsh == 1

    def test_an_empty_comparison_is_undefined(self) -> None:
        with pytest.raises(ValueError, match="nothing to compare"):
            agreement([])


class TestMean:
    def test_averages(self) -> None:
        assert mean([1.0, 0.0, 0.5]) == 0.5

    def test_an_empty_sequence_has_no_mean(self) -> None:
        """Reporting 0.0 for "we measured nothing" would read as a bad score
        rather than as an absent one."""
        assert mean([]) is None


class TestDisagreementDirection:
    def test_names_the_direction_of_each_pair(self) -> None:
        """Reported per pair, not from the aggregate counts. Using the totals
        would label every line in a mixed result the same way."""
        result = agreement([(NONE, STRONG), (STRONG, NONE)])
        assert [result.direction(pair) for pair in result.disagreements] == ["generous", "harsh"]

    def test_a_bigger_claim_than_a_person_made_is_generous(self) -> None:
        assert agreement([(PARTIAL, STRONG)]).direction((PARTIAL, STRONG)) == "generous"

    def test_a_smaller_claim_is_harsh(self) -> None:
        assert agreement([(PARTIAL, NONE)]).direction((PARTIAL, NONE)) == "harsh"
