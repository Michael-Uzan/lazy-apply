from lazyapply_core.hashing import content_hash, normalize


def test_normalize_collapses_whitespace() -> None:
    assert normalize("  a\n\n  b \t c  ") == "a b c"


def test_reflowing_text_does_not_change_the_hash() -> None:
    """Re-wrapping a paragraph must not cost an embedding call."""
    wrapped = "Rebuilt the trading dashboard\nas a React application."
    flowed = "Rebuilt the trading dashboard as a React application."
    assert content_hash(wrapped) == content_hash(flowed)


def test_editing_words_does_change_the_hash() -> None:
    assert content_hash("Rebuilt the dashboard") != content_hash("Rewrote the dashboard")


def test_hash_is_stable_across_calls() -> None:
    assert content_hash("same text") == content_hash("same text")
