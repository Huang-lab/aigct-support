"""The reply parser must not turn prose, a cut-off reply or a reasoning trace into a score."""

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "llm_vep" / "submit_variants.py"
SPEC = importlib.util.spec_from_file_location("submit_variants", MODULE_PATH)
submit_variants = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(submit_variants)

parse = submit_variants.parse_score_and_rationale
NO_SCORE = submit_variants.NO_SCORE


def test_no_score_marker_is_minus_one():
    # compare_runs.py counts exactly this value as "no score".
    assert NO_SCORE == -1.0


@pytest.mark.parametrize(
    ("reply", "score"),
    [
        ('{"score": 0.82, "rationale": "x"}', 0.82),
        ('```json\n{"score": 0.4, "rationale": "x"}\n```', 0.4),
        ('{"score": "0.7", "rationale": "x"}', 0.7),
        ('{"score": 0, "rationale": "benign"}', 0.0),
        ('Result: {"score": 0.7, "rationale": "x"} as asked.', 0.7),
        ('{"score": 0.9, "rationale": "cut off inside the rationale', 0.9),
        ("0.73", 0.73),
        ("1", 1.0),
    ],
)
def test_an_explicit_score_is_read(reply, score):
    assert parse(reply)[0] == pytest.approx(score)


@pytest.mark.parametrize(
    "reply",
    [
        # a stray 0 or 1 in prose
        "On a scale from 0 to 1 I would rate this variant 0.85.",
        "I cannot assess variant 1 from coordinates alone.",
        # a reply cut off by the token limit
        '{"score": 0.',
        '{"score": 0.8',
        # a number in a reasoning trace
        "We need to evaluate this. The base rate for 1 in 20 is low, so the final answer is 0.2",
        # no number at all, or an explicit "no score"
        "I'm sorry, but I can't assess this variant.",
        "I would put it at about 0.7.",
        '{"score": null, "rationale": "I do not know this variant."}',
        "",
    ],
)
def test_no_number_is_invented(reply):
    assert parse(reply)[0] == NO_SCORE


@pytest.mark.parametrize(
    "reply",
    [
        '{"score": true}',
        '{"score": 7}',
        '{"score": NaN}',
        '{"score": -0.2}',
        "85",
        '{"score": "high"}',
    ],
)
def test_an_invalid_score_raises(reply):
    with pytest.raises(ValueError):
        parse(reply)


def test_batch_scores_are_range_checked():
    assert submit_variants.parse_batch_results(
        '[{"row_id": 0, "score": 0.4, "rationale": "x"}]'
    ) == {0: (0.4, "x")}
    for bad in (
        '[{"row_id": 0, "score": 7}]',
        '[{"row_id": 0, "score": true}]',
        '[{"row_id": 0, "score": NaN}]',
        "[0.2, 3]",
    ):
        with pytest.raises(ValueError):
            submit_variants.parse_batch_results(bad)
