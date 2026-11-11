import csv
import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "llm_vep" / "submit_variants.py"
SPEC = importlib.util.spec_from_file_location("submit_variants", MODULE_PATH)
submit_variants = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(submit_variants)


def test_parse_score_from_response_accepts_json_with_markdown_fence():
    response = """```json
{"score": 0.8123, "rationale": "Strong evidence for pathogenicity."}
```"""

    assert submit_variants.parse_score_from_response(response) == pytest.approx(0.8123)


def test_write_output_csv_writes_expected_columns_and_scores(
    tmp_path,
    monkeypatch,
):
    input_path = tmp_path / "variants.csv"
    input_path.write_text(
        "GENOME_ASSEMBLY,CHROMOSOME,POSITION,REFERENCE_NUCLEOTIDE,"
        "ALTERNATE_NUCLEOTIDE,COMMENT\n"
        "GRCh38,1,123,A,G,example\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "scored_variants.csv"

    monkeypatch.setattr(
        submit_variants,
        "score_variants",
        lambda *args, **kwargs: [0.456789],
    )

    submit_variants.write_output_csv(
        str(input_path),
        out_path,
        disease_area="GENERAL",
        api_key="test-key",
        model="test-model",
        temperature=0.1,
        max_retries=1,
    )

    assert out_path.exists()
    with out_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {
            "GENOME_ASSEMBLY": "GRCh38",
            "CHROMOSOME": "1",
            "POSITION": "123",
            "REFERENCE_NUCLEOTIDE": "A",
            "ALTERNATE_NUCLEOTIDE": "G",
            "SCORE_SOURCE": "OPENROUTER",
            "RAW_SCORE": "0.456789",
            "RANK_SCORE": "0.456789",
        }
    ]


def test_write_output_csv_uses_env_api_key_when_api_key_is_none(
    tmp_path,
    monkeypatch,
):
    input_path = tmp_path / "variants.csv"
    input_path.write_text(
        "GENOME_ASSEMBLY,CHROMOSOME,POSITION,REFERENCE_NUCLEOTIDE,"
        "ALTERNATE_NUCLEOTIDE,COMMENT\n"
        "GRCh38,1,123,A,G,example\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "scored_variants.csv"
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-test-key")

    def fake_score_variants(
        rows,
        disease_area,
        api_key,
        model,
        temperature,
        max_retries,
        batch_size=1,
        variant_source=None,
    ):
        assert api_key == "env-test-key"
        return [0.456789]

    monkeypatch.setattr(submit_variants, "score_variants", fake_score_variants)

    submit_variants.write_output_csv(
        str(input_path),
        out_path,
        disease_area="GENERAL",
        api_key=None,
        model="test-model",
        temperature=0.1,
        max_retries=1,
    )

    assert out_path.exists()
    with out_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["SCORE_SOURCE"] == "OPENROUTER"
    assert rows[0]["RAW_SCORE"] == "0.456789"


def test_build_prompt_defaults_to_coding_context_without_explicit_noncoding_annotation():
    prompt = submit_variants.build_prompt(
        "CANCER",
        {
            "GENOME_ASSEMBLY": "GRCh38",
            "CHROMOSOME": "17",
            "POSITION": "7577121",
            "REFERENCE_NUCLEOTIDE": "G",
            "ALTERNATE_NUCLEOTIDE": "A",
        },
    )

    lower_prompt = prompt.lower()
    assert "coding-region submissions" in lower_prompt
    assert "non-coding interpretation" in lower_prompt
    assert "chromosome and position" in lower_prompt


def test_read_variant_rows_skips_leading_rows(tmp_path):
    input_path = tmp_path / "variants.csv"
    input_path.write_text(
        "GENOME_ASSEMBLY,CHROMOSOME,POSITION,REFERENCE_NUCLEOTIDE,ALTERNATE_NUCLEOTIDE\n"
        "skip,1,10,A,C\n"
        "GRCh38,1,123,A,G\n"
        "GRCh38,2,456,T,C\n",
        encoding="utf-8",
    )

    fieldnames, rows = submit_variants.read_variant_rows(str(input_path), skip_rows=1)

    assert fieldnames == [
        "GENOME_ASSEMBLY",
        "CHROMOSOME",
        "POSITION",
        "REFERENCE_NUCLEOTIDE",
        "ALTERNATE_NUCLEOTIDE",
    ]
    assert rows == [
        {
            "GENOME_ASSEMBLY": "GRCh38",
            "CHROMOSOME": "1",
            "POSITION": 123,
            "REFERENCE_NUCLEOTIDE": "A",
            "ALTERNATE_NUCLEOTIDE": "G",
        },
        {
            "GENOME_ASSEMBLY": "GRCh38",
            "CHROMOSOME": "2",
            "POSITION": 456,
            "REFERENCE_NUCLEOTIDE": "T",
            "ALTERNATE_NUCLEOTIDE": "C",
        },
    ]
