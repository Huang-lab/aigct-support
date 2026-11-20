#!/usr/bin/env python3
"""Submit single-nucleotide variants to OpenRouter for pathogenicity scoring."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
import pandas as pd
from typing import Any, Tuple
from urllib import error, request

DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_BATCH_ENDPOINT = "https://openrouter.ai/api/v1/batches"
REQUIRED_COLUMNS = [
    "GENOME_ASSEMBLY",
    "CHROMOSOME",
    "POSITION",
    "REFERENCE_NUCLEOTIDE",
    "ALTERNATE_NUCLEOTIDE",
]
OUTPUT_COLUMNS = [
    "GENOME_ASSEMBLY",
    "CHROMOSOME",
    "POSITION",
    "REFERENCE_NUCLEOTIDE",
    "ALTERNATE_NUCLEOTIDE",
    "SCORE_SOURCE",
    "RAW_SCORE",
    "RANK_SCORE",
    "RS_DBSNP",
    "RATIONALE",
]

VARIANT_CSV_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "aigct_data1"
    / "repo1"
    / "data"
    / "variant.csv"
)


def canonical_name(key: str) -> str:
    return key.strip().upper().replace(" ", "_")


def normalize_cli_args(argv: list[str] | None = None) -> list[str]:
    if argv is None:
        argv = sys.argv[1:]

    normalized: list[str] = []
    for arg in argv:
        normalized.append(
            arg.translate(
                {
                    0x2010: "-",
                    0x2011: "-",
                    0x2012: "-",
                    0x2013: "-",
                    0x2014: "-",
                    0x2015: "-",
                    0x2212: "-",
                    0xFE58: "-",
                    0xFE63: "-",
                    0xFF0D: "-",
                }
            )
        )
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score each single-nucleotide variant in a CSV using an LLM via the OpenRouter API. "
            "The output CSV is written to llm_vep/data/<output_filename>.csv."
        )
    )
    parser.add_argument(
        "input_csv", help="Path to the input CSV file containing variant records."
    )
    parser.add_argument(
        "output_name",
        help="Name of the output CSV file to create under llm_vep/data/ (for example: variants_scores.csv).",
    )
    parser.add_argument(
        "--disease-area",
        default="GENERAL",
        help="Disease context for the pathogenicity scoring prompt (for example: CANCER or GENERAL).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL),
        help="OpenRouter chat model to use. Defaults to OPENROUTER_MODEL or openai/gpt-4o-mini.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENROUTER_API_KEY"),
        help="OpenRouter API key. Defaults to OPENROUTER_API_KEY from the environment.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature for the model call.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Number of retries for an API call before failing the row.",
    )
    parser.add_argument(
        "--api-mode",
        choices=["sync", "batch"],
        default="batch",
        help=(
            "Use the synchronous chat endpoint or the OpenRouter batch API. "
            "Default: batch."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Number of variants to include in each per-request prompt chunk. "
            "Each chunk is a separate request inside an OpenRouter batch call."
        ),
    )
    parser.add_argument(
        "--chunks-per-openrouter-batch",
        type=int,
        default=1,
        help=(
            "How many chunked requests to include in a single OpenRouter batch API "
            "call. Each chunk contains up to --batch-size variants."
        ),
    )
    parser.add_argument(
        "--skip-rows",
        type=int,
        default=0,
        help=("Number of leading data rows to skip before processing variant rows."),
    )
    parser.add_argument(
        "--variant-source",
        default=None,
        help=(
            "Optional source description for the variants, such as a cohort, "
            "study, or database. Included in the prompt context."
        ),
    )
    parser.add_argument(
        "--score-source",
        default="OPENROUTER",
        help="Source label to store in the output CSV for each scored variant.",
    )
    return parser.parse_args(normalize_cli_args())


def resolve_output_path(output_name: str) -> Path:
    if not output_name or output_name in {".", "/"}:
        raise ValueError("The output name must be a file name, not a directory path.")

    candidate = Path(output_name)
    if candidate.name != output_name:
        raise ValueError(
            "Use a file name only; output is always written under llm_vep/data/."
        )

    output_dir = Path(__file__).resolve().parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / output_name


def emit_progress(message: str, progress_file: Path | None = None) -> None:
    del progress_file
    print(message, flush=True)


def write_partial_output_csv(
    output_path: Path,
    rows: list[dict[str, Any]],
    scores: list[float | None],
    rationales: list[str | None],
    score_source: str,
) -> None:
    has_meta = any(
        row.get("RS_DBSNP", "") or (rationale or "")
        for row, rationale in zip(rows, rationales)
    )
    fieldnames = (
        OUTPUT_COLUMNS
        if has_meta
        else [
            field for field in OUTPUT_COLUMNS if field not in {"RS_DBSNP", "RATIONALE"}
        ]
    )

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, score, rationale in zip(rows, scores, rationales):
            if score is None:
                continue
            payload = {
                "GENOME_ASSEMBLY": row.get("GENOME_ASSEMBLY", ""),
                "CHROMOSOME": row.get("CHROMOSOME", ""),
                "POSITION": row.get("POSITION", ""),
                "REFERENCE_NUCLEOTIDE": row.get("REFERENCE_NUCLEOTIDE", ""),
                "ALTERNATE_NUCLEOTIDE": row.get("ALTERNATE_NUCLEOTIDE", ""),
                "SCORE_SOURCE": score_source,
                "RAW_SCORE": f"{score:.6f}",
                "RANK_SCORE": f"{score:.6f}",
            }
            if has_meta:
                payload["RS_DBSNP"] = row.get("RS_DBSNP", "")
                payload["RATIONALE"] = rationale if rationale is not None else ""
            writer.writerow({key: payload.get(key, "") for key in fieldnames})


def normalize_fieldnames(fieldnames: list[str]) -> dict[str, str]:
    return {canonical_name(field): field for field in fieldnames}


def read_variant_rows(
    input_csv: str, skip_rows: int = 0
) -> tuple[list[str], list[dict[str, Any]]]:
    try:
        with open(input_csv, "r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("The CSV is missing a header row.")

            field_map = normalize_fieldnames(reader.fieldnames)
            missing = [
                col for col in REQUIRED_COLUMNS if canonical_name(col) not in field_map
            ]
            if missing:
                raise ValueError(
                    "The input CSV is missing required columns: " + ", ".join(missing)
                )

            rows: list[dict[str, Any]] = []
            for _ in range(skip_rows):
                next(reader, None)
            for row in reader:
                rows.append(row)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Input CSV not found: {input_csv}") from exc

    df_rows = pd.DataFrame(rows)
    df_rows["POSITION"] = df_rows["POSITION"].astype(int)
    df_rows["CHROMOSOME"] = df_rows["CHROMOSOME"].astype(str)
    rows = df_rows.to_dict(orient="records")
    return list(reader.fieldnames), rows


def build_coding_status_instruction(variant: dict[str, Any]) -> str:
    variant_lookup = {canonical_name(str(key)): value for key, value in variant.items()}
    explicit_annotations: list[str] = []
    for key, value in variant_lookup.items():
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        if key in {
            "CONSEQUENCE",
            "VEP_CONSEQUENCE",
            "MOST_SEVERE_CONSEQUENCE",
            "EFFECT",
            "ANNOTATION",
            "IMPACT",
            "PROTEIN_CHANGE",
            "AA_CHANGE",
            "GENE",
            "GENE_SYMBOL",
            "GENE_NAME",
            "TRANSCRIPT",
            "EXON",
            "INTRON",
        }:
            explicit_annotations.append(text)

    noncoding_markers = (
        "intron",
        "intronic",
        "intergenic",
        "upstream",
        "downstream",
        "utr",
        "3_utr",
        "5_utr",
        "noncoding",
        "non-coding",
        "regulatory",
    )
    coding_markers = (
        "missense",
        "nonsense",
        "frameshift",
        "splice",
        "stop",
        "start",
        "coding",
        "exon",
        "protein",
        "aa",
    )

    joined_annotations = " ".join(explicit_annotations).lower()
    if any(marker in joined_annotations for marker in noncoding_markers):
        return (
            "Use explicit non-coding annotation only when the variant "
            "annotation itself says the consequence is intronic, "
            "intergenic, UTR, or another non-coding effect; do not infer "
            "non-coding status from genomic coordinates alone."
        )
    if any(marker in joined_annotations for marker in coding_markers):
        return (
            "The provided annotations indicate a coding-region consequence; "
            "prioritize coding-sequence, splice-site, and protein-level "
            "effects when scoring pathogenicity."
        )
    return (
        "Assume these variants are coding-region submissions unless the "
        "input explicitly marks a non-coding consequence. Do not default "
        "to 'non-coding' based only on chromosome and position."
    )


def build_prompt(
    disease_area: str,
    variant: dict[str, Any],
    variant_source: str | None = None,
) -> str:
    variant_context = "\n".join(
        [
            (
                "- GENOME ASSEMBLY: "
                f"{variant.get('GENOME_ASSEMBLY', variant.get('genome_assembly', ''))}"
            ),
            (
                "- CHROMOSOME: "
                f"{variant.get('CHROMOSOME', variant.get('chromosome', ''))}"
            ),
            ("- POSITION: " f"{variant.get('POSITION', variant.get('position', ''))}"),
            (
                "- REFERENCE NUCLEOTIDE: "
                f"{variant.get('REFERENCE_NUCLEOTIDE', variant.get('reference_nucleotide', ''))}"
            ),
            (
                "- ALTERNATE NUCLEOTIDE: "
                f"{variant.get('ALTERNATE_NUCLEOTIDE', variant.get('alternate_nucleotide', ''))}"
            ),
        ]
    )
    dbs = variant.get("RS_DBSNP", "")
    print(f"Building prompt for variant:\n{dbs}", flush=True)
    # variant_context = "RS DBSNP: " + variant.get("RS_DBSNP", "")
    disease_context = (
        "general disease contexts"
        if disease_area.upper() == "GENERAL"
        else disease_area.upper()
    )
    source_note = (
        f"These variants originated from: {variant_source}. " if variant_source else ""
    )
    coding_instruction = build_coding_status_instruction(variant)

    return f"""
You are a clinical genomics variant prioritization assistant.
Evaluate the pathogenicity of this single nucleotide variant in the context
of {disease_context}.
{source_note}Return a JSON object only, with fields:
  - score: a float between 0.0 and 1.0, where 0.0 is benign or
    non-pathogenic and 1.0 is highly pathogenic.
  - rationale: a brief string explaining the core reasoning in one sentence.

Critical coding-region instruction:
{coding_instruction}

Use this information:
{variant_context}

Rules:
- All variants are single-point nucleotide substitutions.
- Consider known mutational effects on coding sequence, splice sites, and
  regulatory consequence patterns.
- Do not default to a non-coding interpretation from chromosome coordinates
  alone.
- Use the disease context to modulate the score, but do not assume a variant
  is pathogenic without evidence.
- Normalize the final score to a value between 0.0 and 1.0.
- Do not output markdown fences or any extra text.
""".strip()


def build_batch_prompt(
    disease_area: str,
    variants: list[dict[str, Any]],
    variant_source: str | None = None,
) -> str:
    disease_context = (
        "general disease contexts"
        if disease_area.upper() == "GENERAL"
        else disease_area.upper()
    )
    source_note = (
        f"These variants originated from: {variant_source}. " if variant_source else ""
    )

    lines: list[str] = [
        "You are a clinical genomics variant prioritization assistant.",
        "Evaluate the pathogenicity of the following single nucleotide variants in the context of "
        f"{disease_context}.",
        f"{source_note}Return a JSON object only, with fields:",
        "  - score: a float between 0.0 and 1.0, where 0.0 is benign or",
        "    non-pathogenic and 1.0 is highly pathogenic.",
        "  - rationale: a brief string explaining the core reasoning in one sentence.",
        "",
        "Critical coding-region instruction:",
        "Assume these variants are coding-region submissions unless the input explicitly marks a non-coding consequence. Do not default to 'non-coding' based only on chromosome and position.",
        "",
        "Use this information:",
    ]

    for idx, variant in enumerate(variants):
        variant_context = "\n".join(
            [
                f"  - row_id: {idx}",
                (
                    "  - GENOME ASSEMBLY: "
                    f"{variant.get('GENOME_ASSEMBLY', variant.get('genome_assembly', ''))}"
                ),
                (
                    "  - CHROMOSOME: "
                    f"{variant.get('CHROMOSOME', variant.get('chromosome', ''))}"
                ),
                (
                    "  - POSITION: "
                    f"{variant.get('POSITION', variant.get('position', ''))}"
                ),
                (
                    "  - REFERENCE NUCLEOTIDE: "
                    f"{variant.get('REFERENCE_NUCLEOTIDE', variant.get('reference_nucleotide', ''))}"
                ),
                (
                    "  - ALTERNATE NUCLEOTIDE: "
                    f"{variant.get('ALTERNATE_NUCLEOTIDE', variant.get('alternate_nucleotide', ''))}"
                ),
            ]
        )
        lines.append(f"Variant {idx}:\n{variant_context}")

    lines.extend(
        [
            "",
            "Rules:",
            "- All variants are single-point nucleotide substitutions.",
            "- Consider known mutational effects on coding sequence, splice sites, and",
            "  regulatory consequence patterns.",
            "- Do not default to a non-coding interpretation from chromosome coordinates",
            "  alone.",
            "- Use the disease context to modulate the score, but do not assume a variant",
            "  is pathogenic without evidence.",
            "- Normalize the final score to a value between 0.0 and 1.0.",
            "- Do not output markdown fences or any extra text.",
            "",
            "Return JSON only as a list of objects in the same order as the input variants.",
            "Each object must contain exactly these keys: row_id, score, rationale.",
            "The score must have 4 decimal places of precision.",
        ]
    )

    print("Building batch prompt for variants:", flush=True)
    return "\n\n".join(lines)


def parse_score_from_response(response_text: str) -> float:
    return parse_score_and_rationale(response_text)[0]


NO_SCORE = -1.0  # what a reply that carries no usable score is recorded as


def _checked_score(value: Any) -> float:
    """Return `value` as a float in [0, 1]; raise ValueError for anything else.

    A JSON boolean is not a score, and NaN fails the range test.
    """
    if isinstance(value, bool):
        raise ValueError(f"Score is a boolean, not a number: {value!r}")
    score = float(value)
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"Scored value is outside the normalized range: {score}")
    return score


def parse_score_and_rationale(response_text: str) -> tuple[float, str]:
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed: Any = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        score_value = parsed.get("score")
        rationale = str(parsed.get("rationale", "")).strip()
        if score_value is None:
            return NO_SCORE, rationale
        return _checked_score(score_value), rationale

    # Not a JSON object. A score is taken only from a complete `"score": <number>`
    # pair, or from a reply that is nothing but a number. A stray 0 or 1 in prose
    # ("on a scale from 0 to 1", "variant 1"), a number cut off by the token limit
    # (`{"score": 0.`) or a number in a reasoning trace is not a score.
    match = re.search(
        r'"score"\s*:\s*(\d+(?:\.\d+)?)(?=\s*[,}\]])',
        cleaned,
        flags=re.IGNORECASE,
    )
    if match is None:
        match = re.fullmatch(r"(\d+(?:\.\d+)?)", cleaned)
    score = _checked_score(match.group(1)) if match else NO_SCORE

    rationale_match = re.search(
        r'"rationale"\s*:\s*"((?:\\.|[^"\\])*)"',
        cleaned,
        flags=re.IGNORECASE,
    )
    if rationale_match:
        rationale = bytes(rationale_match.group(1), "utf-8").decode("unicode_escape")
    else:
        rationale = bytes(cleaned.strip(), "utf-8").decode("unicode_escape")
    return score, rationale


def parse_batch_scores(response_text: str) -> dict[int, float]:
    return {
        row_id: result[0]
        for row_id, result in parse_batch_results(response_text).items()
    }


def parse_batch_results(response_text: str) -> dict[int, tuple[float, str]]:
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed: Any = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        if cleaned.startswith("["):
            recovered_items: list[Any] = []
            for candidate in re.findall(r"\{[^{}]*\}", cleaned, flags=re.DOTALL):
                try:
                    item = json.loads(candidate)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    recovered_items.append(item)
            if recovered_items:
                parsed = recovered_items
            else:
                raise ValueError(
                    f"Model output was not valid JSON: {response_text!r}"
                ) from exc
        else:
            raise ValueError(
                f"Model output was not valid JSON: {response_text!r}"
            ) from exc

    if isinstance(parsed, dict):
        if "results" in parsed:
            parsed = parsed["results"]
        elif "scores" in parsed:
            parsed = parsed["scores"]
        else:
            parsed = [parsed]

    if not isinstance(parsed, list):
        raise ValueError(f"Batch response did not contain a JSON list: {parsed!r}")

    batch_results: dict[int, tuple[float, str]] = {}
    for idx, item in enumerate(parsed):
        if isinstance(item, dict):
            row_id = item.get("row_id")
            if row_id is None:
                row_id = item.get("index", idx)
            score_value = item.get("score")
            if score_value is None:
                raise ValueError(f"Missing score in batch item: {item!r}")
            rationale = str(item.get("rationale", "")).strip()
            batch_results[int(row_id)] = (_checked_score(score_value), rationale)
        else:
            batch_results[idx] = (_checked_score(item), "")

    return batch_results


def _extract_openrouter_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
                    continue
                nested = item.get("content")
                if isinstance(nested, str):
                    text_parts.append(nested)
                    continue
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
        if text_parts:
            return "\n".join(text_parts)
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        nested = content.get("content")
        if isinstance(nested, str):
            return nested
    return ""


def call_openrouter_with_details(
    api_key: str,
    model: str,
    disease_area: str,
    variant: dict[str, Any],
    temperature: float,
    timeout: int = 60,
    variant_source: str | None = None,
) -> tuple[float, str]:
    if not api_key:
        raise ValueError(
            "No OpenRouter API key was provided. Set OPENROUTER_API_KEY or "
            "pass --api-key."
        )

    content = build_prompt(disease_area, variant, variant_source)
    # print(content)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content,
            },
        ],
        "temperature": temperature,
        "max_tokens": 256,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/",
        "X-Title": "llm_vep",
    }
    request_data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        DEFAULT_ENDPOINT, data=request_data, headers=headers, method="POST"
    )

    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenRouter API request failed with HTTP {exc.code}: {detail}"
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

    api_response = json.loads(body)
    choices = api_response.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenRouter returned an empty response: {api_response}")

    message = choices[0].get("message") or {}
    content = message.get("content")
    text = _extract_openrouter_text(content)
    if not text:
        reasoning = message.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            text = reasoning
        elif message.get("tool_calls"):
            raise RuntimeError(
                "OpenRouter returned a tool-call response without assistant text: "
                f"{message!r}"
            )
        else:
            raise RuntimeError(
                "Unexpected OpenRouter content format: "
                f"content={content!r}, full_message={message!r}"
            )

    return parse_score_and_rationale(text)


def call_openrouter(
    api_key: str,
    model: str,
    disease_area: str,
    variant: dict[str, Any],
    temperature: float,
    timeout: int = 60,
    variant_source: str | None = None,
) -> Tuple[float, str]:
    score, rationale = call_openrouter_with_details(
        api_key,
        model,
        disease_area,
        variant,
        temperature,
        timeout=timeout,
        variant_source=variant_source,
    )
    # print(rationale, flush=True)
    return score, rationale


def _get_openrouter_json(
    api_key: str,
    endpoint: str,
    timeout: int = 60,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/",
        "X-Title": "llm_vep",
    }
    req = request.Request(endpoint, headers=headers, method="GET")

    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenRouter GET request failed with HTTP {exc.code}: {detail}"
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(f"OpenRouter GET request failed: {exc}") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OpenRouter returned non-JSON content: {body!r}") from exc


def _post_openrouter_json(
    api_key: str,
    endpoint: str,
    payload: dict[str, Any],
    timeout: int = 60,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/",
        "X-Title": "llm_vep",
    }
    request_data = json.dumps(payload).encode("utf-8")
    req = request.Request(endpoint, data=request_data, headers=headers, method="POST")

    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenRouter API request failed with HTTP {exc.code}: {detail}"
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OpenRouter returned non-JSON content: {body!r}") from exc


def _extract_batch_response_text(item: Any) -> str:
    if not isinstance(item, dict):
        return ""

    if "output_text" in item and isinstance(item["output_text"], str):
        return item["output_text"]

    response = item.get("response")
    if isinstance(response, dict):
        text = _extract_openrouter_text(response.get("content"))
        if text:
            return text
        text = _extract_openrouter_text(response.get("text"))
        if text:
            return text
        if "choices" in response and isinstance(response["choices"], list):
            message = response["choices"][0].get("message") or {}
            text = _extract_openrouter_text(message.get("content"))
            if text:
                return text

    if "content" in item:
        text = _extract_openrouter_text(item.get("content"))
        if text:
            return text
    if "text" in item:
        text = _extract_openrouter_text(item.get("text"))
        if text:
            return text
    if "output" in item:
        text = _extract_openrouter_text(item.get("output"))
        if text:
            return text

    message = item.get("message")
    if isinstance(message, dict):
        text = _extract_openrouter_text(message.get("content"))
        if text:
            return text

    return ""


def _parse_openrouter_batch_response(
    batch_response: dict[str, Any],
) -> dict[int, tuple[float, str]]:
    items: list[Any] = []
    for key in ("data", "results", "responses", "output", "items"):
        value = batch_response.get(key)
        if isinstance(value, list):
            items = value
            break
    if not items and isinstance(batch_response, list):
        items = batch_response
    if not items:
        raise ValueError(
            f"Batch response did not contain any result items: {batch_response!r}"
        )

    parsed: dict[int, tuple[float, str]] = {}
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        custom_id = item.get("custom_id") or item.get("id") or item.get("request_id")
        if custom_id is None:
            row_id = index
        else:
            match = re.search(r"(\d+)$", str(custom_id))
            row_id = int(match.group(1)) if match else index

        text = _extract_batch_response_text(item)
        if not text:
            continue

        try:
            score, rationale = parse_score_and_rationale(text)
        except ValueError as exc:
            raise ValueError(
                f"Could not parse batch result for row {row_id}: {text!r}"
            ) from exc
        parsed[row_id] = (score, rationale)

    if not parsed:
        raise ValueError(
            f"Batch API response had no parseable scoring output: {batch_response!r}"
        )
    return parsed


def call_openrouter_batch_with_details(
    api_key: str,
    model: str,
    disease_area: str,
    variants: list[dict[str, Any]],
    temperature: float,
    timeout: int = 60,
    variant_source: str | None = None,
    batch_size: int = 1,
    chunks_per_openrouter_batch: int = 1,
) -> dict[int, tuple[float, str]]:
    if not api_key:
        raise ValueError(
            "No OpenRouter API key was provided. Set OPENROUTER_API_KEY or "
            "pass --api-key."
        )
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if chunks_per_openrouter_batch < 1:
        raise ValueError("chunks_per_openrouter_batch must be at least 1.")

    chunked_variants = [
        variants[idx : idx + batch_size] for idx in range(0, len(variants), batch_size)
    ]
    requests: list[dict[str, Any]] = []
    for chunk_index, chunk in enumerate(chunked_variants):
        if len(requests) >= chunks_per_openrouter_batch:
            break
        prompt = build_batch_prompt(disease_area, chunk, variant_source)
        requests.append(
            {
                "custom_id": f"variants-{chunk_index}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": 1024,
                },
            }
        )

    print(
        f"Submitting OpenRouter batch request with {len(requests)} chunks...",
        flush=True,
    )
    batch_response = _post_openrouter_json(
        api_key,
        DEFAULT_BATCH_ENDPOINT,
        {
            "endpoint": "/v1/chat/completions",
            "model": model,
            "requests": requests,
        },
        timeout,
    )

    batch_id = batch_response.get("id")
    if not batch_id:
        raise RuntimeError(f"OpenRouter did not return a batch id: {batch_response!r}")

    status = str(batch_response.get("status") or "").lower()
    if status not in {"completed", "failed", "cancelled"}:
        for attempt in range(20):
            time.sleep(2)
            try:
                batch_response = _get_openrouter_json(
                    api_key,
                    f"{DEFAULT_BATCH_ENDPOINT}/{batch_id}",
                    timeout,
                )
            except RuntimeError as exc:
                if "404" in str(exc):
                    print("failed with 404, retrying...", flush=True)
                    continue
                print(f"Error retrieving batch status: {exc}", flush=True)
                raise
            status = str(batch_response.get("status") or "").lower()
            print(f"Batch status: {status}", flush=True)
            if status in {"completed", "failed", "cancelled"}:
                break
            if status not in {"validating", "queued", "in_progress"}:
                break

    if status == "failed":
        error_info = batch_response.get("error")
        raise RuntimeError(f"OpenRouter batch failed: {error_info!r}")
    if status == "cancelled":
        raise RuntimeError(f"OpenRouter batch was cancelled: {batch_response!r}")
    if status != "completed":
        raise RuntimeError(
            f"OpenRouter batch did not complete in time: {batch_response!r}"
        )

    batch_items: list[Any] = []
    for key in ("data", "results", "responses", "output", "items"):
        value = batch_response.get(key)
        if isinstance(value, list):
            batch_items = value
            break

    if not batch_items and isinstance(batch_response.get("result"), list):
        batch_items = batch_response["result"]

    if not batch_items:
        raise RuntimeError(
            f"Unexpected OpenRouter batch response format: {batch_response!r}"
        )

    parsed: dict[int, tuple[float, str]] = {}
    for item in batch_items:
        item_id = item.get("custom_id") if isinstance(item, dict) else None
        if not item_id:
            continue
        match = re.search(r"variants-(\d+)$", str(item_id))
        if not match:
            continue
        chunk_index = int(match.group(1))
        if chunk_index >= len(chunked_variants):
            continue
        text = _extract_batch_response_text(item)
        if not text:
            continue
        chunk_results = parse_batch_results(text)
        base_index = chunk_index * batch_size
        for local_index, result in chunk_results.items():
            row_index = base_index + local_index
            if row_index < 0 or row_index >= len(variants):
                raise ValueError(
                    "Batch response contained a row_id outside the provided input."
                )
            parsed[row_index] = result

    if not parsed:
        raise RuntimeError(
            f"OpenRouter batch response contained no parseable results: "
            f"{batch_response!r}"
        )
    return parsed


def call_openrouter_batch(
    api_key: str,
    model: str,
    disease_area: str,
    variants: list[dict[str, Any]],
    temperature: float,
    timeout: int = 60,
    variant_source: str | None = None,
    batch_size: int = 1,
    chunks_per_openrouter_batch: int = 1,
) -> dict[int, float]:
    return {
        row_id: result[0]
        for row_id, result in call_openrouter_batch_with_details(
            api_key,
            model,
            disease_area,
            variants,
            temperature,
            timeout=timeout,
            variant_source=variant_source,
            batch_size=batch_size,
            chunks_per_openrouter_batch=chunks_per_openrouter_batch,
        ).items()
    }


def _coerce_score_result(
    result: list[float] | tuple[list[float], list[str]],
) -> tuple[list[float], list[str]]:
    if isinstance(result, tuple):
        scores, rationales = result
        return (
            [float(value) for value in scores],
            [str(value) for value in rationales],
        )
    return [float(value) for value in result], ["" for _ in result]


def score_variants(
    rows: list[dict[str, Any]],
    disease_area: str,
    api_key: str,
    model: str,
    temperature: float,
    max_retries: int,
    batch_size: int = 1,
    chunks_per_openrouter_batch: int = 1,
    variant_source: str | None = None,
    progress_interval: int = 20,
    progress_file: Path | None = None,
    output_path: Path | None = None,
    score_source: str = "OPENROUTER",
) -> tuple[list[float], list[str]]:
    def report_progress(
        completed: int,
        total: int,
        next_threshold: int,
    ) -> int:
        while completed >= next_threshold and next_threshold <= total:
            emit_progress(
                f"Progress: {next_threshold} / {total} variants completed.",
                progress_file,
            )
            if output_path is not None:
                write_partial_output_csv(
                    output_path,
                    rows,
                    scores,
                    rationales,
                    score_source,
                )
            next_threshold += progress_interval
        return next_threshold

    if batch_size <= 1 and chunks_per_openrouter_batch <= 1:
        scores: list[float | None] = [None] * len(rows)
        rationales: list[str | None] = [None] * len(rows)
        emit_progress(
            f"Starting scoring for {len(rows)} variants "
            f"(progress every {progress_interval}).",
            progress_file,
        )
        if output_path is not None:
            write_partial_output_csv(
                output_path,
                rows,
                scores,
                rationales,
                score_source,
            )
        next_threshold = progress_interval
        for idx, row in enumerate(rows, start=1):
            last_error: Exception | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    score, rationale = call_openrouter(
                        api_key,
                        model,
                        disease_area,
                        row,
                        temperature,
                        variant_source=variant_source,
                    )
                    scores[idx - 1] = score
                    rationales[idx - 1] = rationale
                    next_threshold = report_progress(
                        sum(1 for value in scores if value is not None),
                        len(rows),
                        next_threshold,
                    )
                    break
                except Exception as exc:  # pragma: no cover - fallback path
                    last_error = exc
                    if attempt < max_retries:
                        time.sleep(1.5 * attempt)
                        continue
                    raise RuntimeError(
                        f"Failed to score row {idx}: {exc}"
                    ) from last_error
        if len(scores) != len(rows):
            raise RuntimeError("Mismatch between row count and scored row count.")
        emit_progress(
            f"Completed: {len(scores)} / {len(rows)} variants.",
            progress_file,
        )
        if output_path is not None:
            write_partial_output_csv(
                output_path,
                rows,
                scores,
                rationales,
                score_source,
            )
        return [float(value) for value in scores], rationales

    scores: list[float | None] = [None] * len(rows)
    rationales: list[str | None] = [None] * len(rows)
    emit_progress(
        f"Starting chunked batch scoring for {len(rows)} variants "
        f"(progress every {progress_interval}).",
        progress_file,
    )
    if output_path is not None:
        write_partial_output_csv(
            output_path,
            rows,
            scores,
            rationales,
            score_source,
        )
    next_threshold = progress_interval

    if chunks_per_openrouter_batch < 1:
        raise ValueError("chunks_per_openrouter_batch must be at least 1.")

    for start in range(0, len(rows), batch_size * chunks_per_openrouter_batch):
        chunk_group = rows[start : start + batch_size * chunks_per_openrouter_batch]
        chunk_results: dict[int, tuple[float, str]] = {}
        for attempt in range(1, max_retries + 1):
            try:
                chunk_results = call_openrouter_batch_with_details(
                    api_key,
                    model,
                    disease_area,
                    chunk_group,
                    temperature,
                    variant_source=variant_source,
                    batch_size=batch_size,
                    chunks_per_openrouter_batch=chunks_per_openrouter_batch,
                )
                break
            except Exception as exc:  # pragma: no cover - fallback path
                print(
                    f"Error scoring batch {start}:{start + len(chunk_group)}: {exc}",
                    flush=True,
                )
                if attempt < max_retries:
                    time.sleep(1.5 * attempt)
                    continue
                if batch_size > 1:
                    emit_progress(
                        "Batch failed; retrying with smaller " "per-request chunks.",
                        progress_file,
                    )
                    sub_scores, sub_rationales = score_variants(
                        chunk_group,
                        disease_area,
                        api_key,
                        model,
                        temperature,
                        max_retries,
                        batch_size=max(1, batch_size // 2),
                        chunks_per_openrouter_batch=chunks_per_openrouter_batch,
                        variant_source=variant_source,
                        progress_file=progress_file,
                        output_path=output_path,
                        score_source=score_source,
                    )
                    for index, value in enumerate(sub_scores):
                        scores[start + index] = float(value)
                        rationales[start + index] = sub_rationales[index]
                    completed = sum(1 for value in scores if value is not None)
                    next_threshold = report_progress(
                        completed, len(rows), next_threshold
                    )
                    break
                raise RuntimeError(
                    "Failed to score batch group "
                    f"{start}:{start + len(chunk_group)}: {exc}"
                ) from exc

        for row_index, result in chunk_results.items():
            global_index = start + row_index
            if global_index < 0 or global_index >= len(rows):
                raise ValueError(
                    "Batch response contained a row_id outside the input rows."
                )
            value, rationale = result
            scores[global_index] = float(value)
            rationales[global_index] = rationale

        completed = sum(1 for value in scores if value is not None)
        next_threshold = report_progress(completed, len(rows), next_threshold)

    for idx, value in enumerate(scores):
        if value is None:
            raise RuntimeError(
                "No score was produced for row " f"{idx} in batched mode."
            )

    emit_progress(
        f"Completed: {len(rows)} / {len(rows)} variants.",
        progress_file,
    )
    if output_path is not None:
        write_partial_output_csv(
            output_path,
            rows,
            scores,
            rationales,
            score_source,
        )
    return [float(value) for value in scores], rationales


def write_output_csv(
    input_csv: str,
    output_path: Path,
    disease_area: str,
    api_key: str | None,
    model: str,
    temperature: float,
    max_retries: int,
    batch_size: int = 1,
    chunks_per_openrouter_batch: int = 1,
    variant_source: str | None = None,
    score_source: str = "OPENROUTER",
    progress_interval: int = 10,
    api_mode: str = "batch",
    skip_rows: int = 0,
) -> None:
    resolved_api_key = api_key
    if resolved_api_key is None:
        resolved_api_key = os.environ.get("OPENROUTER_API_KEY")

    if api_mode == "sync":
        batch_size = 1
        chunks_per_openrouter_batch = 1

    fieldnames, rows = read_variant_rows(input_csv, skip_rows=skip_rows)
    del fieldnames

    score_kwargs = {
        "batch_size": batch_size,
        "chunks_per_openrouter_batch": chunks_per_openrouter_batch,
        "variant_source": variant_source,
        "progress_interval": progress_interval,
        "output_path": output_path,
        "score_source": score_source,
    }
    supported_kwargs = {}
    try:
        accepted = inspect.signature(score_variants).parameters
    except (TypeError, ValueError):
        accepted = {}
    for key, value in score_kwargs.items():
        if key in accepted:
            supported_kwargs[key] = value

    scoring_result = score_variants(
        rows,
        disease_area,
        resolved_api_key,
        model,
        temperature,
        max_retries,
        **supported_kwargs,
    )
    scores, rationales = _coerce_score_result(scoring_result)

    if output_path is not None:
        write_partial_output_csv(
            output_path,
            rows,
            scores,
            rationales,
            score_source,
        )


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.chunks_per_openrouter_batch < 1:
        raise ValueError("--chunks-per-openrouter-batch must be at least 1.")
    if args.skip_rows < 0:
        raise ValueError("--skip-rows must be non-negative.")

    input_csv = Path(args.input_csv).expanduser().resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    try:
        out_path = resolve_output_path(args.output_name)
        write_output_csv(
            str(input_csv),
            out_path,
            disease_area=args.disease_area,
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
            max_retries=args.max_retries,
            batch_size=args.batch_size,
            chunks_per_openrouter_batch=args.chunks_per_openrouter_batch,
            variant_source=args.variant_source,
            score_source=args.score_source,
            api_mode=args.api_mode,
            skip_rows=args.skip_rows,
        )
        return 0
    except Exception as exc:  # pragma: no cover - CLI error path
        print(f"Error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
