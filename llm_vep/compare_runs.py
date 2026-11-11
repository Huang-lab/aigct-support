#!/usr/bin/env python3
"""Compare repeated LLM variant-scoring runs on the same variants."""

from __future__ import annotations

import argparse
import csv
import math
from itertools import combinations
from pathlib import Path
from typing import Iterable

DEFAULT_SCORE_COLUMN = "RAW_SCORE"
DEFAULT_VARIANT_COLUMNS = [
    "GENOME_ASSEMBLY",
    "CHROMOSOME",
    "POSITION",
    "REFERENCE_NUCLEOTIDE",
    "ALTERNATE_NUCLEOTIDE",
]


def canonical_name(name: str) -> str:
    return name.strip().upper().replace(" ", "_")


def normalize_fieldnames(fieldnames: Iterable[str]) -> dict[str, str]:
    return {canonical_name(field): field for field in fieldnames}


def read_run_csv(
    path: str, score_column: str = DEFAULT_SCORE_COLUMN
) -> dict[str, float]:
    """Read a scoring CSV and return a mapping of variant_id -> score."""
    csv_path = Path(path)
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is missing a header row.")

        field_map = normalize_fieldnames(reader.fieldnames)
        missing = [
            col
            for col in DEFAULT_VARIANT_COLUMNS
            if canonical_name(col) not in field_map
        ]
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {', '.join(missing)}"
            )

        score_key = canonical_name(score_column)
        if score_key not in field_map:
            raise ValueError(
                f"{path} is missing the score column '{score_column}'. "
                f"Available columns: {', '.join(reader.fieldnames)}"
            )

        scores: dict[str, float] = {}
        for row in reader:
            variant_parts = [
                str(row.get(field_map.get(canonical_name(col), col), "")).strip()
                for col in DEFAULT_VARIANT_COLUMNS
            ]
            variant_id = ":".join(variant_parts)
            raw_value = row.get(field_map[score_key], "")
            if raw_value in (None, ""):
                raise ValueError(f"Missing score for variant {variant_id} in {path}.")
            try:
                score = float(raw_value)
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric score '{raw_value}' for variant {variant_id} in {path}."
                ) from exc
            scores[variant_id] = score

    return scores


def pearson_correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")

    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)

    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))

    if den_x == 0.0 or den_y == 0.0:
        return float("nan")
    return num / (den_x * den_y)


def pairwise_metrics(
    scores_a: dict[str, float], scores_b: dict[str, float], tolerance: float
) -> dict[str, float | int]:
    shared = sorted(set(scores_a) & set(scores_b))
    if not shared:
        raise ValueError("No shared variants between the two runs.")

    vals_a = [scores_a[k] for k in shared]
    vals_b = [scores_b[k] for k in shared]

    deltas = [abs(a - b) for a, b in zip(vals_a, vals_b)]
    within = sum(1 for d in deltas if d <= tolerance) / len(deltas)
    minus_one_count = sum(
        1
        for k in shared
        if scores_a[k] == -1.0 or scores_b[k] == -1.0
    )

    return {
        "shared_variants": len(shared),
        "num_minus_one_variants": minus_one_count,
        "mean_abs_delta": sum(deltas) / len(deltas),
        "median_abs_delta": (
            sorted(deltas)[len(deltas) // 2]
            if len(deltas) % 2
            else (
                sorted(deltas)[len(deltas) // 2 - 1] + sorted(deltas)[len(deltas) // 2]
            )
            / 2
        ),
        "max_abs_delta": max(deltas),
        "pearson_r": pearson_correlation(vals_a, vals_b),
        "frac_within_tolerance": within,
    }


def compare_runs(
    paths: list[str], score_column: str = DEFAULT_SCORE_COLUMN, tolerance: float = 0.05
) -> list[dict[str, object]]:
    runs = [
        {"name": Path(p).name, "path": p, "scores": read_run_csv(p, score_column)}
        for p in paths
    ]

    summaries: list[dict[str, object]] = []
    for left, right in combinations(runs, 2):
        metrics = pairwise_metrics(left["scores"], right["scores"], tolerance)
        summaries.append(
            {
                "run_a": left["name"],
                "run_b": right["name"],
                **metrics,
            }
        )

    return summaries


def write_summary_csv(path: str, rows: list[dict[str, object]]) -> None:
    output_path = Path(path)
    fieldnames = [
        "run_a",
        "run_b",
        "shared_variants",
        "num_minus_one_variants",
        "mean_abs_delta",
        "median_abs_delta",
        "max_abs_delta",
        "pearson_r",
        "frac_within_tolerance",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare repeated scoring runs from llm_vep submit_variants.py and summarize "
            "how different they are from each other."
        )
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Two or more CSV files produced by submit_variants.py to compare.",
    )
    parser.add_argument(
        "--score-column",
        default=DEFAULT_SCORE_COLUMN,
        help="Name of the score column in the CSV files (default: RAW_SCORE).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Delta threshold used to report the fraction of variants within tolerance.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV file to write the comparison summary to.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.files) < 2:
        raise SystemExit("Provide at least two CSV files to compare.")

    rows = compare_runs(
        args.files, score_column=args.score_column, tolerance=args.tolerance
    )

    print("Comparison summary")
    for row in rows:
        print(f"run_a: {row['run_a']}")
        print(f"run_b: {row['run_b']}")
        print(f"shared_variants: {row['shared_variants']}")
        print(f"num_minus_one_variants: {row['num_minus_one_variants']}")
        print(f"mean_abs_delta: {row['mean_abs_delta']:.6f}")
        print(f"median_abs_delta: {row['median_abs_delta']:.6f}")
        print(f"max_abs_delta: {row['max_abs_delta']:.6f}")
        print(f"pearson_r: {row['pearson_r']:.6f}")
        print(f"frac_within_tolerance: {row['frac_within_tolerance']:.6f}")
        print()

    if args.output:
        write_summary_csv(args.output, rows)
        print(f"Wrote comparison summary to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
