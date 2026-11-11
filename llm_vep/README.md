# LLM VEP scoring

This helper submits a CSV of single-point variants to OpenRouter and asks the model to return a normalized pathogenicity score between 0.0 and 1.0.

## Required input columns

The input CSV must include:

- `GENOME_ASSEMBLY`
- `CHROMOSOME`
- `POSITION`
- `REFERENCE_NUCLEOTIDE`
- `ALTERNATE_NUCLEOTIDE`

It may include additional columns; they are preserved in the output file.

## Usage

```bash
export OPENROUTER_API_KEY="your_api_key_here"
python llm_vep/submit_variants.py variants.csv scored_variants.csv --disease-area CANCER
```

The script writes output to:

```text
llm_vep/data/scored_variants.csv
```

The output CSV contains the same input columns plus a final `score` column.

## Optional arguments

- `--disease-area`: set to `GENERAL` or a disease context such as `CANCER`.
- `--model`: OpenRouter model name (defaults to `openai/gpt-4o-mini`).
- `--temperature`: sampling temperature.
- `--api-key`: pass the API key directly instead of using the environment variable.

## Notes

- All variants are treated as single-point substitutions.
- The output score is normalized to the range `[0, 1]`.
- The script preserves the input columns and appends `score` as the last column.
