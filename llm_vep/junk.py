import submit_variants

input_path = "../aigct_data1/repo1/data/ADRD/variant_effect_label.csv"
out_path = "llm_vep/data/test_out.csv"

submit_variants.write_output_csv(
    input_path,
    out_path,
    disease_area="GENERAL",
    api_key=None,
    model="google/gemini-2.5-flash",
    temperature=0.1,
    max_retries=1,
    score_source="OPENROUTER",
    progress_interval=2,
)
