# Paths to user-uploaded files
import pathlib
import pandas as pd

add_results_path = pathlib.Path("2_add_results.csv")
choq_results_path = pathlib.Path("2_ChoquetRank_results.csv")

add_diff_path = pathlib.Path("2_add_results_diffvectors.csv")
choq_diff_path = pathlib.Path("2_ChoquetRank_diffvectors.csv")

# Load CSVs
add_results = pd.read_csv(add_results_path)
choq_results = pd.read_csv(choq_results_path)

add_diff = pd.read_csv(add_diff_path)
choq_diff = pd.read_csv(choq_diff_path)

# Check that column sets match
if set(add_results.columns) != set(choq_results.columns):
    raise ValueError(
        "Column mismatch between results files:\n"
        f"- Only in add_results: {set(add_results.columns) - set(choq_results.columns)}\n"
        f"- Only in ChoquetRank_results: {set(choq_results.columns) - set(add_results.columns)}"
    )

if set(add_diff.columns) != set(choq_diff.columns):
    raise ValueError(
        "Column mismatch between diff‑vector files:\n"
        f"- Only in add_diffvectors: {set(add_diff.columns) - set(choq_diff.columns)}\n"
        f"- Only in ChoquetRank_diffvectors: {set(choq_diff.columns) - set(add_diff.columns)}"
    )

# Merge = simple vertical concatenation
merged_results = pd.concat([add_results, choq_results], ignore_index=True)
merged_diffvectors = pd.concat([add_diff, choq_diff], ignore_index=True)

merged_results_path = pathlib.Path("2_merged_results.csv")
merged_diff_path = pathlib.Path("2_merged_diffvectors.csv")

merged_results.to_csv(merged_results_path, index=False)
merged_diffvectors.to_csv(merged_diff_path, index=False)

# Show quick summary to the user
print("✅ Files merged successfully.")
print(f"merged_results.csv rows: {len(merged_results)}")
print(f"merged_diffvectors.csv rows: {len(merged_diffvectors)}")
