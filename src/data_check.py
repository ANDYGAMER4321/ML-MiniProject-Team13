import pandas as pd

file_path = "../data/atp_matches_2019.csv"

df = pd.read_csv(file_path)

required_columns = [
    "winner_name",
    "loser_name",
    "winner_rank",
    "loser_rank",
    "winner_rank_points",
    "loser_rank_points",
    "surface"
]

clean_df = df.dropna(subset=required_columns)

output_path = "../data/cleaned_2019.csv"

clean_df.to_csv(output_path, index=False)

print("Original matches:", len(df))
print("Cleaned matches:", len(clean_df))
print("Saved to:", output_path)