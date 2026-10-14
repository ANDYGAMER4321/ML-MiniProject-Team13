import pandas as pd
import glob
import os

# Load all ATP match files from 1997 to 2019
files = glob.glob("../data/atp_matches_*.csv")

files = [
    file for file in files
    if 1997 <= int(os.path.basename(file).split("_")[-1].replace(".csv", "")) <= 2019
]

files.sort()

all_matches = []

for file in files:
    year = int(os.path.basename(file).split("_")[-1].replace(".csv", ""))

    df = pd.read_csv(file)
    df["data_year"] = year

    all_matches.append(df)

matches = pd.concat(all_matches, ignore_index=True)

# Keep matches with required information
required_columns = [
    "winner_name",
    "loser_name",
    "winner_rank",
    "loser_rank",
    "winner_rank_points",
    "loser_rank_points",
    "surface"
]

matches = matches.dropna(subset=required_columns)

# Only use 1997-2018 for training
train_matches = matches[
    (matches["data_year"] >= 1997) &
    (matches["data_year"] <= 2018)
].copy()

# -------------------------------
# Create two rows for each match
# -------------------------------

# Row 1: Winner is Player 1
row_winner = pd.DataFrame({
    "player_1": train_matches["winner_name"],
    "player_2": train_matches["loser_name"],
    "player_1_rank": train_matches["winner_rank"],
    "player_2_rank": train_matches["loser_rank"],
    "rank_difference": (
        train_matches["winner_rank"] -
        train_matches["loser_rank"]
    ),
    "target": 1
})

# Row 2: Loser is Player 1
row_loser = pd.DataFrame({
    "player_1": train_matches["loser_name"],
    "player_2": train_matches["winner_name"],
    "player_1_rank": train_matches["loser_rank"],
    "player_2_rank": train_matches["winner_rank"],
    "rank_difference": (
        train_matches["loser_rank"] -
        train_matches["winner_rank"]
    ),
    "target": 0
})

# Combine both rows
training_data = pd.concat(
    [row_winner, row_loser],
    ignore_index=True
)

print("Original training matches:", len(train_matches))
print("Training rows after creating two rows per match:", len(training_data))

print("\nFirst 10 rows:")
print(training_data.head(10))

print("\nTarget distribution:")
print(training_data["target"].value_counts())

# Save
output_path = "../data/training_features_basic.csv"

training_data.to_csv(
    output_path,
    index=False
)

print("\nSaved to:", output_path)