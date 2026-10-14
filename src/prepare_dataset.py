import pandas as pd
import glob
import os

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

    # Store the year from the filename
    df["data_year"] = year

    all_matches.append(df)

matches = pd.concat(all_matches, ignore_index=True)

# Training: files from 1997 to 2018
train_data = matches[
    (matches["data_year"] >= 1997) &
    (matches["data_year"] <= 2018)
].copy()

# Testing: file from 2019
test_data = matches[
    matches["data_year"] == 2019
].copy()

print("Total matches:", len(matches))
print("Training matches:", len(train_data))
print("Testing matches:", len(test_data))

print("\nTraining data years:")
print(train_data["data_year"].min(), "to", train_data["data_year"].max())

print("\nTesting data year:")
print(test_data["data_year"].unique())