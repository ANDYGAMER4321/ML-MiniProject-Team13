import pandas as pd
import joblib


# ============================================================
# 1. LOAD FINAL MODEL
# ============================================================

MODEL_PATH = "../data/final_tennis_model.pkl"
DATA_PATH = "../data/test_features_2019_surface_h2h.csv"
THRESHOLD_PATH = "../data/final_threshold.txt"


model = joblib.load(MODEL_PATH)


with open(THRESHOLD_PATH, "r") as file:
    threshold = float(file.read().strip())


# ============================================================
# 2. LOAD FEATURE DATA
# ============================================================

data = pd.read_csv(DATA_PATH)


# ============================================================
# 3. FEATURES
# ============================================================

feature_columns = [

    "player_1_rank",
    "player_2_rank",
    "rank_difference",

    "player_1_rank_points",
    "player_2_rank_points",
    "rank_points_difference",

    "player_1_last5_form",
    "player_2_last5_form",

    "player_1_surface_form",
    "player_2_surface_form",

    "player_1_tournament_form",
    "player_2_tournament_form",

    "player_1_h2h_wins",
    "player_2_h2h_wins",

    "player_1_common_opponent_form",
    "player_2_common_opponent_form",

    "player_1_surface_h2h",
    "player_2_surface_h2h"
]


# ============================================================
# 4. GET PLAYER NAMES
# ============================================================

print()
print("======================================")
print("       TENNIS MATCH PREDICTION")
print("======================================")

print()
player_1 = input("Enter Player 1 name: ").strip()
player_2 = input("Enter Player 2 name: ").strip()


# ============================================================
# 5. FIND MATCH
# ============================================================

match = data[
    (
        (
            data["player_1"].str.lower()
            == player_1.lower()
        )
        &
        (
            data["player_2"].str.lower()
            == player_2.lower()
        )
    )
    |
    (
        (
            data["player_1"].str.lower()
            == player_2.lower()
        )
        &
        (
            data["player_2"].str.lower()
            == player_1.lower()
        )
    )
]


# ============================================================
# 6. CHECK MATCH EXISTS
# ============================================================

if len(match) == 0:

    print()
    print("ERROR: Match not found in feature data.")

    print()
    print(
        "This prediction program currently works with"
    )

    print(
        "matches already present in:"
    )

    print(
        DATA_PATH
    )

    print()
    print(
        "For a completely new match, we need to build"
    )

    print(
        "a live feature-generation system."
    )

    exit()


# ============================================================
# 7. SELECT CORRECT ORIENTATION
# ============================================================

row = match.iloc[0]


# ============================================================
# 8. CREATE MODEL INPUT
# ============================================================

X = pd.DataFrame(
    [row[feature_columns].values],
    columns=feature_columns
)


# ============================================================
# 9. PREDICT
# ============================================================

probability = model.predict_proba(X)[0][1]


# ============================================================
# 10. APPLY THRESHOLD
# ============================================================

if probability >= threshold:

    predicted_winner = row["player_1"]

    player_1_probability = probability
    player_2_probability = 1 - probability

else:

    predicted_winner = row["player_2"]

    player_1_probability = probability
    player_2_probability = 1 - probability


# ============================================================
# 11. DISPLAY RESULT
# ============================================================

print()
print("======================================")
print("           PREDICTION RESULT")
print("======================================")

print()
print(
    "Player 1:",
    row["player_1"]
)

print(
    "Player 2:",
    row["player_2"]
)

print()

print(
    f"{row['player_1']} win probability: "
    f"{player_1_probability * 100:.2f}%"
)

print(
    f"{row['player_2']} win probability: "
    f"{player_2_probability * 100:.2f}%"
)

print()

print(
    "Prediction threshold:",
    threshold
)

print()

print(
    "Predicted winner:",
    predicted_winner
)

print()
print("======================================")