import pandas as pd
import joblib


# ============================================================
# 1. LOAD MODEL, DATA, THRESHOLD
# ============================================================

model = joblib.load(
    "../data/final_tennis_model.pkl"
)

data = pd.read_csv(
    "../data/test_features_2019_surface_h2h.csv"
)

with open(
    "../data/final_threshold.txt",
    "r"
) as file:

    threshold = float(
        file.read().strip()
    )


# ============================================================
# 2. FEATURES
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
# 3. TEST 10 MATCHES
# ============================================================

correct = 0
total = 0


print()
print("======================================")
print("FINAL MODEL TEST - 10 MATCHES")
print("======================================")


for i in range(0, 20, 2):

    row1 = data.iloc[i]
    row2 = data.iloc[i + 1]


    # --------------------------------------------------------
    # Choose alphabetical orientation
    # --------------------------------------------------------

    if str(row1["player_1"]).lower() <= str(
        row2["player_1"]
    ).lower():

        row = row1

    else:

        row = row2


    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    X = pd.DataFrame(
        [row[feature_columns].values],
        columns=feature_columns
    )


    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probability = model.predict_proba(X)[0][1]


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if probability >= threshold:

        predicted = row["player_1"]

    else:

        predicted = row["player_2"]


    # --------------------------------------------------------
    # Actual winner
    # --------------------------------------------------------

    if row["target"] == 1:

        actual = row["player_1"]

    else:

        actual = row["player_2"]


    is_correct = (
        predicted == actual
    )


    if is_correct:

        correct += 1


    total += 1


    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print()
    print(
        f"Match {total}:"
    )

    print(
        "Player 1:",
        row["player_1"]
    )

    print(
        "Player 2:",
        row["player_2"]
    )

    print(
        f"Player 1 probability: "
        f"{probability * 100:.2f}%"
    )

    print(
        f"Predicted winner: {predicted}"
    )

    print(
        f"Actual winner:    {actual}"
    )

    print(
        "Result:",
        "CORRECT" if is_correct else "WRONG"
    )


# ============================================================
# 4. SUMMARY
# ============================================================

accuracy = (
    correct / total
) * 100


print()
print("======================================")
print("TEST SUMMARY")
print("======================================")

print(
    "Total tested:",
    total
)

print(
    "Correct:",
    correct
)

print(
    "Wrong:",
    total - correct
)

print(
    f"Accuracy on these {total} matches: "
    f"{accuracy:.2f}%"
)

print(
    "Model threshold:",
    threshold
)

print("======================================")
