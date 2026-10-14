import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 1. LOAD DATA
# ============================================================

train_data = pd.read_csv(
    "../data/training_features_surface_h2h.csv"
)

test_data = pd.read_csv(
    "../data/test_features_2019_surface_h2h.csv"
)


print("Training data:", train_data.shape)
print("Testing data:", test_data.shape)


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
# 3. TRAIN
# ============================================================

X_train = train_data[feature_columns]
y_train = train_data["target"]

X_test = test_data[feature_columns]


print("\nX_train:", X_train.shape)
print("X_test:", X_test.shape)


print("\nTraining Random Forest...")


model = RandomForestClassifier(

    n_estimators=200,
    max_depth=10,
    min_samples_leaf=2,

    random_state=42,
    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


print("Training completed!")


# ============================================================
# 4. PREDICT BOTH ROWS
# ============================================================

probabilities = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 5. CORRECT MATCH-LEVEL EVALUATION
# ============================================================

correct = 0
total_matches = 0


for i in range(
    0,
    len(test_data),
    2
):

    # --------------------------------------------------------
    # First row
    # --------------------------------------------------------

    player_1_probability = probabilities[i]


    # --------------------------------------------------------
    # Second row
    #
    # This row has the players reversed.
    # Therefore its probability represents the other
    # player's chance of winning.
    # --------------------------------------------------------

    player_2_probability = 1 - probabilities[i + 1]


    # --------------------------------------------------------
    # Choose the player with the higher probability
    # --------------------------------------------------------

    if player_1_probability >= player_2_probability:

        predicted_winner = 1

    else:

        predicted_winner = 2


    # --------------------------------------------------------
    # Actual winner
    #
    # First row target = 1 means player 1 won.
    # --------------------------------------------------------

    actual_target = test_data.iloc[i]["target"]


    if actual_target == 1:

        actual_winner = 1

    else:

        actual_winner = 2


    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    if predicted_winner == actual_winner:

        correct += 1


    total_matches += 1


# ============================================================
# 6. ACCURACY
# ============================================================

accuracy = (
    correct / total_matches
) * 100


# ============================================================
# 7. RESULT
# ============================================================

print("\n======================================")
print("CORRECT MATCH-LEVEL EVALUATION")
print("======================================")

print(
    "Total matches:",
    total_matches
)

print(
    "Correct predictions:",
    correct
)

print(
    "Incorrect predictions:",
    total_matches - correct
)

print(
    "Actual match-level accuracy:",
    accuracy,
    "%"
)

print("======================================")