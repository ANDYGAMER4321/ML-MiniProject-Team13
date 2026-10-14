import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# LOAD DATA
# ============================================================

train_data = pd.read_csv(
    "../data/training_features_attributes.csv"
)

test_data = pd.read_csv(
    "../data/test_features_2019_attributes.csv"
)


print("Training data:", train_data.shape)
print("Testing data:", test_data.shape)


# ============================================================
# FEATURES
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
    "player_2_surface_h2h",

    # NEW PLAYER ATTRIBUTES
    "player_1_height",
    "player_2_height",

    "player_1_seed",
    "player_2_seed",

    "player_1_hand",
    "player_2_hand"
]


X_train = train_data[feature_columns]
y_train = train_data["target"]

X_test = test_data[feature_columns]


print("\nX_train:", X_train.shape)
print("X_test:", X_test.shape)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining player-attributes model...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("Training completed!")


# ============================================================
# PREDICTION
# ============================================================

predictions = model.predict(X_test)


# ============================================================
# MATCH-LEVEL ACCURACY
# ============================================================

correct = 0

total_matches = len(test_data) // 2


for i in range(0, len(test_data), 2):

    # Prediction for player 1
    prediction_player_1 = predictions[i]

    # Actual target
    actual_target = test_data.iloc[i]["target"]


    if prediction_player_1 == actual_target:
        correct += 1


accuracy = (
    correct / total_matches
) * 100


# ============================================================
# COMPARE WITH CURRENT BEST
# ============================================================

previous_best = 64.02173913043478

improvement = accuracy - previous_best


# ============================================================
# RESULT
# ============================================================

print("\n======================================")
print("PLAYER ATTRIBUTES MODEL")
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
    "New accuracy:",
    accuracy,
    "%"
)

print(
    "\nPrevious best:",
    previous_best,
    "%"
)

print(
    "Improvement:",
    improvement,
    "percentage points"
)


if improvement > 0:

    print(
        "\nRESULT: Accuracy improved."
    )

elif improvement < 0:

    print(
        "\nRESULT: Accuracy decreased."
    )

else:

    print(
        "\nRESULT: Accuracy stayed the same."
    )

print("======================================")