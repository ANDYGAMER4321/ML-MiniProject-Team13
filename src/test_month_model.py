import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 1. LOAD DATA
# ============================================================

train_data = pd.read_csv(
    "../data/training_features_month.csv"
)

test_data = pd.read_csv(
    "../data/test_features_2019_month.csv"
)


print("Training data:", train_data.shape)
print("Testing data:", test_data.shape)


# ============================================================
# 2. FEATURES
# ============================================================

feature_columns = [

    # Ranking
    "player_1_rank",
    "player_2_rank",
    "rank_difference",

    # Ranking points
    "player_1_rank_points",
    "player_2_rank_points",
    "rank_points_difference",

    # Last 5 form
    "player_1_last5_form",
    "player_2_last5_form",

    # Surface form
    "player_1_surface_form",
    "player_2_surface_form",

    # H2H
    "player_1_h2h_wins",
    "player_2_h2h_wins",

    # NEW: last 30 days
    "player_1_last_month_form",
    "player_2_last_month_form"
]


# ============================================================
# 3. CREATE TRAINING / TEST DATA
# ============================================================

X_train = train_data[feature_columns]

y_train = train_data["target"]

X_test = test_data[feature_columns]

y_test = test_data["target"]


# ============================================================
# 4. TRAIN MODEL
# ============================================================

print("\nTraining model...")

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
# 5. PREDICT PROBABILITIES
# ============================================================

probabilities = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 6. MATCH-LEVEL EVALUATION
# ============================================================

correct = 0

total_matches = len(test_data) // 2


for i in range(0, len(test_data), 2):

    row = test_data.iloc[i]

    player_1 = row["player_1"]

    probability_player_1 = probabilities[i]


    if probability_player_1 >= 0.5:

        predicted_winner = player_1

    else:

        predicted_winner = row["player_2"]


    actual_winner = player_1


    if predicted_winner == actual_winner:

        correct += 1


# ============================================================
# 7. ACCURACY
# ============================================================

accuracy = (
    correct / total_matches
) * 100


old_accuracy = 63.3695652173913

improvement = (
    accuracy - old_accuracy
)


# ============================================================
# 8. RESULT
# ============================================================

print("\n======================================")
print("LAST-MONTH FEATURE MODEL")
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
    "\nPrevious baseline:",
    old_accuracy,
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