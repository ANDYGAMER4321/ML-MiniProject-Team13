import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ============================================================
# 1. LOAD EXTENDED TRAINING DATA
# ============================================================

train_data = pd.read_csv(
    "../data/training_features_extended.csv"
)


# ============================================================
# 2. LOAD EXTENDED 2019 TEST DATA
# ============================================================

test_data = pd.read_csv(
    "../data/test_features_2019_extended.csv"
)


print("Training data:", train_data.shape)
print("Testing data:", test_data.shape)


# ============================================================
# 3. FEATURES
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

    # Recent form
    "player_1_last5_form",
    "player_2_last5_form",

    "player_1_last10_form",
    "player_2_last10_form",

    "player_1_last15_form",
    "player_2_last15_form",

    "player_1_last25_form",
    "player_2_last25_form",

    # Surface
    "player_1_surface_form",
    "player_2_surface_form",

    # H2H
    "player_1_h2h_wins",
    "player_2_h2h_wins"
]


# ============================================================
# 4. CREATE X AND y
# ============================================================

X_train = train_data[feature_columns]

y_train = train_data["target"]

X_test = test_data[feature_columns]

y_test = test_data["target"]


print("\nX_train:", X_train.shape)
print("X_test:", X_test.shape)


# ============================================================
# 5. TRAIN RANDOM FOREST
# ============================================================

print("\nTraining extended Random Forest...")

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
# 6. PREDICT
# ============================================================

probabilities = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 7. MATCH-LEVEL EVALUATION
# ============================================================

results = []

for i in range(0, len(test_data), 2):

    row1 = test_data.iloc[i]

    player_1 = row1["player_1"]

    player_2 = row1["player_2"]

    probability_player_1 = probabilities[i]


    if probability_player_1 >= 0.5:

        predicted_winner = player_1

    else:

        predicted_winner = player_2


    actual_winner = player_1


    correct = (
        predicted_winner == actual_winner
    )


    results.append(correct)


# ============================================================
# 8. CALCULATE ACCURACY
# ============================================================

accuracy = sum(results) / len(results)


print("\n================================")
print("EXTENDED MODEL RESULT")
print("================================")

print(
    "Total matches:",
    len(results)
)

print(
    "Correct predictions:",
    sum(results)
)

print(
    "Incorrect predictions:",
    len(results) - sum(results)
)

print(
    "Match-level accuracy:",
    accuracy * 100,
    "%"
)

print("================================")


# ============================================================
# 9. COMPARE WITH OLD MODEL
# ============================================================

old_accuracy = 63.3695652173913

improvement = (
    accuracy * 100
) - old_accuracy


print("\nPrevious accuracy:")
print(old_accuracy, "%")

print("\nNew accuracy:")
print(accuracy * 100, "%")

print("\nImprovement:")
print(improvement, "percentage points")


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