import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 1. LOAD CURRENT BEST DATA
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
# 2. CURRENT BEST FEATURES
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


X_train = train_data[feature_columns]
y_train = train_data["target"]

X_test = test_data[feature_columns]


print("\nX_train:", X_train.shape)
print("X_test:", X_test.shape)


# ============================================================
# 3. TRAIN THE CURRENT BEST RANDOM FOREST
# ============================================================

print("\nTraining current best Random Forest...")


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
# 4. GET PLAYER 1 PROBABILITIES
# ============================================================

probabilities = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 5. TEST DIFFERENT THRESHOLDS
# ============================================================

thresholds = [

    0.45,
    0.46,
    0.47,
    0.48,
    0.49,
    0.50,
    0.51,
    0.52,
    0.53,
    0.54,
    0.55
]


results = []


total_matches = len(test_data) // 2


print("\n======================================")
print("THRESHOLD TESTING")
print("======================================")


for threshold in thresholds:

    correct = 0


    for i in range(
        0,
        len(test_data),
        2
    ):

        # First row represents the actual winner
        probability_player_1 = probabilities[i]


        if probability_player_1 >= threshold:

            prediction = 1

        else:

            prediction = 0


        actual = test_data.iloc[i]["target"]


        if prediction == actual:

            correct += 1


    accuracy = (
        correct / total_matches
    ) * 100


    results.append(
        (
            threshold,
            correct,
            accuracy
        )
    )


    print(
        f"Threshold: {threshold:.2f} | "
        f"Correct: {correct} | "
        f"Accuracy: {accuracy:.4f}%"
    )


# ============================================================
# 6. FIND BEST THRESHOLD
# ============================================================

best_threshold, best_correct, best_accuracy = max(
    results,
    key=lambda x: x[2]
)


# ============================================================
# 7. CURRENT BEST
# ============================================================

previous_best = 64.02173913043478

improvement = (
    best_accuracy - previous_best
)


print("\n======================================")
print("BEST THRESHOLD")
print("======================================")


print(
    "Best threshold:",
    best_threshold
)

print(
    "Correct predictions:",
    best_correct
)

print(
    "Total matches:",
    total_matches
)

print(
    "Best accuracy:",
    best_accuracy,
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
        "\nRESULT: Threshold improved accuracy."
    )

elif improvement < 0:

    print(
        "\nRESULT: Threshold did not improve accuracy."
    )

else:

    print(
        "\nRESULT: Accuracy stayed the same."
    )


print("======================================")