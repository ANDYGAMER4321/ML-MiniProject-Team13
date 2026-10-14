import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier


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


# ============================================================
# 3. PREPARE DATA
# ============================================================

X_train = train_data[
    feature_columns
]

y_train = train_data[
    "target"
]

X_test = test_data[
    feature_columns
]


print("\nX_train:", X_train.shape)
print("X_test:", X_test.shape)


# ============================================================
# 4. TRAIN EXTRA TREES
# ============================================================

print("\nTraining Extra Trees model...")


model = ExtraTreesClassifier(

    n_estimators=300,

    max_depth=12,

    min_samples_leaf=1,

    random_state=42,

    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


print("Training completed!")


# ============================================================
# 5. PREDICT
# ============================================================

probabilities = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 6. MATCH-LEVEL ACCURACY
# ============================================================

correct = 0

total_matches = len(test_data) // 2


for i in range(
    0,
    len(test_data),
    2
):

    probability_player_1 = probabilities[i]


    if probability_player_1 >= 0.5:

        prediction = 1

    else:

        prediction = 0


    actual = test_data.iloc[i]["target"]


    if prediction == actual:

        correct += 1


accuracy = (
    correct / total_matches
) * 100


# ============================================================
# 7. COMPARE WITH BEST
# ============================================================

previous_best = 64.02173913043478

improvement = (
    accuracy - previous_best
)


# ============================================================
# 8. RESULT
# ============================================================

print("\n======================================")
print("EXTRA TREES MODEL")
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