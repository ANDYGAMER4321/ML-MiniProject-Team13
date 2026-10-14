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


# ============================================================
# 2. SPLIT 2018 VALIDATION FROM TRAINING
# ============================================================

development_data = train_data[
    train_data["data_year"] <= 2017
].copy()

validation_data = train_data[
    train_data["data_year"] == 2018
].copy()


print("Development data:", development_data.shape)
print("Validation data:", validation_data.shape)
print("2019 test data:", test_data.shape)


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
# 4. TRAIN ON 1997-2017
# ============================================================

X_train = development_data[feature_columns]
y_train = development_data["target"]

X_val = validation_data[feature_columns]


print("\nX_train:", X_train.shape)
print("X_val:", X_val.shape)


# ============================================================
# 5. TRAIN RANDOM FOREST
# ============================================================

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
# 6. VALIDATION PROBABILITIES
# ============================================================

val_probabilities = model.predict_proba(
    X_val
)[:, 1]


# ============================================================
# 7. TEST THRESHOLDS ON 2018 ONLY
# ============================================================

thresholds = [
    0.30,
    0.31,
    0.32,
    0.33,
    0.34,
    0.35,
    0.36,
    0.37,
    0.38,
    0.39,
    0.40,
    0.41,
    0.42,
    0.43,
    0.44,
    0.45,
    0.46,
    0.47,
    0.48,
    0.49,
    0.50
]

best_threshold = 0
best_accuracy = 0


print("\n======================================")
print("2018 VALIDATION THRESHOLD TEST")
print("======================================")


for threshold in thresholds:

    correct = 0

    total_matches = len(validation_data) // 2


    for i in range(
        0,
        len(validation_data),
        2
    ):

        probability = val_probabilities[i]


        if probability >= threshold:
            prediction = 1
        else:
            prediction = 0


        actual = validation_data.iloc[i]["target"]


        if prediction == actual:
            correct += 1


    accuracy = (
        correct / total_matches
    ) * 100


    print(
        f"Threshold: {threshold:.2f} | "
        f"Correct: {correct} | "
        f"Accuracy: {accuracy:.4f}%"
    )


    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_threshold = threshold


# ============================================================
# 8. BEST THRESHOLD FROM 2018
# ============================================================

print("\n======================================")
print("BEST 2018 THRESHOLD")
print("======================================")

print(
    "Best threshold:",
    best_threshold
)

print(
    "2018 validation accuracy:",
    best_accuracy,
    "%"
)


# ============================================================
# 9. NOW TRAIN ON ALL 1997-2018 DATA
# ============================================================

X_full = train_data[feature_columns]
y_full = train_data["target"]

X_test = test_data[feature_columns]


print("\nTraining final model on 1997-2018...")


final_model = RandomForestClassifier(

    n_estimators=200,
    max_depth=10,
    min_samples_leaf=2,

    random_state=42,
    n_jobs=-1
)


final_model.fit(
    X_full,
    y_full
)


print("Final training completed!")


# ============================================================
# 10. TEST 2019
# ============================================================

test_probabilities = final_model.predict_proba(
    X_test
)[:, 1]


correct = 0

total_matches = len(test_data) // 2


for i in range(
    0,
    len(test_data),
    2
):

    probability = test_probabilities[i]


    if probability >= best_threshold:
        prediction = 1
    else:
        prediction = 0


    actual = test_data.iloc[i]["target"]


    if prediction == actual:
        correct += 1


accuracy_2019 = (
    correct / total_matches
) * 100


# ============================================================
# 11. FINAL RESULT
# ============================================================

print("\n======================================")
print("FINAL 2019 RESULT")
print("======================================")

print(
    "Threshold selected from 2018:",
    best_threshold
)

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
    "2019 accuracy:",
    accuracy_2019,
    "%"
)

print("======================================")
