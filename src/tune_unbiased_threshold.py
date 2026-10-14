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
# 2. SPLIT TRAINING DATA
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
# 4. TRAIN MODEL ON 1997–2017
# ============================================================

X_train = development_data[
    feature_columns
]

y_train = development_data[
    "target"
]


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
# 5. FUNCTION TO MAKE UNBIASED MATCH DATA
# ============================================================

def make_unbiased_matches(data):

    rows = []


    for i in range(
        0,
        len(data),
        2
    ):

        row1 = data.iloc[i]
        row2 = data.iloc[i + 1]


        name1 = str(
            row1["player_1"]
        )

        name2 = str(
            row2["player_1"]
        )


        # Alphabetical ordering
        if name1.lower() <= name2.lower():

            rows.append(row1)

        else:

            rows.append(row2)


    return pd.DataFrame(
        rows
    ).reset_index(drop=True)


# ============================================================
# 6. CREATE UNBIASED 2018 VALIDATION DATA
# ============================================================

validation_matches = make_unbiased_matches(
    validation_data
)


print(
    "\nUnbiased validation matches:",
    validation_matches.shape
)


X_validation = validation_matches[
    feature_columns
]

y_validation = validation_matches[
    "target"
]


# ============================================================
# 7. VALIDATION PROBABILITIES
# ============================================================

validation_probabilities = model.predict_proba(
    X_validation
)[:, 1]


# ============================================================
# 8. TEST THRESHOLDS
# ============================================================

thresholds = [

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
    0.50,
    0.51,
    0.52,
    0.53,
    0.54,
    0.55,
    0.56,
    0.57,
    0.58,
    0.59,
    0.60
]


best_threshold = 0.50
best_accuracy = 0


print("\n======================================")
print("UNBIASED 2018 THRESHOLD TEST")
print("======================================")


for threshold in thresholds:

    predictions = (
        validation_probabilities
        >= threshold
    ).astype(int)


    correct = (
        predictions
        == y_validation.values
    ).sum()


    accuracy = (
        correct
        / len(y_validation)
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
# 9. BEST THRESHOLD
# ============================================================

print("\n======================================")
print("BEST UNBIASED THRESHOLD")
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
# 10. TRAIN FINAL MODEL ON 1997–2018
# ============================================================

print(
    "\nTraining final model on 1997–2018..."
)


X_full = train_data[
    feature_columns
]

y_full = train_data[
    "target"
]


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


print(
    "Final training completed!"
)


# ============================================================
# 11. CREATE UNBIASED 2019 TEST DATA
# ============================================================

test_matches = make_unbiased_matches(
    test_data
)


X_test = test_matches[
    feature_columns
]

y_test = test_matches[
    "target"
]


# ============================================================
# 12. FINAL 2019 PREDICTION
# ============================================================

test_probabilities = final_model.predict_proba(
    X_test
)[:, 1]


predictions = (
    test_probabilities
    >= best_threshold
).astype(int)


correct = (
    predictions
    == y_test.values
).sum()


total_matches = len(
    test_matches
)


accuracy = (
    correct
    / total_matches
) * 100


# ============================================================
# 13. FINAL RESULT
# ============================================================

print("\n======================================")
print("FINAL UNBIASED 2019 RESULT")
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
    accuracy,
    "%"
)

print("======================================")