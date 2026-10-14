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
# 3. TRAIN MODEL
# ============================================================

X_train = train_data[feature_columns]
y_train = train_data["target"]


print("\nX_train:", X_train.shape)


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
# 4. CREATE UNBIASED TEST SET
# ============================================================
#
# Each real match occurs twice:
#
# A vs B
# B vs A
#
# We choose the row whose player_1 name comes first
# alphabetically.
#
# This makes the orientation independent of the winner.
# ============================================================

unbiased_rows = []


for i in range(
    0,
    len(test_data),
    2
):

    row1 = test_data.iloc[i]
    row2 = test_data.iloc[i + 1]


    player1_row1 = str(row1["player_1"])
    player1_row2 = str(row2["player_1"])


    if player1_row1.lower() <= player1_row2.lower():

        selected_row = row1

    else:

        selected_row = row2


    unbiased_rows.append(
        selected_row
    )


unbiased_test = pd.DataFrame(
    unbiased_rows
).reset_index(drop=True)


print(
    "\nUnbiased test rows:",
    unbiased_test.shape
)


# ============================================================
# 5. PREDICT
# ============================================================

X_test = unbiased_test[
    feature_columns
]

y_test = unbiased_test[
    "target"
]


probabilities = model.predict_proba(
    X_test
)[:, 1]


predictions = (
    probabilities >= 0.50
).astype(int)


# ============================================================
# 6. ACCURACY
# ============================================================

correct = (
    predictions == y_test.values
).sum()


total_matches = len(
    unbiased_test
)


incorrect = (
    total_matches - correct
)


accuracy = (
    correct / total_matches
) * 100


# ============================================================
# 7. RESULT
# ============================================================

print("\n======================================")
print("UNBIASED MATCH-LEVEL EVALUATION")
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
    incorrect
)

print(
    "Accuracy:",
    accuracy,
    "%"
)


# ============================================================
# 8. SHOW FIRST 10 PREDICTIONS
# ============================================================

result = pd.DataFrame({

    "player_1":
        unbiased_test["player_1"].values,

    "player_2":
        unbiased_test["player_2"].values,

    "actual_target":
        y_test.values,

    "player_1_probability":
        probabilities,

    "predicted_target":
        predictions

})


print("\nFirst 10 predictions:")

print(
    result.head(10).to_string(
        index=False
    )
)


print("\n======================================")