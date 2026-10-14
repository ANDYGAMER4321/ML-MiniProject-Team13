import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ============================================================
# 1. LOAD DATA
# ============================================================

train_data = pd.read_csv(
    "../data/training_features_h2h.csv"
)

test_data = pd.read_csv(
    "../data/test_features_2019.csv"
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

    "player_1_h2h_wins",
    "player_2_h2h_wins"
]


X_train = train_data[feature_columns]
y_train = train_data["target"]

X_test = test_data[feature_columns]
y_test = test_data["target"]


# ============================================================
# 3. TRAIN MODEL
# ============================================================

print("Training Random Forest...")

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
# 4. ROW-LEVEL PREDICTIONS
# ============================================================

row_predictions = model.predict(X_test)

row_accuracy = accuracy_score(
    y_test,
    row_predictions
)

print("\nRow-level accuracy:")
print(row_accuracy * 100)


# ============================================================
# 5. GET PREDICTION PROBABILITIES
# ============================================================

probabilities = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 6. ACTUAL MATCH-LEVEL PREDICTION
# ============================================================

# Every real match has TWO rows:
#
# Row 1:
# Player A vs Player B
#
# Row 2:
# Player B vs Player A
#
# We use the probability from the first row.
# If probability >= 0.5:
#     Player 1 is predicted to win
# Otherwise:
#     Player 2 is predicted to win.


match_results = []

for i in range(0, len(test_data), 2):

    row1 = test_data.iloc[i]
    row2 = test_data.iloc[i + 1]

    probability_player1 = probabilities[i]

    if probability_player1 >= 0.5:

        predicted_winner = row1["player_1"]

    else:

        predicted_winner = row1["player_2"]


    # The first row always has target = 1
    # because it represents the actual winner as Player 1.

    actual_winner = row1["player_1"]


    correct = (
        predicted_winner == actual_winner
    )


    match_results.append({

        "player_1": row1["player_1"],

        "player_2": row1["player_2"],

        "actual_winner": actual_winner,

        "predicted_winner": predicted_winner,

        "player_1_probability": probability_player1,

        "correct": correct
    })


# ============================================================
# 7. MATCH-LEVEL DATAFRAME
# ============================================================

match_results = pd.DataFrame(
    match_results
)


# ============================================================
# 8. MATCH-LEVEL ACCURACY
# ============================================================

match_accuracy = (
    match_results["correct"].mean()
)

print("\nActual match-level accuracy:")
print(match_accuracy * 100)


print("\nCorrect matches:")
print(
    match_results["correct"].sum()
)

print("\nTotal matches:")
print(
    len(match_results)
)


# ============================================================
# 9. SHOW SAMPLE PREDICTIONS
# ============================================================

print("\nFirst 10 match predictions:")

print(
    match_results.head(10)
)


# ============================================================
# 10. SAVE MATCH PREDICTIONS
# ============================================================

match_results.to_csv(
    "../data/2019_match_predictions.csv",
    index=False
)

print(
    "\nSaved predictions to:"
)

print(
    "../data/2019_match_predictions.csv"
)


# ============================================================
# 11. SAVE TRAINED MODEL
# ============================================================

joblib.dump(
    model,
    "../data/tennis_model.pkl"
)

print(
    "\nSaved model to:"
)

print(
    "../data/tennis_model.pkl"
)