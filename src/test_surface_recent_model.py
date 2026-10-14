import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# Load data
train_data = pd.read_csv(
    "../data/training_features_surface_recent.csv"
)

test_data = pd.read_csv(
    "../data/test_features_2019_surface_recent.csv"
)

print("Training data:", train_data.shape)
print("Testing data:", test_data.shape)


# Features
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

    # New features
    "player_1_surface_recent_form",
    "player_2_surface_recent_form",

    "player_1_h2h_wins",
    "player_2_h2h_wins",

    # Tournament features
    "player_1_tournament_form",
    "player_2_tournament_form"
]


# Prepare data
X_train = train_data[feature_columns]
y_train = train_data["target"]

X_test = test_data[feature_columns]


# Train
print("\nTraining surface-recent model...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training completed!")


# Predictions
probabilities = model.predict_proba(X_test)[:, 1]


# Match-level evaluation
correct = 0

total_matches = len(test_data) // 2

for i in range(0, len(test_data), 2):

    row = test_data.iloc[i]

    player_1 = row["player_1"]
    player_2 = row["player_2"]

    probability_player_1 = probabilities[i]

    if probability_player_1 >= 0.5:
        predicted_winner = player_1
    else:
        predicted_winner = player_2

    actual_winner = player_1

    if predicted_winner == actual_winner:
        correct += 1


# Accuracy
accuracy = (
    correct / total_matches
) * 100


# Current best
previous_best = 63.69565217391304

improvement = accuracy - previous_best


# Result
print("\n======================================")
print("SURFACE RECENT FEATURE MODEL")
print("======================================")

print("Total matches:", total_matches)

print("Correct predictions:", correct)

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

    print("\nRESULT: Accuracy improved.")

elif improvement < 0:

    print("\nRESULT: Accuracy decreased.")

else:

    print("\nRESULT: Accuracy stayed the same.")

print("======================================")