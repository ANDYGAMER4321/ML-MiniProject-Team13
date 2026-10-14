import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 1. LOAD TRAINING DATA
# ============================================================

train_data = pd.read_csv(
    "../data/training_features_surface_h2h.csv"
)

print("Training data:", train_data.shape)


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
# 3. PREPARE DATA
# ============================================================

X = train_data[
    feature_columns
]

y = train_data[
    "target"
]


print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# 4. TRAIN FINAL RANDOM FOREST
# ============================================================

print("\nTraining final Random Forest...")


model = RandomForestClassifier(

    n_estimators=200,
    max_depth=10,
    min_samples_leaf=2,

    random_state=42,
    n_jobs=-1
)


model.fit(
    X,
    y
)


print("Training completed!")


# ============================================================
# 5. SAVE MODEL
# ============================================================

model_path = "../data/final_tennis_model.pkl"

joblib.dump(
    model,
    model_path
)


print("\nModel saved to:")
print(model_path)


# ============================================================
# 6. SAVE THRESHOLD
# ============================================================

best_threshold = 0.48

threshold_path = "../data/final_threshold.txt"


with open(
    threshold_path,
    "w"
) as file:

    file.write(
        str(best_threshold)
    )


print("\nThreshold saved to:")
print(threshold_path)


# ============================================================
# 7. SAVE FEATURE NAMES
# ============================================================

features_path = "../data/final_features.txt"


with open(
    features_path,
    "w"
) as file:

    for feature in feature_columns:

        file.write(
            feature + "\n"
        )


print("\nFeatures saved to:")
print(features_path)


# ============================================================
# 8. FINAL INFORMATION
# ============================================================

print("\n======================================")
print("FINAL MODEL SAVED")
print("======================================")

print(
    "Model:",
    model_path
)

print(
    "Threshold:",
    best_threshold
)

print(
    "Number of features:",
    len(feature_columns)
)

print("======================================")