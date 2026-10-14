import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 1. LOAD CURRENT BEST FEATURES
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
# 3. SPLIT TRAINING INTO TRAIN + VALIDATION
#
# 1997-2017 = training
# 2018       = validation
# 2019       = final test
# ============================================================

development_data = train_data[
    train_data["data_year"] <= 2017
].copy()

validation_data = train_data[
    train_data["data_year"] == 2018
].copy()


print("\nDevelopment data:", development_data.shape)
print("Validation data:", validation_data.shape)


# ============================================================
# 4. PREPARE DATA
# ============================================================

X_dev = development_data[
    feature_columns
]

y_dev = development_data[
    "target"
]


X_val = validation_data[
    feature_columns
]

y_val = validation_data[
    "target"
]


# ============================================================
# 5. FUNCTION FOR MATCH-LEVEL ACCURACY
# ============================================================

def match_accuracy(
    model,
    data,
    probabilities
):

    correct = 0

    total_matches = len(data) // 2

    for i in range(
        0,
        len(data),
        2
    ):

        # probability that player 1 wins
        p1_probability = probabilities[i]

        if p1_probability >= 0.5:

            prediction = 1

        else:

            prediction = 0


        actual = data.iloc[i]["target"]


        if prediction == actual:

            correct += 1


    return (
        correct / total_matches
    ) * 100


# ============================================================
# 6. RANDOM FOREST SETTINGS TO TEST
# ============================================================

models = [

    {
        "name": "Model 1",
        "n_estimators": 200,
        "max_depth": 8,
        "min_samples_leaf": 2
    },

    {
        "name": "Model 2",
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_leaf": 2
    },

    {
        "name": "Model 3",
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 2
    },

    {
        "name": "Model 4",
        "n_estimators": 300,
        "max_depth": 15,
        "min_samples_leaf": 2
    },

    {
        "name": "Model 5",
        "n_estimators": 400,
        "max_depth": 10,
        "min_samples_leaf": 1
    },

    {
        "name": "Model 6",
        "n_estimators": 400,
        "max_depth": 12,
        "min_samples_leaf": 1
    }
]


# ============================================================
# 7. TEST PARAMETERS ON 2018 VALIDATION DATA
# ============================================================

best_accuracy = 0

best_parameters = None


print("\n======================================")
print("RANDOM FOREST PARAMETER TUNING")
print("======================================")


for settings in models:

    print(
        "\nTraining",
        settings["name"],
        "..."
    )


    model = RandomForestClassifier(

        n_estimators=
            settings["n_estimators"],

        max_depth=
            settings["max_depth"],

        min_samples_leaf=
            settings["min_samples_leaf"],

        random_state=42,

        n_jobs=-1
    )


    model.fit(
        X_dev,
        y_dev
    )


    probabilities = model.predict_proba(
        X_val
    )[:, 1]


    accuracy = match_accuracy(
        model,
        validation_data,
        probabilities
    )


    print(
        settings
    )

    print(
        "2018 validation accuracy:",
        round(accuracy, 4),
        "%"
    )


    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_parameters = settings.copy()


# ============================================================
# 8. BEST PARAMETERS
# ============================================================

print("\n======================================")
print("BEST PARAMETERS")
print("======================================")

print(
    best_parameters
)

print(
    "Validation accuracy:",
    round(best_accuracy, 4),
    "%"
)


# ============================================================
# 9. TRAIN FINAL MODEL ON ALL 1997-2018 DATA
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


print("\nTraining final model...")


final_model = RandomForestClassifier(

    n_estimators=
        best_parameters["n_estimators"],

    max_depth=
        best_parameters["max_depth"],

    min_samples_leaf=
        best_parameters["min_samples_leaf"],

    random_state=42,

    n_jobs=-1
)


final_model.fit(
    X_train,
    y_train
)


print("Final model training completed!")


# ============================================================
# 10. TEST ON 2019
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

    probability_player_1 = (
        test_probabilities[i]
    )


    if probability_player_1 >= 0.5:

        prediction = 1

    else:

        prediction = 0


    actual = test_data.iloc[i]["target"]


    if prediction == actual:

        correct += 1


final_accuracy = (
    correct / total_matches
) * 100


# ============================================================
# 11. COMPARE WITH CURRENT BEST
# ============================================================

previous_best = 64.02173913043478

improvement = (
    final_accuracy
    - previous_best
)


print("\n======================================")
print("FINAL TUNED RANDOM FOREST")
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
    final_accuracy,
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