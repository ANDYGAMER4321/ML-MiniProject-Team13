import pandas as pd
import glob
import os


# ============================================================
# 1. LOAD MATCH FILES
# ============================================================

files = glob.glob("../data/atp_matches_*.csv")

files = [
    file
    for file in files
    if 1997 <= int(
        os.path.basename(file).split("_")[-1].replace(".csv", "")
    ) <= 2019
]

files.sort()

all_matches = []

for file in files:

    year = int(
        os.path.basename(file).split("_")[-1].replace(".csv", "")
    )

    df = pd.read_csv(file)

    df["data_year"] = year

    all_matches.append(df)


matches = pd.concat(
    all_matches,
    ignore_index=True
)


# ============================================================
# 2. REMOVE INCOMPLETE MATCHES
# ============================================================

required_columns = [
    "winner_name",
    "loser_name",
    "winner_rank",
    "loser_rank",
    "winner_rank_points",
    "loser_rank_points",
    "surface"
]

matches = matches.dropna(
    subset=required_columns
)


# ============================================================
# 3. SORT CHRONOLOGICALLY
# ============================================================

matches["tourney_date"] = pd.to_datetime(
    matches["tourney_date"].astype(str),
    format="%Y%m%d"
)

matches = matches.sort_values(
    ["tourney_date", "match_num"]
).reset_index(drop=True)


# ============================================================
# 4. HISTORY
# ============================================================

# Overall match history
player_history = {}

# Surface history
surface_history = {}

# Head-to-head
h2h_history = {}

# Last-month history
#
# Each entry will look like:
# player_history_dates["Player"] = [
#     (date, 1),
#     (date, 0),
#     ...
# ]

player_month_history = {}


# ============================================================
# 5. FORM FUNCTION
# ============================================================

def calculate_form(history, number_of_matches):

    recent = history[-number_of_matches:]

    if len(recent) == 0:
        return 0.5

    return sum(recent) / len(recent)


# ============================================================
# 6. LAST 30 DAYS FORM
# ============================================================

def calculate_last_month_form(
    history,
    current_date
):

    cutoff_date = (
        current_date
        - pd.Timedelta(days=30)
    )

    recent_results = [
        result
        for date, result in history
        if cutoff_date <= date < current_date
    ]

    if len(recent_results) == 0:
        return 0.5

    return (
        sum(recent_results)
        / len(recent_results)
    )


# ============================================================
# 7. CREATE FEATURE ROWS
# ============================================================

rows = []


# ============================================================
# 8. PROCESS MATCHES
# ============================================================

for _, match in matches.iterrows():

    winner = match["winner_name"]

    loser = match["loser_name"]

    surface = match["surface"]

    current_date = match["tourney_date"]


    # ========================================================
    # OVERALL LAST-5 FORM
    # ========================================================

    winner_history = player_history.get(
        winner,
        []
    )

    loser_history = player_history.get(
        loser,
        []
    )


    winner_last5 = calculate_form(
        winner_history,
        5
    )

    loser_last5 = calculate_form(
        loser_history,
        5
    )


    # ========================================================
    # SURFACE FORM
    # ========================================================

    winner_surface_history = surface_history.get(
        (winner, surface),
        []
    )

    loser_surface_history = surface_history.get(
        (loser, surface),
        []
    )


    if winner_surface_history:

        winner_surface_form = (
            sum(winner_surface_history)
            / len(winner_surface_history)
        )

    else:

        winner_surface_form = 0.5


    if loser_surface_history:

        loser_surface_form = (
            sum(loser_surface_history)
            / len(loser_surface_history)
        )

    else:

        loser_surface_form = 0.5


    # ========================================================
    # H2H
    # ========================================================

    winner_h2h = h2h_history.get(
        (winner, loser),
        0
    )

    loser_h2h = h2h_history.get(
        (loser, winner),
        0
    )


    # ========================================================
    # LAST MONTH FORM
    # ========================================================

    winner_month_history = player_month_history.get(
        winner,
        []
    )

    loser_month_history = player_month_history.get(
        loser,
        []
    )


    winner_last_month = calculate_last_month_form(
        winner_month_history,
        current_date
    )

    loser_last_month = calculate_last_month_form(
        loser_month_history,
        current_date
    )


    # ========================================================
    # WINNER = PLAYER 1
    # ========================================================

    rows.append({

        "data_year": match["data_year"],

        "tourney_date": match["tourney_date"],

        "surface": surface,

        "player_1": winner,

        "player_2": loser,


        # Ranking
        "player_1_rank": match["winner_rank"],

        "player_2_rank": match["loser_rank"],

        "rank_difference": (
            match["winner_rank"]
            - match["loser_rank"]
        ),


        # Ranking points
        "player_1_rank_points": (
            match["winner_rank_points"]
        ),

        "player_2_rank_points": (
            match["loser_rank_points"]
        ),

        "rank_points_difference": (
            match["winner_rank_points"]
            - match["loser_rank_points"]
        ),


        # Last 5
        "player_1_last5_form": winner_last5,

        "player_2_last5_form": loser_last5,


        # Surface
        "player_1_surface_form": winner_surface_form,

        "player_2_surface_form": loser_surface_form,


        # H2H
        "player_1_h2h_wins": winner_h2h,

        "player_2_h2h_wins": loser_h2h,


        # Last month
        "player_1_last_month_form": (
            winner_last_month
        ),

        "player_2_last_month_form": (
            loser_last_month
        ),


        # Target
        "target": 1
    })


    # ========================================================
    # LOSER = PLAYER 1
    # ========================================================

    rows.append({

        "data_year": match["data_year"],

        "tourney_date": match["tourney_date"],

        "surface": surface,

        "player_1": loser,

        "player_2": winner,


        # Ranking
        "player_1_rank": match["loser_rank"],

        "player_2_rank": match["winner_rank"],

        "rank_difference": (
            match["loser_rank"]
            - match["winner_rank"]
        ),


        # Ranking points
        "player_1_rank_points": (
            match["loser_rank_points"]
        ),

        "player_2_rank_points": (
            match["winner_rank_points"]
        ),

        "rank_points_difference": (
            match["loser_rank_points"]
            - match["winner_rank_points"]
        ),


        # Last 5
        "player_1_last5_form": loser_last5,

        "player_2_last5_form": winner_last5,


        # Surface
        "player_1_surface_form": loser_surface_form,

        "player_2_surface_form": winner_surface_form,


        # H2H
        "player_1_h2h_wins": loser_h2h,

        "player_2_h2h_wins": winner_h2h,


        # Last month
        "player_1_last_month_form": (
            loser_last_month
        ),

        "player_2_last_month_form": (
            winner_last_month
        ),


        # Target
        "target": 0
    })


    # ========================================================
    # UPDATE HISTORY AFTER CURRENT MATCH
    # ========================================================

    player_history.setdefault(
        winner,
        []
    ).append(1)

    player_history.setdefault(
        loser,
        []
    ).append(0)


    # Surface history
    surface_history.setdefault(
        (winner, surface),
        []
    ).append(1)

    surface_history.setdefault(
        (loser, surface),
        []
    ).append(0)


    # H2H history
    h2h_history[(winner, loser)] = (
        h2h_history.get(
            (winner, loser),
            0
        ) + 1
    )


    # Last-month history
    player_month_history.setdefault(
        winner,
        []
    ).append(
        (current_date, 1)
    )

    player_month_history.setdefault(
        loser,
        []
    ).append(
        (current_date, 0)
    )


# ============================================================
# 9. CREATE DATAFRAME
# ============================================================

features = pd.DataFrame(rows)


# ============================================================
# 10. TRAINING DATA: 1997-2018
# ============================================================

training_features = features[
    (features["data_year"] >= 1997)
    &
    (features["data_year"] <= 2018)
].copy()


# ============================================================
# 11. TEST DATA: 2019
# ============================================================

test_features = features[
    features["data_year"] == 2019
].copy()


# ============================================================
# 12. CHECK MISSING VALUES
# ============================================================

print("\nTraining missing values:")

print(
    training_features.isnull().sum()
)


print("\nTest missing values:")

print(
    test_features.isnull().sum()
)


# ============================================================
# 13. DISPLAY SHAPES
# ============================================================

print(
    "\nTraining shape:",
    training_features.shape
)

print(
    "Test shape:",
    test_features.shape
)


# ============================================================
# 14. DISPLAY NEW FEATURE
# ============================================================

print("\nLast-month features:")

print(
    training_features[
        [
            "player_1",
            "player_2",
            "player_1_last_month_form",
            "player_2_last_month_form",
            "target"
        ]
    ].head(10)
)


# ============================================================
# 15. SAVE
# ============================================================

training_features.to_csv(
    "../data/training_features_month.csv",
    index=False
)

test_features.to_csv(
    "../data/test_features_2019_month.csv",
    index=False
)


print(
    "\nSaved training file:"
)

print(
    "../data/training_features_month.csv"
)


print(
    "\nSaved test file:"
)

print(
    "../data/test_features_2019_month.csv"
)