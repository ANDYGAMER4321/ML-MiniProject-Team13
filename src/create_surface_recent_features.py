import pandas as pd
import glob
import os


# ============================================================
# 1. LOAD MATCHES
# ============================================================

files = glob.glob("../data/atp_matches_*.csv")

files = [
    file for file in files
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
# 2. CLEAN
# ============================================================

required_columns = [
    "winner_name",
    "loser_name",
    "winner_rank",
    "loser_rank",
    "winner_rank_points",
    "loser_rank_points",
    "surface",
    "tourney_name"
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
# 4. HISTORIES
# ============================================================

player_history = {}

surface_history = {}

h2h_history = {}

tournament_history = {}


# ============================================================
# 5. FORM FUNCTION
# ============================================================

def recent_form(history, n=5):

    recent = history[-n:]

    if len(recent) == 0:
        return 0.5

    return sum(recent) / len(recent)


def overall_form(history):

    if len(history) == 0:
        return 0.5

    return sum(history) / len(history)


# ============================================================
# 6. CREATE FEATURES
# ============================================================

rows = []


for _, match in matches.iterrows():

    winner = match["winner_name"]
    loser = match["loser_name"]

    surface = match["surface"]
    tournament = match["tourney_name"]


    # ========================================================
    # OVERALL LAST 5
    # ========================================================

    winner_history = player_history.get(
        winner, []
    )

    loser_history = player_history.get(
        loser, []
    )


    winner_last5 = recent_form(
        winner_history, 5
    )

    loser_last5 = recent_form(
        loser_history, 5
    )


    # ========================================================
    # OVERALL SURFACE FORM
    # ========================================================

    winner_surface_history = surface_history.get(
        (winner, surface), []
    )

    loser_surface_history = surface_history.get(
        (loser, surface), []
    )


    winner_surface_form = overall_form(
        winner_surface_history
    )

    loser_surface_form = overall_form(
        loser_surface_history
    )


    # ========================================================
    # NEW: RECENT SURFACE FORM
    # ========================================================

    winner_surface_recent = recent_form(
        winner_surface_history, 5
    )

    loser_surface_recent = recent_form(
        loser_surface_history, 5
    )


    # ========================================================
    # H2H
    # ========================================================

    winner_h2h = h2h_history.get(
        (winner, loser), 0
    )

    loser_h2h = h2h_history.get(
        (loser, winner), 0
    )


    # ========================================================
    # TOURNAMENT FORM
    # ========================================================

    winner_tournament_history = tournament_history.get(
        (winner, tournament), []
    )

    loser_tournament_history = tournament_history.get(
        (loser, tournament), []
    )


    winner_tournament_form = overall_form(
        winner_tournament_history
    )

    loser_tournament_form = overall_form(
        loser_tournament_history
    )


    # ========================================================
    # WINNER = PLAYER 1
    # ========================================================

    rows.append({

        "data_year": match["data_year"],
        "tourney_date": match["tourney_date"],
        "tourney_name": tournament,
        "surface": surface,

        "player_1": winner,
        "player_2": loser,

        "player_1_rank": match["winner_rank"],
        "player_2_rank": match["loser_rank"],

        "rank_difference":
            match["winner_rank"]
            - match["loser_rank"],

        "player_1_rank_points":
            match["winner_rank_points"],

        "player_2_rank_points":
            match["loser_rank_points"],

        "rank_points_difference":
            match["winner_rank_points"]
            - match["loser_rank_points"],

        # Last 5 overall
        "player_1_last5_form": winner_last5,
        "player_2_last5_form": loser_last5,

        # Overall surface
        "player_1_surface_form":
            winner_surface_form,

        "player_2_surface_form":
            loser_surface_form,

        # NEW recent surface
        "player_1_surface_recent_form":
            winner_surface_recent,

        "player_2_surface_recent_form":
            loser_surface_recent,

        # H2H
        "player_1_h2h_wins": winner_h2h,
        "player_2_h2h_wins": loser_h2h,

        # Tournament
        "player_1_tournament_form":
            winner_tournament_form,

        "player_2_tournament_form":
            loser_tournament_form,

        "target": 1
    })


    # ========================================================
    # LOSER = PLAYER 1
    # ========================================================

    rows.append({

        "data_year": match["data_year"],
        "tourney_date": match["tourney_date"],
        "tourney_name": tournament,
        "surface": surface,

        "player_1": loser,
        "player_2": winner,

        "player_1_rank": match["loser_rank"],
        "player_2_rank": match["winner_rank"],

        "rank_difference":
            match["loser_rank"]
            - match["winner_rank"],

        "player_1_rank_points":
            match["loser_rank_points"],

        "player_2_rank_points":
            match["winner_rank_points"],

        "rank_points_difference":
            match["loser_rank_points"]
            - match["winner_rank_points"],

        # Last 5 overall
        "player_1_last5_form": loser_last5,
        "player_2_last5_form": winner_last5,

        # Overall surface
        "player_1_surface_form":
            loser_surface_form,

        "player_2_surface_form":
            winner_surface_form,

        # NEW recent surface
        "player_1_surface_recent_form":
            loser_surface_recent,

        "player_2_surface_recent_form":
            winner_surface_recent,

        # H2H
        "player_1_h2h_wins": loser_h2h,
        "player_2_h2h_wins": winner_h2h,

        # Tournament
        "player_1_tournament_form":
            loser_tournament_form,

        "player_2_tournament_form":
            winner_tournament_form,

        "target": 0
    })


    # ========================================================
    # UPDATE HISTORY AFTER CURRENT MATCH
    # ========================================================

    player_history.setdefault(
        winner, []
    ).append(1)

    player_history.setdefault(
        loser, []
    ).append(0)


    # Surface history
    surface_history.setdefault(
        (winner, surface), []
    ).append(1)

    surface_history.setdefault(
        (loser, surface), []
    ).append(0)


    # H2H
    h2h_history[(winner, loser)] = (
        h2h_history.get(
            (winner, loser), 0
        ) + 1
    )


    # Tournament history
    tournament_history.setdefault(
        (winner, tournament), []
    ).append(1)

    tournament_history.setdefault(
        (loser, tournament), []
    ).append(0)


# ============================================================
# 7. DATAFRAME
# ============================================================

features = pd.DataFrame(rows)


# ============================================================
# 8. SPLIT TRAIN / TEST
# ============================================================

training_features = features[
    (features["data_year"] >= 1997)
    &
    (features["data_year"] <= 2018)
].copy()


test_features = features[
    features["data_year"] == 2019
].copy()


# ============================================================
# 9. CHECK
# ============================================================

print(
    "\nTraining shape:",
    training_features.shape
)

print(
    "Test shape:",
    test_features.shape
)


print("\nMissing values:")

print(
    training_features.isnull().sum()
)


print("\nRecent surface features:")

print(
    training_features[
        [
            "surface",
            "player_1",
            "player_2",
            "player_1_surface_recent_form",
            "player_2_surface_recent_form",
            "target"
        ]
    ].head(10)
)


# ============================================================
# 10. SAVE
# ============================================================

training_features.to_csv(
    "../data/training_features_surface_recent.csv",
    index=False
)

test_features.to_csv(
    "../data/test_features_2019_surface_recent.csv",
    index=False
)


print(
    "\nSaved training file:"
)

print(
    "../data/training_features_surface_recent.csv"
)


print(
    "\nSaved test file:"
)

print(
    "../data/test_features_2019_surface_recent.csv"
)