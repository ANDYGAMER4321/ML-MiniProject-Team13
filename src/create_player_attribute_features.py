import pandas as pd
import glob
import os


# ============================================================
# 1. LOAD ALL MATCHES
# ============================================================

files = glob.glob("../data/atp_matches_*.csv")

files = [
    f for f in files
    if 1997 <= int(
        os.path.basename(f).split("_")[-1].replace(".csv", "")
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

required = [
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
    subset=required
)


# ============================================================
# 3. SORT BY DATE
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
tournament_history = {}
h2h_history = {}
opponent_history = {}
surface_h2h_history = {}


# ============================================================
# 5. HELPER
# ============================================================

def form(history):

    if len(history) == 0:
        return 0.5

    return sum(history) / len(history)


def common_opponent_score(player_1, player_2):

    opponents_1 = set(
        opponent_history.get(player_1, {}).keys()
    )

    opponents_2 = set(
        opponent_history.get(player_2, {}).keys()
    )

    common = opponents_1.intersection(opponents_2)

    if len(common) == 0:
        return 0.5, 0.5

    results_1 = []
    results_2 = []

    for opponent in common:

        results_1.extend(
            opponent_history[player_1][opponent]
        )

        results_2.extend(
            opponent_history[player_2][opponent]
        )

    score_1 = sum(results_1) / len(results_1)
    score_2 = sum(results_2) / len(results_2)

    return score_1, score_2


def surface_h2h(player_1, player_2, surface):

    p1 = surface_h2h_history.get(
        (player_1, player_2, surface),
        0
    )

    p2 = surface_h2h_history.get(
        (player_2, player_1, surface),
        0
    )

    total = p1 + p2

    if total == 0:
        return 0.5, 0.5

    return p1 / total, p2 / total


# ============================================================
# 6. CREATE FEATURES
# ============================================================

rows = []


for _, match in matches.iterrows():

    winner = match["winner_name"]
    loser = match["loser_name"]

    surface = match["surface"]
    tournament = match["tourney_name"]


    # --------------------------------------------------------
    # EXISTING FEATURES
    # --------------------------------------------------------

    winner_history = player_history.get(winner, [])
    loser_history = player_history.get(loser, [])

    winner_last5 = form(winner_history[-5:])
    loser_last5 = form(loser_history[-5:])


    winner_surface_history = surface_history.get(
        (winner, surface), []
    )

    loser_surface_history = surface_history.get(
        (loser, surface), []
    )

    winner_surface = form(winner_surface_history)
    loser_surface = form(loser_surface_history)


    winner_tournament_history = tournament_history.get(
        (winner, tournament), []
    )

    loser_tournament_history = tournament_history.get(
        (loser, tournament), []
    )

    winner_tournament = form(
        winner_tournament_history
    )

    loser_tournament = form(
        loser_tournament_history
    )


    winner_h2h = h2h_history.get(
        (winner, loser), 0
    )

    loser_h2h = h2h_history.get(
        (loser, winner), 0
    )


    winner_common, loser_common = (
        common_opponent_score(
            winner,
            loser
        )
    )


    winner_surface_h2h, loser_surface_h2h = (
        surface_h2h(
            winner,
            loser,
            surface
        )
    )


    # ========================================================
    # PLAYER ATTRIBUTES
    # ========================================================

    winner_height = match.get("winner_ht", 0)
    loser_height = match.get("loser_ht", 0)

    winner_seed = match.get("winner_seed", 0)
    loser_seed = match.get("loser_seed", 0)

    winner_hand = match.get("winner_hand", "U")
    loser_hand = match.get("loser_hand", "U")


    # Missing height -> 0
    if pd.isna(winner_height):
        winner_height = 0

    if pd.isna(loser_height):
        loser_height = 0


    # Missing seed -> 0
    if pd.isna(winner_seed):
        winner_seed = 0

    if pd.isna(loser_seed):
        loser_seed = 0


    # Missing hand -> U
    if pd.isna(winner_hand):
        winner_hand = "U"

    if pd.isna(loser_hand):
        loser_hand = "U"


    # --------------------------------------------------------
    # Handedness encoding
    #
    # R = 1
    # L = 0
    # U = -1
    # --------------------------------------------------------

    winner_hand_encoded = {
        "R": 1,
        "L": 0,
        "U": -1
    }.get(winner_hand, -1)


    loser_hand_encoded = {
        "R": 1,
        "L": 0,
        "U": -1
    }.get(loser_hand, -1)


    # ========================================================
    # PLAYER 1 = WINNER
    # ========================================================

    rows.append({

        "data_year": match["data_year"],
        "tourney_date": match["tourney_date"],
        "surface": surface,

        "player_1": winner,
        "player_2": loser,

        "player_1_rank": match["winner_rank"],
        "player_2_rank": match["loser_rank"],

        "rank_difference":
            match["winner_rank"] -
            match["loser_rank"],

        "player_1_rank_points":
            match["winner_rank_points"],

        "player_2_rank_points":
            match["loser_rank_points"],

        "rank_points_difference":
            match["winner_rank_points"] -
            match["loser_rank_points"],

        "player_1_last5_form": winner_last5,
        "player_2_last5_form": loser_last5,

        "player_1_surface_form": winner_surface,
        "player_2_surface_form": loser_surface,

        "player_1_tournament_form": winner_tournament,
        "player_2_tournament_form": loser_tournament,

        "player_1_h2h_wins": winner_h2h,
        "player_2_h2h_wins": loser_h2h,

        "player_1_common_opponent_form":
            winner_common,

        "player_2_common_opponent_form":
            loser_common,

        "player_1_surface_h2h":
            winner_surface_h2h,

        "player_2_surface_h2h":
            loser_surface_h2h,

        # NEW ATTRIBUTES
        "player_1_height": winner_height,
        "player_2_height": loser_height,

        "player_1_seed": winner_seed,
        "player_2_seed": loser_seed,

        "player_1_hand": winner_hand_encoded,
        "player_2_hand": loser_hand_encoded,

        "target": 1
    })


    # ========================================================
    # PLAYER 1 = LOSER
    # ========================================================

    rows.append({

        "data_year": match["data_year"],
        "tourney_date": match["tourney_date"],
        "surface": surface,

        "player_1": loser,
        "player_2": winner,

        "player_1_rank": match["loser_rank"],
        "player_2_rank": match["winner_rank"],

        "rank_difference":
            match["loser_rank"] -
            match["winner_rank"],

        "player_1_rank_points":
            match["loser_rank_points"],

        "player_2_rank_points":
            match["winner_rank_points"],

        "rank_points_difference":
            match["loser_rank_points"] -
            match["winner_rank_points"],

        "player_1_last5_form": loser_last5,
        "player_2_last5_form": winner_last5,

        "player_1_surface_form": loser_surface,
        "player_2_surface_form": winner_surface,

        "player_1_tournament_form": loser_tournament,
        "player_2_tournament_form": winner_tournament,

        "player_1_h2h_wins": loser_h2h,
        "player_2_h2h_wins": winner_h2h,

        "player_1_common_opponent_form":
            loser_common,

        "player_2_common_opponent_form":
            winner_common,

        "player_1_surface_h2h":
            loser_surface_h2h,

        "player_2_surface_h2h":
            winner_surface_h2h,

        # NEW ATTRIBUTES
        "player_1_height": loser_height,
        "player_2_height": winner_height,

        "player_1_seed": loser_seed,
        "player_2_seed": winner_seed,

        "player_1_hand": loser_hand_encoded,
        "player_2_hand": winner_hand_encoded,

        "target": 0
    })


    # ========================================================
    # UPDATE HISTORY
    # ========================================================

    player_history.setdefault(
        winner, []
    ).append(1)

    player_history.setdefault(
        loser, []
    ).append(0)


    surface_history.setdefault(
        (winner, surface), []
    ).append(1)

    surface_history.setdefault(
        (loser, surface), []
    ).append(0)


    tournament_history.setdefault(
        (winner, tournament), []
    ).append(1)

    tournament_history.setdefault(
        (loser, tournament), []
    ).append(0)


    h2h_history[(winner, loser)] = (
        h2h_history.get(
            (winner, loser), 0
        ) + 1
    )


    opponent_history.setdefault(
        winner, {}
    )

    opponent_history.setdefault(
        loser, {}
    )

    opponent_history[winner].setdefault(
        loser, []
    ).append(1)

    opponent_history[loser].setdefault(
        winner, []
    ).append(0)


    # Surface H2H
    surface_h2h_history[
        (winner, loser, surface)
    ] = surface_h2h_history.get(
        (winner, loser, surface),
        0
    ) + 1


    surface_h2h_history[
        (loser, winner, surface)
    ] = surface_h2h_history.get(
        (loser, winner, surface),
        0
    )


# ============================================================
# 7. DATAFRAME
# ============================================================

features = pd.DataFrame(rows)


# ============================================================
# 8. TRAIN / TEST
# ============================================================

training = features[
    features["data_year"] <= 2018
].copy()

testing = features[
    features["data_year"] == 2019
].copy()


# ============================================================
# 9. CHECK
# ============================================================

print(
    "Training shape:",
    training.shape
)

print(
    "Testing shape:",
    testing.shape
)

print("\nPlayer attribute features:")

print(
    training[
        [
            "player_1",
            "player_2",
            "player_1_height",
            "player_2_height",
            "player_1_seed",
            "player_2_seed",
            "player_1_hand",
            "player_2_hand",
            "target"
        ]
    ].head(10)
)


print("\nMissing values:")

print(
    training.isnull().sum()
)


# ============================================================
# 10. SAVE
# ============================================================

training.to_csv(
    "../data/training_features_attributes.csv",
    index=False
)

testing.to_csv(
    "../data/test_features_2019_attributes.csv",
    index=False
)


print(
    "\nSaved training file:"
)

print(
    "../data/training_features_attributes.csv"
)

print(
    "\nSaved test file:"
)

print(
    "../data/test_features_2019_attributes.csv"
)