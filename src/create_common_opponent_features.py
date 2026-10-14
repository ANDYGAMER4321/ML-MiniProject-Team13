import pandas as pd
import glob
import os


# ============================================================
# 1. LOAD DATA
# ============================================================

files = glob.glob("../data/atp_matches_*.csv")

files = [
    f for f in files
    if 1997 <= int(
        os.path.basename(f)
        .split("_")[-1]
        .replace(".csv", "")
    ) <= 2019
]

files.sort()

all_matches = []

for file in files:

    year = int(
        os.path.basename(file)
        .split("_")[-1]
        .replace(".csv", "")
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
# 3. SORT
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

player_history = {}

surface_history = {}

tournament_history = {}

h2h_history = {}

# Player -> opponent -> wins/losses
opponent_history = {}


# ============================================================
# 5. HELPER
# ============================================================

def form(history):

    if len(history) == 0:
        return 0.5

    return sum(history) / len(history)


# ============================================================
# 6. COMMON OPPONENT FUNCTION
# ============================================================

def common_opponent_score(
    player_1,
    player_2
):

    opponents_1 = set(
        opponent_history.get(
            player_1,
            {}
        ).keys()
    )

    opponents_2 = set(
        opponent_history.get(
            player_2,
            {}
        ).keys()
    )

    common = opponents_1.intersection(
        opponents_2
    )

    if len(common) == 0:
        return 0.5, 0.5


    player_1_results = []

    player_2_results = []


    for opponent in common:

        player_1_results.extend(
            opponent_history[
                player_1
            ][opponent]
        )

        player_2_results.extend(
            opponent_history[
                player_2
            ][opponent]
        )


    player_1_score = (
        sum(player_1_results)
        / len(player_1_results)
    )

    player_2_score = (
        sum(player_2_results)
        / len(player_2_results)
    )


    return (
        player_1_score,
        player_2_score
    )


# ============================================================
# 7. CREATE ROWS
# ============================================================

rows = []


for _, match in matches.iterrows():

    winner = match["winner_name"]

    loser = match["loser_name"]

    surface = match["surface"]

    tournament = match["tourney_name"]


    # --------------------------------------------------------
    # LAST 5
    # --------------------------------------------------------

    winner_history = player_history.get(
        winner,
        []
    )

    loser_history = player_history.get(
        loser,
        []
    )


    winner_last5 = form(
        winner_history[-5:]
    )

    loser_last5 = form(
        loser_history[-5:]
    )


    # --------------------------------------------------------
    # SURFACE
    # --------------------------------------------------------

    winner_surface_history = surface_history.get(
        (winner, surface),
        []
    )

    loser_surface_history = surface_history.get(
        (loser, surface),
        []
    )


    winner_surface = form(
        winner_surface_history
    )

    loser_surface = form(
        loser_surface_history
    )


    # --------------------------------------------------------
    # TOURNAMENT
    # --------------------------------------------------------

    winner_tournament_history = (
        tournament_history.get(
            (winner, tournament),
            []
        )
    )

    loser_tournament_history = (
        tournament_history.get(
            (loser, tournament),
            []
        )
    )


    winner_tournament = form(
        winner_tournament_history
    )

    loser_tournament = form(
        loser_tournament_history
    )


    # --------------------------------------------------------
    # H2H
    # --------------------------------------------------------

    winner_h2h = h2h_history.get(
        (winner, loser),
        0
    )

    loser_h2h = h2h_history.get(
        (loser, winner),
        0
    )


    # --------------------------------------------------------
    # COMMON OPPONENTS
    # --------------------------------------------------------

    winner_common, loser_common = (
        common_opponent_score(
            winner,
            loser
        )
    )


    # ========================================================
    # WINNER AS PLAYER 1
    # ========================================================

    rows.append({

        "data_year": match["data_year"],

        "tourney_date": match["tourney_date"],

        "surface": surface,

        "player_1": winner,

        "player_2": loser,

        "player_1_rank":
            match["winner_rank"],

        "player_2_rank":
            match["loser_rank"],

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

        "player_1_last5_form":
            winner_last5,

        "player_2_last5_form":
            loser_last5,

        "player_1_surface_form":
            winner_surface,

        "player_2_surface_form":
            loser_surface,

        "player_1_tournament_form":
            winner_tournament,

        "player_2_tournament_form":
            loser_tournament,

        "player_1_h2h_wins":
            winner_h2h,

        "player_2_h2h_wins":
            loser_h2h,

        "player_1_common_opponent_form":
            winner_common,

        "player_2_common_opponent_form":
            loser_common,

        "target": 1
    })


    # ========================================================
    # LOSER AS PLAYER 1
    # ========================================================

    rows.append({

        "data_year": match["data_year"],

        "tourney_date": match["tourney_date"],

        "surface": surface,

        "player_1": loser,

        "player_2": winner,

        "player_1_rank":
            match["loser_rank"],

        "player_2_rank":
            match["winner_rank"],

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

        "player_1_last5_form":
            loser_last5,

        "player_2_last5_form":
            winner_last5,

        "player_1_surface_form":
            loser_surface,

        "player_2_surface_form":
            winner_surface,

        "player_1_tournament_form":
            loser_tournament,

        "player_2_tournament_form":
            winner_tournament,

        "player_1_h2h_wins":
            loser_h2h,

        "player_2_h2h_wins":
            winner_h2h,

        "player_1_common_opponent_form":
            loser_common,

        "player_2_common_opponent_form":
            winner_common,

        "target": 0
    })


    # ========================================================
    # UPDATE PLAYER HISTORY
    # ========================================================

    player_history.setdefault(
        winner,
        []
    ).append(1)

    player_history.setdefault(
        loser,
        []
    ).append(0)


    # ========================================================
    # SURFACE HISTORY
    # ========================================================

    surface_history.setdefault(
        (winner, surface),
        []
    ).append(1)

    surface_history.setdefault(
        (loser, surface),
        []
    ).append(0)


    # ========================================================
    # TOURNAMENT HISTORY
    # ========================================================

    tournament_history.setdefault(
        (winner, tournament),
        []
    ).append(1)

    tournament_history.setdefault(
        (loser, tournament),
        []
    ).append(0)


    # ========================================================
    # H2H
    # ========================================================

    h2h_history[(winner, loser)] = (
        h2h_history.get(
            (winner, loser),
            0
        ) + 1
    )


    # ========================================================
    # OPPONENT HISTORY
    # ========================================================

    opponent_history.setdefault(
        winner,
        {}
    )

    opponent_history.setdefault(
        loser,
        {})


    opponent_history[
        winner
    ].setdefault(
        loser,
        []
    ).append(1)


    opponent_history[
        loser
    ].setdefault(
        winner,
        []
    ).append(0)


# ============================================================
# 8. DATAFRAME
# ============================================================

features = pd.DataFrame(rows)


# ============================================================
# 9. TRAIN / TEST
# ============================================================

training = features[
    features["data_year"] <= 2018
].copy()

testing = features[
    features["data_year"] == 2019
].copy()


# ============================================================
# 10. CHECK
# ============================================================

print(
    "Training shape:",
    training.shape
)

print(
    "Testing shape:",
    testing.shape
)


print("\nCommon-opponent features:")

print(
    training[
        [
            "player_1",
            "player_2",
            "player_1_common_opponent_form",
            "player_2_common_opponent_form",
            "target"
        ]
    ].head(20)
)


print("\nMissing values:")

print(
    training.isnull().sum()
)


# ============================================================
# 11. SAVE
# ============================================================

training.to_csv(
    "../data/training_features_common.csv",
    index=False
)

testing.to_csv(
    "../data/test_features_2019_common.csv",
    index=False
)


print(
    "\nSaved training file:"
)

print(
    "../data/training_features_common.csv"
)


print(
    "\nSaved test file:"
)

print(
    "../data/test_features_2019_common.csv"
)