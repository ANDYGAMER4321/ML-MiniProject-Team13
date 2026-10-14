import pandas as pd
import glob
import os


# ============================================================
# 1. LOAD ATP MATCH FILES (1997-2019)
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
# 2. REMOVE MISSING ESSENTIAL DATA
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

before_cleaning = len(matches)

matches = matches.dropna(
    subset=required_columns
)

after_cleaning = len(matches)

print("Matches before cleaning:", before_cleaning)
print("Matches after cleaning:", after_cleaning)
print("Matches removed:", before_cleaning - after_cleaning)


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

player_history = {}

surface_history = {}

h2h_history = {}


# ============================================================
# 5. FORM FUNCTION
# ============================================================

def calculate_form(history, number_of_matches):

    recent_matches = history[-number_of_matches:]

    if len(recent_matches) == 0:
        return 0.5

    return sum(recent_matches) / len(recent_matches)


# ============================================================
# 6. STORE FEATURES
# ============================================================

rows = []


# ============================================================
# 7. PROCESS ALL MATCHES
# ============================================================

for _, match in matches.iterrows():

    winner = match["winner_name"]
    loser = match["loser_name"]
    surface = match["surface"]


    # ========================================================
    # RECENT OVERALL FORM
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


    winner_last10 = calculate_form(
        winner_history,
        10
    )

    loser_last10 = calculate_form(
        loser_history,
        10
    )


    winner_last15 = calculate_form(
        winner_history,
        15
    )

    loser_last15 = calculate_form(
        loser_history,
        15
    )


    winner_last25 = calculate_form(
        winner_history,
        25
    )

    loser_last25 = calculate_form(
        loser_history,
        25
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
    # HEAD-TO-HEAD
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


        # Last 10
        "player_1_last10_form": winner_last10,

        "player_2_last10_form": loser_last10,


        # Last 15
        "player_1_last15_form": winner_last15,

        "player_2_last15_form": loser_last15,


        # Last 25
        "player_1_last25_form": winner_last25,

        "player_2_last25_form": loser_last25,


        # Surface
        "player_1_surface_form": winner_surface_form,

        "player_2_surface_form": loser_surface_form,


        # H2H
        "player_1_h2h_wins": winner_h2h,

        "player_2_h2h_wins": loser_h2h,


        # Winner
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


        # Last 10
        "player_1_last10_form": loser_last10,

        "player_2_last10_form": winner_last10,


        # Last 15
        "player_1_last15_form": loser_last15,

        "player_2_last15_form": winner_last15,


        # Last 25
        "player_1_last25_form": loser_last25,

        "player_2_last25_form": winner_last25,


        # Surface
        "player_1_surface_form": loser_surface_form,

        "player_2_surface_form": winner_surface_form,


        # H2H
        "player_1_h2h_wins": loser_h2h,

        "player_2_h2h_wins": winner_h2h,


        # Loser
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


    # H2H
    h2h_history[(winner, loser)] = (
        h2h_history.get(
            (winner, loser),
            0
        ) + 1
    )


# ============================================================
# 8. CREATE DATAFRAME
# ============================================================

features = pd.DataFrame(rows)


# ============================================================
# 9. SELECT 2019 TEST DATA
# ============================================================

test_features = features[
    features["data_year"] == 2019
].copy()


# ============================================================
# 10. CHECK MISSING VALUES
# ============================================================

print("\nMissing values:")

print(
    test_features.isnull().sum()
)


# ============================================================
# 11. DISPLAY INFORMATION
# ============================================================

print(
    "\n2019 test rows:",
    len(test_features)
)

print(
    "\nNumber of columns:",
    len(test_features.columns)
)

print("\nColumns:")

print(
    test_features.columns.tolist()
)


print("\nFirst 10 rows:")

print(
    test_features.head(10)
)


# ============================================================
# 12. TARGET DISTRIBUTION
# ============================================================

print("\nTarget distribution:")

print(
    test_features["target"].value_counts()
)


# ============================================================
# 13. SAVE
# ============================================================

output_path = "../data/test_features_2019_extended.csv"

test_features.to_csv(
    output_path,
    index=False
)

print(
    "\nSaved to:",
    output_path
)