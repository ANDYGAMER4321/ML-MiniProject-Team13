"""
Feature generation for ANY two players.

Builds the exact 18 features the final Random Forest was trained on, using
the same definitions as src/create_surface_h2h_features.py, but only from
matches that were completed BEFORE the prediction date (no data leakage).

Training-definition notes (kept on purpose so features mean the same thing):
  * Matches without rank, rank points or surface are dropped (same cleaning).
  * Only 1997+ matches are used (the training window starts in 1997).
  * "Form" features use 0.5 when a player has NO relevant history
    (e.g. never played on this surface). That is the trained convention; the
    app reports every such case in `notes` instead of hiding it.
  * surface_h2h is 0.5 for both players in EVERY training row (the training
    script appends a win for both players), so it is always 0.5 here too.
  * Ranking and ranking points come from each player's most recent match
    before the date (the data has no separate weekly rankings file).
"""
from __future__ import annotations

import glob
import os
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

FEATURE_COLUMNS = [
    "player_1_rank", "player_2_rank", "rank_difference",
    "player_1_rank_points", "player_2_rank_points", "rank_points_difference",
    "player_1_last5_form", "player_2_last5_form",
    "player_1_surface_form", "player_2_surface_form",
    "player_1_tournament_form", "player_2_tournament_form",
    "player_1_h2h_wins", "player_2_h2h_wins",
    "player_1_common_opponent_form", "player_2_common_opponent_form",
    "player_1_surface_h2h", "player_2_surface_h2h",
]

FIRST_TRAINING_YEAR = 1997
MIN_PRIOR_MATCHES = int(os.getenv("MIN_PRIOR_MATCHES", "5"))
MAX_RANK_AGE_DAYS = int(os.getenv("MAX_RANK_AGE_DAYS", "365"))

INSUFFICIENT = "Insufficient historical data to generate a reliable prediction."


class PredictionError(Exception):
    """Base class: message is safe to show to the user."""
    status = 400


class ValidationError(PredictionError):
    status = 400


class UnknownPlayerError(PredictionError):
    status = 404


class InsufficientDataError(PredictionError):
    status = 422

    def __init__(self, detail: str = ""):
        super().__init__(f"{INSUFFICIENT} {detail}".strip())


@dataclass
class FeatureResult:
    player1: str
    player2: str
    surface: str
    tournament: str | None
    prediction_date: str
    features: dict
    notes: list = field(default_factory=list)
    matches_used: dict = field(default_factory=dict)
    latest_available_from: pd.Timestamp | None = None


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip().lower()


class HistoryStore:
    """All cleaned ATP matches, indexed per player, loaded once at startup."""

    REQUIRED = ["winner_name", "loser_name", "winner_rank", "loser_rank",
                "winner_rank_points", "loser_rank_points", "surface", "tourney_name"]

    def __init__(self, data_dir: Path | str = DATA_DIR):
        pattern = os.path.join(str(data_dir), "atp_matches_[0-9][0-9][0-9][0-9].csv")
        files = [f for f in sorted(glob.glob(pattern))
                 if int(re.search(r"(\d{4})\.csv$", f).group(1)) >= FIRST_TRAINING_YEAR]
        if not files:
            raise FileNotFoundError(f"No atp_matches_YYYY.csv files found in {data_dir}")

        cols = ["tourney_name", "surface", "tourney_level", "tourney_date", "match_num",
                "winner_name", "loser_name", "winner_rank", "loser_rank",
                "winner_rank_points", "loser_rank_points"]
        m = pd.concat([pd.read_csv(f, usecols=cols) for f in files], ignore_index=True)
        m = m.dropna(subset=self.REQUIRED)                      # same cleaning as training
        m["tourney_date"] = pd.to_datetime(m["tourney_date"].astype(str), format="%Y%m%d")
        m = m.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)
        m["seq"] = m.index

        # A tournament's matches only count as "already played" once the whole
        # event is over. The data has the tournament START date only, so this
        # is the conservative way to guarantee no future match leaks in.
        days = m["tourney_level"].map({"G": 14, "M": 14, "F": 10}).fillna(7)
        m["available_from"] = m["tourney_date"] + pd.to_timedelta(days, unit="D")
        self.matches = m

        base = {"surface": m["surface"], "tourney": m["tourney_name"], "date": m["tourney_date"],
                "available_from": m["available_from"], "seq": m["seq"]}
        winners = pd.DataFrame({**base, "player": m["winner_name"], "opp": m["loser_name"], "win": 1,
                                "rank": m["winner_rank"], "points": m["winner_rank_points"]})
        losers = pd.DataFrame({**base, "player": m["loser_name"], "opp": m["winner_name"], "win": 0,
                               "rank": m["loser_rank"], "points": m["loser_rank_points"]})
        games = pd.concat([winners, losers], ignore_index=True).sort_values("seq", kind="stable")
        self.by_player = {n: g.reset_index(drop=True) for n, g in games.groupby("player", sort=False)}

        self.names = sorted(self.by_player)
        self._norm = {n: _normalise(n) for n in self.names}
        self._lookup = {v: k for k, v in self._norm.items()}
        self.surfaces = sorted(m["surface"].unique())
        self.data_through = m["tourney_date"].max()
        self.first_year = int(m["tourney_date"].dt.year.min())

    # ----- lookups -------------------------------------------------------
    def resolve_player(self, name: str) -> str:
        key = _normalise(name or "")
        if key in self._lookup:
            return self._lookup[key]
        raise UnknownPlayerError(f"Player not found: {name!r}. Pick a name from the suggestions.")

    def search(self, query: str, limit: int = 10) -> list[dict]:
        q = _normalise(query)
        if not q:
            return []
        scored = []
        for name, norm in self._norm.items():
            if q not in norm:
                continue
            if norm.startswith(q):
                rank = 0
            elif any(tok.startswith(q) for tok in norm.split()):
                rank = 1
            else:
                rank = 2
            scored.append((rank, -int(self.by_player[name]["date"].iloc[-1].year), -len(self.by_player[name]), name))
        scored.sort()
        # Best text match first, then most recently active player, then most matches.
        return [{"name": n, "matches": -c, "last_year": -y} for _, y, c, n in scored[:limit]]

    def tournaments(self) -> list[str]:
        latest_year = self.matches["tourney_date"].dt.year.max()
        recent = self.matches[self.matches["tourney_date"].dt.year == latest_year]
        return sorted(recent["tourney_name"].unique())

    def resolve_tournament(self, name: str | None) -> str | None:
        if not name:
            return None
        known = {_normalise(t): t for t in self.matches["tourney_name"].unique()}
        key = _normalise(name)
        if key not in known:
            raise ValidationError(f"Unknown tournament: {name!r}.")
        return known[key]


_default_store: HistoryStore | None = None


def get_store() -> HistoryStore:
    global _default_store
    if _default_store is None:
        _default_store = HistoryStore()
    return _default_store


def parse_date(value) -> pd.Timestamp:
    if value is None or value == "":
        return pd.Timestamp(date.today())
    if isinstance(value, (date, datetime)):
        return pd.Timestamp(value).normalize()
    try:
        return pd.Timestamp(datetime.strptime(str(value).strip(), "%Y-%m-%d"))
    except ValueError:
        raise ValidationError("Invalid date. Use the format YYYY-MM-DD.") from None


def build_match_features(player1, player2, surface, prediction_date, tournament=None,
                         *, store: HistoryStore | None = None, _cutoff_seq: int | None = None) -> FeatureResult:
    """
    Build the 18 model features for `player1` (as player_1) vs `player2`.

    Only matches from tournaments completed before `prediction_date` are used.
    `_cutoff_seq` is a test hook that reproduces the training pipeline's
    match-by-match ordering exactly (used to verify parity with the CSVs).
    """
    store = store or get_store()
    when = parse_date(prediction_date)
    a_name, b_name = store.resolve_player(player1), store.resolve_player(player2)
    if a_name == b_name:
        raise ValidationError("Choose two different players.")

    surface = str(surface or "").strip().title()
    if surface not in store.surfaces:
        raise ValidationError(f"Invalid surface {surface!r}. Choose one of: {', '.join(store.surfaces)}.")
    tournament = store.resolve_tournament(tournament)

    def visible(name):
        g = store.by_player[name]
        if _cutoff_seq is not None:
            return g[g["seq"] < _cutoff_seq]
        return g[g["available_from"] <= when]

    a, b = visible(a_name), visible(b_name)
    data_end = f"{store.data_through:%Y-%m-%d}"
    when_s = f"{when:%Y-%m-%d}"
    notes: list[str] = []

    if _cutoff_seq is None:
        for name, g in ((a_name, a), (b_name, b)):
            if len(g) < MIN_PRIOR_MATCHES:
                raise InsufficientDataError(
                    f"{name} has only {len(g)} completed ATP matches in the data before {when_s} "
                    f"(minimum {MIN_PRIOR_MATCHES}). The data ends {data_end}.")
            age = (when - g.iloc[-1]["date"]).days
            if age > MAX_RANK_AGE_DAYS:
                raise InsufficientDataError(
                    f"{name} has no ranking information within {MAX_RANK_AGE_DAYS} days before {when_s} "
                    f"(latest match in the data: {g.iloc[-1]['date']:%Y-%m-%d}). Add newer atp_matches_YYYY.csv "
                    f"files to the data folder.")

    def neutral(text):
        notes.append(f"{text} The neutral value 0.5 was used (same convention as training).")
        return 0.5

    def last_row(g):
        return g.iloc[-1] if len(g) else None

    ra, rb = last_row(a), last_row(b)
    # `ra`/`rb` can only be None in the parity-test hook (a player's first ever
    # match); real predictions are stopped earlier by MIN_PRIOR_MATCHES.
    nan = float("nan")
    rank_a, rank_b = (float(r["rank"]) if r is not None else nan for r in (ra, rb))
    pts_a, pts_b = (float(r["points"]) if r is not None else nan for r in (ra, rb))

    def last5(g):
        return float(g["win"].tail(5).mean()) if len(g) else 0.5      # training: 0.5 with no history

    last5_a, last5_b = last5(a), last5(b)

    def surface_form(g, name):
        s = g[g["surface"] == surface]
        return float(s["win"].mean()) if len(s) else neutral(f"{name} has no earlier {surface}-court matches in the data.")

    surf_a, surf_b = surface_form(a, a_name), surface_form(b, b_name)

    if tournament is None:
        tour_a = tour_b = neutral("No tournament was selected, so tournament form is unknown.")
    else:
        def tournament_form(g, name):
            t = g[g["tourney"] == tournament]
            return float(t["win"].mean()) if len(t) else neutral(f"{name} has no earlier matches at {tournament}.")
        tour_a, tour_b = tournament_form(a, a_name), tournament_form(b, b_name)

    h2h_a = int(((a["opp"] == b_name) & (a["win"] == 1)).sum())
    h2h_b = int(((b["opp"] == a_name) & (b["win"] == 1)).sum())

    common = set(a["opp"]) & set(b["opp"])
    if common:
        com_a = float(a[a["opp"].isin(common)]["win"].mean())
        com_b = float(b[b["opp"].isin(common)]["win"].mean())
    else:
        com_a = com_b = neutral(f"{a_name} and {b_name} share no earlier opponents in the data.")

    features = {
        "player_1_rank": rank_a, "player_2_rank": rank_b, "rank_difference": rank_a - rank_b,
        "player_1_rank_points": pts_a, "player_2_rank_points": pts_b, "rank_points_difference": pts_a - pts_b,
        "player_1_last5_form": last5_a, "player_2_last5_form": last5_b,
        "player_1_surface_form": surf_a, "player_2_surface_form": surf_b,
        "player_1_tournament_form": tour_a, "player_2_tournament_form": tour_b,
        "player_1_h2h_wins": h2h_a, "player_2_h2h_wins": h2h_b,
        "player_1_common_opponent_form": com_a, "player_2_common_opponent_form": com_b,
        "player_1_surface_h2h": 0.5, "player_2_surface_h2h": 0.5,   # constant in training (see module docstring)
    }
    assert list(features) == FEATURE_COLUMNS

    latest = max((g["available_from"].max() for g in (a, b) if len(g)), default=None)
    return FeatureResult(a_name, b_name, surface, tournament, when_s, features, notes,
                         {a_name: len(a), b_name: len(b)}, latest)
