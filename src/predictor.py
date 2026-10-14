"""Loads the saved Random Forest once and turns two players into a prediction."""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd

from src.prediction_features import (DATA_DIR, FEATURE_COLUMNS, HistoryStore,
                                     build_match_features)

log = logging.getLogger(__name__)

# Result of the unbiased 2019 evaluation (1779 / 2760 matches). Display only.
HISTORICAL_TEST_ACCURACY = 64.46


class Predictor:
    def __init__(self, store: HistoryStore, model_path=None, threshold_path=None):
        model_path = Path(model_path or DATA_DIR / "final_tennis_model.pkl")
        threshold_path = Path(threshold_path or DATA_DIR / "final_threshold.txt")
        self.store = store
        self.model = joblib.load(model_path)
        self.threshold = float(threshold_path.read_text().strip())
        if list(self.model.feature_names_in_) != FEATURE_COLUMNS:
            raise RuntimeError("Model features do not match the feature generator.")
        if list(self.model.classes_) != [0, 1]:
            raise RuntimeError("Unexpected model classes.")

    def predict(self, player1, player2, surface, prediction_date=None, tournament=None) -> dict:
        # Resolve names first so the orientation uses canonical names.
        n1, n2 = self.store.resolve_player(player1), self.store.resolve_player(player2)

        # The evaluation (64.46%) always put the alphabetically-first player
        # in the "player_1" slot, so we do the same. Probabilities are then
        # mapped back to the order the user typed.
        first, second = (n1, n2) if n1.lower() <= n2.lower() else (n2, n1)
        res = build_match_features(first, second, surface, prediction_date, tournament, store=self.store)

        X = pd.DataFrame([res.features], columns=FEATURE_COLUMNS)
        p_first = float(self.model.predict_proba(X)[0][1])          # P(first wins)
        winner = first if p_first >= self.threshold else second

        p1 = p_first if n1 == first else 1.0 - p_first
        return {
            "player1": n1,
            "player2": n2,
            "player1_probability": p1,
            "player2_probability": 1.0 - p1,
            "predicted_winner": winner,
            "threshold": self.threshold,
            "historical_test_accuracy": HISTORICAL_TEST_ACCURACY,
            "surface": res.surface,
            "date": res.prediction_date,
            "notes": res.notes,
            "model_input": {"player_1": first, "player_2": second, "features": res.features},
        }
