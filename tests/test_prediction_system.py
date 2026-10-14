"""
Run from the project root:   python -m unittest tests.test_prediction_system -v
"""
import unittest
import warnings
from unittest import mock

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from app import create_app
from src.prediction_features import (FEATURE_COLUMNS, DATA_DIR, HistoryStore, InsufficientDataError,
                                     UnknownPlayerError, ValidationError, build_match_features)
from src.predictor import Predictor

DATE = "2019-12-01"          # just after the last match in the supplied data


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = HistoryStore()
        cls.pred = Predictor(cls.store)


class TestModelAndThreshold(Base):
    def test_model_loads_with_18_features_in_order(self):
        self.assertEqual(len(FEATURE_COLUMNS), 18)
        self.assertEqual(list(self.pred.model.feature_names_in_), FEATURE_COLUMNS)
        file_features = [l.strip() for l in open(DATA_DIR / "final_features.txt") if l.strip()]
        self.assertEqual(file_features, FEATURE_COLUMNS)

    def test_threshold_is_048(self):
        self.assertEqual(self.pred.threshold, 0.48)

    def test_verified_2019_accuracy_is_reproduced(self):
        """Backward compatibility: the saved model + 0.48 still gives 1779/2760."""
        te = pd.read_csv(DATA_DIR / "test_features_2019_surface_h2h.csv")
        rows = []
        for i in range(0, len(te), 2):
            r1, r2 = te.iloc[i], te.iloc[i + 1]
            rows.append(r1 if str(r1.player_1).lower() <= str(r2.player_1).lower() else r2)
        ub = pd.DataFrame(rows)
        p = self.pred.model.predict_proba(ub[FEATURE_COLUMNS])[:, 1]
        correct = int(((p >= 0.48).astype(int) == ub.target.values).sum())
        self.assertEqual((len(ub), correct), (2760, 1779))
        self.assertAlmostEqual(correct / len(ub) * 100, 64.4565, places=3)


class TestFeatureParityWithTrainingCSV(Base):
    """History features rebuilt from raw matches must equal the training pipeline's values."""

    def test_parity_on_sampled_2019_matches(self):
        csv = pd.read_csv(DATA_DIR / "test_features_2019_surface_h2h.csv", parse_dates=["tourney_date"])
        csv = csv[csv.target == 1].drop_duplicates(["tourney_date", "player_1", "player_2"], keep=False)
        key = csv.set_index(["tourney_date", "player_1", "player_2"])
        m = self.store.matches
        sample = m[m.tourney_date.dt.year == 2019].sample(400, random_state=0)
        hist = ["last5_form", "surface_form", "h2h_wins", "common_opponent_form", "surface_h2h"]
        checked = bad = 0
        for _, row in sample.iterrows():
            k = (row.tourney_date, row.winner_name, row.loser_name)
            if k not in key.index:
                continue
            ref = key.loc[k]
            got = build_match_features(row.winner_name, row.loser_name, row.surface, row.tourney_date,
                                       store=self.store, _cutoff_seq=int(row.seq)).features
            checked += 1
            for h in hist:
                for side in ("player_1_", "player_2_"):
                    if abs(float(ref[side + h]) - got[side + h]) > 1e-9:
                        bad += 1
        self.assertGreater(checked, 300)
        self.assertEqual(bad, 0, f"{bad} feature mismatches over {checked} matches")

    def test_tournament_form_parity(self):
        m = self.store.matches
        sample = m[m.tourney_date.dt.year == 2019].sample(300, random_state=1)
        # recompute tournament form independently from the raw match table
        for _, row in sample.iterrows():
            prior = m[m.seq < row.seq]
            def form(p):
                g = prior[(prior.tourney_name == row.tourney_name) & ((prior.winner_name == p) | (prior.loser_name == p))]
                return 0.5 if len(g) == 0 else float((g.winner_name == p).mean())
            got = build_match_features(row.winner_name, row.loser_name, row.surface, row.tourney_date,
                                       row.tourney_name, store=self.store, _cutoff_seq=int(row.seq)).features
            self.assertAlmostEqual(got["player_1_tournament_form"], form(row.winner_name), places=9)
            self.assertAlmostEqual(got["player_2_tournament_form"], form(row.loser_name), places=9)


class TestNoLeakage(Base):
    def test_only_completed_tournaments_are_used(self):
        for d in ("2019-03-20", "2019-07-10", DATE):
            r = build_match_features("Novak Djokovic", "Rafael Nadal", "Hard", d, store=self.store)
            self.assertLessEqual(r.latest_available_from, pd.Timestamp(d))

    def test_features_ignore_matches_after_the_date(self):
        # Truncating the store at the date must not change the features.
        d = pd.Timestamp("2018-06-01")
        full = build_match_features("Novak Djokovic", "Rafael Nadal", "Clay", d, store=self.store).features
        cut = HistoryStore.__new__(HistoryStore)
        cut.__dict__.update(self.store.__dict__)
        cut.by_player = {n: g[g.available_from <= d] for n, g in self.store.by_player.items()}
        cut.by_player = {n: g for n, g in cut.by_player.items() if len(g)}
        self.assertEqual(full, build_match_features("Novak Djokovic", "Rafael Nadal", "Clay", d, store=cut).features)


class TestPredictions(Base):
    def check(self, r):
        self.assertAlmostEqual(r["player1_probability"] + r["player2_probability"], 1.0, places=12)
        self.assertTrue(0 < r["player1_probability"] < 1)
        self.assertIn(r["predicted_winner"], (r["player1"], r["player2"]))

    def test_djokovic_vs_medvedev_and_threshold_rule(self):
        r = self.pred.predict("Novak Djokovic", "Daniil Medvedev", "Hard", DATE)
        self.check(r)
        first = min(r["player1"], r["player2"], key=str.lower)
        p_first = r["player1_probability"] if r["player1"] == first else r["player2_probability"]
        self.assertEqual(r["predicted_winner"], first if p_first >= 0.48 else (r["player2"] if first == r["player1"] else r["player1"]))

    def test_order_independent(self):
        a = self.pred.predict("Roger Federer", "Rafael Nadal", "Clay", DATE)
        b = self.pred.predict("Rafael Nadal", "Roger Federer", "Clay", DATE)
        self.assertAlmostEqual(a["player1_probability"], b["player2_probability"], places=12)
        self.assertEqual(a["predicted_winner"], b["predicted_winner"])

    def test_surfaces_change_the_output(self):
        ps = {s: self.pred.predict("Rafael Nadal", "Roger Federer", s, DATE)["player1_probability"] for s in ("Hard", "Clay", "Grass")}
        self.assertGreater(len(set(round(v, 6) for v in ps.values())), 1, ps)

    def test_dates_change_the_output(self):
        a = self.pred.predict("Novak Djokovic", "Rafael Nadal", "Hard", "2012-01-10")
        b = self.pred.predict("Novak Djokovic", "Rafael Nadal", "Hard", "2018-01-10")
        self.check(a); self.check(b)
        self.assertNotEqual(a["model_input"]["features"], b["model_input"]["features"])

    def test_player_with_few_matches_works_when_enough_history(self):
        self.check(self.pred.predict("Jannik Sinner", "Daniil Medvedev", "Hard", DATE))

    def test_tournament_is_optional_and_used_when_given(self):
        with_t = self.pred.predict("Novak Djokovic", "Rafael Nadal", "Hard", DATE, "Australian Open")
        without = self.pred.predict("Novak Djokovic", "Rafael Nadal", "Hard", DATE)
        self.assertTrue(any("No tournament" in n for n in without["notes"]))
        self.assertFalse(any("No tournament" in n for n in with_t["notes"]))

    def test_probabilities_come_from_predict_proba(self):
        fake = mock.Mock(return_value=np.array([[0.1, 0.9]]))
        with mock.patch.object(self.pred.model, "predict_proba", fake):
            r = self.pred.predict("Novak Djokovic", "Rafael Nadal", "Hard", DATE)
        fake.assert_called_once()
        first = "Novak Djokovic"                      # alphabetically before Rafael Nadal
        self.assertAlmostEqual(r["player1_probability"], 0.9)
        self.assertEqual(r["predicted_winner"], first)


class TestErrors(Base):
    def assertErr(self, exc, *args):
        with self.assertRaises(exc):
            self.pred.predict(*args)

    def test_unknown_player(self):
        self.assertErr(UnknownPlayerError, "Carlos Alcaraz", "Novak Djokovic", "Hard", DATE)   # not in the 2019 data
        self.assertErr(UnknownPlayerError, "Nobody Real", "Novak Djokovic", "Hard", DATE)

    def test_same_player(self):
        self.assertErr(ValidationError, "Novak Djokovic", "novak djokovic", "Hard", DATE)

    def test_invalid_surface_and_date_and_tournament(self):
        self.assertErr(ValidationError, "Novak Djokovic", "Rafael Nadal", "Snow", DATE)
        self.assertErr(ValidationError, "Novak Djokovic", "Rafael Nadal", "Hard", "not-a-date")
        self.assertErr(ValidationError, "Novak Djokovic", "Rafael Nadal", "Hard", DATE, "Made Up Open")

    def test_insufficient_data_message(self):
        with self.assertRaises(InsufficientDataError) as c:
            self.pred.predict("Novak Djokovic", "Rafael Nadal", "Hard", "2026-10-04")           # data ends 2019
        self.assertTrue(str(c.exception).startswith("Insufficient historical data to generate a reliable prediction."))
        self.assertErr(InsufficientDataError, "Novak Djokovic", "Roger Federer", "Hard", "1998-06-01")


class TestFlaskApi(Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = create_app().test_client()

    def post(self, **body):
        return self.client.post("/api/predict", json=body)

    def test_pages_and_search(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        names = [p["name"] for p in self.client.get("/api/players/search?q=djo").get_json()["players"]]
        self.assertIn("Novak Djokovic", names)
        self.assertEqual(self.client.get("/api/players/search?q=").get_json()["players"], [])
        self.assertGreater(self.client.get("/api/players").get_json()["count"], 1000)
        self.assertIn("Australian Open", self.client.get("/api/tournaments").get_json()["tournaments"])
        self.assertTrue(self.client.get("/api/status").get_json()["model_loaded"])

    def test_predict_response_shape(self):
        r = self.post(player1="Novak Djokovic", player2="Rafael Nadal", surface="Hard", date=DATE)
        self.assertEqual(r.status_code, 200)
        j = r.get_json()
        for k in ("player1", "player2", "player1_probability", "player2_probability",
                  "predicted_winner", "threshold", "historical_test_accuracy"):
            self.assertIn(k, j)
        self.assertEqual((j["threshold"], j["historical_test_accuracy"]), (0.48, 64.46))
        self.assertAlmostEqual(j["player1_probability"] + j["player2_probability"], 1.0)

    def test_error_statuses_and_no_tracebacks(self):
        cases = [({"player1": "Carlos Alcaraz", "player2": "Novak Djokovic", "surface": "Hard", "date": DATE}, 404),
                 ({"player1": "Novak Djokovic", "player2": "Novak Djokovic", "surface": "Hard", "date": DATE}, 400),
                 ({"player1": "Novak Djokovic", "player2": "Rafael Nadal", "surface": "Mud", "date": DATE}, 400),
                 ({"player1": "Novak Djokovic", "player2": "Rafael Nadal", "surface": "Hard", "date": "2019-13-45"}, 400),
                 ({"player1": "Novak Djokovic", "player2": "Rafael Nadal", "surface": "Hard", "date": "2026-10-04"}, 422),
                 ({"player1": "", "player2": "Rafael Nadal", "surface": "Hard", "date": DATE}, 400)]
        for body, status in cases:
            r = self.post(**body)
            self.assertEqual(r.status_code, status, body)
            self.assertNotIn("Traceback", r.get_data(as_text=True))
        self.assertEqual(self.client.post("/api/predict", data="garbage").status_code, 400)

    def test_unexpected_exception_is_hidden(self):
        app = create_app()
        with mock.patch.object(app.config["PREDICTOR"], "predict", side_effect=RuntimeError("secret internals")):
            r = app.test_client().post("/api/predict", json={"player1": "Novak Djokovic", "player2": "Rafael Nadal", "surface": "Hard"})
        self.assertEqual(r.status_code, 500)
        self.assertNotIn("secret internals", r.get_data(as_text=True))

    def test_model_loading_failure(self):
        c = create_app(model_path="does_not_exist.pkl").test_client()
        r = c.post("/api/predict", json={"player1": "Novak Djokovic", "player2": "Rafael Nadal", "surface": "Hard"})
        self.assertEqual(r.status_code, 503)
        self.assertFalse(c.get("/api/status").get_json()["model_loaded"])

    def test_data_source_failure(self):
        c = create_app(data_dir="/nonexistent").test_client()
        self.assertEqual(c.get("/api/players/search?q=djo").status_code, 503)
        self.assertEqual(c.post("/api/predict", json={"player1": "a", "player2": "b"}).status_code, 503)


if __name__ == "__main__":
    unittest.main()
