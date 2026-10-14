"""Flask backend for the Tennis Match Predictor."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

try:                                    # optional: loads values from a local .env file
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:                     # pragma: no cover
    pass

from src.prediction_features import HistoryStore, PredictionError
from src.predictor import Predictor

BASE_DIR = Path(__file__).resolve().parent
log = logging.getLogger("tennis-app")


def create_app(model_path=None, threshold_path=None, data_dir=None) -> Flask:
    app = Flask(__name__,
                template_folder=str(BASE_DIR / "web" / "templates"),
                static_folder=str(BASE_DIR / "web" / "static"))

    # ---- load everything ONCE at startup (never per request) -------------
    store = predictor = None
    try:
        store = HistoryStore(data_dir) if data_dir else HistoryStore()
        log.info("Loaded %d matches for %d players", len(store.matches), len(store.names))
    except Exception:
        log.exception("Could not load match history")
    if store is not None:
        try:
            predictor = Predictor(store, model_path, threshold_path)
            log.info("Model loaded (threshold %.2f)", predictor.threshold)
        except Exception:
            log.exception("Could not load the model")
    app.config["STORE"], app.config["PREDICTOR"] = store, predictor

    def error(message: str, status: int):
        return jsonify({"error": message}), status

    # ---- pages -------------------------------------------------------------
    @app.get("/")
    def index():
        return render_template("index.html")

    # ---- read-only data endpoints -----------------------------------------
    @app.get("/api/status")
    def status():
        s = app.config["STORE"]
        return jsonify({
            "model_loaded": app.config["PREDICTOR"] is not None,
            "data_through": f"{s.data_through:%Y-%m-%d}" if s else None,
            "players": len(s.names) if s else 0,
            "matches": len(s.matches) if s else 0,
        })

    @app.get("/api/players")
    def players():
        s = app.config["STORE"]
        if s is None:
            return error("Match data is unavailable.", 503)
        return jsonify({"count": len(s.names), "players": s.names})

    @app.get("/api/players/search")
    def search_players():
        s = app.config["STORE"]
        if s is None:
            return error("Match data is unavailable.", 503)
        return jsonify({"players": s.search(request.args.get("q", ""), limit=10)})

    @app.get("/api/tournaments")
    def tournaments():
        s = app.config["STORE"]
        if s is None:
            return error("Match data is unavailable.", 503)
        return jsonify({"tournaments": s.tournaments()})

    # ---- prediction ---------------------------------------------------------
    @app.post("/api/predict")
    def predict():
        pred = app.config["PREDICTOR"]
        if pred is None:
            return error("The prediction model is unavailable right now.", 503)

        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return error("Send a JSON body with player1, player2, surface and date.", 400)
        p1, p2 = str(body.get("player1") or "").strip(), str(body.get("player2") or "").strip()
        if not p1:
            return error("Player 1 is required.", 400)
        if not p2:
            return error("Player 2 is required.", 400)

        try:
            result = pred.predict(p1, p2, body.get("surface"), body.get("date"), body.get("tournament") or None)
        except PredictionError as exc:                       # expected, user-facing problems
            return error(str(exc), exc.status)
        except Exception:                                    # anything else: log, don't leak
            log.exception("Unexpected error while predicting")
            return error("Something went wrong while making the prediction. Please try again.", 500)
        return jsonify(result)

    @app.errorhandler(404)
    def not_found(_):
        return error("Not found.", 404) if request.path.startswith("/api/") else ("Page not found", 404)

    @app.errorhandler(405)
    def wrong_method(_):
        return error("Method not allowed.", 405)

    @app.errorhandler(500)
    def server_error(_):
        return error("Something went wrong. Please try again.", 500)

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    application = create_app()
    application.run(host=os.getenv("HOST", "127.0.0.1"),
                    port=int(os.getenv("PORT", "5000")),
                    debug=os.getenv("FLASK_DEBUG") == "1")
