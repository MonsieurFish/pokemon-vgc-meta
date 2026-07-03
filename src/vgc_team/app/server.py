"""Flask server for the live meta-forecast web app.

Routes:
  GET /                      -> the single-page Plotly UI
  GET /api/forecast?lam=     -> forecast from the currently loaded meta (fast)
  GET /api/refresh?source=&days=&lam=  -> re-scrape/re-embed then forecast
"""

from __future__ import annotations

import threading
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from vgc_team.app.service import AppState

app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
# Re-read templates on every request so HTML/JS/CSS edits show up on a plain
# browser refresh — no server restart, no model reload.
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True

_state: AppState | None = None
_lock = threading.Lock()


def get_state() -> AppState:
    global _state
    with _lock:
        if _state is None:
            _state = AppState()
        if _state.embedded is None:
            _state.refresh(source="cached")
        return _state


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/forecast")
def api_forecast():
    lam = float(request.args.get("lam", 0.3))
    return jsonify(get_state().forecast(lam=lam))


@app.route("/api/refresh")
def api_refresh():
    state = get_state()
    source = request.args.get("source", "live")
    days = int(request.args.get("days", 45))
    lam = float(request.args.get("lam", 0.3))
    with _lock:
        state.refresh(source=source, days=days)
    return jsonify(state.forecast(lam=lam))


@app.route("/api/rate", methods=["POST"])
def api_rate():
    body = request.get_json(force=True, silent=True) or {}
    paste = body.get("paste", "")
    lam = float(body.get("lam", 0.3))
    return jsonify(get_state().rate_team(paste, lam))


@app.route("/api/complete", methods=["POST"])
def api_complete():
    body = request.get_json(force=True, silent=True) or {}
    paste = body.get("paste", "")
    lam = float(body.get("lam", 0.3))
    return jsonify(get_state().recommend_completion(paste, lam))
