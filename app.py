from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional
import os

from flask import Flask, flash, jsonify, make_response, redirect, render_template, request, session, url_for


BASE_DIR = Path(__file__).resolve().parent
PREDICT_PY = BASE_DIR / "predict.py"
OUTPUT_JSON = BASE_DIR / "output.json"

ALLOWED_TICKERS = {"AAPL", "NVDA", "TSLA", "ALL"}
ALLOWED_HORIZONS = {1, 7, 30}
ALLOWED_MODELS = {"logistic_regression", "decision_tree", "xgboost", "llama"}

app = Flask(__name__)

# app.secret_key = "change-this-to-a-random-secret"  # needed for session + flash

# # simple in-memory user store for demo (not for production)
# users: Dict[str, Dict[str, str]] = {}

# @app.route("/")
# def home():
#     if "user" not in session:
#         return redirect(url_for("login"))
#     return render_template("index.html")


# @app.route("/login", methods=["GET", "POST"])
# def login():
#     if request.method == "POST":
#         email = request.form.get("email", "").strip()
#         password = request.form.get("password", "").strip()

#         if email in users and users[email]["password"] == password:
#             session["user"] = email
#             return redirect(url_for("home"))
#         else:
#             flash("Invalid email or password")

#     return render_template("login.html")


# @app.route("/signup", methods=["GET", "POST"])
# def signup():
#     if request.method == "POST":
#         username = request.form.get("username", "").strip()
#         email = request.form.get("email", "").strip()
#         password = request.form.get("password", "").strip()
#         confirm_password = request.form.get("confirm_password", "").strip()

#         if not username or not email or not password or not confirm_password:
#             flash("Please fill in all fields")
#             return redirect(url_for("signup"))

#         if password != confirm_password:
#             flash("Passwords do not match")
#             return redirect(url_for("signup"))

#         if email in users:
#             flash("Email already registered")
#             return redirect(url_for("signup"))

#         users[email] = {
#             "username": username,
#             "password": password
#         }

#         flash("Account created successfully. Please login.")
#         return redirect(url_for("login"))

#     return render_template("signup.html")


# @app.route("/logout")
# def logout():
#     session.pop("user", None)
#     flash("You have been logged out")
#     return redirect(url_for("login"))

# @app.get("/__debug_users")
# def __debug_users():
#     # DO NOT use in production, only for testing
#     return jsonify(users)



@app.get("/__debug_paths")
def __debug_paths():
    return jsonify({
        "cwd": str(Path.cwd()),
        "app_root_path": app.root_path,
        "template_folder": app.template_folder,
        "static_folder": app.static_folder,
        "index_exists": (Path(app.root_path) / (app.template_folder or "templates") / "index.html").exists(),
        "index_path": str(Path(app.root_path) / (app.template_folder or "templates") / "index.html"),
    })


@app.get("/__debug_index_len")
def __debug_index_len():
    p = Path(app.root_path) / (app.template_folder or "templates") / "index.html"
    if not p.exists():
        return jsonify({"exists": False, "path": str(p)})
    txt = p.read_text(encoding="utf-8", errors="replace")
    return jsonify({
        "exists": True,
        "path": str(p),
        "length": len(txt),
        "first_120": txt[:120]
    })


def _no_cache(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "updated_at": None, "runs": []}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {"schema_version": 1, "updated_at": None, "runs": []}


def _latest_run(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    runs = data.get("runs")
    if not isinstance(runs, list) or not runs:
        return None
    last = runs[-1]
    return last if isinstance(last, dict) else None


def _pick_result(run: Dict[str, Any], ticker: str, horizon: Optional[int] = None) -> Optional[Dict[str, Any]]:
    results = run.get("results")
    if not isinstance(results, list):
        return None

    t = ticker.strip().upper()
    matched = [
        r for r in results
        if isinstance(r, dict) and str(r.get("ticker", "")).upper() == t
    ]

    if not matched:
        return None

    if horizon is not None:
        for r in matched:
            try:
                if int(r.get("horizon_days", -1)) == int(horizon):
                    return r
            except Exception:
                pass

    return matched[-1]


def _pick_error(run: Dict[str, Any], ticker: str) -> Optional[Dict[str, Any]]:
    errors = run.get("errors")
    if not isinstance(errors, list):
        return None

    t = ticker.strip().upper()
    for e in errors:
        if isinstance(e, dict) and str(e.get("ticker", "")).upper() == t:
            return e
    return None


def _validate_inputs(tickers: str, horizon: int, model_name: str) -> Optional[str]:
    tickers = (tickers or "ALL").strip().upper()
    model_name = (model_name or "").strip().lower()

    if tickers == "ALL":
        ok_tickers = True
    else:
        parts = [p.strip().upper() for p in tickers.split(",") if p.strip()]
        ok_tickers = len(parts) > 0 and all(p in ALLOWED_TICKERS and p != "ALL" for p in parts)

    if not ok_tickers:
        return "Invalid tickers. Use ALL or comma list of: AAPL,NVDA,TSLA"

    if horizon not in ALLOWED_HORIZONS:
        return "Invalid horizon. Use 1, 7, or 30."

    if model_name not in ALLOWED_MODELS:
        return "Invalid model selected."

    return None


def _run_predict_py(tickers: str, horizon: int, model_name: str) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable, str(PREDICT_PY),
        "--tickers", tickers,
        "--horizon", str(horizon),
        "--model-name", model_name,
        "--output", str(OUTPUT_JSON),
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/output.json")
def output_json():
    data = _read_json(OUTPUT_JSON)
    resp = make_response(jsonify(data))
    return _no_cache(resp)


@app.post("/run_predict")
def run_predict():
    body = request.get_json(silent=True) or {}

    tickers = str(body.get("tickers", "ALL")).strip().upper()
    horizon = int(body.get("horizon", 7))
    ticker_for_ui = str(body.get("ticker", "")).strip().upper()
    model_name = str(body.get("model", "xgboost")).strip().lower()

    err = _validate_inputs(tickers, horizon, model_name)
    if err:
        return _no_cache(make_response(jsonify({"ok": False, "error": err}), 400))

    if not PREDICT_PY.exists():
        return _no_cache(make_response(jsonify({"ok": False, "error": "predict.py not found"}), 500))

    proc = _run_predict_py(tickers, horizon, model_name)
    if proc.returncode != 0:
        return _no_cache(make_response(jsonify({
            "ok": False,
            "error": "predict.py failed",
            "stdout": proc.stdout,
            "stderr": proc.stderr
        }), 500))

    data = _read_json(OUTPUT_JSON)
    last = _latest_run(data)

    picked_result = None
    picked_error = None

    if last and ticker_for_ui:
        picked_result = _pick_result(last, ticker_for_ui, horizon)
        picked_error = _pick_error(last, ticker_for_ui)

    target_date = None
    try:
        today = datetime.now().date()
        target_date = (today + timedelta(days=int(horizon))).isoformat()
    except Exception:
        target_date = None

    return _no_cache(make_response(jsonify({
        "ok": True,
        "data": data,
        "latest_run": last,
        "picked": {
            "ticker": ticker_for_ui or None,
            "result": picked_result,
            "error": picked_error,
            "target_date": target_date
        }
    }), 200))


if __name__ == "__main__":
    app.jinja_env.auto_reload = True
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)))
