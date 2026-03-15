from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DEFAULT_MODEL_PATH = "model_multi.pkl"
DEFAULT_OUTPUT_PATH = "output.json"
VALID_TICKERS = ["AAPL", "NVDA", "TSLA"]
VALID_HORIZONS = [1, 7, 30]
VALID_MODEL_NAMES = ["logistic_regression", "decision_tree", "xgboost", "llama"]
LLAMA_MODEL_NAME = "llama-3.1-8b-instant"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load trained models, fetch live data, predict, call Groq, and write output.json"
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="ALL",
        help="ALL or comma-separated tickers, e.g. AAPL,NVDA",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=7,
        choices=VALID_HORIZONS,
        help="Prediction horizon in days: 1, 7, or 30",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained combined model file",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="xgboost",
        choices=VALID_MODEL_NAMES,
        help="Which prediction model to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to output JSON",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="6mo",
        help="Yahoo Finance period for fetching recent history, e.g. 3mo, 6mo, 1y",
    )
    return parser.parse_args()


def interactive_prompt() -> tuple[str, int, str, str, str, str]:
    print("\n=== Stock Predictor ===")
    print("Tickers available: AAPL, NVDA, TSLA")
    tickers = input("Choose ticker(s) [ALL/AAPL/NVDA/TSLA or comma separated] (default ALL): ").strip()
    if not tickers:
        tickers = "ALL"

    print("\nPrediction horizon:")
    print("1 = 1 day")
    print("7 = 7 days")
    print("30 = 30 days")
    horizon_raw = input("Choose horizon (default 7): ").strip()

    try:
        horizon = int(horizon_raw) if horizon_raw else 7
    except ValueError:
        horizon = 7

    print("\nPrediction model:")
    print("1 = xgboost")
    print("2 = logistic_regression")
    print("3 = decision_tree")
    print("4 = llama")
    model_raw = input("Choose model (default xgboost): ").strip().lower()

    model_map = {
        "1": "xgboost",
        "2": "logistic_regression",
        "3": "decision_tree",
        "4": "llama",
        "xgboost": "xgboost",
        "logistic_regression": "logistic_regression",
        "decision_tree": "decision_tree",
        "llama": "llama",
    }
    model_name = model_map.get(model_raw, "xgboost")

    model_path = DEFAULT_MODEL_PATH
    output_path = DEFAULT_OUTPUT_PATH
    period = "6mo"

    return tickers, horizon, model_path, model_name, output_path, period


def normalize_requested_tickers(tickers_raw: str) -> List[str]:
    raw = tickers_raw.strip().upper()
    if raw == "ALL":
        return VALID_TICKERS.copy()

    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    invalid = [t for t in tickers if t not in VALID_TICKERS]
    if invalid:
        raise ValueError(f"Invalid tickers: {invalid}. Allowed: {VALID_TICKERS}")
    return tickers


def load_artifact(model_path: str) -> Dict[str, Any]:
    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"model file not found: {path}")

    artifact = joblib.load(path)

    required_keys = {"models", "feature_columns", "base_features", "targets"}
    missing = required_keys - set(artifact.keys())
    if missing:
        raise ValueError(f"model artifact missing keys: {sorted(missing)}")

    return artifact


def fetch_live_data(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        raise ValueError(f"No Yahoo Finance data returned for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    df = df.reset_index()

    expected_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected Yahoo columns for {ticker}: {missing}")

    df["Ticker"] = ticker
    return df


def build_live_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ret_1"] = df["Close"].pct_change(1)
    df["ret_5"] = df["Close"].pct_change(5)
    df["ret_20"] = df["Close"].pct_change(20)

    df["vol_5"] = df["ret_1"].rolling(5).std()
    df["vol_20"] = df["ret_1"].rolling(20).std()

    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["ma_ratio"] = df["ma_5"] / df["ma_20"]

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def build_model_input_row(
    live_df: pd.DataFrame,
    ticker: str,
    feature_columns: List[str],
    base_features: List[str],
) -> tuple[pd.DataFrame, pd.Series]:
    working = live_df.copy()

    missing_base = [c for c in base_features if c not in working.columns]
    if missing_base:
        raise ValueError(f"Missing base feature columns for {ticker}: {missing_base}")

    working = working.dropna(subset=base_features).reset_index(drop=True)
    if working.empty:
        raise ValueError(f"Not enough recent data to compute features for {ticker}")

    latest_row = working.iloc[[-1]].copy()
    as_of_row = latest_row.iloc[0]

    x = latest_row[base_features + ["Ticker"]].copy()
    x = pd.get_dummies(x, columns=["Ticker"], prefix="Ticker")

    for col in feature_columns:
        if col not in x.columns:
            x[col] = 0

    extra_cols = [c for c in x.columns if c not in feature_columns]
    if extra_cols:
        x = x.drop(columns=extra_cols)

    x = x[feature_columns]
    return x, as_of_row


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in LLM response.")
        return json.loads(match.group(0))


def generate_llama_prediction(
    client: Groq | None,
    *,
    ticker: str,
    horizon: int,
    latest_row: pd.Series,
) -> Dict[str, Any]:
    if client is None:
        raise ValueError("GROQ_API_KEY is not set.")

    feature_summary = {
        "Open": float(latest_row.get("Open", 0)),
        "High": float(latest_row.get("High", 0)),
        "Low": float(latest_row.get("Low", 0)),
        "Close": float(latest_row.get("Close", 0)),
        "Volume": float(latest_row.get("Volume", 0)),
        "ret_1": float(latest_row.get("ret_1", 0)),
        "ret_5": float(latest_row.get("ret_5", 0)),
        "ret_20": float(latest_row.get("ret_20", 0)),
        "vol_5": float(latest_row.get("vol_5", 0)),
        "vol_20": float(latest_row.get("vol_20", 0)),
        "ma_5": float(latest_row.get("ma_5", 0)),
        "ma_20": float(latest_row.get("ma_20", 0)),
        "ma_ratio": float(latest_row.get("ma_ratio", 0)),
    }

    prompt = f"""
You are helping with a university stock prediction project.

Ticker: {ticker}
Prediction horizon: {horizon} day(s)

Latest engineered features:
{json.dumps(feature_summary, indent=2)}

Return ONLY valid JSON in this format:
{{
  "direction": "UP" or "DOWN",
  "prob_up": number between 0 and 1,
  "reason": "one short sentence"
}}

Be cautious, neutral, and do not give financial advice.
"""

    response = client.chat.completions.create(
        model=LLAMA_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a careful financial prediction assistant. Return valid JSON only.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
        max_completion_tokens=180,
    )

    text = response.choices[0].message.content.strip()
    parsed = extract_json_object(text)

    direction = str(parsed["direction"]).upper().strip()
    prob_up = float(parsed["prob_up"])
    reason = str(parsed.get("reason", "")).strip()

    if direction not in {"UP", "DOWN"}:
        raise ValueError("LLaMA returned invalid direction.")

    prob_up = max(0.0, min(1.0, prob_up))

    return {
        "direction": direction,
        "prob_up": prob_up,
        "reason": reason or "LLM-based prediction",
    }


def generate_llm_note(
    client: Groq | None,
    *,
    ticker: str,
    horizon: int,
    direction: str,
    prob_up: float,
    expected_return: float | None,
    as_of_date: str,
    model_name: str,
    model_reason: str | None = None,
) -> str:
    if client is None:
        return "Groq note unavailable because GROQ_API_KEY is not set."

    expected_return_text = f"{expected_return:.4f}" if expected_return is not None else "N/A"
    model_reason_text = model_reason if model_reason else "No extra model reason available."

    prompt = (
        f"You are helping explain a stock prediction to a university project user.\n"
        f"Ticker: {ticker}\n"
        f"Model used: {model_name}\n"
        f"Horizon: {horizon} day(s)\n"
        f"Predicted direction: {direction}\n"
        f"Probability of UP: {prob_up:.4f}\n"
        f"Expected return estimate: {expected_return_text}\n"
        f"Model reason: {model_reason_text}\n"
        f"Data as of: {as_of_date}\n\n"
        f"Write 2 short sentences only. Keep it neutral. "
        f"Do not claim certainty. Mention this is model-based and not financial advice."
    )

    try:
        response = client.chat.completions.create(
            model=LLAMA_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You write short, neutral financial notes for dashboard output.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.3,
            max_completion_tokens=120,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Groq note unavailable: {e}"


def append_run(output_path: Path, run_entry: Dict[str, Any]) -> None:
    if output_path.exists():
        try:
            data = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception:
            data = {"runs": []}
    else:
        data = {"runs": []}

    if "runs" not in data or not isinstance(data["runs"], list):
        data = {"runs": []}

    data["runs"].append(run_entry)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_pipeline(
    tickers_raw: str,
    horizon: int,
    model_path: str,
    model_name: str,
    output_path: str,
    period: str,
) -> int:
    artifact = load_artifact(model_path)
    models_by_name: Dict[str, Dict[int, Any]] = artifact["models"]
    feature_columns: List[str] = artifact["feature_columns"]
    base_features: List[str] = artifact["base_features"]

    if model_name != "llama":
        if model_name not in models_by_name:
            raise ValueError(f"No trained model group found for model '{model_name}'")
        models_for_selected_name = models_by_name[model_name]
        if horizon not in models_for_selected_name:
            raise ValueError(f"No trained model found for model '{model_name}' and horizon {horizon}")
        selected_model = models_for_selected_name[horizon]
    else:
        selected_model = None

    requested_tickers = normalize_requested_tickers(tickers_raw)

    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_client = Groq(api_key=groq_api_key) if groq_api_key else None

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    for ticker in requested_tickers:
        try:
            raw_df = fetch_live_data(ticker, period)
            feat_df = build_live_features(raw_df)
            x_latest, latest_row = build_model_input_row(
                feat_df,
                ticker=ticker,
                feature_columns=feature_columns,
                base_features=base_features,
            )

            if model_name == "llama":
                llama_pred = generate_llama_prediction(
                    groq_client,
                    ticker=ticker,
                    horizon=horizon,
                    latest_row=latest_row,
                )
                direction = llama_pred["direction"]
                prob_up = llama_pred["prob_up"]
                pred = 1 if direction == "UP" else 0
                model_reason = llama_pred.get("reason")
            else:
                pred = int(selected_model.predict(x_latest)[0])
                prob_up = None
                if hasattr(selected_model, "predict_proba"):
                    proba = selected_model.predict_proba(x_latest)[0]
                    if len(proba) >= 2:
                        prob_up = float(proba[1])

                direction = "UP" if pred == 1 else "DOWN"
                model_reason = None

            expected_return = None
            as_of_date = pd.to_datetime(latest_row["Date"]).strftime("%Y-%m-%d")

            llm_note = generate_llm_note(
                groq_client,
                ticker=ticker,
                horizon=horizon,
                direction=direction,
                prob_up=prob_up if prob_up is not None else 0.5,
                expected_return=expected_return,
                as_of_date=as_of_date,
                model_name=model_name,
                model_reason=model_reason,
            )

            result = {
                "ticker": ticker,
                "horizon_days": horizon,
                "model_name": model_name,
                "as_of_date": as_of_date,
                "direction": direction,
                "prob_up": prob_up,
                "expected_return": expected_return,
                "model_reason": model_reason,
                "note": llm_note,
                "llm_note": llm_note,
                "source": "Yahoo Finance live/recent market data",
            }
            results.append(result)

        except Exception as e:
            errors.append({"ticker": ticker, "error": str(e)})

    run_entry = {
        "run_at": utc_now_iso(),
        "params": {
            "tickers": requested_tickers,
            "horizon_days": horizon,
            "model_name": model_name,
            "model_path": str(Path(model_path).expanduser().resolve()),
            "period": period,
        },
        "results": results,
        "errors": errors,
    }

    output_path_p = Path(output_path).expanduser().resolve()
    append_run(output_path_p, run_entry)

    print(f"[OK] Wrote {len(results)} prediction(s) to: {output_path_p}")
    if errors:
        print(f"[WARN] {len(errors)} ticker(s) failed. See 'errors' in the JSON.")

    if results:
        print("\nLatest prediction summary:")
        for item in results:
            print(
                f"- {item['ticker']} ({item['horizon_days']}d, {item['model_name']}): {item['direction']}"
                f" | prob_up={item['prob_up']}"
                f" | as_of={item['as_of_date']}"
            )

    return 0


def main() -> int:
    args = parse_args()
    return run_pipeline(
        tickers_raw=args.tickers,
        horizon=args.horizon,
        model_path=args.model_path,
        model_name=args.model_name,
        output_path=args.output,
        period=args.period,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        tickers, horizon, model_path, model_name, output_path, period = interactive_prompt()
        raise SystemExit(
            run_pipeline(
                tickers_raw=tickers,
                horizon=horizon,
                model_path=model_path,
                model_name=model_name,
                output_path=output_path,
                period=period,
            )
        )
    else:
        raise SystemExit(main())