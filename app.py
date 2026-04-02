import os

import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

from micro_pricing import compute_risk_only

app = Flask(__name__)

_BASE = os.path.dirname(os.path.abspath(__file__))
_bundle = joblib.load(os.path.join(_BASE, "premium_model.pkl"))
_model = _bundle["model"]


def _premium_from_model(rainfall: float, aqi: float, area_risk: int, past_disruptions: int) -> str:
    sample = pd.DataFrame(
        [
            {
                "Rainfall": rainfall,
                "AQI": aqi,
                "Area_Risk": area_risk,
                "Past_Disruptions": past_disruptions,
            }
        ]
    )
    raw = float(_model.predict(sample)[0])
    rounded = int(round(raw))
    rounded = max(20, min(70, rounded))
    return f"₹{rounded}"

_cors = os.environ.get("CORS_ORIGINS", "*").strip()
if _cors == "*":
    CORS(app)
else:
    CORS(app, origins=[o.strip() for o in _cors.split(",") if o.strip()])

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Weekly micro-insurance premium API",
        "status": "running",
    })

@app.route("/predict-price", methods=["POST"])
def predict():
    data = request.get_json()

    required = ["Rainfall", "AQI", "Area_Risk", "Past_Disruptions"]
    missing  = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": "Missing fields", "missing": missing}), 400

    rainfall = float(data["Rainfall"])
    aqi = float(data["AQI"])
    area_risk = int(data["Area_Risk"])
    past_disruptions = int(data["Past_Disruptions"])

    premium_str = _premium_from_model(rainfall, aqi, area_risk, past_disruptions)
    risk = compute_risk_only(rainfall, aqi, area_risk)

    return jsonify({"premium": premium_str, "risk": risk})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
