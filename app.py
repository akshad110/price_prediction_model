import os

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)

_cors = os.environ.get("CORS_ORIGINS", "*").strip()
if _cors == "*":
    CORS(app)
else:
    CORS(app, origins=[o.strip() for o in _cors.split(",") if o.strip()])

_BASE = os.path.dirname(os.path.abspath(__file__))
bundle = joblib.load(os.path.join(_BASE, "premium_model.pkl"))
model = bundle["model"]

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message" : "Insurance Premium Prediction API",
        "status"  : "running"
    })

@app.route("/predict-price", methods=["POST"])
def predict():
    data = request.get_json()

    required = ["Rainfall", "AQI", "Area_Risk", "Past_Disruptions"]
    missing  = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": "Missing fields", "missing": missing}), 400

    sample = pd.DataFrame([{
        "Rainfall"        : float(data["Rainfall"]),
        "AQI"             : float(data["AQI"]),
        "Area_Risk"       : int(data["Area_Risk"]),
        "Past_Disruptions": int(data["Past_Disruptions"])
    }])

    predicted_premium = model.predict(sample)[0]

    if predicted_premium < 4000:
        risk = "LOW RISK"
    elif predicted_premium < 8000:
        risk = "MEDIUM RISK"
    else:
        risk = "HIGH RISK"

    return jsonify({
        "Premium Price" : f"Rs. {predicted_premium:,.2f}",
        "Risk Level"    : risk
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
