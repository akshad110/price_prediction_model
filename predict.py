import joblib
import pandas as pd

from micro_pricing import compute_risk_only

bundle = joblib.load("premium_model.pkl")
model = bundle["model"]

rainfall = 94
aqi = 336
area_risk = 1
past_disruptions = 0

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
raw = float(model.predict(sample)[0])
premium_val = max(20, min(70, int(round(raw))))
risk = compute_risk_only(rainfall, aqi, area_risk)

print("Premium : Rs.", premium_val, "(ML, clamped 20-70)")
print("Risk    :", risk)
