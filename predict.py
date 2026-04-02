import pandas as pd
import joblib

bundle = joblib.load("premium_model.pkl")
model  = bundle["model"]

rainfall         = 180
aqi              = 320
area_risk        =   4
past_disruptions =   7

sample = pd.DataFrame([{
    "Rainfall"        : rainfall,
    "AQI"             : aqi,
    "Area_Risk"       : area_risk,
    "Past_Disruptions": past_disruptions
}])

predicted_premium = model.predict(sample)[0]

if predicted_premium < 4000:
    risk = "LOW RISK"
elif predicted_premium < 8000:
    risk = "MEDIUM RISK"
else:
    risk = "HIGH RISK"

print(f"Premium Price : Rs. {predicted_premium:,.2f}")
print(f"Risk Level    : {risk}")
