import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset.csv")

target_col = "Premium" if "Premium" in df.columns else "Premium_Price"
feature_cols = ["Rainfall", "AQI", "Area_Risk", "Past_Disruptions"]
X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
)
model.fit(X_train, y_train)

train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

mae_train = mean_absolute_error(y_train, train_preds)
mae_test = mean_absolute_error(y_test, test_preds)
r2_train = r2_score(y_train, train_preds)
r2_test = r2_score(y_test, test_preds)
rmse_test = float(np.sqrt(mean_squared_error(y_test, test_preds)))

print("--- Random Forest (weekly Premium) ---")
print(f"Rows: {len(df)} | Target: {target_col}")
print(f"Train MAE:  {mae_train:.3f}")
print(f"Test MAE:   {mae_test:.3f}")
print(f"Train R2:   {r2_train:.4f}  (~{r2_train * 100:.2f}% variance explained)")
print(f"Test R2:    {r2_test:.4f}  (~{r2_test * 100:.2f}% variance explained)")
print(f"Test RMSE:  {rmse_test:.3f}")

joblib.dump(
    {
        "model": model,
        "mae": mae_test,
        "r2": r2_test,
        "rmse": rmse_test,
        "target_col": target_col,
    },
    "premium_model.pkl",
)

print("Saved: premium_model.pkl")
