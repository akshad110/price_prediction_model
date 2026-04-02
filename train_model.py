import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np

df = pd.read_csv("dataset.csv")

X = df[["Rainfall", "AQI", "Area_Risk", "Past_Disruptions"]]
y = df["Premium_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

train_preds = model.predict(X_train)
test_preds  = model.predict(X_test)

mae      = mean_absolute_error(y_test, test_preds)
r2       = r2_score(y_test, test_preds)
rmse     = np.sqrt(mean_squared_error(y_test, test_preds))
train_r2 = r2_score(y_train, train_preds)
accuracy = r2 * 100

joblib.dump({"model": model, "mae": mae, "r2": r2, "rmse": rmse}, "premium_model.pkl")

