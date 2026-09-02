"""
train_model.py

Trains an XGBoost binary classifier on vellore_training_data.csv and saves it
to irrigation_model.json, which server.py loads at startup.

Re-run this any time you regenerate the dataset (generate_vellore_data.py) or,
later, once you have real logged sensor data -- just point CSV_PATH at your
real CSV as long as it has the same 7 feature columns + an 'irrigate' label column.
"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CSV_PATH = "vellore_training_data.csv"
FEATURES = ["soil", "temp", "hum", "leaf", "light", "reservoir", "rain"]
LABEL = "irrigate"
MODEL_OUT = "irrigation_model.json"

df = pd.read_csv(CSV_PATH)
X = df[FEATURES]
y = df[LABEL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print()
print(classification_report(y_test, preds, target_names=["no_irrigate", "irrigate"]))
print("Confusion matrix (rows=actual, cols=predicted):")
print(confusion_matrix(y_test, preds))

# feature importance -- useful sanity check that soil/rain dominate, as expected
importances = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
print("\nFeature importances:")
for feat, imp in importances:
    print(f"  {feat:12s} {imp:.3f}")

model.save_model(MODEL_OUT)
print(f"\nSaved model -> {MODEL_OUT}")
