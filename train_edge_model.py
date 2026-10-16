"""
train_edge_model.py - trains a SMALL XGBoost model for the ESP32 and exports it as model.h.

Uses only features the ESP32 can get itself (no leaf / light, which the server estimates).
Picks the smallest model whose accuracy is within TOLERANCE of your full 7-feature model.

    python train_edge_model.py
Needs export_xgb_to_c.py in the same folder. Output: irrigation_edge_model.json, model.h
"""
import subprocess, sys
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CSV_PATH = "vellore_training_data.csv"
FULL_FEATURES = ["soil", "temp", "hum", "leaf", "light", "reservoir", "rain"]
EDGE_FEATURES = ["soil", "temp", "hum", "reservoir", "rain"]   # order must match the sketch
LABEL = "irrigate"
TOLERANCE = 0.01          # edge model may be at most 1 point less accurate than the full model

df = pd.read_csv(CSV_PATH)
tr, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[LABEL])
tr, val = train_test_split(tr, test_size=0.2, random_state=42, stratify=tr[LABEL])

def fit(feats, n, d, lr=0.1):
    m = xgb.XGBClassifier(n_estimators=n, max_depth=d, learning_rate=lr,
                          subsample=0.9, colsample_bytree=0.9, eval_metric="logloss")
    return m.fit(tr[feats], tr[LABEL])

def acc(m, feats, part):
    return accuracy_score(part[LABEL], m.predict(part[feats]))

# reference: your current model (same settings as train_model.py)
full = fit(FULL_FEATURES, 100, 4)
full_val = acc(full, FULL_FEATURES, val)
print(f"Full 7-feature model   val acc {full_val:.3f}  test acc {acc(full, FULL_FEATURES, test):.3f}")

# smallest edge model that stays within tolerance (size ~ trees * 2^depth)
grid = sorted(((n, d) for n in (10, 20, 30, 50) for d in (2, 3, 4)), key=lambda p: p[0] * 2 ** p[1])
chosen = None
for n, d in grid:
    m = fit(EDGE_FEATURES, n, d)
    a = acc(m, EDGE_FEATURES, val)
    print(f"  edge {n:3d} trees depth {d}: val acc {a:.3f}")
    if a >= full_val - TOLERANCE:
        chosen = (n, d, m)
        break
if chosen is None:
    n, d = grid[-1]; chosen = (n, d, fit(EDGE_FEATURES, n, d))
    print("No small model met the tolerance; using the largest tried. Consider keeping the server model.")
n, d, edge = chosen

preds = edge.predict(test[EDGE_FEATURES])
print(f"\nChosen: {n} trees, depth {d}")
print("Edge model test accuracy:", round(accuracy_score(test[LABEL], preds), 3))
print(classification_report(test[LABEL], preds, target_names=["no_irrigate", "irrigate"]))
print("Confusion matrix (rows=actual, cols=predicted):\n", confusion_matrix(test[LABEL], preds))
agree = (preds == full.predict(test[FULL_FEATURES])).mean()
print(f"Agreement with full model on test set: {agree:.3f}")

edge.save_model("irrigation_edge_model.json")
subprocess.run([sys.executable, "export_xgb_to_c.py", "irrigation_edge_model.json", "model.h"], check=True)
print("\nIn the sketch: float x[N_FEATURES] = {soil, temp, hum, reservoir, rain};  // this exact order")