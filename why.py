import numpy as np, xgboost as xgb
m = xgb.XGBClassifier(); m.load_model("irrigation_model.json")
names = ["soil","temp","hum","leaf","light","reservoir","rain"]
base = [27, 26.5, 75, 91, 0, 91, 24]
p = lambda v: m.predict_proba(np.array([v]))[0][1]

print("now: P(irrigate) =", round(p(base), 3), "(needs 0.5+ to say yes)")
for n, vals in {"leaf":[10,30], "hum":[50,40], "light":[2000], "rain":[0], "temp":[34]}.items():
    for val in vals:
        v = base.copy(); v[names.index(n)] = val
        print(f"  if {n}={val}: P = {p(v):.3f}")

c = m.get_booster().predict(xgb.DMatrix(np.array([base])), pred_contribs=True, validate_features=False)[0]
print("pushes (negative = toward 'no'):")
for n, v in sorted(zip(names, c), key=lambda t: t[1]):
    print(f"  {n:10s} {v:+.2f}")