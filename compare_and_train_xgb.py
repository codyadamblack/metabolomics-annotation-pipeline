#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import uniform, randint

# ─── SETTINGS ────────────────────────────────────────────────────────────────
INPUT_CSV   = "sig_met_candidates_with_rdkit3d_v2.csv"
OUTPUT_CSV  = "sig_met_candidates_final_scored_ensemble_v2.csv"
TARGET      = "RT"
GROUP       = "uniq_name"
CORRECT_COL = "correct_annotation"

# include your physicochemical + 3D‐shape descriptors
FEATURE_COLS = [
    "logp","logs","polar_surface_area",
    "donor_count","acceptor_count","rotatable_bond_count",
    "C","H","N","O","P","S","H_C_ratio","heteroatom_count",
    "PMI1","PMI2","asphericity"
]

# ─── 1) LOAD DATA ────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)

# ─── 2) PREPARE TRAINING DATA ───────────────────────────────────────────────
# keep only confirmed true hits for RT regression
train_df = (
    df[df[CORRECT_COL] == 1]
      .dropna(subset=FEATURE_COLS + [TARGET, GROUP])
)
X = train_df[FEATURE_COLS]
y = train_df[TARGET]
groups = train_df[GROUP]

# ─── 3) SPLIT INTO TRAIN/TEST (group‐aware) ─────────────────────────────────
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test   = X.iloc[train_idx],   X.iloc[test_idx]
y_train, y_test   = y.iloc[train_idx],   y.iloc[test_idx]
groups_train      = groups.iloc[train_idx]

# ─── 4) TRAIN A RANDOM FOREST ───────────────────────────────────────────────
rf = RandomForestRegressor(n_estimators=200, random_state=0)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

# ─── 5) HYPERPARAM‐RANDOMIZED SEARCH FOR XGBOOST ─────────────────────────────
xgb = XGBRegressor(objective="reg:squarederror", random_state=0)
param_dist = {
    "n_estimators":    randint(50, 501),
    "max_depth":       randint(3, 11),
    "learning_rate":   uniform(0.01, 0.29),
    "subsample":       uniform(0.5, 0.5),
    "colsample_bytree":uniform(0.5, 0.5),
}
cv = GroupKFold(n_splits=min(5, groups_train.nunique()))
rs = RandomizedSearchCV(
    xgb,
    param_distributions=param_dist,
    n_iter=30,
    cv=cv,
    scoring="neg_root_mean_squared_error",
    random_state=0,
    n_jobs=-1
)
rs.fit(X_train, y_train, groups=groups_train)
best_xgb = rs.best_estimator_

xgb_pred = best_xgb.predict(X_test)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))

# ─── 6) ENSEMBLE ON TEST ────────────────────────────────────────────────────
ens_pred = 0.5 * rf_pred + 0.5 * xgb_pred
ens_rmse = np.sqrt(mean_squared_error(y_test, ens_pred))

print(f"RF Test RMSE:    {rf_rmse:.3f}")
print(f"XGB (tuned) RMSE:{xgb_rmse:.3f}")
print(f"Ensemble RMSE:   {ens_rmse:.3f}")

# ─── 7) PREDICT RT FOR ALL CANDIDATES ────────────────────────────────────────
# fill any missing features with zero
X_all   = df[FEATURE_COLS].fillna(0)
rf_all  = rf.predict(X_all)
xgb_all = best_xgb.predict(X_all)
ens_all = 0.5 * rf_all + 0.5 * xgb_all

df["RT_pred"]  = ens_all
df["rt_error"] = np.abs(df["RT_pred"] - df["RT"])
sigma = ens_rmse
df["rt_score"] = np.exp(-0.5 * (df["rt_error"] / sigma) ** 2)

# ─── 8) FINAL COMPOSITE & RANKING ───────────────────────────────────────────
df["composite_score3"] = df["composite_score"] + df["rt_score"]
df["composite_rank3"]  = (
    df.groupby(GROUP)["composite_score3"]
      .rank("dense", ascending=False)
      .astype(int)
)

# ─── 9) SAVE RESULTS ────────────────────────────────────────────────────────
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved scored table → {OUTPUT_CSV}")
