# scripts/train_tabular_single_target_clean.py
import argparse, os, joblib, json
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

def load_clean_df(path, target, require_clean=True):
    df = pd.read_csv(path, low_memory=False)
    if require_clean and "snr" in df.columns:
        df = df[df["snr"].astype(str).str.lower().eq("clean")].copy()

    drop_cols = {target, "filename", "reference", "hypothesis", "snr"}
    feat_cols = [c for c in df.columns if c not in drop_cols and np.issubdtype(df[c].dtype, np.number)]

    X = df[feat_cols].astype(np.float32).values
    y = df[target].astype(np.float32).values
    groups = df["filename"] if "filename" in df.columns else pd.Series(np.arange(len(df)))
    return df, X, y, groups, feat_cols

def split_grouped(X, y, groups, test_size=0.2, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    (train_idx, val_idx) = next(gss.split(X, y, groups))
    return train_idx, val_idx

def metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae

def main():
    ap = argparse.ArgumentParser(description="Train LGBM & CatBoost only on CLEAN rows (no SNR variants). Saves feature list.")
    ap.add_argument("--dataset", required=True, help="CSV mit sigmos+wavlm (clean+noisy erlaubt)")
    ap.add_argument("--target", required=True, help="z.B. wer_tiny")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--no_filter_clean", action="store_true", help="Falls gesetzt, KEIN Filter auf snr=clean.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"=== Train (clean-only) auf {args.dataset} (Target: {args.target}) ===")

    df, X, y, groups, feat_cols = load_clean_df(args.dataset, args.target, require_clean=not args.no_filter_clean)
    print(f"Datensatz (nach Filter): {df.shape[0]} Zeilen, {len(feat_cols)} Features")
    print(f"Beispiel-Features: {feat_cols[:5]} ...")

    tr_idx, va_idx = split_grouped(X, y, groups)
    X_train, y_train = X[tr_idx], y[tr_idx]
    X_val,   y_val   = X[va_idx], y[va_idx]
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # Featureliste speichern
    feat_path = os.path.join(args.out_dir, f"feature_columns_{args.target}_clean.json")
    with open(feat_path, "w") as f:
        json.dump(feat_cols, f)
    print(f"Featureliste gespeichert unter: {feat_path}")

    # LightGBM
    print("\nTrainiere LightGBM (clean-only) ...")
    lgb = LGBMRegressor(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=48,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=42,
    )
    lgb.fit(X_train, y_train)
    pred_lgb = lgb.predict(X_val)
    r2_l, rmse_l, mae_l = metrics(y_val, pred_lgb)
    print(f"LightGBM (Val): R2={r2_l:.4f}, RMSE={rmse_l:.4f}, MAE={mae_l:.4f}")
    joblib.dump(lgb, os.path.join(args.out_dir, f"lgbm_{args.target}_clean.pkl"))

    # CatBoost
    print("\nTrainiere CatBoost (clean-only) ...")
    cat = CatBoostRegressor(
        loss_function="RMSE",
        iterations=600,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=False
    )
    cat.fit(X_train, y_train)
    pred_cat = cat.predict(X_val)
    r2_c, rmse_c, mae_c = metrics(y_val, pred_cat)
    print(f"CatBoost (Val): R2={r2_c:.4f}, RMSE={rmse_c:.4f}, MAE={mae_c:.4f}")
    joblib.dump(cat, os.path.join(args.out_dir, f"catboost_{args.target}_clean.pkl"))

    pd.DataFrame([
        {"model":"lgbm_clean","r2":r2_l,"rmse":rmse_l,"mae":mae_l},
        {"model":"catboost_clean","r2":r2_c,"rmse":rmse_c,"mae":mae_c},
    ]).to_csv(os.path.join(args.out_dir, f"validation_metrics_{args.target}_clean.csv"), index=False)

if __name__ == "__main__":
    main()