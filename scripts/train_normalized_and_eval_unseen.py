# scripts/train_normalized_and_eval_unseen.py
import os
import argparse
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


def pick_feature_cols(df, target):
    exclude = {target, "filename", "snr", "wer_tiny", "wer_base", "wer_small", "reference", "hypothesis"}
    feat_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
    return feat_cols


def fit_per_snr_scalers(df_train, feat_cols, snr_levels=("clean", "0", "10", "20")):
    """
    Z-Score pro SNR-Stufe. mean/std nur aus dem Trainingssplit.
    Gibt dict: snr -> (mean, std) als np.ndarray.
    """
    scalers = {}
    snr_series = df_train.get("snr")
    if snr_series is None:
        # Falls es keine SNR-Spalte gibt (sollte hier nicht passieren), alles als 'clean' behandeln
        X = df_train[feat_cols].to_numpy(dtype=np.float32)
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
        scalers["clean"] = (mean, std)
        return scalers

    # Sicherstellen, dass snr Strings sind (z.B. 0/10/20 -> "0"/"10"/"20")
    snr_series = snr_series.astype(str)

    for snr in snr_levels:
        subset = df_train.loc[snr_series == snr, feat_cols]
        if subset.empty:
            # Fallback: falls eine Stufe im Train nicht vorkommt, verwende globalen Train-Mittelwert/Std
            X_all = df_train[feat_cols].to_numpy(dtype=np.float32)
            mean = X_all.mean(axis=0)
            std = X_all.std(axis=0) + 1e-8
        else:
            X = subset.to_numpy(dtype=np.float32)
            mean = X.mean(axis=0)
            std = X.std(axis=0) + 1e-8
        scalers[snr] = (mean, std)

    return scalers


def apply_per_snr_scaling(df, feat_cols, scalers, default_snr="clean"):
    """
    Wendet die pro-SNR Z-Score-Normalisierung an.
    Wenn df keine 'snr'-Spalte hat, wird alles mit default_snr normalisiert.
    """
    X = df[feat_cols].to_numpy(dtype=np.float32)
    if "snr" not in df.columns:
        mean, std = scalers.get(default_snr, list(scalers.values())[0])
        return (X - mean) / std

    snr_vec = df["snr"].astype(str).values
    X_scaled = np.empty_like(X, dtype=np.float32)

    # Vektorisiert pro SNR blockweise anwenden
    unique = pd.unique(snr_vec)
    for snr in unique:
        idx = (snr_vec == snr)
        mean, std = scalers.get(snr, scalers.get(default_snr))
        X_scaled[idx] = (X[idx] - mean) / std
    return X_scaled


def metrics(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def main():
    parser = argparse.ArgumentParser(description="Train normalized LGBM & CatBoost (single target) and eval on unseen (normalized with train stats).")
    parser.add_argument("--train_csv", required=True, help="Pfad zum clean+noisy Trainingsdatensatz")
    parser.add_argument("--unseen_csv", required=True, help="Pfad zum Unseen-Datensatz (i.d.R. nur clean)")
    parser.add_argument("--target", required=True, help="Zielspalte, z.B. wer_tiny")
    parser.add_argument("--out_dir", required=True, help="Ausgabeverzeichnis")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"=== Normalized-Training & Unseen-Eval (Target: {args.target}) ===")

    # -------------------------------
    # 1) Daten laden + Split (grouped)
    # -------------------------------
    df = pd.read_csv(args.train_csv, low_memory=False)
    print(f"Train CSV: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' nicht im Trainingsdatensatz gefunden.")

    feat_cols = pick_feature_cols(df, args.target)

    groups = df["filename"] if "filename" in df.columns else np.arange(len(df))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    (train_idx, val_idx) = next(gss.split(df, groups=groups))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val   = df.iloc[val_idx].reset_index(drop=True)

    # -------------------------------
    # 2) Skaler (nur auf Trainingssplit) fitten und anwenden
    # -------------------------------
    print("Fit per-SNR scalers auf Trainingssplit ...")
    scalers = fit_per_snr_scalers(df_train, feat_cols)

    X_train = apply_per_snr_scaling(df_train, feat_cols, scalers)
    y_train = df_train[args.target].to_numpy(dtype=np.float32)
    X_val   = apply_per_snr_scaling(df_val, feat_cols, scalers)
    y_val   = df_val[args.target].to_numpy(dtype=np.float32)

    # -------------------------------
    # 3) Modelle trainieren
    # -------------------------------
    print("\nTrainiere LightGBM (normalized) ...")
    lgb = LGBMRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=48,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=42,
    )
    lgb.fit(X_train, y_train)
    pred_val_lgb = lgb.predict(X_val)
    m_lgb_val = metrics(y_val, pred_val_lgb)
    print(f"LightGBM Val: R2={m_lgb_val['r2']:.4f}, RMSE={m_lgb_val['rmse']:.4f}, MAE={m_lgb_val['mae']:.4f}")

    print("\nTrainiere CatBoost (normalized) ...")
    cat = CatBoostRegressor(
        iterations=600,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=False,
    )
    cat.fit(X_train, y_train)
    pred_val_cat = cat.predict(X_val)
    m_cat_val = metrics(y_val, pred_val_cat)
    print(f"CatBoost Val: R2={m_cat_val['r2']:.4f}, RMSE={m_cat_val['rmse']:.4f}, MAE={m_cat_val['mae']:.4f}")

    # -------------------------------
    # 4) Artefakte speichern (Modelle + Skaler)
    # -------------------------------
    joblib.dump({"scalers": scalers, "feat_cols": feat_cols}, os.path.join(args.out_dir, "snr_scalers.joblib"))
    joblib.dump(lgb, os.path.join(args.out_dir, "lgbm_normalized.joblib"))
    joblib.dump(cat, os.path.join(args.out_dir, "catboost_normalized.joblib"))
    print(f"\nGespeichert unter: {args.out_dir}")

    # -------------------------------
    # 5) Unseen laden, mit TRAIN-Skaler normalisieren, evaluieren
    # -------------------------------
    print(f"\n=== Unseen-Evaluation auf {args.unseen_csv} ===")
    df_unseen = pd.read_csv(args.unseen_csv, low_memory=False)
    print(f"Unseen CSV: {df_unseen.shape[0]} Zeilen, {df_unseen.shape[1]} Spalten")

    # Falls 'snr' fehlt, setze alles auf 'clean'
    if "snr" not in df_unseen.columns:
        df_unseen = df_unseen.copy()
        df_unseen["snr"] = "clean"

    # Warnung, falls Target auf Unseen fehlt (dann nur Vorhersage ohne Metriken)
    have_labels = args.target in df_unseen.columns

    # Feature-Schnittmenge sicherstellen (falls unseen weniger Spalten hat)
    feat_cols_in_unseen = [c for c in df_unseen.columns if c in feat_cols and np.issubdtype(df_unseen[c].dtype, np.number)]
    if set(feat_cols_in_unseen) != set(feat_cols):
        # Align: fehlende Spalten auf 0 setzen (nach Normalisierung macht das wenig Sinn,
        # aber verhindert harte Abbrüche; für strikte Replizierbarkeit sollte man identische Pipelines nutzen)
        for c in feat_cols:
            if c not in df_unseen.columns:
                df_unseen[c] = 0.0
        feat_cols_in_unseen = feat_cols

    X_unseen = apply_per_snr_scaling(df_unseen, feat_cols_in_unseen, scalers, default_snr="clean")

    # Evaluate
    res_rows = []

    # LightGBM
    pred_unseen_lgb = lgb.predict(X_unseen)
    if have_labels:
        y_unseen = df_unseen[args.target].to_numpy(dtype=np.float32)
        m = metrics(y_unseen, pred_unseen_lgb)
        print(f"LightGBM Unseen: R2={m['r2']:.4f}, RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}")
        res_rows.append({"model": "LightGBM (normalized)", **m})

    # CatBoost
    pred_unseen_cat = cat.predict(X_unseen)
    if have_labels:
        y_unseen = df_unseen[args.target].to_numpy(dtype=np.float32)
        m = metrics(y_unseen, pred_unseen_cat)
        print(f"CatBoost Unseen: R2={m['r2']:.4f}, RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}")
        res_rows.append({"model": "CatBoost (normalized)", **m})

    if have_labels:
        out_csv = os.path.join(args.out_dir, f"unseen_metrics_{args.target}_normalized.csv")
        pd.DataFrame(res_rows).to_csv(out_csv, index=False)
        print(f"Unseen-Metriken gespeichert unter: {out_csv}")
    else:
        # Nur Vorhersagen speichern
        out_pred = os.path.join(args.out_dir, f"unseen_predictions_{args.target}_normalized.csv")
        df_pred = pd.DataFrame({"filename": df_unseen.get("filename", pd.Series(np.arange(len(df_unseen)))),
                                "pred": pred_unseen_lgb, "model": "lgbm_normalized"})
        df_pred.to_csv(out_pred, index=False)
        print(f"Unseen-Vorhersagen gespeichert unter: {out_pred}")


if __name__ == "__main__":
    main()