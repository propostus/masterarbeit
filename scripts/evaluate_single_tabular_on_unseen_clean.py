# scripts/evaluate_single_tabular_on_unseen_clean.py
import argparse, joblib, json, os
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def load_features_with_template(path, target, feature_cols_path):
    df = pd.read_csv(path, low_memory=False)

    # Featureliste laden
    with open(feature_cols_path, "r") as f:
        feat_cols = json.load(f)

    # Nicht-numerische Ziel-/Meta-Spalten entfernen (werden sowieso nicht in feat_cols stehen)
    # Aus df nur die feat_cols in der richtigen Reihenfolge nehmen
    missing = [c for c in feat_cols if c not in df.columns]
    extra   = [c for c in df.columns if c not in feat_cols and np.issubdtype(df[c].dtype, np.number)]

    # Fehlende als 0 ergänzen
    for c in missing:
        df[c] = 0.0

    # Jetzt exakt reindexieren
    X = df.reindex(columns=feat_cols).astype(np.float32).values

    # Ziel, falls vorhanden
    y = df[target].astype(np.float32).values if target in df.columns else None

    # Diagnoseausgabe
    if missing:
        print(f"Hinzugefügte fehlende Spalten (als 0): {missing[:5]}{' ...' if len(missing)>5 else ''}")
    if extra:
        print(f"Im Unseen vorhandene zusätzliche numerische Spalten (ignoriert): {extra[:5]}{' ...' if len(extra)>5 else ''}")

    return X, y

def evaluate(model_path, X, y, name):
    model = joblib.load(model_path)
    preds = model.predict(X)
    if y is None:
        print(f"{name}: Vorhersagen erzeugt (kein Ground Truth im Unseen-CSV vorhanden).")
        return None
    r2 = r2_score(y, preds)
    rmse = mean_squared_error(y, preds, squared=False)
    mae = mean_absolute_error(y, preds)
    print(f"{name} Unseen (clean): R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
    return {"model": name, "r2": r2, "rmse": rmse, "mae": mae}

def main():
    ap = argparse.ArgumentParser(description="Evaluate clean-only tabular models on unseen clean dataset using saved feature list.")
    ap.add_argument("--unseen_csv", required=True, help="Unseen-CSV (sigmos+wavlm, clean)")
    ap.add_argument("--target", required=True, help="z.B. wer_tiny")
    ap.add_argument("--feature_cols", required=True, help="Pfad zur gespeicherten Featureliste (JSON)")
    ap.add_argument("--lgbm_model", required=True, help=".pkl")
    ap.add_argument("--catboost_model", required=True, help=".pkl")
    ap.add_argument("--out_csv", required=False, help="Speicherort für Metriken (optional)")
    args = ap.parse_args()

    X_unseen, y_unseen = load_features_with_template(args.unseen_csv, args.target, args.feature_cols)
    print(f"Unseen geladen (ausgerichtet): {X_unseen.shape[0]} Zeilen, {X_unseen.shape[1]} Features")

    rows = []
    r = evaluate(args.lgbm_model, X_unseen, y_unseen, "LightGBM (clean-only)")
    if r: rows.append(r)
    r = evaluate(args.catboost_model, X_unseen, y_unseen, "CatBoost (clean-only)")
    if r: rows.append(r)

    if args.out_csv and rows:
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        print(f"Metriken gespeichert unter: {args.out_csv}")

if __name__ == "__main__":
    main()