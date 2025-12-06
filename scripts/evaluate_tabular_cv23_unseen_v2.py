import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib


def concordance_correlation_coefficient(y_true, y_pred):
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-12)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluiert Tabular v2 (LightGBM + CatBoost) auf Unseen mit Handcrafted."
    )
    parser.add_argument("--unseen_csv", required=True,
                        help="merged_sigmos_wavlm_unseen.csv")
    parser.add_argument("--handcrafted_unseen_csv", required=True,
                        help="handcrafted_audio_features_unseen_filtered.csv")
    parser.add_argument("--models_dir", required=True,
                        help="results/model_exports_cv23_tabular_v2")
    parser.add_argument("--out_csv", required=True,
                        help="Output-Metriken-CSV")
    args = parser.parse_args()

    print(f"Lade Unseen-Basisdaten von: {args.unseen_csv}")
    df_unseen = pd.read_csv(args.unseen_csv, low_memory=False)
    print(f"Unseen-Basis: {df_unseen.shape[0]} Zeilen, {df_unseen.shape[1]} Spalten")

    print(f"Lade Handcrafted-Features von: {args.handcrafted_unseen_csv}")
    df_hand = pd.read_csv(args.handcrafted_unseen_csv)
    print(f"Handcrafted: {df_hand.shape[0]} Zeilen, {df_hand.shape[1]} Spalten")

    df_unseen["filename"] = df_unseen["filename"].astype(str)
    df_hand["filename"] = df_hand["filename"].astype(str)

    df = df_unseen.merge(df_hand, on="filename", how="left")
    print(f"Nach Merge (Unseen + Handcrafted): {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    # Featureliste aus Training laden
    feature_path = os.path.join(args.models_dir, "feature_cols.txt")
    with open(feature_path, "r") as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    print(f"Anzahl Feature-Spalten laut Training: {len(feature_cols)}")

    # Konsistenzcheck
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print("WARNUNG: folgende Features fehlen im Unseen-DF:", missing)

    X = df[feature_cols].astype(np.float32).values
    targets = ["wer_tiny", "wer_base", "wer_small"]

    results = []

    for target in targets:
        y = df[target].astype(np.float32).values

        for model_name in ["LGBM_v2", "CatBoost_v2"]:
            model_path = os.path.join(args.models_dir, f"{model_name}_{target}.pkl")
            if not os.path.exists(model_path):
                print(f"Modell nicht gefunden: {model_path}")
                continue

            print(f"\n=== Evaluation: {model_name} ({target}) ===")
            model = joblib.load(model_path)
            preds = model.predict(X)

            r2 = r2_score(y, preds)
            rmse = mean_squared_error(y, preds, squared=False)
            mae = mean_absolute_error(y, preds)
            ccc = concordance_correlation_coefficient(y, preds)

            print(f"R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.44}, CCC={ccc:.4f}")

            results.append({
                "model": model_name,
                "target": target,
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "ccc": ccc,
            })

    if not results:
        print("Keine Ergebnisse – vermutlich keine Modelle gefunden.")
        return

    res_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    res_df.to_csv(args.out_csv, index=False)
    print(f"\nGesamtübersicht gespeichert unter: {args.out_csv}")
    print(res_df)


if __name__ == "__main__":
    main()