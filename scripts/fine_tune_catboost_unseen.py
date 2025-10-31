# scripts/fine_tune_catboost_unseen.py
import argparse
import os
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def main():
    parser = argparse.ArgumentParser(description="Fine-Tune ein trainiertes CatBoost-Modell auf einem Teil des Unseen-Datensatzes")
    parser.add_argument("--unseen_dataset", required=True, help="Pfad zur CSV-Datei mit Features und echten WERs (z.B. merged_sigmos_wavlm_unseen.csv)")
    parser.add_argument("--model_path", required=True, help="Pfad zum gespeicherten CatBoost-Modell (.cbm)")
    parser.add_argument("--out_dir", required=True, help="Pfad zum Ordner für Ergebnisse")
    parser.add_argument("--target", default="wer_tiny", help="Zielspalte für Fine-Tuning (z.B. wer_tiny)")
    parser.add_argument("--fine_tune_ratio", type=float, default=0.1, help="Anteil der Daten, die zum Fine-Tuning verwendet werden (Default: 0.1)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"=== Fine-Tuning CatBoost auf {args.unseen_dataset} (Target: {args.target}) ===")

    # --- Daten laden ---
    df = pd.read_csv(args.unseen_dataset)
    exclude_cols = ["filename", "snr", "wer_tiny", "wer_base", "wer_small", "reference", "hypothesis"]
    feature_cols = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]

    X = df[feature_cols].astype(np.float32)
    y = df[args.target].astype(np.float32)

    # --- Split 10% Fine-Tune / 90% Test ---
    X_ft, X_test, y_ft, y_test = train_test_split(X, y, test_size=(1 - args.fine_tune_ratio), random_state=42)

    # --- Modell laden ---
    model = CatBoostRegressor()
    model.load_model(args.model_path)
    print("Modell geladen")

    # --- Fine-Tuning ---
    print(f"Fine-Tune auf {len(X_ft)} Beispielen...")
    model.fit(X_ft, y_ft, init_model=model, verbose=False)

    # --- Evaluation ---
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)

    print("\n=== Fine-Tuning Ergebnisse (auf 90% Unseen-Test) ===")
    print(f"R2:   {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")

    # --- Speichern ---
    result_path = os.path.join(args.out_dir, f"fine_tune_results_{args.target}.csv")
    pd.DataFrame([{"target": args.target, "r2": r2, "rmse": rmse, "mae": mae}]).to_csv(result_path, index=False)
    print(f"\nErgebnisse gespeichert unter: {result_path}")


if __name__ == "__main__":
    main()