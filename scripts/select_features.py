# scripts/select_features.py
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA


def select_features_rf(X, y, top_n):
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X.fillna(0), y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    return importances.sort_values(ascending=False).head(top_n).index.tolist()


def select_features_corr(X, y, top_n):
    corrs = X.fillna(0).corrwith(pd.Series(y))
    return corrs.abs().sort_values(ascending=False).head(top_n).index.tolist()


def select_features_mi(X, y, top_n):
    mi = mutual_info_regression(X.fillna(0), y, random_state=42)
    scores = pd.Series(mi, index=X.columns)
    return scores.sort_values(ascending=False).head(top_n).index.tolist()


def select_features_pca(X, top_n):
    pca = PCA(n_components=top_n, random_state=42)
    X_pca = pca.fit_transform(X.fillna(0))
    cols = [f"pca_{i+1}" for i in range(X_pca.shape[1])]
    return pd.DataFrame(X_pca, columns=cols), cols


def process_dataset(dataset_csv, out_dir, top_ns, target_col="wer", methods=None):
    df = pd.read_csv(dataset_csv)
    if target_col not in df.columns:
        raise ValueError(f"Dataset {dataset_csv} hat keine Spalte '{target_col}'")

    # Zeilen mit fehlendem Zielwert entfernen
    df = df.dropna(subset=[target_col])

    # Numerische Features auswählen, ohne filename und Zielspalte
    X = df.drop(columns=["filename", target_col], errors="ignore").select_dtypes(include=[np.number])

    # Alle anderen WER-Spalten entfernen (nur solche, die mit 'wer_' anfangen)
    wer_cols = [c for c in X.columns if c.startswith("wer_")]
    if wer_cols:
        print(f"Entferne {len(wer_cols)} andere WER-Spalten: {wer_cols}")
        X = X.drop(columns=wer_cols, errors="ignore")

    y = df[target_col].values

    os.makedirs(out_dir, exist_ok=True)

    # Falls keine Methoden angegeben, alle vier verwenden
    if not methods:
        methods = ["rf", "corr", "mi", "pca"]

    for top_n in top_ns:
        print(f"Verarbeite top_n = {top_n} ({', '.join(methods)})")

        if "rf" in methods:
            sel_rf = select_features_rf(X, y, top_n)
            df_rf = df[["filename", target_col] + sel_rf]
            df_rf.to_csv(os.path.join(out_dir, f"dataset_rf_top{top_n}.csv"), index=False)

        if "corr" in methods:
            sel_corr = select_features_corr(X, y, top_n)
            df_corr = df[["filename", target_col] + sel_corr]
            df_corr.to_csv(os.path.join(out_dir, f"dataset_corr_top{top_n}.csv"), index=False)

        if "mi" in methods:
            sel_mi = select_features_mi(X, y, top_n)
            df_mi = df[["filename", target_col] + sel_mi]
            df_mi.to_csv(os.path.join(out_dir, f"dataset_mi_top{top_n}.csv"), index=False)

        if "pca" in methods:
            df_pca, cols_pca = select_features_pca(X, top_n)
            df_pca[target_col] = y
            if "filename" in df.columns:
                df_pca["filename"] = df["filename"].values
                df_pca = df_pca[["filename"] + cols_pca + [target_col]]
            else:
                df_pca = df_pca[cols_pca + [target_col]]
            df_pca.to_csv(os.path.join(out_dir, f"dataset_pca_top{top_n}.csv"), index=False)

        print(f"Top {top_n} gespeichert.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_csv", type=str, required=True, help="Pfad zum Input-Dataset")
    parser.add_argument("--out_dir", type=str, required=True, help="Output-Verzeichnis")
    parser.add_argument("--target_col", type=str, default="wer", help="Name der Zielspalte")
    parser.add_argument("--top_n", type=int, nargs="+", required=True, help="Liste an Feature-Zahlen")
    parser.add_argument("--methods", type=str, nargs="+", default=None,
                        help="Feature-Selection-Methoden: rf, corr, mi, pca")
    args = parser.parse_args()

    process_dataset(args.dataset_csv, args.out_dir, args.top_n, args.target_col, args.methods)