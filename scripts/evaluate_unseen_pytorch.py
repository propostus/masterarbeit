# scripts/evaluate_unseen_pytorch.py

import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ------------------------------------------------------------
# Definiere MultiOutputRegressor für den sicheren Import
# ------------------------------------------------------------
class MultiOutputRegressor(nn.Module):
    def __init__(self, input_dim, hidden1=512, hidden2=256, hidden3=128, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Linear(hidden3, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------
# Modell sicher laden
# ------------------------------------------------------------
def load_model(model_path, input_dim, device):
    """Lädt automatisch ein komplettes Modellobjekt oder ein state_dict."""

    # explizit erlauben, MultiOutputRegressor beim Unpickling zu verwenden
    torch.serialization.add_safe_globals([MultiOutputRegressor])

    try:
        # Erst versuchen, das vollständige Modell zu laden
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, nn.Module):
            print("→ Lade komplettes gespeichertes Modellobjekt.")
            model = checkpoint.to(device)
            model.eval()
            return model
    except Exception as e:
        print(f"Konnte Modell nicht direkt laden ({e}). Versuche state_dict...")

    # Wenn das fehlschlägt: state_dict laden
    print("→ Lade state_dict und rekonstruiere Modellarchitektur.")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    output_dim = 3
    model = MultiOutputRegressor(input_dim=input_dim, out_dim=output_dim).to(device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------
def evaluate_model(model, X, y, device):
    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy()

    results = []
    for i, target in enumerate(y.columns):
        y_true = y.iloc[:, i].values
        y_pred = preds[:, i]
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        results.append({"target": target, "r2": r2, "rmse": rmse, "mae": mae})

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    print(f"=== Evaluation auf {args.dataset} ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.dataset, low_memory=False)
    print(f"Datensatz: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    target_cols = ["wer_tiny", "wer_base", "wer_small"]
    exclude_cols = target_cols + ["filename", "snr", "reference", "hypothesis"]
    feature_cols = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]

    X = df[feature_cols]
    y = df[target_cols]
    print(f"Verwendete Feature-Dimensionen: {X.shape[1]}")

    model = load_model(args.model_path, input_dim=X.shape[1], device=device)
    results = evaluate_model(model, X, y, device)

    print("\nErgebnisse (Unseen Dataset):")
    print(results.to_string(index=False))

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    results.to_csv(args.out_csv, index=False)
    print(f"\nErgebnisse gespeichert unter: {args.out_csv}")


if __name__ == "__main__":
    main()