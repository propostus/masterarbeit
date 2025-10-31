# scripts/merge_multiwer_embeddings.py
import pandas as pd

def merge_embeddings_with_multiwer():
    paths = {
        "tiny": "results/datasets/clean_and_noisy/embeddings_sigmos_wavlm_clean_and_noisy_tiny.csv",
        "base": "results/datasets/clean_and_noisy/embeddings_sigmos_wavlm_clean_and_noisy_base.csv",
        "small": "results/datasets/clean_and_noisy/embeddings_sigmos_wavlm_clean_and_noisy_small.csv",
    }

    dfs = {}
    for model, path in paths.items():
        df = pd.read_csv(path)
        dfs[model] = df[["filename", "snr", f"wer_{model}"]]

    merged = pd.read_csv(paths["tiny"])
    merged = merged.drop(columns=["wer_tiny"], errors="ignore")

    merged = merged.merge(dfs["tiny"], on=["filename", "snr"])
    merged = merged.merge(dfs["base"], on=["filename", "snr"])
    merged = merged.merge(dfs["small"], on=["filename", "snr"])

    merged.to_csv("results/datasets/merged_sigmos_wavlm_multiwer.csv", index=False)
    print("✅ Multi-WER-Dataset gespeichert unter results/datasets/merged_sigmos_wavlm_multiwer.csv")

if __name__ == "__main__":
    merge_embeddings_with_multiwer()