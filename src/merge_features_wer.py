import pandas as pd

def main():
    # Load both CSVs
    features_df = pd.read_csv("results/subset_50h/features_250728.csv")
    wer_df = pd.read_csv("results/subset_50h/wer_test_50_tiny.csv")
    
    # Merge on 'filename'
    merged_df = pd.merge(features_df, wer_df, on="filename")
    
    # Save df
    merged_df.to_csv("results/subset_50h/merged_features_wer.csv", index=False)

if __name__ == '__main__':
    main()