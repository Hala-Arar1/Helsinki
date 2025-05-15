import os
import re
import pandas as pd
import numpy as np
import pyreadr

def load_data():
    """
    Load data from RDS files.
    """
    print("Step 1: Loading data from RDS files")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    data_dir = os.path.join(project_root, "data", "raw")
    
    result_pubmed = pyreadr.read_r(os.path.join(data_dir, "pubmed.rds"))
    df_pubmed = list(result_pubmed.values())[0]
    
    result_autoreg = pyreadr.read_r(os.path.join(data_dir, "autoregulatoryDB.rds"))
    df_autoreg = list(result_autoreg.values())[0]
    
    print(f"Loaded PubMed data: {df_pubmed.shape}")
    print(f"Loaded Autoregulatory data: {df_autoreg.shape}")
    
    return df_pubmed, df_autoreg

def merge_datasets(df_pubmed, df_autoreg):
    """
    Extract PMID and merge PubMed and autoregulatory datasets.
    """
    print("\nStep 2: Merging datasets")
    df_autoreg['PMID'] = df_autoreg['RX'].str.extract(r'PubMed=(\d+)')
    df_autoreg['PMID'] = df_autoreg['PMID'].astype(str)
    df_pubmed['PMID'] = df_pubmed['PMID'].astype(str)
    
    df_merged = pd.merge(df_autoreg, df_pubmed, on='PMID', how='left')
    print(f"Merged dataset shape: {df_merged.shape}")
    
    columns_to_keep = ['AC', 'PMID', 'Title', 'Abstract', 'Term_in_RP', 'Term_in_RT', 'Term_in_RC']
    df_selected = df_merged[columns_to_keep].copy()
    print(f"Selected columns dataset shape: {df_selected.shape}")
    
    return df_selected

def consolidate_terms(df):
    """
    Merge terms from different columns into a single column.
    """
    print("\nStep 3: Consolidating terms")
    
    def merge_terms(row):
        cols = ['Term_in_RP', 'Term_in_RT', 'Term_in_RC']
        terms = []
        for col in cols:
            val = row[col]
            if pd.notna(val):
                split_terms = [t.strip() for t in str(val).split(',') if t.strip()]
                terms.extend(split_terms)
        return ', '.join(sorted(set(terms))) if terms else ''
    
    df['Terms'] = df.apply(merge_terms, axis=1)
    df_cleaned = df.drop(columns=['Term_in_RP', 'Term_in_RT', 'Term_in_RC'])
    
    labeled = sum((df_cleaned['Terms'].notna()) & (df_cleaned['Terms'] != ''))
    unlabeled = len(df_cleaned) - labeled
    print(f"Records with terms (labeled): {labeled}")
    print(f"Records without terms (unlabeled): {unlabeled}")
    
    return df_cleaned

def clean_text_data(df):
    """
    Clean and preprocess text data.
    """
    print("\nStep 4: Cleaning text data")
    
    df = df.dropna(subset=['Title', 'Abstract'])
    df = df[
        df['Title'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0) &
        df['Abstract'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)
    ].copy()
    
    def preprocess_text(text):
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        return text
    
    df['Text_combined'] = df['Title'].apply(preprocess_text) + " " + df['Abstract'].apply(preprocess_text)
    print(f"Cleaned dataset shape: {df.shape}")
    return df

def remove_autophosphatase_term(df):
    """
    Remove 'autophosphatase' term from the Terms column.
    """
    print("\nStep 5: Removing autophosphatase term")
    df['Terms'] = df['Terms'].apply(lambda x: ', '.join([t for t in x.split(',') if t.strip() != 'autophosphatase']) if pd.notna(x) else x)
    return df

def create_test_set(df, random_seed=42):
    """
    Create a test set with 10 labeled and 10 unlabeled samples.
    """
    print("\nStep 6: Creating test set")
    labeled_data = df[(df['Terms'].notna()) & (df['Terms'] != '')]
    unlabeled_data = df[(df['Terms'].isna()) | (df['Terms'] == '')]
    
    labeled_test = labeled_data.sample(n=min(10, len(labeled_data)), random_state=random_seed)
    unlabeled_test = unlabeled_data.sample(n=min(10, len(unlabeled_data)), random_state=random_seed)
    
    test_set = pd.concat([labeled_test, unlabeled_test])
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed', "test_data.csv")
    test_set.to_csv(output_path, index=False)
    print(f"Test set saved to {output_path}")
    
    return df.drop(test_set.index)

def create_balanced_dataset(df, ratio=2, random_seed=None, batch_number=None):
    """
    Process a dataset to create a balanced dataset with a specified ratio of unlabeled to labeled data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing labeled and unlabeled data
    ratio : int, default=2
        The ratio of unlabeled to labeled samples in the final dataset
    random_seed : int, optional
        Random seed for reproducibility. If None, shuffling will be completely random.
    batch_number : int, optional
        Batch number for print messages. If None, won't show batch number.
        
    Returns:
    --------
    pandas.DataFrame
        The balanced dataset with specified ratio of labeled to unlabeled samples
    """
    print(f"\nProcessing Batch {batch_number} with ratio {ratio} (random seed: {random_seed})")
    
    # Split into labeled and unlabeled datasets
    labeled = df[(df['Terms'].notna()) & (df['Terms'] != '')].reset_index(drop=True)
    unlabeled = df[(df['Terms'].isna()) | (df['Terms'] == '')].reset_index(drop=True)
    
    # Shuffle both datasets
    labeled = labeled.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    unlabeled = unlabeled.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Calculate required unlabeled samples
    num_labeled = len(labeled)
    num_unlabeled = min(len(unlabeled), num_labeled * ratio)
    
    # Ensure there are enough samples for the ratio
    if num_unlabeled == 0:
        print(f"Insufficient unlabeled samples for batch {batch_number}. Skipping...")
        return pd.DataFrame()
    
    # Select unlabeled samples
    unlabeled_selected = unlabeled.iloc[:num_unlabeled]
    
    # Combine and shuffle final dataset
    balanced_df = pd.concat([labeled, unlabeled_selected], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Print statistics
    print(f"Batch {batch_number}: Labeled samples: {len(labeled)}, Unlabeled samples: {len(unlabeled_selected)}")
    
    return balanced_df

def create_multiple_shuffled_datasets(df, n_shuffles=10, ratio=2):
    """
    Create multiple shuffled datasets with different random seeds.
    """
    print("\nStep 7: Creating multiple shuffled datasets")
    
    seeds = [12345, 67890, 24680, 13579, 98765, 43210, 11111, 33333, 55555, 77777]
    all_shuffled_datasets = []
    
    for batch_idx, seed in enumerate(seeds[:n_shuffles], 1):
        shuffled_df = create_balanced_dataset(df, ratio=ratio, random_seed=seed, batch_number=batch_idx)
        if not shuffled_df.empty:
            shuffled_df['batch_number'] = batch_idx
            all_shuffled_datasets.append(shuffled_df)
    
    # Combine all shuffled datasets
    combined_df = pd.concat(all_shuffled_datasets, ignore_index=True)
    print(f"\nFinal combined dataset shape: {combined_df.shape}")
    
    return combined_df

def save_processed_data(processed_df):
    """
    Save processed data to CSV.
    """
    print("\nStep 8: Saving processed data")
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "shuffled_10_data.csv")
    
    processed_df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

def main():
    """
    Main function to run the entire data processing pipeline.
    """
    print("Starting data processing pipeline")
    
    # Load and preprocess data
    df_pubmed, df_autoreg = load_data()
    df_merged = merge_datasets(df_pubmed, df_autoreg)
    df_consolidated = consolidate_terms(df_merged)
    df_cleaned = clean_text_data(df_consolidated)
    df_no_autophos = remove_autophosphatase_term(df_cleaned)
    df_for_shuffling = create_test_set(df_no_autophos)
    
    # Create and save multiple shuffled datasets
    shuffled_data = create_multiple_shuffled_datasets(df_for_shuffling, n_shuffles=10, ratio=2)
    save_processed_data(shuffled_data)
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()