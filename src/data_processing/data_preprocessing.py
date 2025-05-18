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
    Clean and preprocess text data, including term normalization.
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
    
    def normalize_terms(terms):
        if pd.isna(terms) or terms == '':
            return terms
            
        normalization_rules = {
            'autocatalytic': 'autocatalysis',
            'autoinhibitory': 'autoinhibition',
            'autoregulatory': 'autoregulation',
            'autoinducer': 'autoinduction'
        }
        
        normalized_terms = []
        for term in terms.split(','):
            term = term.strip().lower()
            normalized_term = normalization_rules.get(term, term)
            normalized_terms.append(normalized_term)
            
        return ', '.join(sorted(set(normalized_terms)))
    
    # First normalize the Terms
    df['Terms'] = df['Terms'].apply(normalize_terms)
    
    # Then create Text_combined
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
    Create a test set with specific requirements:
    Total 20 samples:
    - 15 labeled samples:
      - 5 samples with terms in text (all if_contain_keyterm=1):
        * 3 single-term samples with different terms
        * 2 multiple-terms samples
      - 10 samples without terms in text:
        * Each findable single term appears at least once
        * Remaining samples must be multiple terms
    - 5 unlabeled samples
    """
    print("\nStep 6: Creating test set")
    
    def check_terms_in_text(row):
        if pd.isna(row['Terms']) or pd.isna(row['Text_combined']):
            return 0
        terms = [term.strip() for term in row['Terms'].split(',')]
        return 1 if any(term.lower() in row['Text_combined'].lower() for term in terms) else 0
    
    def get_term_count(terms_str):
        if pd.isna(terms_str) or terms_str == '':
            return 0
        return len([t for t in terms_str.split(',') if t.strip()])
    
    def get_terms(terms_str):
        if pd.isna(terms_str) or terms_str == '':
            return set()
        return {t.strip() for t in terms_str.split(',') if t.strip()}
    
    # Split into labeled and unlabeled data
    labeled_data = df[(df['Terms'].notna()) & (df['Terms'] != '')].copy()
    unlabeled_data = df[(df['Terms'].isna()) | (df['Terms'] == '')].copy()
    
    # Add term presence indicator and term count for labeled data
    labeled_data['if_contain_keyterm'] = labeled_data.apply(check_terms_in_text, axis=1)
    labeled_data['term_count'] = labeled_data['Terms'].apply(get_term_count)
    labeled_data['terms_set'] = labeled_data['Terms'].apply(get_terms)
    
    # Get all unique terms and separate single terms
    all_terms = set()
    single_terms = set()
    for _, row in labeled_data.iterrows():
        terms = row['terms_set']
        all_terms.update(terms)
        if len(terms) == 1:
            single_terms.update(terms)
    
    # Split data based on term presence in text
    terms_present = labeled_data[labeled_data['if_contain_keyterm'] == 1]
    terms_absent = labeled_data[labeled_data['if_contain_keyterm'] == 0]
    
    # First select 5 samples with terms in text
    # 1. Select 3 single-term samples with different terms
    single_term_present = terms_present[terms_present['term_count'] == 1]
    if len(single_term_present) < 3:
        raise ValueError(f"Insufficient single-term samples with terms in text. Need at least 3")
    
    # Group samples by their terms
    term_groups = {}
    for _, row in single_term_present.iterrows():
        term = next(iter(row['terms_set']))  # Get the single term
        if term not in term_groups:
            term_groups[term] = []
        term_groups[term].append(row)
    
    # Check if we have enough different terms
    if len(term_groups) < 3:
        raise ValueError(f"Insufficient different single terms in samples with terms in text. Need at least 3, but only found {len(term_groups)}")
    
    # Select one sample from each of 3 different terms
    selected_terms = list(term_groups.keys())[:3]
    selected_present_rows = []
    for term in selected_terms:
        # Randomly select one sample for this term
        samples = term_groups[term]
        selected_sample = pd.DataFrame([samples[0] if len(samples) == 1 else samples[np.random.randint(len(samples))]])
        selected_present_rows.append(selected_sample)
    
    selected_present = pd.concat(selected_present_rows)
    selected_present['if_contain_keyterm'] = 1
    
    # 2. Select 2 multiple-terms samples
    multi_term_present = terms_present[terms_present['term_count'] > 1]
    if len(multi_term_present) < 2:
        raise ValueError(f"Insufficient multiple-terms samples with terms in text. Need at least 2")
    additional_present = multi_term_present.sample(n=2, random_state=random_seed)
    additional_present['if_contain_keyterm'] = 1  # All samples with terms in text should have if_contain_keyterm=1
    selected_present = pd.concat([selected_present, additional_present])
    
    # Now select 10 samples without terms in text
    # 1. First try to get one sample for each single term
    selected_absent_samples = []
    covered_terms = set()
    
    # Try to get one sample for each single term
    for term in single_terms:
        term_samples = terms_absent[
            (terms_absent['Terms'].str.contains(term, na=False)) &
            (terms_absent['term_count'] == 1)  # Only single-term samples
        ]
        if not term_samples.empty:
            # Select sample that hasn't been used yet
            unused_samples = term_samples[~term_samples['PMID'].isin([s['PMID'] for s in selected_absent_samples])]
            if not unused_samples.empty:
                selected_sample = unused_samples.iloc[0]
                selected_absent_samples.append(selected_sample)
                covered_terms.add(term)
    
    # Report which single terms were not covered
    uncovered_terms = single_terms - covered_terms
    
    # Convert to DataFrame
    selected_absent_df = pd.DataFrame(selected_absent_samples) if selected_absent_samples else pd.DataFrame()
    
    # 2. Fill remaining slots with multiple-terms samples
    remaining_needed = 10 - len(selected_absent_df)
    if remaining_needed > 0:
        multi_term_absent = terms_absent[
            (terms_absent['term_count'] > 1) &  # Multiple terms
            (~terms_absent['PMID'].isin(selected_absent_df['PMID']))  # Not already selected
        ]
        if len(multi_term_absent) < remaining_needed:
            raise ValueError(f"Insufficient multiple-terms samples without terms in text. Need {remaining_needed} more")
        additional_absent = multi_term_absent.sample(n=remaining_needed, random_state=random_seed)
        selected_absent_df = pd.concat([selected_absent_df, additional_absent])
    
    # Select unlabeled samples
    unlabeled_sample = unlabeled_data.sample(n=5, random_state=random_seed)
    unlabeled_sample['if_contain_keyterm'] = 0
    
    # Combine all samples
    test_set = pd.concat([selected_absent_df, selected_present, unlabeled_sample])
    
    # Clean up and reorder columns to ensure if_contain_keyterm is after Abstract
    all_columns = df.columns.tolist()
    abstract_index = all_columns.index('Abstract')
    
    # Create the new column order
    reordered_columns = all_columns[:abstract_index + 1]  # Columns up to and including Abstract
    reordered_columns.append('if_contain_keyterm')  # Add if_contain_keyterm after Abstract
    reordered_columns.extend([col for col in all_columns[abstract_index + 1:] if col != 'if_contain_keyterm'])  # Add remaining columns
    
    # Reorder the columns
    test_set = test_set[reordered_columns]
    
    # Save test set
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed', "test_data.csv")
    test_set.to_csv(output_path, index=False)
    
    print(f"\nTest set created with:")
    print(f"- 15 labeled samples:")
    print(f"  - 5 samples with terms in text (all if_contain_keyterm=1):")
    print(f"    * 3 single-term samples with terms: {', '.join(selected_terms)}")
    print(f"    * 2 multiple-terms samples")
    print(f"  - 10 samples without terms in text:")
    if uncovered_terms:
        print(f"    Note: Could not find samples without terms in text for these single terms: {', '.join(sorted(uncovered_terms))}")
    print(f"    * {len(selected_absent_samples)} single-term samples")
    print(f"    * {10 - len(selected_absent_samples)} multiple-terms samples")
    print(f"- 5 unlabeled samples")
    print(f"Test set saved to {output_path}")
    
    # Return the remaining data
    return df.drop(test_set.index)

def create_balanced_dataset(df, ratio=2, random_seed=None, batch_number=None):
    """
    Process a dataset to create a balanced dataset with a specified ratio of unlabeled to labeled data,
    excluding PMIDs that appear in the test set.
    
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
    
    # Load test data to get test PMIDs
    test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed', "test_data.csv")
    test_pmids = set(pd.read_csv(test_data_path)['PMID'].astype(str))
    
    # Filter out PMIDs that are in test set
    df = df[~df['PMID'].astype(str).isin(test_pmids)].copy()
    
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
    output_path = os.path.join(output_dir, "train_data.csv")
    
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