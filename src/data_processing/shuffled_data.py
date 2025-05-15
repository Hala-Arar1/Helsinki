import os
import re
import pandas as pd
import numpy as np
import pyreadr
from sklearn.preprocessing import MultiLabelBinarizer

def load_data():
    """
    Load data from RDS files.
    
    Returns:
    --------
    tuple
        (pubmed_dataframe, autoregulatory_dataframe)
    """
    print("Step 1: Loading data from RDS files")
    # Define the absolute path to the data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    # Define data directories using absolute paths
    data_dir = os.path.join(project_root, "data", "raw")
    
    # Load pubmed data
    result_pubmed = pyreadr.read_r(os.path.join(data_dir, "pubmed.rds"))
    df_pubmed = list(result_pubmed.values())[0]
    
    # Load autoregulatory data
    result_autoreg = pyreadr.read_r(os.path.join(data_dir, "autoregulatoryDB.rds"))
    df_autoreg = list(result_autoreg.values())[0]
    
    print(f"Loaded PubMed data: {df_pubmed.shape}")
    print(f"Loaded Autoregulatory data: {df_autoreg.shape}")
    
    return df_pubmed, df_autoreg

def merge_datasets(df_pubmed, df_autoreg):
    """
    Extract PMID and merge PubMed and autoregulatory datasets.
    
    Parameters:
    -----------
    df_pubmed : pandas.DataFrame
        PubMed dataframe
    df_autoreg : pandas.DataFrame
        Autoregulatory dataframe
        
    Returns:
    --------
    pandas.DataFrame
        Merged dataframe with selected columns
    """
    print("\nStep 2: Merging datasets")
    # Extract PMID from RX column
    df_autoreg['PMID'] = df_autoreg['RX'].str.extract(r'PubMed=(\d+)')
    
    # Convert to string for proper matching
    df_autoreg['PMID'] = df_autoreg['PMID'].astype(str)
    df_pubmed['PMID'] = df_pubmed['PMID'].astype(str)
    
    # Merge datasets on PMID
    df_merged = pd.merge(df_autoreg, df_pubmed, on='PMID', how='left')
    print(f"Merged dataset shape: {df_merged.shape}")
    
    # Select only relevant columns
    columns_to_keep = ['AC', 'PMID', 'Title', 'Abstract', 'Term_in_RP', 'Term_in_RT', 'Term_in_RC']
    df_selected = df_merged[columns_to_keep].copy()
    print(f"Selected columns dataset shape: {df_selected.shape}")
    
    return df_selected

def consolidate_terms(df):
    """
    Merge terms from different columns into a single column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with term columns
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with consolidated terms
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
    
    # Count labeled and unlabeled records
    labeled = sum((df_cleaned['Terms'].notna()) & (df_cleaned['Terms'] != ''))
    unlabeled = len(df_cleaned) - labeled
    print(f"Records with terms (labeled): {labeled}")
    print(f"Records without terms (unlabeled): {unlabeled}")
    
    return df_cleaned

def clean_text_data(df):
    """
    Clean and preprocess text data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with text columns
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with cleaned text data
    """
    print("\nStep 4: Cleaning text data")
    
    # Drop rows with missing or empty Title or Abstract
    df = df.dropna(subset=['Title', 'Abstract'])
    df = df[
        df['Title'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0) &
        df['Abstract'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)
    ].copy()
    
    # Define preprocessing function
    def preprocess_text(text):
        if not isinstance(text, str):
            return ""
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)             # remove extra whitespace/newlines
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)   # remove non-printable characters
        return text
    
    # Apply preprocessing to Title and Abstract
    df['Title_clean'] = df['Title'].apply(preprocess_text)
    df['Abstract_clean'] = df['Abstract'].apply(preprocess_text)
    
    # Concatenate cleaned Title and Abstract
    df['Text_combined'] = df['Title_clean'] + " " + df['Abstract_clean']
    
    print(f"Cleaned dataset shape: {df.shape}")
    return df

def process_labels(df):
    """
    Process term labels using MultiLabelBinarizer.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with terms column
        
    Returns:
    --------
    tuple
        (processed_dataframe, binary_labels, label_classes)
    """
    print("\nStep 5: Processing labels")
    
    # Split terms into lists
    df['Term_list'] = df['Terms'].apply(
        lambda x: [t.strip() for t in x.split(',')] if isinstance(x, str) else []
    )
    
    # Initialize and fit binarizer
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df['Term_list'])
    
    # Save label classes for later use
    label_classes = mlb.classes_
    
    # Remove specific term if needed
    term_to_drop = "autophosphatase"
    if term_to_drop in label_classes:
        print(f"\nRemoving term: {term_to_drop}")
        drop_idx = list(label_classes).index(term_to_drop)
        Y = np.delete(Y, drop_idx, axis=1)
        label_classes = [label for i, label in enumerate(label_classes) if i != drop_idx]
    
    return df, Y, label_classes

def create_balanced_dataset(df, ratio=2):
    """
    Process a dataset to create a balanced dataset with a specified ratio of unlabeled to labeled data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing labeled and unlabeled data
    ratio : int, default=2
        The ratio of unlabeled to labeled samples in the final dataset
        
    Returns:
    --------
    pandas.DataFrame
        The final balanced dataset
    """
    print("\nStep 6: Creating balanced dataset")
    
    # Consider both NaN and empty strings as unlabeled
    df_labeled = df[(df['Terms'].notna()) & (df['Terms'] != '')].reset_index(drop=True)
    df_unlabeled = df[(df['Terms'].isna()) | (df['Terms'] == '')].reset_index(drop=True)
    
    # Shuffle the labeled and unlabeled datasets
    df_labeled = df_labeled.sample(frac=1, random_state=42).reset_index(drop=True)
    df_unlabeled = df_unlabeled.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate how many unlabeled samples we need
    num_labeled = len(df_labeled)
    num_unlabeled_needed = num_labeled * ratio
    
    # Check if we have enough unlabeled samples
    if num_unlabeled_needed > len(df_unlabeled):
        print(f"Warning: Not enough unlabeled samples. Need {num_unlabeled_needed}, but only have {len(df_unlabeled)}.")
        print(f"Using all available unlabeled samples ({len(df_unlabeled)}).")
        df_unlabeled_selected = df_unlabeled
    else:
        print(f"Selecting {num_unlabeled_needed} unlabeled samples out of {len(df_unlabeled)} available.")
        df_unlabeled_selected = df_unlabeled.iloc[:num_unlabeled_needed]
    
    # Concatenate labeled and unlabeled data
    final_data = pd.concat([df_labeled, df_unlabeled_selected], ignore_index=True)
    
    # Final shuffle
    final_data = final_data.sample(frac=1, random_state=42).reset_index(drop=True)
    

    return final_data

def save_processed_data(final_data):
    """
    Save processed data to CSV.
    
    Parameters:
    -----------
    final_data : pandas.DataFrame
        Processed dataframe to save
    """
    print("\nStep 7: Saving processed data")
    
    # Define the output directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    output_dir = os.path.join(project_root, "data", "processed")
    
    # Define the output path
    output_path = os.path.join(output_dir, "shuffled_data.csv")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the shuffled dataset to a CSV file
    final_data.to_csv(output_path, index=False)
    
    print(f"Shuffled data has been saved to: {output_path}")

def main():
    """
    Main function to run the entire data processing pipeline.
    """
    print("Starting publication data processing pipeline")
    
    # Step 1: Load data
    df_pubmed, df_autoreg = load_data()
    
    
    # Step 2: Merge datasets
    df_selected = merge_datasets(df_pubmed, df_autoreg)
    
    # Step 3: Consolidate terms
    df_cleaned = consolidate_terms(df_selected)
    
    # Step 4: Clean text data
    df_cleaned = clean_text_data(df_cleaned)
    
    # Step 5: Process labels
    df_cleaned, Y, label_classes = process_labels(df_cleaned)
    
    # Step 6: Create balanced dataset
    final_data = create_balanced_dataset(df_cleaned, ratio=2)
    
    # ★ Step 6.5: **NEW** — Binarize multi-labels on the balanced set
    print("\nStep 6.5: Binarizing multi-labels on balanced data")
    final_data['Term_list'] = final_data['Terms'].apply(
        lambda x: [t.strip() for t in str(x).split(',') if t.strip()]
    )
    mlb = MultiLabelBinarizer()
    Y2 = mlb.fit_transform(final_data['Term_list'])
    label_classes2 = list(mlb.classes_)
    
    # remove 'autophosphatase' if present
    if 'autophosphatase' in label_classes2:
        idx = label_classes2.index('autophosphatase')
        print("Removing extremely rare term: autophosphatase")
        Y2 = np.delete(Y2, idx, axis=1)
        del label_classes2[idx]
    
    # append binary-label columns
    labels_df = pd.DataFrame(Y2, columns=label_classes2)
    final_data = pd.concat([final_data.reset_index(drop=True), labels_df], axis=1)
    print("Balanced set now has label matrix shape:", Y2.shape)
    


    # Step 7: Save processed data
    save_processed_data(final_data)
    
    print("\nData processing pipeline completed successfully!")

if __name__ == "__main__":
    main()