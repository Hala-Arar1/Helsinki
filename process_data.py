#!/usr/bin/env python3

import os
import pyreadr
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

def main():
    data_dir = os.path.join("data/raw_data")

    # Load data
    result_pubmed = pyreadr.read_r(os.path.join(data_dir, "pubmed.rds"))
    df_pubmed = list(result_pubmed.values())[0]

    result_autoreg = pyreadr.read_r(os.path.join(data_dir, "autoregulatoryDB.rds"))
    df_autoreg = list(result_autoreg.values())[0]

    # Extract PMID from RX column
    df_autoreg['PMID'] = df_autoreg['RX'].str.extract(r'PubMed=(\d+)')
    df_autoreg['PMID'] = df_autoreg['PMID'].astype(str)
    df_pubmed['PMID'] = df_pubmed['PMID'].astype(str)

    # Merge dataframes
    df_merged = pd.merge(df_autoreg, df_pubmed, on='PMID', how='left')

    # Select relevant columns
    columns_to_keep = ['AC', 'OS', 'PMID', 'Title', 'Abstract', 'Term_in_RP', 'Term_in_RT', 'Term_in_RC']
    df_selected = df_merged[columns_to_keep].copy()

    # Merge terms
    def merge_terms(row):
        cols = ['Term_in_RP', 'Term_in_RT', 'Term_in_RC']
        terms = []

        for col in cols:
            val = row[col]
            if pd.notna(val):
                split_terms = [t.strip() for t in str(val).split(',') if t.strip()]
                terms.extend(split_terms)

        return ', '.join(sorted(set(terms))) if terms else ''

    df_selected['Terms'] = df_selected.apply(merge_terms, axis=1)

    # Drop unnecessary columns
    df_cleaned = df_selected.drop(columns=['Term_in_RP', 'Term_in_RT', 'Term_in_RC'])

    # Save the cleaned data
    output_path = 'data/processed_data/protein_autoregulatory_terms.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_cleaned.to_csv(output_path, index=False)

    print(f"Data processing completed. Output saved to {output_path}")

if __name__ == "__main__":
    main()
