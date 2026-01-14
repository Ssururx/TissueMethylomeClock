"""
Goal: Extract 335 healthy samples with the top 50,000 most variable CpG sites.

Information: This script performs selective loading of methylation data: it identifies
the most variable CpG sites across samples, then saves only those sites
to create a more manageable dataset for downstream analysis. The full
485,000-site dataset is never stored entirely in memory.
"""

import pandas as pd
import numpy as np
from google.colab import drive

def main():
    # Mount Google Drive to access data files
    print("Mounting Google Drive...")
    drive.mount('/content/drive')

    # Define file paths
    metadata_path = '/content/drive/MyDrive/GSE74193_EXACT_335_CONTROLS_FINAL.csv'
    beta_values_path = '/content/drive/MyDrive/GSE74193_GEO_procData.csv.gz'

    print(f"\nMetadata file: {metadata_path}")
    print(f"Beta values file: {beta_values_path}")

    # Load the sample metadata for the 335 control samples
    print("\nLoading sample metadata...")
    metadata = pd.read_csv(metadata_path)

    # Map metadata sample names to beta file column names
    # Beta file uses '_Beta' suffix where metadata uses '_Control'
    sample_names = metadata['Sample_Title'].tolist()
    beta_columns = [name.replace('_Control', '_Beta') for name in sample_names]

    print(f"Loaded metadata for {len(metadata)} control samples")

    # Examine the beta file header to identify columns
    print("\nExamining beta values file structure...")
    beta_header = pd.read_csv(beta_values_path, compression='gzip', nrows=0)
    cpg_id_column = beta_header.columns[0]  # Typically the CpG identifier column

    # Select only columns we need: CpG IDs plus our 335 samples
    columns_to_load = [cpg_id_column] + [col for col in beta_columns if col in beta_header.columns]

    found_samples = len(columns_to_load) - 1
    print(f"Found {found_samples} of {len(beta_columns)} sample columns in the beta file")

    # Calculate variance across samples to identify most variable CpGs
    print("\nCalculating variance for each CpG site...")
    print("Processing data in chunks to manage memory...")

    chunk_size = 50000
    processed_cpgs = 0
    variance_data = None

    # First pass: calculate variance chunk by chunk
    for chunk in pd.read_csv(beta_values_path,
                           compression='gzip',
                           usecols=columns_to_load,
                           chunksize=chunk_size):

        chunk = chunk.set_index(cpg_id_column)

        # Standardize column names to match metadata
        chunk.columns = [col.replace('_Beta', '_Control') for col in chunk.columns]

        # Calculate variance for each CpG in this chunk
        chunk_variance = chunk.var(axis=1)

        # Accumulate variance results
        if variance_data is None:
            variance_data = chunk_variance
        else:
            variance_data = pd.concat([variance_data, chunk_variance])

        processed_cpgs += len(chunk)
        if processed_cpgs % 100000 == 0:
            print(f"  Processed {processed_cpgs:,} CpG sites")

    print(f"Variance calculation complete for {processed_cpgs:,} CpG sites")

    # Identify the most variable CpG sites
    print("\nIdentifying the 50,000 most variable CpG sites...")

    # Sort CpGs by variance (descending)
    variance_sorted = variance_data.sort_values(ascending=False)

    # Select top 50,000 CpGs
    top_cpg_list = variance_sorted.head(50000).index.tolist()
    top_cpg_set = set(top_cpg_list)

    min_variance = variance_sorted.iloc[49999]
    max_variance = variance_sorted.iloc[0]

    print(f"Selected 50,000 CpG sites from {processed_cpgs:,} total")
    print(f"Variance range in selected sites: {min_variance:.6f} to {max_variance:.6f}")

    # Second pass: load only the selected CpG sites
    print("\nLoading data for the selected 50,000 CpG sites...")

    # Define filter function for selective loading
    def filter_selected_cpgs(chunk):
        return chunk[chunk[cpg_id_column].isin(top_cpg_set)]

    # Read only rows for our selected CpGs
    selected_beta_data = pd.read_csv(beta_values_path,
                                    compression='gzip',
                                    usecols=columns_to_load)

    # Filter to keep only selected CpGs
    selected_beta_data = selected_beta_data[selected_beta_data[cpg_id_column].isin(top_cpg_set)]

    # Prepare final dataset
    selected_beta_data = selected_beta_data.set_index(cpg_id_column)
    selected_beta_data.columns = [col.replace('_Beta', '_Control') for col in selected_beta_data.columns]

    print(f"Loaded beta values matrix: {len(selected_beta_data)} CpGs × {len(selected_beta_data.columns)} samples")

    # Prepare metadata in same order as beta matrix columns
    print("\nPreparing aligned metadata...")
    aligned_metadata = metadata.set_index('Sample_Title').loc[selected_beta_data.columns].reset_index()

    # Save output files
    print("\nSaving output files...")

    # Save the reduced beta values matrix
    beta_output_path = '/content/drive/MyDrive/GSE74193_BETA_50K_VARIABLE.csv'
    selected_beta_data.to_csv(beta_output_path)
    print(f"  Beta values saved to: {beta_output_path}")
    print(f"    Dimensions: {selected_beta_data.shape[0]:,} CpGs × {selected_beta_data.shape[1]} samples")

    # Save aligned metadata
    meta_output_path = '/content/drive/MyDrive/GSE74193_METADATA_335.csv'
    aligned_metadata.to_csv(meta_output_path, index=False)
    print(f"  Metadata saved to: {meta_output_path}")
    print(f"    Contains: {len(aligned_metadata)} samples")

    # Save variance information for reference
    variance_output_path = '/content/drive/MyDrive/GSE74193_CPG_VARIANCE_RANKING.csv'
    variance_ranking = pd.DataFrame({
        'cpg_site': variance_sorted.index,
        'variance': variance_sorted.values,
        'rank': range(1, len(variance_sorted) + 1)
    })
    variance_ranking.head(50000).to_csv(variance_output_path, index=False)
    print(f"  Variance ranking saved to: {variance_output_path}")

    # Provide analysis summary
    print("\n" + "-" * 60)
    print("Processing Summary")
    print("-" * 60)

    print(f"\nData reduction:")
    print(f"  Initial CpG sites: {processed_cpgs:,}")
    print(f"  Retained CpG sites: {len(selected_beta_data):,}")
    print(f"  Reduction: {100 - (len(selected_beta_data)/processed_cpgs*100):.1f}% of sites removed")

    print(f"\nMatrix characteristics:")
    print(f"  Sample count: {selected_beta_data.shape[1]}")
    print(f"  Missing values: {selected_beta_data.isna().sum().sum()/(selected_beta_data.shape[0]*selected_beta_data.shape[1])*100:.1f}%")
    print(f"  Memory usage: {selected_beta_data.memory_usage().sum()/1024/1024:.1f} MB")

    print(f"\nOutput files created:")
    print(f"  1. {beta_output_path} - Beta values for 50,000 variable CpG sites")
    print(f"  2. {meta_output_path} - Sample metadata aligned with beta matrix")
    print(f"  3. {variance_output_path} - Variance ranking of CpG sites")

    print("\nProcessing complete. The reduced dataset is ready for analysis.")

if __name__ == '__main__':
    main()
