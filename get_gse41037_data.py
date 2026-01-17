"""
GSE41037 Healthy Control Sample Extraction

This script processes the GSE41037 whole-blood methylation dataset to identify
and extract samples from healthy control individuals. The original dataset
includes both healthy controls and schizophrenia patients.

Healthy control samples are selected based on sample identifiers reported
in the associated study literature. The resulting subset is intended for
downstream analysis and integration with other blood-based datasets.
"""

import os
import tarfile
import pandas as pd
import numpy as np
import shutil
from datetime import datetime

def extract_gse41037_archive():
    """Extract the GSE41037 compressed archive file."""

    print("Extracting GSE41037 archive.")

    file_path = '/content/drive/MyDrive/GSE41037.tar.gz'
    if not os.path.exists(file_path):
        for root, dirs, files in os.walk('/content/drive/MyDrive'):
            for file in files:
                if 'GSE41037' in file and file.endswith('.tar.gz'):
                    file_path = os.path.join(root, file)
                    break

    print(f"Archive path: {file_path}")

    extract_directory = 'GSE41037_extracted'
    if os.path.exists(extract_directory):
        shutil.rmtree(extract_directory)
    os.makedirs(extract_directory)

    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(extract_directory)

    print(f"Extraction complete: {extract_directory}")

    return extract_directory

def load_and_inspect_metadata(extract_directory):
    """Load and examine the dataset metadata."""

    print("Loading and examining metadata.")

    metadata_file = None
    for root, dirs, files in os.walk(extract_directory):
        for file in files:
            if 'meta' in file.lower() and file.endswith('.csv'):
                metadata_file = os.path.join(root, file)
                break

    if not metadata_file:
        print("Metadata file not found")
        return None

    print(f"Metadata file located: {metadata_file}")

    metadata = pd.read_csv(metadata_file)
    print(f"Metadata dimensions: {metadata.shape}")

    print(f"Metadata columns ({len(metadata.columns)} total):")
    for i, column in enumerate(metadata.columns, 1):
        print(f"  {i:2d}. {column}")

    if 'GSM_number' in metadata.columns:
        gsm_samples = metadata['GSM_number'].dropna().unique()
        print(f"Unique GSM identifiers: {len(gsm_samples)}")

    categorical_variables = []
    for column in metadata.columns:
        if metadata[column].dtype == 'object':
            unique_count = metadata[column].nunique()
            if unique_count < 20:
                categorical_variables.append((column, unique_count))

    if categorical_variables:
        print("Potential categorical variables for sample grouping:")
        for column, count in categorical_variables:
            print(f"  {column}: {count} unique values")

    return metadata

def load_methylation_matrix(extract_directory):
    """Load the methylation data matrix."""

    print("Loading methylation data matrix.")

    methylation_file = None
    for root, dirs, files in os.walk(extract_directory):
        for file in files:
            if any(term in file.lower() for term in ['count', 'matrix', 'methylation']):
                if file.endswith('.csv'):
                    methylation_file = os.path.join(root, file)
                    break

    if not methylation_file:
        print("Methylation data file not found")
        return None, None

    print(f"Methylation data file: {methylation_file}")

    file_size_mb = os.path.getsize(methylation_file) / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")

    with open(methylation_file, 'r') as f:
        first_line = f.readline().strip()

    separator = '\t' if '\t' in first_line else ','
    print(f"File format: {'tab-separated' if separator == '\t' else 'comma-separated'}")

    print("Loading methylation data.")
    try:
        methylation_data = pd.read_csv(methylation_file, sep=separator, low_memory=False)
    except:
        methylation_data = pd.read_csv(methylation_file, low_memory=False)

    print(f"Methylation data loaded: {methylation_data.shape}")

    if methylation_data.shape[1] == 1:
        print("Detected concatenated data, performing column separation.")
        separated_data = methylation_data.iloc[:, 0].str.split('\t', expand=True)

        column_names = separated_data.iloc[0].tolist()
        separated_data = separated_data[1:].reset_index(drop=True)
        separated_data.columns = column_names

        methylation_data = separated_data
        print(f"After separation: {methylation_data.shape}")

    cpg_identifier = None
    for column in methylation_data.columns:
        column_str = str(column).lower()
        if any(pattern in column_str for pattern in ['targetid', 'cg', 'ch.', 'ilmn', 'probe']):
            cpg_identifier = column
            break

    if not cpg_identifier:
        cpg_identifier = methylation_data.columns[0]

    print(f"CpG identifier column: {cpg_identifier}")

    sample_columns = [col for col in methylation_data.columns
                     if 'GSM' in str(col) and col != cpg_identifier]

    print(f"Sample columns identified: {len(sample_columns)}")

    return methylation_data, cpg_identifier

def identify_healthy_samples(metadata, methylation_data, cpg_identifier):
    """Automatically identify healthy control samples based on documented GSM ranges."""

    print("Identifying healthy control samples.")

    healthy_control_gsms = []
    for col in methylation_data.columns:
        if 'GSM' in str(col) and col != cpg_identifier:
            try:
                gsm_num = int(str(col)[3:])
                if 1007129 <= gsm_num <= 1007522:
                    healthy_control_gsms.append(str(col).strip())
            except ValueError:
                continue

    print(f"Identified {len(healthy_control_gsms)} healthy control GSM identifiers from documented range GSM1007129 to GSM1007522.")

    if 'GSM_number' in metadata.columns:
        filtered_metadata = metadata[metadata['GSM_number'].astype(str).isin(healthy_control_gsms)].copy()
        print(f"Found {len(filtered_metadata)} matching samples in metadata.")
    else:
        filtered_metadata = metadata.copy()
        print("Could not filter metadata by GSM, using all metadata entries.")

    return filtered_metadata, healthy_control_gsms

def filter_methylation_data(methylation_data, cpg_identifier, sample_identifiers):
    """Extract methylation data for selected samples."""

    print("Filtering methylation data for healthy control samples.")

    if not sample_identifiers:
        print("No sample identifiers provided.")
        return None

    matching_columns = []
    for identifier in sample_identifiers:
        for column in methylation_data.columns:
            if identifier in str(column):
                matching_columns.append(column)
                break

    print(f"Found {len(matching_columns)} matching columns in methylation data.")

    if not matching_columns:
        print("No matching columns found in methylation data.")
        return None

    columns_to_include = [cpg_identifier] + matching_columns
    filtered_data = methylation_data[columns_to_include].copy()

    print(f"Filtered data dimensions: {filtered_data.shape}")
    print(f"  CpG sites: {filtered_data.shape[0]}")
    print(f"  Samples: {filtered_data.shape[1] - 1}")

    sample_columns = [col for col in filtered_data.columns if col != cpg_identifier]

    for column in sample_columns:
        filtered_data[column] = pd.to_numeric(filtered_data[column], errors='coerce')

    if sample_columns:
        sample_means = filtered_data[sample_columns].mean()
        overall_mean = filtered_data[sample_columns].values.mean()
        missing_values = filtered_data[sample_columns].isna().sum().sum()

        print("Data statistics:")
        print(f"  Overall mean methylation: {overall_mean:.4f}")
        print(f"  Sample mean range: {sample_means.min():.4f} to {sample_means.max():.4f}")
        print(f"  Missing values: {missing_values:,}")

    return filtered_data

def save_processed_data(metadata, methylation_data, output_directory='GSE41037_processed'):
    """Save the processed data to files."""

    print("Saving processed data.")

    os.makedirs(output_directory, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    if metadata is not None:
        metadata_path = os.path.join(output_directory, f'GSE41037_healthy_metadata_{timestamp}.csv')
        metadata.to_csv(metadata_path, index=False)
        print(f"Metadata saved: {metadata_path}")
        print(f"  Samples: {len(metadata)}")

    if methylation_data is not None:
        methylation_path = os.path.join(output_directory, f'GSE41037_healthy_methylation_{timestamp}.csv')
        methylation_data.to_csv(methylation_path, index=False)
        print(f"Methylation data saved: {methylation_path}")
        print(f"  Dimensions: {methylation_data.shape}")

    summary_path = os.path.join(output_directory, f'processing_summary_{timestamp}.txt')

    with open(summary_path, 'w') as summary_file:
        summary_file.write("GSE41037 Healthy Control Data Processing Summary\n")
        summary_file.write("=" * 60 + "\n\n")

        summary_file.write(f"Processing completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        summary_file.write("Dataset Information\n")
        summary_file.write("-" * 40 + "\n")
        summary_file.write("Accession: GSE41037\n")
        summary_file.write("Description: Whole blood DNA methylation in schizophrenia\n")
        summary_file.write("Platform: Illumina HumanMethylation27 BeadChip\n")
        summary_file.write("Total reported samples: 720\n")
        summary_file.write("Reported healthy controls: 394\n")
        summary_file.write("Reported schizophrenia cases: 325\n")
        summary_file.write("Reported bipolar cases: 1\n\n")

        summary_file.write("Processing Results\n")
        summary_file.write("-" * 40 + "\n")
        summary_file.write("Sample selection: Automatic extraction of healthy controls.\n")
        summary_file.write("Selection basis: Documented GSM range for controls (GSM1007129 to GSM1007522).\n")

        if metadata is not None:
            summary_file.write(f"Selected metadata samples: {len(metadata)}\n")

        if methylation_data is not None:
            summary_file.write(f"Methylation data samples: {methylation_data.shape[1] - 1}\n")
            summary_file.write(f"Methylation CpG sites: {methylation_data.shape[0]}\n")

        summary_file.write("\nNotes\n")
        summary_file.write("-" * 40 + "\n")
        summary_file.write("Healthy control samples were automatically extracted based on\n")
        summary_file.write("the sample identifier range published in the associated study.\n")
        summary_file.write("This corresponds to 394 healthy control samples.\n")

    print(f"Processing summary saved: {summary_path}")
    print(f"All output files saved in: {output_directory}")

    return output_directory

def main():
    """Main processing pipeline for GSE41037 healthy control extraction."""

    try:
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print("Google Drive mounted.")
        except ImportError:
            print("Running in local environment.")

        print("Step 1: Extracting dataset archive.")
        extracted_directory = extract_gse41037_archive()

        print("Step 2: Loading metadata.")
        study_metadata = load_and_inspect_metadata(extracted_directory)
        if study_metadata is None:
            print("Failed to load metadata.")
            return None, None

        print("Step 3: Loading methylation data.")
        methylation_result = load_methylation_matrix(extracted_directory)
        if methylation_result is None:
            print("Failed to load methylation data.")
            return None, None

        methylation_matrix, cpg_column = methylation_result

        print("Step 4: Identifying healthy control samples.")
        selected_metadata, selected_samples = identify_healthy_samples(
            study_metadata, methylation_matrix, cpg_column
        )

        if not selected_samples:
            print("No healthy control samples identified.")
            return None, None

        print("Step 5: Filtering methylation data.")
        filtered_methylation = filter_methylation_data(
            methylation_matrix, cpg_column, selected_samples
        )

        print("Step 6: Saving results.")
        output_directory = save_processed_data(selected_metadata, filtered_methylation)

        print("Processing completed successfully.")
        return selected_metadata, filtered_methylation

    except Exception as error:
        print(f"Processing error: {error}")
        import traceback
        traceback.print_exc()
        return None, None

print("Beginning GSE41037 healthy control data processing.")
results = main()

if results[0] is not None or results[1] is not None:
    print("Processing Results")
    print("-" * 50)

    if results[0] is not None:
        print(f"Selected metadata samples: {len(results[0])}")

    if results[1] is not None:
        print(f"Methylation data dimensions: {results[1].shape}")
        print(f"  CpG sites: {results[1].shape[0]}")
        print(f"  Samples: {results[1].shape[1] - 1}")

    print(f"Output directory: GSE41037_processed/")
else:
    print("Processing did not produce output.")
