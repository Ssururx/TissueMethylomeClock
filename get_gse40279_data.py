# ----------------------------------------------------------------------
# GSE40279 Data Download - 50000 CpGs Version
# ----------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import requests
import gzip
from io import BytesIO, StringIO
import re

print("="*80)
print("GSE40279 DATASET DOWNLOAD - 50000 CPGS")
print("="*80)

# Setup output directory
OUTPUT_DIR = '/content/GSE40279_50000_cpgs/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Download metadata from series matrix
# ----------------------------------------------------------------------

print("\n1. Downloading metadata")

SERIES_MATRIX_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE40nnn/GSE40279/matrix/GSE40279_series_matrix.txt.gz"

try:
    # Attempt to download the gzipped file
    response = requests.get(SERIES_MATRIX_URL, timeout=30)
    response.raise_for_status()

    # Check if content is actually gzipped by checking the magic bytes
    if response.content[:2] == b'\x1f\x8b':
        # Content is gzipped
        with gzip.open(BytesIO(response.content), 'rt') as f:
            content = f.read()
    else:
        # Content is not gzipped, read as plain text
        content = response.content.decode('utf-8')

except Exception as e:
    print(f"   Error downloading metadata: {e}")
    print(f"   Attempting alternative download method...")

    # Try alternative URL
    ALTERNATIVE_URL = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE40279&format=file&file=GSE40279_series_matrix.txt.gz"
    response = requests.get(ALTERNATIVE_URL, timeout=30)

    if response.content[:2] == b'\x1f\x8b':
        with gzip.open(BytesIO(response.content), 'rt') as f:
            content = f.read()
    else:
        content = response.content.decode('utf-8')

# Extract sample titles from series matrix
lines = content.split('\n')
titles = []
for line in lines:
    if line.startswith('!Sample_title'):
        titles = line.split('\t')[1:]
        break

# Create metadata dataframe
metadata = []
for i, title in enumerate(titles):
    gsm_id = f"GSM{989827 + i}"

    # Extract age information from sample title
    age_match = re.search(r'age\s*(\d+)\s*y', str(title), re.IGNORECASE)
    if age_match:
        age = float(age_match.group(1))
    else:
        # Try alternative pattern
        age_match = re.search(r'(\d+)\s*y[^a-z]', str(title))
        age = float(age_match.group(1)) if age_match else np.nan

    metadata.append({
        'GSM': gsm_id,
        'Title': title.strip('"'),
        'Age': age,
        'Gender': 'N/A',
        'Tissue': 'whole blood',
        'Dataset': 'GSE40279',
        'Health_Status': 'Healthy'
    })

metadata_df = pd.DataFrame(metadata)
print(f"   Samples: {len(metadata_df)}")
print(f"   Age range: {metadata_df['Age'].min():.0f} to {metadata_df['Age'].max():.0f}")

# ----------------------------------------------------------------------
# 2. Download first methylation file to identify top 50000 CpGs
# ----------------------------------------------------------------------

print("\n2. Identifying top 50000 CpGs by variance")

beta_files = [
    "GSE40279_average_beta_GSM989827-GSM989990.txt.gz",
    "GSE40279_average_beta_GSM989991-GSM990299.txt.gz",
    "GSE40279_average_beta_GSM990300-GSM990463.txt.gz",
    "GSE40279_average_beta_GSM990464-GSM990627.txt.gz"
]

BASE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE40nnn/GSE40279/suppl/"

# Download first file to calculate CpG variances
print(f"   Processing first file for CpG selection")
try:
    response = requests.get(BASE_URL + beta_files[0], timeout=30)
    response.raise_for_status()

    cpgs = []
    variances = []

    # Check if content is gzipped
    if response.content[:2] == b'\x1f\x8b':
        with gzip.open(BytesIO(response.content), 'rt') as f:
            header = f.readline().strip().split('\t')

            # Read all CpGs to calculate variance
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue

                cpg = parts[0]
                try:
                    values = np.array([float(x) if x != 'NA' and x != '' and x != 'null' else np.nan for x in parts[1:]])

                    # Calculate variance of non-NaN values
                    valid_values = values[~np.isnan(values)]
                    if len(valid_values) > 10:
                        variance = np.var(valid_values)
                    else:
                        variance = 0

                    cpgs.append(cpg)
                    variances.append(variance)
                except ValueError:
                    continue
    else:
        # Handle non-gzipped content
        content = response.content.decode('utf-8')
        lines = content.split('\n')
        header = lines[0].strip().split('\t')

        for line in lines[1:]:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue

            cpg = parts[0]
            try:
                values = np.array([float(x) if x != 'NA' and x != '' and x != 'null' else np.nan for x in parts[1:]])

                valid_values = values[~np.isnan(values)]
                if len(valid_values) > 10:
                    variance = np.var(valid_values)
                else:
                    variance = 0

                cpgs.append(cpg)
                variances.append(variance)
            except ValueError:
                continue

except Exception as e:
    print(f"   Error downloading first file: {e}")
    # Try alternative URL
    ALT_BASE_URL = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE40279&format=file&file="
    response = requests.get(ALT_BASE_URL + beta_files[0], timeout=30)

    cpgs = []
    variances = []

    if response.content[:2] == b'\x1f\x8b':
        with gzip.open(BytesIO(response.content), 'rt') as f:
            header = f.readline().strip().split('\t')

            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue

                cpg = parts[0]
                try:
                    values = np.array([float(x) if x != 'NA' and x != '' and x != 'null' else np.nan for x in parts[1:]])

                    valid_values = values[~np.isnan(values)]
                    if len(valid_values) > 10:
                        variance = np.var(valid_values)
                    else:
                        variance = 0

                    cpgs.append(cpg)
                    variances.append(variance)
                except ValueError:
                    continue

# Check if we have enough CpGs
if len(cpgs) == 0:
    raise ValueError("No CpGs were successfully read from the data file")

print(f"   Total CpGs available: {len(cpgs):,}")

# Select top 50000 CpGs by highest variance
print(f"   Selecting top 50000 CpGs by variance")
variances_array = np.array(variances)
target_cpgs = 50000

if len(variances_array) < target_cpgs:
    print(f"   Warning: Only {len(variances_array)} CpGs available, using all")
    top_indices = np.argsort(variances_array)
else:
    top_indices = np.argsort(variances_array)[-target_cpgs:]

selected_cpgs = [cpgs[i] for i in top_indices]
selected_cpgs_set = set(selected_cpgs)

print(f"   Selected {len(selected_cpgs)} CpGs")
print(f"   Variance range: {variances_array[top_indices].min():.6f} to {variances_array[top_indices].max():.6f}")

# ----------------------------------------------------------------------
# 3. Extract selected CpGs from all data files
# ----------------------------------------------------------------------

print("\n3. Extracting selected CpGs from all files")

all_data = {}
samples_processed = 0

for file_idx, file in enumerate(beta_files):
    print(f"   Processing file {file_idx+1}/4: {file}")

    try:
        # Try primary URL
        response = requests.get(BASE_URL + file, timeout=30)
        if response.status_code != 200:
            # Try alternative URL
            ALT_BASE_URL = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE40279&format=file&file="
            response = requests.get(ALT_BASE_URL + file, timeout=30)

        if response.content[:2] == b'\x1f\x8b':
            with gzip.open(BytesIO(response.content), 'rt') as f:
                header = f.readline().strip().split('\t')
                file_samples = header[1:]

                # Initialize data storage for samples in this file
                for sample in file_samples:
                    all_data[sample] = {}

                # Process each CpG in the file
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 2:
                        continue

                    cpg = parts[0]

                    if cpg in selected_cpgs_set:
                        values = parts[1:]

                        for i, sample in enumerate(file_samples):
                            if i < len(values):
                                val = values[i]
                                if val != 'NA' and val != '' and val != 'null':
                                    all_data[sample][cpg] = float(val)
                                else:
                                    all_data[sample][cpg] = np.nan
        else:
            # Handle non-gzipped content
            content = response.content.decode('utf-8')
            lines = content.split('\n')
            header = lines[0].strip().split('\t')
            file_samples = header[1:]

            for sample in file_samples:
                all_data[sample] = {}

            for line in lines[1:]:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue

                cpg = parts[0]

                if cpg in selected_cpgs_set:
                    values = parts[1:]

                    for i, sample in enumerate(file_samples):
                        if i < len(values):
                            val = values[i]
                            if val != 'NA' and val != '' and val != 'null':
                                all_data[sample][cpg] = float(val)
                            else:
                                all_data[sample][cpg] = np.nan

    except Exception as e:
        print(f"     Error processing file {file}: {e}")
        continue

    samples_processed += len(file_samples)
    print(f"     Processed {len(file_samples)} samples (total: {samples_processed})")

# ----------------------------------------------------------------------
# 4. Create final methylation dataframe
# ----------------------------------------------------------------------

print("\n4. Creating final dataframe")

if not all_data:
    raise ValueError("No data was successfully extracted from any file")

samples_list = list(all_data.keys())
cpgs_list = list(selected_cpgs)

# Create array for methylation data
data_array = np.full((len(samples_list), len(cpgs_list)), np.nan)

print(f"   Filling data array")
for i, sample in enumerate(samples_list):
    sample_data = all_data[sample]
    for j, cpg in enumerate(cpgs_list):
        if cpg in sample_data:
            data_array[i, j] = sample_data[cpg]

    # Progress update
    if (i + 1) % 100 == 0:
        print(f"     Processed {i + 1}/{len(samples_list)} samples")

# Create dataframe with samples as columns, CpGs as rows
meth_df = pd.DataFrame(data_array, index=samples_list, columns=cpgs_list)
meth_df = meth_df.T

print(f"   Final shape: {meth_df.shape}")
print(f"   Memory usage: {meth_df.memory_usage().sum() / 1024**2:.1f} MB")
print(f"   Missing values: {meth_df.isna().sum().sum():,} ({meth_df.isna().mean().mean()*100:.2f}%)")

# ----------------------------------------------------------------------
# 5. Map sample IDs to GSM IDs
# ----------------------------------------------------------------------

print("\n5. Mapping samples to GSM IDs")

# Create mapping from sample IDs to GSM IDs
sample_to_gsm = {}
for i, sample_id in enumerate(meth_df.columns):
    gsm_id = f"GSM{989827 + i}"
    sample_to_gsm[sample_id] = gsm_id

# Rename columns with GSM IDs
meth_df_renamed = meth_df.rename(columns=sample_to_gsm)
print(f"   Mapped {len(sample_to_gsm)} samples")

# ----------------------------------------------------------------------
# 6. Save all output files
# ----------------------------------------------------------------------

print("\n6. Saving files")

# Save metadata
metadata_file = os.path.join(OUTPUT_DIR, 'GSE40279_metadata.csv')
metadata_df.to_csv(metadata_file, index=False)
print(f"   Metadata: {metadata_file}")

# Save methylation data with samples as rows
meth_file = os.path.join(OUTPUT_DIR, 'GSE40279_methylation_50000.csv.gz')
meth_df_renamed.T.to_csv(meth_file, compression='gzip')
print(f"   Methylation data: {meth_file}")
print(f"     CpGs: {meth_df_renamed.shape[0]:,}")
print(f"     Samples: {meth_df_renamed.shape[1]}")

# Save CpG list
cpg_file = os.path.join(OUTPUT_DIR, 'GSE40279_cpgs_50000.txt')
with open(cpg_file, 'w') as f:
    for cpg in meth_df_renamed.index:
        f.write(f"{cpg}\n")
print(f"   CpG list: {cpg_file}")

# ----------------------------------------------------------------------
# 7. Print format information for alignment
# ----------------------------------------------------------------------

print("\n" + "="*80)
print("Data format information")
print("="*80)

print("\nMethylation data format:")
print("-" * 40)
print("Rows: Samples (GSM IDs)")
print("Columns: CpG probes")
print("Values: Beta values (0-1)")

print("\nFirst 2 rows and 5 columns of methylation data:")
print("-" * 40)
sample_data_preview = meth_df_renamed.T.iloc[:2, :5]
print(sample_data_preview)

print("\nColumn names (first 5 CpGs):")
print("-" * 40)
print(list(meth_df_renamed.index[:5]))

print("\nMetadata format:")
print("-" * 40)
print("Columns:", list(metadata_df.columns))
print("\nFirst 2 rows:")
print(metadata_df.head(2))

# ----------------------------------------------------------------------
# 8. Summary
# ----------------------------------------------------------------------

print("\n" + "="*80)
print("Summary")
print("="*80)
print(f"Metadata samples: {len(metadata_df)}")
print(f"Methylation samples: {meth_df_renamed.shape[1]}")
print(f"CpGs: {meth_df_renamed.shape[0]:,}")
print(f"Age range: {metadata_df['Age'].min():.0f}-{metadata_df['Age'].max():.0f} years")
print(f"Output directory: {OUTPUT_DIR}")
print("\nData preparation complete.")
