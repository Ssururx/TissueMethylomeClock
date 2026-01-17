import pandas as pd
import numpy as np
from pathlib import Path
import gzip
import re
import warnings
import tarfile
import urllib.request
import os
import shutil
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

project_dir = '/content/drive/MyDrive/epigenetics_project'
gse_dir = Path(project_dir) / 'external_datasets' / 'GSE19711'
raw_dir = gse_dir / 'raw_data'
processed_dir = gse_dir / 'processed_data'

# Create directories
raw_dir.mkdir(parents=True, exist_ok=True)
processed_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("GSE19711 COMPLETE PROCESSING - Healthy Controls Only")
print("Platform: Illumina HumanMethylation27 (~27,000 CpGs)")
print("=" * 80)

# ============================================================================
# 1. DOWNLOAD FILES FROM NCBI - THESE ARE THE LINKS
# ============================================================================

print("\n1. DOWNLOADING FILES FROM NCBI")

# METADATA DOWNLOAD URL
series_matrix_url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE19nnn/GSE19711/matrix/GSE19711_series_matrix.txt.gz"

# RAW METHYLATION DATA DOWNLOAD URL (249.6 MB)
raw_data_url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE19nnn/GSE19711/suppl/GSE19711_RAW.tar"

# Define local paths
series_matrix_path = raw_dir / 'GSE19711_series_matrix.txt.gz'
raw_archive_path = raw_dir / 'GSE19711_RAW.tar'

# Download series matrix (metadata)
if not series_matrix_path.exists():
    print("  Downloading metadata from:", series_matrix_url)
    urllib.request.urlretrieve(series_matrix_url, series_matrix_path)
    print(f"  Downloaded: {series_matrix_path}")
else:
    print(f"  Metadata already exists: {series_matrix_path}")

# Download raw methylation data
if not raw_archive_path.exists():
    print("  Downloading methylation data from:", raw_data_url)
    print("  File size: 249.6 MB")
    urllib.request.urlretrieve(raw_data_url, raw_archive_path)
    print(f"  Downloaded: {raw_archive_path}")
else:
    print(f"  Methylation data already exists: {raw_archive_path}")

# ============================================================================
# 2. EXTRACT DATA
# ============================================================================

print("\n2. EXTRACTING DATA")

extracted_dir = raw_dir / 'extracted_GSE19711'
if extracted_dir.exists():
    shutil.rmtree(extracted_dir)
extracted_dir.mkdir(exist_ok=True)

# Extract TAR file
print("  Extracting TAR archive...")
with tarfile.open(raw_archive_path, 'r') as tar:
    tar.extractall(path=extracted_dir)

# Find all .txt.gz files
gz_files = list(extracted_dir.glob('*.txt.gz'))
print(f"  Found {len(gz_files)} methylation data files")

# ============================================================================
# 3. PARSE METADATA
# ============================================================================

print("\n3. PARSING METADATA")

with gzip.open(series_matrix_path, 'rt', encoding='utf-8', errors='ignore') as f:
    content = f.read()

lines = content.split('\n')
gsm_ids = []

# Extract GSM IDs
for line in lines:
    if line.startswith('!Sample_geo_accession'):
        parts = line.split('\t')
        gsm_ids = [x.strip().strip('"') for x in parts[1:]]
        break

print(f"  Found {len(gsm_ids)} samples in metadata")

# ============================================================================
# 4. IDENTIFY HEALTHY CONTROLS
# ============================================================================

print("\n4. IDENTIFYING HEALTHY CONTROLS")

metadata_by_gsm = {}
for gsm in gsm_ids:
    metadata_by_gsm[gsm] = {'gsm_id': gsm}

# Parse characteristics from series matrix
for line in lines:
    if line.startswith('!Sample_characteristics_ch1'):
        parts = line.split('\t')
        if len(parts) > 1:
            first_part = parts[1].strip().strip('"')
            if ':' in first_part:
                field_name = first_part.split(':')[0].strip()
                for i, value in enumerate(parts[1:], 0):
                    if i < len(gsm_ids):
                        gsm = gsm_ids[i]
                        metadata_by_gsm[gsm][field_name] = value.strip().strip('"')

# Identify healthy vs cancer samples
healthy_gsms = []
cancer_gsms = []

for gsm, meta in metadata_by_gsm.items():
    if 'sample type' in meta:
        sample_type = meta['sample type'].lower()
        if 'control' in sample_type:
            healthy_gsms.append(gsm)
        elif 'case' in sample_type:
            cancer_gsms.append(gsm)
        else:
            if 'ageatdiagnosis' in meta and meta['ageatdiagnosis'].strip():
                cancer_gsms.append(gsm)
            else:
                healthy_gsms.append(gsm)
    else:
        healthy_gsms.append(gsm)

print(f"  Identified: {len(healthy_gsms)} healthy controls, {len(cancer_gsms)} cancer samples")

# ============================================================================
# 5. CREATE METADATA DATAFRAME
# ============================================================================

print(f"\n5. CREATING METADATA FOR {len(healthy_gsms)} HEALTHY CONTROLS")

metadata_list = []

for gsm in healthy_gsms:
    if gsm in metadata_by_gsm:
        meta = metadata_by_gsm[gsm]

        # Extract age
        age = np.nan
        age_fields = ['ageatrecruitment', 'agegroupatsampledraw']

        for field in age_fields:
            if field in meta:
                try:
                    age_val = str(meta[field])
                    age_match = re.search(r'(\d+)', age_val)
                    if age_match:
                        age = int(age_match.group(1))
                        break
                except:
                    pass

        metadata_entry = {
            'sample_id': gsm,
            'status': 'healthy',
            'age': age,
            'gender': 'female',
            'dataset': 'GSE19711',
            'tissue': 'peripheral_whole_blood',
            'platform': 'Illumina HumanMethylation27'
        }

        metadata_list.append(metadata_entry)

metadata_df = pd.DataFrame(metadata_list)
print(f"  Created metadata for {len(metadata_df)} healthy controls")

# ============================================================================
# 6. PROCESS METHYLATION FILES
# ============================================================================

print(f"\n6. PROCESSING METHYLATION FILES")

# Map GSM IDs to .gz file paths
gsm_to_gzfile = {}
for gz_file in gz_files:
    gsm_match = re.search(r'(GSM\d+)', gz_file.name)
    if gsm_match:
        gsm = gsm_match.group(1)
        gsm_to_gzfile[gsm] = gz_file

print(f"  Mapped {len(gsm_to_gzfile)} GSM IDs to data files")

# Filter to only healthy controls with data files
healthy_with_files = [gsm for gsm in healthy_gsms if gsm in gsm_to_gzfile]
print(f"  {len(healthy_with_files)} healthy controls have data files")

methylation_data = []
processed_samples = []
cpg_ids = None

for gsm in healthy_with_files:
    gz_file = gsm_to_gzfile[gsm]

    try:
        with gzip.open(gz_file, 'rt') as f:
            df = pd.read_csv(f, sep="\t")

        # Find CpG ID column
        cpg_col = None
        for col in ['IlmnID', 'ID_REF', 'TargetID', 'Probe_ID']:
            if col in df.columns:
                cpg_col = col
                break

        if cpg_col is None:
            continue

        # Find beta value column
        beta_col = None
        for col in ['Beta', 'VALUE', 'Beta_value', 'BetaValue']:
            if col in df.columns:
                beta_col = col
                break

        if beta_col is None:
            continue

        # Store CpG IDs from first sample
        if cpg_ids is None:
            cpg_ids = df[cpg_col].tolist()

        # Extract beta values
        beta_values = pd.to_numeric(df[beta_col], errors='coerce').values
        methylation_data.append(beta_values)
        processed_samples.append(gsm)

    except Exception as e:
        continue

print(f"  Successfully processed {len(processed_samples)} samples")

# ============================================================================
# 7. CREATE METHYLATION MATRIX
# ============================================================================

print("\n7. CREATING METHYLATION MATRIX")

methylation_df = pd.DataFrame(
    np.column_stack(methylation_data),
    index=cpg_ids,
    columns=processed_samples
)

print(f"  Methylation matrix shape: {methylation_df.shape}")
print(f"  CpGs: {methylation_df.shape[0]:,}, Samples: {methylation_df.shape[1]}")

# Update metadata
metadata_df = metadata_df[metadata_df['sample_id'].isin(processed_samples)].copy()

# ============================================================================
# 8. SAVE PROCESSED FILES
# ============================================================================

print("\n8. SAVING PROCESSED FILES")

# Save metadata
metadata_output_path = processed_dir / 'GSE19711_metadata.csv'
metadata_df.to_csv(metadata_output_path, index=False)
print(f"  Metadata saved: {metadata_output_path}")

# Save methylation data
methylation_output_path = processed_dir / 'GSE19711_methylation.csv'
methylation_df.to_csv(methylation_output_path)
print(f"  Methylation data saved: {methylation_output_path}")

print("\n" + "=" * 80)
print("PROCESSING COMPLETE")
print("=" * 80)
print(f"Metadata: {len(metadata_df)} healthy controls")
print(f"Methylation: {methylation_df.shape[0]:,} CpGs × {methylation_df.shape[1]} samples")
print(f"Download URLs used:")
print(f"1. Metadata: {series_matrix_url}")
print(f"2. Raw data: {raw_data_url}")
