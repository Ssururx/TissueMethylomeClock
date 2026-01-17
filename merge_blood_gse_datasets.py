# ----------------------------------------------------------------------------
# Processes: GSE40279, GSE19711, and GSE41037 from existing files
# Uses only shared CpGs for cross-platform analysis
# ----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Real blood methylation dataset processing with shared CpGs")
print("=" * 80)
print("Processing existing files:")
print("1. GSE40279 - from /content/GSE40279_50000_cpgs/")
print("2. GSE19711 - from Google Drive processed_data/")
print("3. GSE41037 - from GSE41037_results/ directory")
print("=" * 80)

# ----------------------------------------------------------------------------
# Step 1: Verify and load existing files
# ----------------------------------------------------------------------------

print("\n1. Verifying all existing files")
print("-" * 40)

# Define all expected file paths
file_paths = {
    # GSE40279 files (450k array)
    'gse40279_meta': Path('/content/GSE40279_50000_cpgs/GSE40279_metadata.csv'),
    'gse40279_meth': Path('/content/GSE40279_50000_cpgs/GSE40279_methylation_50000.csv.gz'),

    # GSE19711 files (27k array)
    'gse19711_meta': Path('/content/drive/MyDrive/epigenetics_project/external_datasets/GSE19711/processed_data/GSE19711_metadata.csv'),
    'gse19711_meth': Path('/content/drive/MyDrive/epigenetics_project/external_datasets/GSE19711/processed_data/GSE19711_methylation.csv'),

    # GSE41037 files (27k array)
    'gse41037_meta': Path('GSE41037_results/GSE41037_healthy_controls_metadata.csv'),
    'gse41037_meth': Path('GSE41037_results/GSE41037_healthy_controls_methylation.csv'),
}

# Check which files exist
existing_files = {}
for name, path in file_paths.items():
    if path.exists():
        existing_files[name] = path
        print(f"Found: {name} -> {path}")
    else:
        print(f"Not found: {name}")

print(f"\nTotal files found: {len(existing_files)}/{len(file_paths)}")

# ----------------------------------------------------------------------------
# Step 2: Load all datasets
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("2. Loading all datasets")
print("=" * 80)

# Load GSE41037 data (27k)
print("Loading GSE41037 data...")
gse41037_meta = pd.read_csv(file_paths['gse41037_meta'])
gse41037_meth = pd.read_csv(file_paths['gse41037_meth'], index_col=0)
print(f"  GSE41037: {gse41037_meth.shape[0]:,} CpGs × {gse41037_meth.shape[1]} samples")

# Load GSE19711 data (27k)
print("Loading GSE19711 data...")
gse19711_meta = pd.read_csv(file_paths['gse19711_meta'])
gse19711_meth = pd.read_csv(file_paths['gse19711_meth'], index_col=0)
print(f"  GSE19711: {gse19711_meth.shape[0]:,} CpGs × {gse19711_meth.shape[1]} samples")

# Load GSE40279 data (450k)
print("Loading GSE40279 data...")
gse40279_meta = pd.read_csv(file_paths['gse40279_meta'])
gse40279_meth = pd.read_csv(file_paths['gse40279_meth'], compression='gzip', index_col=0)
print(f"  GSE40279: {gse40279_meth.shape[0]} samples × {gse40279_meth.shape[1]:,} CpGs")

# Transpose GSE40279 to match format
print("Transposing GSE40279 to match 27k format...")
gse40279_meth = gse40279_meth.T
print(f"  After transpose: {gse40279_meth.shape[0]:,} CpGs × {gse40279_meth.shape[1]} samples")

# ----------------------------------------------------------------------------
# Step 3: Standardize metadata (extract only essential columns)
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("3. Standardizing metadata with essential columns only")
print("=" * 80)

def extract_essential_metadata(meta_df, dataset_name):
    """Extract only essential columns from metadata"""
    essential = pd.DataFrame()

    # Extract sample ID
    if 'GSM_number' in meta_df.columns:
        essential['sample_id'] = meta_df['GSM_number']
    elif 'GSM' in meta_df.columns:
        essential['sample_id'] = meta_df['GSM']
    elif 'sample_id' in meta_df.columns:
        essential['sample_id'] = meta_df['sample_id']
    else:
        # Try to find GSM pattern in column names
        gsm_cols = [col for col in meta_df.columns if 'GSM' in col.upper()]
        if gsm_cols:
            essential['sample_id'] = meta_df[gsm_cols[0]]
        else:
            # Last resort: use first column
            essential['sample_id'] = meta_df.iloc[:, 0]

    # Extract age
    age_cols = ['Age', 'age', 'AGE']
    for col in age_cols:
        if col in meta_df.columns:
            essential['age'] = pd.to_numeric(meta_df[col], errors='coerce')
            break
    else:
        essential['age'] = np.nan

    # Extract sex/gender
    sex_cols = ['Gender', 'gender', 'sex', 'SEX', 'GENDER']
    for col in sex_cols:
        if col in meta_df.columns:
            sex_data = meta_df[col].astype(str).str.upper()
            essential['sex'] = sex_data.map({
                'M': 'M', 'MALE': 'M',
                'F': 'F', 'FEMALE': 'F',
                'U': 'U', 'UNKNOWN': 'U',
                'N/A': 'U', 'NA': 'U'
            }).fillna('U')
            break
    else:
        essential['sex'] = 'U'

    # Add dataset and platform info
    essential['dataset'] = dataset_name
    if dataset_name == 'GSE40279':
        essential['platform'] = 'HumanMethylation450'
    else:
        essential['platform'] = 'HumanMethylation27'

    essential['tissue'] = 'whole_blood'
    essential['status'] = 'healthy'

    return essential

print("\nExtracting essential metadata columns...")
gse41037_meta_clean = extract_essential_metadata(gse41037_meta, 'GSE41037')
gse19711_meta_clean = extract_essential_metadata(gse19711_meta, 'GSE19711')
gse40279_meta_clean = extract_essential_metadata(gse40279_meta, 'GSE40279')

print(f"GSE41037 metadata: {gse41037_meta_clean.shape}")
print(f"GSE19711 metadata: {gse19711_meta_clean.shape}")
print(f"GSE40279 metadata: {gse40279_meta_clean.shape}")

print("\nEssential columns in cleaned metadata:")
print(f"  {', '.join(gse41037_meta_clean.columns.tolist())}")

# ----------------------------------------------------------------------------
# Step 4: Identify shared CpGs across platforms
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("4. Identifying shared CpGs across platforms")
print("=" * 80)

# Get CpG sets
cpgs_41037 = set(gse41037_meth.index)
cpgs_19711 = set(gse19711_meth.index)
cpgs_40279 = set(gse40279_meth.index)

print(f"\nOriginal CpG counts:")
print(f"  GSE41037 (27k): {len(cpgs_41037):,} CpGs")
print(f"  GSE19711 (27k): {len(cpgs_19711):,} CpGs")
print(f"  GSE40279 (450k): {len(cpgs_40279):,} CpGs")

# Find shared CpGs
shared_27k = cpgs_41037.intersection(cpgs_19711)
shared_all = shared_27k.intersection(cpgs_40279)

print(f"\nShared CpGs:")
print(f"  Between 27k datasets: {len(shared_27k):,} CpGs")
print(f"  Across all platforms: {len(shared_all):,} CpGs")

if len(shared_all) == 0:
    print("\nWarning: No CpGs shared across all platforms")
    print("Using only shared CpGs between 27k datasets...")
    shared_cpgs = sorted(list(shared_27k))
else:
    shared_cpgs = sorted(list(shared_all))

print(f"\nUsing {len(shared_cpgs):,} shared CpGs for cross-platform analysis")

# ----------------------------------------------------------------------------
# Step 5: Create datasets with shared CpGs only
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("5. Creating datasets with shared CpGs only")
print("=" * 80)

print("\nSubsetting methylation data to shared CpGs...")

# Subset each dataset to shared CpGs
gse41037_meth_shared = gse41037_meth.loc[shared_cpgs]
gse19711_meth_shared = gse19711_meth.loc[shared_cpgs]
gse40279_meth_shared = gse40279_meth.loc[shared_cpgs]

print(f"  GSE41037 shared: {gse41037_meth_shared.shape}")
print(f"  GSE19711 shared: {gse19711_meth_shared.shape}")
print(f"  GSE40279 shared: {gse40279_meth_shared.shape}")

# Check for missing CpGs
missing_in_41037 = [cpg for cpg in shared_cpgs if cpg not in gse41037_meth.index]
missing_in_19711 = [cpg for cpg in shared_cpgs if cpg not in gse19711_meth.index]
missing_in_40279 = [cpg for cpg in shared_cpgs if cpg not in gse40279_meth.index]

if missing_in_41037:
    print(f"  Warning: {len(missing_in_41037)} CpGs missing in GSE41037")
if missing_in_19711:
    print(f"  Warning: {len(missing_in_19711)} CpGs missing in GSE19711")
if missing_in_40279:
    print(f"  Warning: {len(missing_in_40279)} CpGs missing in GSE40279")

# ----------------------------------------------------------------------------
# Step 6: Combine datasets
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("6. Combining all datasets")
print("=" * 80)

print("\nCombining methylation data...")
combined_meth = pd.concat([gse41037_meth_shared, gse19711_meth_shared, gse40279_meth_shared], axis=1)
combined_meta = pd.concat([gse41037_meta_clean, gse19711_meta_clean, gse40279_meta_clean], ignore_index=True)

print(f"Combined methylation: {combined_meth.shape}")
print(f"Combined metadata: {combined_meta.shape}")

# ----------------------------------------------------------------------------
# Step 7: Verify alignment and data quality
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("7. Verifying alignment and data quality")
print("=" * 80)

# Align samples
meth_samples = set(combined_meth.columns)
meta_samples = set(combined_meta['sample_id'])

print(f"\nSample alignment:")
print(f"  Methylation samples: {len(meth_samples)}")
print(f"  Metadata samples: {len(meta_samples)}")
print(f"  Common samples: {len(meth_samples.intersection(meta_samples))}")

# Keep only aligned samples
common_samples = list(meth_samples.intersection(meta_samples))
combined_meth = combined_meth[common_samples]
combined_meta = combined_meta[combined_meta['sample_id'].isin(common_samples)]

print(f"\nAfter alignment:")
print(f"  Methylation: {combined_meth.shape}")
print(f"  Metadata: {combined_meta.shape}")

# Data quality checks
print(f"\nData quality checks:")
print(f"  Methylation range: [{combined_meth.min().min():.4f}, {combined_meth.max().max():.4f}]")
print(f"  Mean methylation: {combined_meth.mean().mean():.4f}")

missing_vals = combined_meth.isna().sum().sum()
total_vals = combined_meth.size
missing_pct = (missing_vals / total_vals) * 100
print(f"  Missing values: {missing_vals:,} ({missing_pct:.2f}%)")

print(f"\nDataset composition:")
for dataset in combined_meta['dataset'].unique():
    n = (combined_meta['dataset'] == dataset).sum()
    age_data = combined_meta[combined_meta['dataset'] == dataset]['age']
    if age_data.notna().any():
        age_min = age_data.min()
        age_max = age_data.max()
        age_mean = age_data.mean()
        print(f"  {dataset}: {n} samples, age {age_min:.0f}-{age_max:.0f} (mean: {age_mean:.1f})")

print(f"\nSex distribution:")
sex_counts = combined_meta['sex'].value_counts()
for sex, count in sex_counts.items():
    print(f"  {sex}: {count} samples")

print(f"\nPlatform distribution:")
platform_counts = combined_meta['platform'].value_counts()
for platform, count in platform_counts.items():
    print(f"  {platform}: {count} samples")

# ----------------------------------------------------------------------------
# Step 8: Create additional datasets for platform-specific analysis
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("8. Creating platform-specific datasets")
print("=" * 80)

# Create 27k-only dataset (GSE41037 + GSE19711)
print("\nCreating 27k-only dataset...")
mask_27k = combined_meta['platform'] == 'HumanMethylation27'
meta_27k = combined_meta[mask_27k]
meth_27k = combined_meth[meta_27k['sample_id']]
print(f"  27k dataset: {meth_27k.shape[0]:,} CpGs × {meth_27k.shape[1]} samples")

# Create 450k-only dataset (GSE40279)
print("\nCreating 450k-only dataset...")
mask_450k = combined_meta['platform'] == 'HumanMethylation450'
meta_450k = combined_meta[mask_450k]
meth_450k = combined_meth[meta_450k['sample_id']]
print(f"  450k dataset: {meth_450k.shape[0]:,} CpGs × {meth_450k.shape[1]} samples")

# ----------------------------------------------------------------------------
# Step 9: Save all datasets
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("9. Saving all datasets")
print("=" * 80)

# Create output directory
output_dir = Path('unified_blood_methylation_clean')
output_dir.mkdir(exist_ok=True)

# Save 1: Combined dataset with shared CpGs
print("\n1. Saving combined dataset with shared CpGs...")
meta_combined_path = output_dir / 'unified_metadata_shared.csv'
meth_combined_path = output_dir / 'unified_methylation_shared.csv'

combined_meta.to_csv(meta_combined_path, index=False)
combined_meth.to_csv(meth_combined_path)

print(f"  Metadata saved: {meta_combined_path}")
print(f"  Methylation saved: {meth_combined_path}")
print(f"  Dimensions: {combined_meth.shape[0]:,} CpGs × {combined_meth.shape[1]} samples")

# Save 2: 27k-only dataset
print("\n2. Saving 27k-only dataset...")
meta_27k_path = output_dir / 'metadata_27k.csv'
meth_27k_path = output_dir / 'methylation_27k.csv'

meta_27k.to_csv(meta_27k_path, index=False)
meth_27k.to_csv(meth_27k_path)

print(f"  Metadata saved: {meta_27k_path}")
print(f"  Methylation saved: {meth_27k_path}")
print(f"  Dimensions: {meth_27k.shape[0]:,} CpGs × {meth_27k.shape[1]} samples")

# Save 3: 450k-only dataset
print("\n3. Saving 450k-only dataset...")
meta_450k_path = output_dir / 'metadata_450k.csv'
meth_450k_path = output_dir / 'methylation_450k.csv'

meta_450k.to_csv(meta_450k_path, index=False)
meth_450k.to_csv(meth_450k_path)

print(f"  Metadata saved: {meta_450k_path}")
print(f"  Methylation saved: {meth_450k_path}")
print(f"  Dimensions: {meth_450k.shape[0]:,} CpGs × {meth_450k.shape[1]} samples")

# Save ML-ready formats
print("\n4. Saving ML-ready formats...")

# Combined dataset
X_combined = combined_meth.T
y_combined = combined_meta.set_index('sample_id').loc[X_combined.index, 'age']
X_combined.to_csv(output_dir / 'X_features_combined.csv')
y_combined.to_csv(output_dir / 'y_age_combined.csv')
print(f"  Combined: X={X_combined.shape}, y={y_combined.shape}")

# 27k dataset
X_27k = meth_27k.T
y_27k = meta_27k.set_index('sample_id').loc[X_27k.index, 'age']
X_27k.to_csv(output_dir / 'X_features_27k.csv')
y_27k.to_csv(output_dir / 'y_age_27k.csv')
print(f"  27k only: X={X_27k.shape}, y={y_27k.shape}")

# 450k dataset
X_450k = meth_450k.T
y_450k = meta_450k.set_index('sample_id').loc[X_450k.index, 'age']
X_450k.to_csv(output_dir / 'X_features_450k.csv')
y_450k.to_csv(output_dir / 'y_age_450k.csv')
print(f"  450k only: X={X_450k.shape}, y={y_450k.shape}")

# ----------------------------------------------------------------------------
# Step 10: Create summary report
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("10. Creating summary report")
print("=" * 80)

summary_path = output_dir / 'processing_summary.txt'
with open(summary_path, 'w') as f:
    f.write("Blood Methylation Dataset Processing Summary\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Processing date: {pd.Timestamp.now()}\n\n")

    f.write("INPUT DATASETS:\n")
    f.write("-" * 40 + "\n")
    f.write(f"GSE41037 (27k): {gse41037_meth.shape[0]:,} CpGs × {gse41037_meth.shape[1]} samples\n")
    f.write(f"GSE19711 (27k): {gse19711_meth.shape[0]:,} CpGs × {gse19711_meth.shape[1]} samples\n")
    f.write(f"GSE40279 (450k): {gse40279_meth.shape[0]:,} CpGs × {gse40279_meth.shape[1]} samples\n\n")

    f.write("SHARED CPG ANALYSIS:\n")
    f.write("-" * 40 + "\n")
    f.write(f"CpGs shared across all platforms: {len(shared_all):,}\n")
    f.write(f"CpGs shared between 27k datasets: {len(shared_27k):,}\n")
    f.write(f"CpGs unique to 27k arrays: {len(shared_27k) - len(shared_all):,}\n")
    f.write(f"CpGs unique to 450k array: {len(cpgs_40279) - len(shared_all):,}\n\n")

    f.write("OUTPUT DATASETS:\n")
    f.write("-" * 40 + "\n")
    f.write(f"1. Combined dataset (shared CpGs):\n")
    f.write(f"   Samples: {combined_meth.shape[1]}\n")
    f.write(f"   CpGs: {combined_meth.shape[0]:,}\n")
    f.write(f"   Missing values: {missing_pct:.2f}%\n\n")

    f.write(f"2. 27k-only dataset:\n")
    f.write(f"   Samples: {meth_27k.shape[1]}\n")
    f.write(f"   CpGs: {meth_27k.shape[0]:,}\n")
    f.write(f"   Age range: {meta_27k['age'].min():.0f}-{meta_27k['age'].max():.0f}\n\n")

    f.write(f"3. 450k-only dataset:\n")
    f.write(f"   Samples: {meth_450k.shape[1]}\n")
    f.write(f"   CpGs: {meth_450k.shape[0]:,}\n")
    f.write(f"   Age range: {meta_450k['age'].min():.0f}-{meta_450k['age'].max():.0f}\n\n")

    f.write("SAMPLE COMPOSITION:\n")
    f.write("-" * 40 + "\n")
    for dataset in combined_meta['dataset'].unique():
        n = (combined_meta['dataset'] == dataset).sum()
        age_stats = combined_meta[combined_meta['dataset'] == dataset]['age'].describe()
        f.write(f"{dataset}: {n} samples, age {age_stats['min']:.0f}-{age_stats['max']:.0f}\n")

    f.write(f"\nTotal samples: {len(combined_meta)}\n")

    f.write("\nSEX DISTRIBUTION:\n")
    f.write("-" * 40 + "\n")
    for sex, count in sex_counts.items():
        f.write(f"{sex}: {count} samples\n")

    f.write("\nPLATFORM DISTRIBUTION:\n")
    f.write("-" * 40 + "\n")
    for platform, count in platform_counts.items():
        f.write(f"{platform}: {count} samples\n")

    f.write("\nFILES CREATED:\n")
    f.write("-" * 40 + "\n")
    f.write("1. unified_metadata_shared.csv - Combined metadata\n")
    f.write("2. unified_methylation_shared.csv - Combined methylation (shared CpGs)\n")
    f.write("3. metadata_27k.csv - 27k platform metadata\n")
    f.write("4. methylation_27k.csv - 27k platform methylation\n")
    f.write("5. metadata_450k.csv - 450k platform metadata\n")
    f.write("6. methylation_450k.csv - 450k platform methylation\n")
    f.write("7. X_features_*.csv - ML-ready feature matrices\n")
    f.write("8. y_age_*.csv - ML-ready age targets\n")

print(f"Summary saved: {summary_path}")

# ----------------------------------------------------------------------------
# Final output
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("PROCESSING COMPLETE")
print("=" * 80)

print(f"\nAll files saved to: {output_dir.absolute()}")

print(f"\nSUMMARY OF CREATED DATASETS:")
print(f"\n1. COMBINED DATASET (Shared CpGs):")
print(f"   Samples: {combined_meth.shape[1]}")
print(f"   CpGs: {combined_meth.shape[0]:,}")
print(f"   Missing values: {missing_pct:.2f}%")

print(f"\n2. 27K-ONLY DATASET (GSE41037 + GSE19711):")
print(f"   Samples: {meth_27k.shape[1]}")
print(f"   CpGs: {meth_27k.shape[0]:,}")
print(f"   Age range: {meta_27k['age'].min():.0f}-{meta_27k['age'].max():.0f}")

print(f"\n3. 450K-ONLY DATASET (GSE40279):")
print(f"   Samples: {meth_450k.shape[1]}")
print(f"   CpGs: {meth_450k.shape[0]:,}")
print(f"   Age range: {meta_450k['age'].min():.0f}-{meta_450k['age'].max():.0f}")

print(f"\nCpG Statistics:")
print(f"  Total unique CpGs across platforms: {len(cpgs_41037.union(cpgs_40279)):,}")
print(f"  Shared CpGs (cross-platform): {len(shared_all):,}")
print(f"  27k-only CpGs: {len(shared_27k) - len(shared_all):,}")
print(f"  450k-only CpGs: {len(cpgs_40279) - len(shared_all):,}")

print(f"\nDataset Statistics:")
print(f"  Total samples: {len(combined_meta)}")
print(f"  Age range: {combined_meta['age'].min():.0f}-{combined_meta['age'].max():.0f}")
print(f"  Mean age: {combined_meta['age'].mean():.1f}")

print(f"\nPlatform Statistics:")
for platform, count in platform_counts.items():
    platform_meta = combined_meta[combined_meta['platform'] == platform]
    print(f"  {platform}: {count} samples, age {platform_meta['age'].min():.0f}-{platform_meta['age'].max():.0f}")

print("\nNotes to self:")
print("1. The combined dataset uses only CpGs shared across all platforms")
print("2. Platform-specific datasets allow analysis of unique CpGs per platform")
print("3. ML-ready formats (X_features_*.csv, y_age_*.csv) are provided")
print("4. All metadata has been standardized to essential columns only")

