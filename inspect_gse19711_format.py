# ============================================================================
# GSE19711 Data Loading and Format Verification
# ============================================================================

import pandas as pd
import numpy as np
import os

print("=" * 80)
print("GSE19711 METHYLATION DATA LOADING AND FORMAT VERIFICATION")
print("=" * 80)

# Paths
methylation_path = '/content/drive/MyDrive/epigenetics_project/external_datasets/GSE19711/processed_data/GSE19711_methylation.csv'
metadata_path = '/content/drive/MyDrive/epigenetics_project/external_datasets/GSE19711/processed_data/GSE19711_metadata.csv'

# Check if files exist
print("\n1. CHECKING FOR EXISTING FILES")
print("-" * 40)

if os.path.exists(methylation_path):
    print(f"Methylation file found: {methylation_path}")
    file_size = os.path.getsize(methylation_path) / (1024*1024)
    print(f"File size: {file_size:.2f} MB")
else:
    print(f"Warning: Methylation file not found at {methylation_path}")

if os.path.exists(metadata_path):
    print(f"Metadata file found: {metadata_path}")
else:
    print(f"Warning: Metadata file not found at {metadata_path}")

print("\n2. LOADING DATA CORRECTLY")
print("-" * 40)

print("\nMethod 1: Load without index_col to inspect structure")
df_raw = pd.read_csv(methylation_path)
print(f"Raw data shape: {df_raw.shape}")
print(f"Raw columns first 5: {df_raw.columns.tolist()[:5]}")
print(f"First column name: '{df_raw.columns[0]}'")
print(f"First few rows of first column:")
print(df_raw.iloc[:5, 0].tolist())

print("\nMethod 2: Correct loading with index_col=0")
methylation_correct = pd.read_csv(
    methylation_path,
    index_col=0,
    header=0
)

print(f"Processed data shape: {methylation_correct.shape}")
print(f"Index name: {methylation_correct.index.name}")
print(f"Index first 5 values: {methylation_correct.index[:5].tolist()}")
print(f"Columns first 5 values: {methylation_correct.columns[:5].tolist()}")

print(f"\nValue statistics:")
print(f"Minimum: {methylation_correct.min().min():.6f}")
print(f"Maximum: {methylation_correct.max().max():.6f}")
print(f"Mean: {methylation_correct.mean().mean():.6f}")
print(f"NaN percentage: {methylation_correct.isna().mean().mean()*100:.2f}%")

print("\n" + "=" * 80)
print("3. DATA FORMAT PREVIEW")
print("=" * 80)

print("\nA. Methylation Data Format:")
print("-" * 40)
print("Structure: Rows = CpG probes, Columns = Samples (GSM IDs)")
print("Values: Beta values ranging from 0 to 1")

print(f"\nFirst 2 rows (CpGs), first 5 columns (samples):")
print("Row headers (CpG IDs) are in the index")
print("Column headers (sample IDs) are in the columns")
preview_cpgs = methylation_correct.iloc[:2, :5]
print(preview_cpgs)

print(f"\nData types:")
print(f"Index dtype: {methylation_correct.index.dtype}")
sample_value = methylation_correct.iloc[0,0]
print(f"Values dtype example: {sample_value} (type: {type(sample_value)})")

print("\n" + "=" * 80)
print("4. METADATA VERIFICATION")
print("=" * 80)

metadata = pd.read_csv(metadata_path)
print(f"\nMetadata shape: {metadata.shape}")
print(f"Metadata columns: {list(metadata.columns)}")

print(f"\nFirst 2 rows of metadata:")
print("Columns show sample information")
print(metadata.head(2))

metadata_samples = set(metadata['sample_id'])
methylation_samples = set(methylation_correct.columns)

print(f"\nSample matching between datasets:")
print(f"Metadata samples count: {len(metadata_samples)}")
print(f"Methylation samples count: {len(methylation_samples)}")
print(f"Common samples count: {len(metadata_samples.intersection(methylation_samples))}")
print(f"Samples missing in methylation data: {len(metadata_samples - methylation_samples)}")
print(f"Extra samples in methylation data: {len(methylation_samples - metadata_samples)}")

if len(metadata_samples - methylation_samples) > 0:
    print(f"\nSamples in metadata but not in methylation (first 3):")
    print(list(metadata_samples - methylation_samples)[:3])

print("\n" + "=" * 80)
print("5. DATA ANALYSIS SUMMARY")
print("=" * 80)

if 'age' in metadata.columns:
    age_stats = metadata['age'].describe()
    print(f"\nAge distribution statistics:")
    print(f"  Count: {age_stats['count']:.0f}")
    print(f"  Mean: {age_stats['mean']:.1f} plus or minus {age_stats['std']:.1f}")
    print(f"  Range: {age_stats['min']:.0f} to {age_stats['max']:.0f}")
    print(f"  Quartiles: {age_stats['25%']:.0f}, {age_stats['50%']:.0f}, {age_stats['75%']:.0f}")

print(f"\nMethylation data statistics:")
meth_stats = methylation_correct.stack().describe()
print(f"  Count: {meth_stats['count']:,}")
print(f"  Mean: {meth_stats['mean']:.4f}")
print(f"  Standard deviation: {meth_stats['std']:.4f}")
print(f"  Minimum: {meth_stats['min']:.4f}")
print(f"  25th percentile: {meth_stats['25%']:.4f}")
print(f"  Median: {meth_stats['50%']:.4f}")
print(f"  75th percentile: {meth_stats['75%']:.4f}")
print(f"  Maximum: {meth_stats['max']:.4f}")

print("\n" + "=" * 80)
print("6. SAVING CORRECTED VERSION")
print("=" * 80)

corrected_path = '/content/drive/MyDrive/epigenetics_project/external_datasets/GSE19711/processed_data/GSE19711_methylation_CORRECT.csv'
methylation_correct.to_csv(corrected_path)
print(f"Saved corrected version to: {corrected_path}")

print("\n" + "=" * 80)
print("7. MACHINE LEARNING PREPARATION")
print("=" * 80)

X = methylation_correct.T
y = metadata.set_index('sample_id').loc[methylation_correct.columns, 'age']

print(f"Feature matrix X shape: {X.shape}")
print(f"Target vector y shape: {y.shape}")

print(f"\nFirst 2 rows of feature matrix X:")
print(X.iloc[:2, :5])

print(f"\nFirst 2 values of target vector y:")
print(y.head(2))

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(f"\nMETADATA SUMMARY:")
print(f"  Samples: {len(metadata)} healthy controls")
print(f"  Age range: {metadata['age'].min():.0f} to {metadata['age'].max():.0f}")
print(f"  Gender: All {metadata['gender'].iloc[0]} (ovarian cancer study)")

print(f"\nMETHYLATION DATA SUMMARY:")
print(f"  CpG sites: {methylation_correct.shape[0]:,}")
print(f"  Samples: {methylation_correct.shape[1]}")
print(f"  Methylation range: [{methylation_correct.min().min():.3f}, {methylation_correct.max().max():.3f}]")
print(f"  Mean methylation: {methylation_correct.mean().mean():.3f}")

print(f"\nSAMPLE MATCHING SUMMARY:")
common_samples = metadata_samples.intersection(methylation_samples)
print(f"  Matching samples: {len(common_samples)} out of {len(metadata)}")

print(f"\nDESCRIPTION/SIZE:")
print(f"  X shape: {X.shape} (samples by features)")
print(f"  y shape: {y.shape} (age labels)")

print("\n" + "=" * 80)
print("USAGE INSTRUCTIONS")
print("=" * 80)

print("\n" + "=" * 80)
print("PROCESS COMPLETED SUCCESSFULLY")
print("=" * 80)
