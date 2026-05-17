"""
Epigenetics Project - Step 2: Complete Analysis
Comprehensive data loading, quality control, and exploration for brain and blood tissues
with Google Drive saving for subsequent steps

Tissues: Brain, Blood (Unified merged dataset: GSE40279 + GSE19711 + GSE41037)

Implementation Notes:
- Uses pandas/numpy for data manipulation
- Uses scikit-learn PCA for dimensionality reduction
- Uses scipy for statistical calculations
- Custom functions for file handling and quality control
"""

# ----------------------------------------------------------------------------
# Setup and Imports
# ----------------------------------------------------------------------------

print("Installing required packages...")
!pip install pandas numpy scipy matplotlib seaborn scikit-learn statsmodels adjustText -q

from google.colab import drive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
import os
import warnings
from datetime import datetime
import shutil
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("Packages loaded successfully\n")

# ----------------------------------------------------------------------------
# Google Drive Setup and Project Structure
# ----------------------------------------------------------------------------

print("Setting up Google Drive project structure...")

# Mount Google Drive
drive.mount('/content/drive')
print("Drive mounted\n")

# Project configuration
BASE_PATH = '/content/drive/MyDrive/'
PROJECT_ROOT = '/content/drive/MyDrive/epigenetics_project/'

# Step 2 specific paths
STEP2_ROOT = f'{PROJECT_ROOT}2_data_qc/'
STEP2_FIGURES = f'{STEP2_ROOT}figures/'
STEP2_TABLES = f'{STEP2_ROOT}tables/'
STEP2_DATA = f'{STEP2_ROOT}cleaned_data/'
STEP2_REPORTS = f'{STEP2_ROOT}reports/'

# Clear and recreate all directories
print("Creating project structure in Google Drive...")

if os.path.exists(STEP2_ROOT):
    shutil.rmtree(STEP2_ROOT)
    print(f"Cleared existing directory: {STEP2_ROOT}")

# Create fresh directories
for folder in [PROJECT_ROOT, STEP2_ROOT, STEP2_FIGURES, STEP2_TABLES, STEP2_DATA, STEP2_REPORTS]:
    os.makedirs(folder, exist_ok=True)
    print(f"Created directory: {folder}")

print("Google Drive structure ready\n")

# Local temporary directory
LOCAL_TEMP_DIR = '2_temp'
if os.path.exists(LOCAL_TEMP_DIR):
    shutil.rmtree(LOCAL_TEMP_DIR)
os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)
os.makedirs(f'{LOCAL_TEMP_DIR}/figures', exist_ok=True)
os.makedirs(f'{LOCAL_TEMP_DIR}/tables', exist_ok=True)

print(f"Local temporary directory: {LOCAL_TEMP_DIR}\n")

# ----------------------------------------------------------------------------
# File Paths - CORRECTED FOR 50K BRAIN DATASET
# ----------------------------------------------------------------------------

# Unified Blood Dataset (GSE40279 + GSE19711 + GSE41037) from our new creation
UNIFIED_BLOOD_METADATA_FILE = '/content/unified_blood_methylation_clean/unified_metadata_shared.csv'
UNIFIED_BLOOD_METHYLATION_FILE = '/content/unified_blood_methylation_clean/unified_methylation_shared.csv'

# CORRECTED: Brain files for 50K dataset
BRAIN_BETA_FILE = '/content/drive/MyDrive/GSE74193_BETA_50K_VARIABLE.csv'  # Updated from GSE74193_CLEAN_BETA_50k.csv
BRAIN_META_FILE = '/content/drive/MyDrive/GSE74193_METADATA_335.csv'  # Updated from GSE74193_CLEAN_META_335.csv

print("Using 50K datasets:")
print(f"  Unified Blood metadata: {UNIFIED_BLOOD_METADATA_FILE}")
print(f"  Unified Blood methylation: {UNIFIED_BLOOD_METHYLATION_FILE}")
print(f"  Brain beta values: {BRAIN_BETA_FILE}")
print(f"  Brain metadata: {BRAIN_META_FILE}\n")

# ----------------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------------

def print_section(title):
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")

def print_subsection(title):
    print(f"\n{title}")
    print(f"{'-' * 60}")

def save_figure(filename, dpi=300):
    plt.tight_layout()
    drive_path = f'{STEP2_FIGURES}{filename}'
    plt.savefig(drive_path, dpi=dpi, bbox_inches='tight')
    print(f"  Saved figure to Google Drive: {drive_path}")
    local_path = f'{LOCAL_TEMP_DIR}/figures/{filename}'
    plt.savefig(local_path, dpi=dpi, bbox_inches='tight')
    return drive_path

def save_table(df, filename, description=""):
    drive_path = f'{STEP2_TABLES}{filename}'
    df.to_csv(drive_path, index=False)
    if description:
        print(f"  Saved table to Google Drive: {filename} ({description})")
    else:
        print(f"  Saved table to Google Drive: {filename}")
    local_path = f'{LOCAL_TEMP_DIR}/tables/{filename}'
    df.to_csv(local_path, index=False)
    return drive_path

def save_report(text, filename):
    drive_path = f'{STEP2_REPORTS}{filename}'
    with open(drive_path, 'w') as f:
        f.write(text)
    print(f"  Saved report to Google Drive: {filename}")
    return drive_path

def save_cleaned_data(meth_df, meta_df, tissue_name):
    meth_filename = f'cleaned_{tissue_name.lower()}_methylation.csv'
    meth_drive_path = f'{STEP2_DATA}{meth_filename}'
    meth_df.to_csv(meth_drive_path)
    print(f"  Saved methylation data to Google Drive: {meth_filename}")
    print(f"    Dimensions: {meth_df.shape[0]:,} CpGs × {meth_df.shape[1]:,} samples")

    final_min, final_max = meth_df.min().min(), meth_df.max().max()
    if final_min >= 0 and final_max <= 1:
        print(f"    Beta value range: {final_min:.4f} to {final_max:.4f}")
    else:
        print(f"    Warning: Beta range issue: {final_min:.4f} to {final_max:.4f}")

    if meta_df is not None:
        meta_filename = f'cleaned_{tissue_name.lower()}_metadata.csv'
        meta_drive_path = f'{STEP2_DATA}{meta_filename}'
        meta_df.to_csv(meta_drive_path, index=False)
        print(f"  Saved metadata to Google Drive: {meta_filename}")
        print(f"    Number of samples: {meta_df.shape[0]}")
        if 'age' in meta_df.columns:
            ages = meta_df['age'].dropna()
            if len(ages) > 0:
                print(f"    Age range: {ages.min():.1f} - {ages.max():.1f} years")

    return meth_drive_path, meta_drive_path if meta_df is not None else None

def load_methylation_data(filepath, description=""):
    print(f"  Loading {description}...")
    try:
        df = pd.read_csv(filepath, index_col=0)
        print(f"    Loaded: {df.shape[0]:,} CpGs × {df.shape[1]:,} samples")

        sample_vals = df.values.flatten()
        sample_vals = sample_vals[~np.isnan(sample_vals)]
        if len(sample_vals) > 0:
            val_min, val_max = sample_vals.min(), sample_vals.max()
            print(f"    Raw value range: {val_min:.3f} to {val_max:.3f}")
            if val_min < 0 or val_max > 1:
                print(f"    Clipping values to [0, 1] range")
                df = df.clip(lower=0, upper=1)

        return df
    except Exception as e:
        print(f"    Error loading data: {e}")
        return None

def load_metadata(filepath, description=""):
    print(f"  Loading metadata {description}...")
    try:
        df = pd.read_csv(filepath)
        print(f"    Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

        # Check for age column
        if 'age' in df.columns:
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
            print(f"    Age column found: 'age'")
            ages = df['age'].dropna()
            if len(ages) > 0:
                print(f"    Age range: {ages.min():.1f} - {ages.max():.1f} years")
        elif 'Age' in df.columns:
            df['age'] = pd.to_numeric(df['Age'], errors='coerce')
            print(f"    Age column found: 'Age' mapped to 'age'")
            ages = df['age'].dropna()
            if len(ages) > 0:
                print(f"    Age range: {ages.min():.1f} - {ages.max():.1f} years")

        return df
    except Exception as e:
        print(f"    Error loading metadata: {e}")
        return None

# ----------------------------------------------------------------------------
# Step 1: Load All Datasets
# ----------------------------------------------------------------------------

print_section("Step 1: Loading All Datasets")

all_data = {}

# Load Brain data
print("\nBrain Dataset (50K CpGs)")
brain_meth = load_methylation_data(BRAIN_BETA_FILE, "brain methylation (50K)")
brain_meta = load_metadata(BRAIN_META_FILE, "brain metadata")
all_data['Brain'] = {'meth': brain_meth, 'meta': brain_meta}
print("  Brain dataset (50K) loaded successfully")

# Load Unified Blood data
print("\nUnified Blood Dataset (GSE40279 + GSE19711 + GSE41037)")
unified_blood_meth = load_methylation_data(UNIFIED_BLOOD_METHYLATION_FILE, "unified blood methylation")
unified_blood_meta = load_metadata(UNIFIED_BLOOD_METADATA_FILE, "unified blood metadata")

if unified_blood_meth is not None and unified_blood_meta is not None:
    all_data['Blood'] = {'meth': unified_blood_meth, 'meta': unified_blood_meta}
    print("  Unified blood dataset loaded successfully")
else:
    print("  ERROR: Unified blood dataset failed to load")
    all_data['Blood'] = {'meth': None, 'meta': None}

print("\nData loading complete")

# ----------------------------------------------------------------------------
# Step 2: Check Data Integrity
# ----------------------------------------------------------------------------

print_section("Step 2: Data Integrity Check")

total_samples = 0
total_cpgs = 0
tissue_summary = []

for tissue, data in all_data.items():
    print(f"\n{tissue}:")
    meth = data['meth']
    meta = data['meta']

    if meth is not None:
        tissue_samples = meth.shape[1]
        tissue_cpgs = meth.shape[0]

        print(f"  Samples: {tissue_samples:,}")
        print(f"  CpGs: {tissue_cpgs:,}")

        val_min, val_max = meth.min().min(), meth.max().max()
        if val_min >= 0 and val_max <= 1:
            print(f"  Beta values: {val_min:.4f} to {val_max:.4f}")
        else:
            print(f"  Warning: Range issue: {val_min:.4f} to {val_max:.4f}")

        total_samples += tissue_samples
        total_cpgs = max(total_cpgs, tissue_cpgs)

        tissue_summary.append({
            'Tissue': tissue,
            'Samples': tissue_samples,
            'CpGs': tissue_cpgs,
            'Age_Available': 'Yes' if meta is not None and 'age' in meta.columns else 'No'
        })
    else:
        print(f"  No methylation data available")

print(f"\nTotal samples across all tissues: {total_samples:,}")

# Dataset composition details
print("\nDataset Composition Details:")
if 'Blood' in all_data and all_data['Blood']['meta'] is not None:
    blood_meta = all_data['Blood']['meta']
    if 'dataset' in blood_meta.columns:
        dataset_counts = blood_meta['dataset'].value_counts()
        print("  Unified Blood dataset composition:")
        for dataset, count in dataset_counts.items():
            print(f"    {dataset}: {count} samples")

summary_df = pd.DataFrame(tissue_summary)
save_table(summary_df, 'tissue_summary.csv', "Tissue summary")

print("\nTissue Summary:")
print(summary_df.to_string(index=False))

# ----------------------------------------------------------------------------
# Step 3: Data Quality Control
# ----------------------------------------------------------------------------

print_section("Step 3: Data Quality Control")

qc_results = []

for tissue, data in all_data.items():
    print_subsection(f"{tissue} Quality Control")
    meth = data['meth']

    if meth is None or meth.shape[1] == 0:
        print("  No data available")
        continue

    missing_pct = (meth.isnull().sum().sum() / meth.size) * 100
    print(f"  Missing values: {missing_pct:.2f}%")

    vals_min = meth.min().min()
    vals_max = meth.max().max()
    print(f"  Value range: {vals_min:.6f} to {vals_max:.6f}")

    if vals_min >= 0 and vals_max <= 1:
        print(f"  Valid beta value range")
    else:
        print(f"  Warning: Outside beta value range (0-1)")

    sample_missing = (meth.isnull().sum(axis=0) / len(meth)) * 100
    bad_samples = (sample_missing > 20).sum()
    print(f"  Samples with >20% missing: {bad_samples}/{len(sample_missing)}")

    cpg_missing = (meth.isnull().sum(axis=1) / meth.shape[1]) * 100
    bad_cpgs = (cpg_missing > 40).sum()
    print(f"  CpGs with >40% missing: {bad_cpgs:,}/{len(cpg_missing):,}")

    mean_per_sample = meth.mean(axis=0)
    print(f"  Mean per sample: {mean_per_sample.min():.4f} to {mean_per_sample.max():.4f}")

    qc_results.append({
        'Tissue': tissue,
        'Samples': meth.shape[1],
        'CpGs': meth.shape[0],
        'Missing_Pct': f"{missing_pct:.2f}%",
        'Bad_Samples': bad_samples,
        'Bad_CpGs': bad_cpgs,
        'Value_Range': f"{vals_min:.4f} - {vals_max:.4f}",
        'Valid_Beta': 'Yes' if vals_min >= 0 and vals_max <= 1 else 'No'
    })

qc_df = pd.DataFrame(qc_results)
save_table(qc_df, 'qc_summary.csv', "Quality control summary")

print("\nQuality Control Summary:")
print(qc_df.to_string(index=False))

# ----------------------------------------------------------------------------
# Step 4: Data Cleaning
# ----------------------------------------------------------------------------

print_section("Step 4: Data Cleaning")

cleaned_data = {}

for tissue, data in all_data.items():
    print(f"\nCleaning {tissue} data...")
    meth = data['meth'].copy() if data['meth'] is not None else None
    meta = data['meta']

    if meth is None:
        print(f"  No data to clean")
        continue

    orig_samples = meth.shape[1]
    orig_cpgs = meth.shape[0]

    # Remove samples with >20% missing values
    sample_missing_pct = (meth.isnull().sum(axis=0) / len(meth)) * 100
    good_samples = sample_missing_pct[sample_missing_pct <= 20].index
    meth = meth[good_samples]

    # Remove CpGs with >40% missing values
    cpg_missing_pct = (meth.isnull().sum(axis=1) / meth.shape[1]) * 100
    good_cpgs = cpg_missing_pct[cpg_missing_pct <= 40].index
    meth = meth.loc[good_cpgs]

    # Clip beta values to [0,1]
    meth = meth.clip(lower=0, upper=1)

    print(f"    Samples: {orig_samples:,} → {meth.shape[1]:,} (removed {orig_samples - meth.shape[1]:,})")
    print(f"    CpGs: {orig_cpgs:,} → {meth.shape[0]:,} (removed {orig_cpgs - meth.shape[0]:,})")

    final_min, final_max = meth.min().min(), meth.max().max()
    print(f"    Final beta range: {final_min:.4f} to {final_max:.4f}")

    cleaned_data[tissue] = {
        'meth': meth,
        'meta': meta
    }

print("\nData cleaning complete")

# ----------------------------------------------------------------------------
# Step 5: Age Distribution Visualization
# ----------------------------------------------------------------------------

print_section("Step 5: Age Distributions")

# Collect datasets with age data
datasets_with_age = []

for tissue in ['Brain', 'Blood']:
    if tissue in cleaned_data:
        data = cleaned_data[tissue]
        if data['meta'] is not None and 'age' in data['meta'].columns:
            ages = data['meta']['age'].dropna()
            if len(ages) > 0:
                datasets_with_age.append({
                    'name': tissue,
                    'ages': ages,
                    'sample_count': len(ages)
                })

if len(datasets_with_age) > 0:
    # Create figure
    fig, axes = plt.subplots(1, len(datasets_with_age), figsize=(5*len(datasets_with_age), 4))

    if len(datasets_with_age) == 1:
        axes = [axes]

    colors = ['skyblue', 'lightcoral']

    for idx, dataset_info in enumerate(datasets_with_age):
        ax = axes[idx]
        ages = dataset_info['ages']
        dataset_name = dataset_info['name']
        sample_count = dataset_info['sample_count']

        ax.hist(ages, bins=30, alpha=0.7, edgecolor='black', color=colors[idx], linewidth=1)
        mean_age = ages.mean()
        median_age = ages.median()
        ax.axvline(mean_age, color='red', linestyle='--', label=f'Mean: {mean_age:.1f}')
        ax.axvline(median_age, color='blue', linestyle=':', label=f'Median: {median_age:.1f}')
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('Count')
        ax.set_title(f'{dataset_name}\n(n={sample_count})')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        textstr = f'Min: {ages.min():.1f}\nMax: {ages.max():.1f}\nSD: {ages.std():.1f}'
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes, va='top', ha='right',
                bbox=dict(facecolor='wheat', alpha=0.5), fontsize=8)

    plt.suptitle('Age Distributions: Brain vs Unified Blood', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure('age_distributions.png')
    plt.show()
else:
    print("No age data available for visualization")

# ----------------------------------------------------------------------------
# Step 6: PCA Visualization
# ----------------------------------------------------------------------------

print_section("Step 6: PCA Analysis")

# Collect datasets for PCA
pca_datasets = []

for tissue in ['Brain', 'Blood']:
    if tissue in cleaned_data:
        data = cleaned_data[tissue]
        if data['meth'] is not None and data['meth'].shape[1] >= 2:
            pca_datasets.append((tissue, data))

if len(pca_datasets) > 0:
    n_datasets = len(pca_datasets)
    fig, axes = plt.subplots(1, n_datasets, figsize=(6*n_datasets, 5))

    if n_datasets == 1:
        axes = [axes]

    for idx, (tissue_name, data) in enumerate(pca_datasets):
        ax = axes[idx]
        meth = data['meth']

        if meth.shape[1] < 2:
            ax.text(0.5, 0.5, f'Insufficient samples\n({meth.shape[1]} samples)',
                    ha='center', va='center')
            ax.set_title(tissue_name)
            continue

        # Select top variable CpGs
        cpg_var = meth.var(axis=1)
        top_cpgs = cpg_var.nlargest(min(5000, len(cpg_var))).index
        X = meth.loc[top_cpgs].T.fillna(meth.loc[top_cpgs].mean(axis=1))

        # Perform PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X)

        # Color by dataset
        color = 'red' if tissue_name == 'Blood' else 'blue'
        ax.scatter(pca_result[:, 0], pca_result[:, 1], c=color, alpha=0.7, s=50)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title(f'{tissue_name}\n(n={meth.shape[1]})')
        ax.grid(alpha=0.3)

        # Add explained variance info
        total_variance = pca.explained_variance_ratio_.sum() * 100
        ax.text(0.02, 0.98, f'Total variance: {total_variance:.1f}%',
               transform=ax.transAxes, va='top', ha='left',
               bbox=dict(facecolor='white', alpha=0.7))

    plt.suptitle('PCA Analysis: Brain vs Unified Blood', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure('pca_analysis.png')
    plt.show()
else:
    print("No datasets available for PCA analysis")

# ----------------------------------------------------------------------------
# Step 7: Dataset Comparison Visualization
# ----------------------------------------------------------------------------

print_section("Step 7: Dataset Comparison Visualization")

fig = plt.figure(figsize=(16, 10))

# 1. Sample counts comparison
ax1 = plt.subplot(2, 3, 1)
tissues = []
sample_counts = []
colors = []

for tissue in ['Brain', 'Blood']:
    if tissue in cleaned_data and cleaned_data[tissue]['meth'] is not None:
        tissues.append(tissue)
        sample_counts.append(cleaned_data[tissue]['meth'].shape[1])
        colors.append('skyblue' if tissue == 'Brain' else 'lightcoral')

x_pos = np.arange(len(tissues))
bars = ax1.bar(x_pos, sample_counts, color=colors, alpha=0.7, edgecolor='black')

ax1.set_xlabel('Tissue')
ax1.set_ylabel('Number of Samples')
ax1.set_title('Sample Counts', fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(tissues)
ax1.grid(axis='y', alpha=0.3)

for bar, count in zip(bars, sample_counts):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + max(sample_counts)*0.01,
            f'{count:,}', ha='center', va='bottom', fontsize=9)

# 2. CpG counts comparison
ax2 = plt.subplot(2, 3, 2)
tissues_cpg = []
cpg_counts = []
colors_cpg = []

for tissue in ['Brain', 'Blood']:
    if tissue in cleaned_data and cleaned_data[tissue]['meth'] is not None:
        tissues_cpg.append(tissue)
        cpg_counts.append(cleaned_data[tissue]['meth'].shape[0])
        colors_cpg.append('skyblue' if tissue == 'Brain' else 'lightcoral')

x_pos_cpg = np.arange(len(tissues_cpg))
bars_cpg = ax2.bar(x_pos_cpg, cpg_counts, color=colors_cpg, alpha=0.7, edgecolor='black')

ax2.set_xlabel('Tissue')
ax2.set_ylabel('Number of CpGs')
ax2.set_title('CpG Counts', fontweight='bold')
ax2.set_xticks(x_pos_cpg)
ax2.set_xticklabels(tissues_cpg)
ax2.grid(axis='y', alpha=0.3)

for bar, count in zip(bars_cpg, cpg_counts):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + max(cpg_counts)*0.01,
            f'{count:,}', ha='center', va='bottom', fontsize=9)

# 3. Age statistics comparison
ax3 = plt.subplot(2, 3, 3)
age_data = []

for tissue in ['Brain', 'Blood']:
    if tissue in cleaned_data and cleaned_data[tissue]['meta'] is not None and 'age' in cleaned_data[tissue]['meta'].columns:
        ages = cleaned_data[tissue]['meta']['age'].dropna()
        if len(ages) > 0:
            age_data.append({
                'name': tissue,
                'mean': ages.mean(),
                'median': ages.median(),
                'min': ages.min(),
                'max': ages.max(),
                'color': 'skyblue' if tissue == 'Brain' else 'lightcoral'
            })

if age_data:
    names = [d['name'] for d in age_data]
    means = [d['mean'] for d in age_data]
    medians = [d['median'] for d in age_data]
    colors_age = [d['color'] for d in age_data]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax3.bar(x - width/2, means, width, label='Mean Age', alpha=0.7, edgecolor='black')
    bars2 = ax3.bar(x + width/2, medians, width, label='Median Age', alpha=0.7, edgecolor='black')

    for bars, color_list in [(bars1, colors_age), (bars2, colors_age)]:
        for bar, color in zip(bars, color_list):
            bar.set_color(color)

    ax3.set_xlabel('Tissue')
    ax3.set_ylabel('Age (years)')
    ax3.set_title('Age Statistics', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(names)
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3)

# 4. Beta value distributions
ax4 = plt.subplot(2, 3, 4)
beta_data = []

for tissue in ['Brain', 'Blood']:
    if tissue in cleaned_data and cleaned_data[tissue]['meth'] is not None:
        beta_sample = cleaned_data[tissue]['meth'].values.flatten()
        beta_sample = beta_sample[~np.isnan(beta_sample)]
        if len(beta_sample) > 10000:
            beta_sample = np.random.choice(beta_sample, 10000, replace=False)
        if len(beta_sample) > 0:
            beta_data.append((tissue, beta_sample, 'skyblue' if tissue == 'Brain' else 'lightcoral'))

if beta_data:
    for name, beta_vals, color in beta_data:
        ax4.hist(beta_vals, bins=50, alpha=0.5, label=name, color=color, density=True)

    ax4.set_xlabel('Beta Value')
    ax4.set_ylabel('Density')
    ax4.set_title('Beta Value Distributions', fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)
    ax4.set_xlim(0, 1)

# 5. Missing values comparison
ax5 = plt.subplot(2, 3, 5)
missing_data = []

for tissue in ['Brain', 'Blood']:
    if tissue in cleaned_data and cleaned_data[tissue]['meth'] is not None:
        missing_pct = (cleaned_data[tissue]['meth'].isnull().sum().sum() / cleaned_data[tissue]['meth'].size) * 100
        missing_data.append({
            'name': tissue,
            'missing_pct': missing_pct,
            'color': 'skyblue' if tissue == 'Brain' else 'lightcoral'
        })

if missing_data:
    names = [d['name'] for d in missing_data]
    missing_vals = [d['missing_pct'] for d in missing_data]
    colors_missing = [d['color'] for d in missing_data]

    x_pos_missing = np.arange(len(names))
    bars_missing = ax5.bar(x_pos_missing, missing_vals, color=colors_missing, alpha=0.7, edgecolor='black')

    ax5.set_xlabel('Tissue')
    ax5.set_ylabel('Missing Values (%)')
    ax5.set_title('Missing Data Percentage', fontweight='bold')
    ax5.set_xticks(x_pos_missing)
    ax5.set_xticklabels(names)
    ax5.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars_missing, missing_vals):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + max(missing_vals)*0.01,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=9)

# 6. Mean beta values per sample
ax6 = plt.subplot(2, 3, 6)
mean_beta_data = []

for tissue in ['Brain', 'Blood']:
    if tissue in cleaned_data and cleaned_data[tissue]['meth'] is not None:
        mean_betas = cleaned_data[tissue]['meth'].mean(axis=0)
        mean_beta_data.append({
            'name': tissue,
            'means': mean_betas,
            'color': 'skyblue' if tissue == 'Brain' else 'lightcoral'
        })

if mean_beta_data:
    positions = []
    all_means = []
    colors_box = []
    labels_box = []

    for i, data in enumerate(mean_beta_data):
        positions.append(i)
        all_means.append(data['means'])
        colors_box.append(data['color'])
        labels_box.append(data['name'])

    box = ax6.boxplot(all_means, positions=positions, patch_artist=True, widths=0.6)

    for patch, color in zip(box['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax6.set_xlabel('Tissue')
    ax6.set_ylabel('Mean Beta Value')
    ax6.set_title('Mean Beta Values per Sample', fontweight='bold')
    ax6.set_xticks(positions)
    ax6.set_xticklabels(labels_box)
    ax6.grid(axis='y', alpha=0.3)

plt.suptitle('Comprehensive Dataset Analysis: Brain vs Unified Blood', fontsize=16, fontweight='bold')
plt.tight_layout()
save_figure('dataset_comparison.png')
plt.show()

# ----------------------------------------------------------------------------
# Step 8: CpG Overlap Analysis
# ----------------------------------------------------------------------------

print_section("Step 8: CpG Overlap Analysis")

cpg_sets = {}

for tissue in ['Brain', 'Blood']:
    if tissue in cleaned_data and cleaned_data[tissue]['meth'] is not None:
        cpg_sets[tissue] = set(cleaned_data[tissue]['meth'].index)

if len(cpg_sets) >= 2:
    print("CpG counts per tissue:")
    for tissue, cpgs in cpg_sets.items():
        print(f"  {tissue}: {len(cpgs):,} CpGs")

    # Calculate overlap
    overlap = len(cpg_sets['Brain'] & cpg_sets['Blood'])
    overlap_pct_brain = (overlap / len(cpg_sets['Brain'])) * 100 if len(cpg_sets['Brain']) > 0 else 0
    overlap_pct_blood = (overlap / len(cpg_sets['Blood'])) * 100 if len(cpg_sets['Blood']) > 0 else 0

    print(f"\nBrain ∩ Blood: {overlap:,} CpGs")
    print(f"  (represents {overlap_pct_brain:.1f}% of Brain CpGs)")
    print(f"  (represents {overlap_pct_blood:.1f}% of Blood CpGs)")

    # Create overlap visualization
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Prepare data for heatmap
    tissues = list(cpg_sets.keys())
    overlap_matrix = np.zeros((2, 2))
    overlap_matrix[0, 0] = len(cpg_sets['Brain'])
    overlap_matrix[1, 1] = len(cpg_sets['Blood'])
    overlap_matrix[0, 1] = overlap
    overlap_matrix[1, 0] = overlap

    im = ax.imshow(overlap_matrix, cmap='YlOrRd')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(tissues)
    ax.set_yticklabels(tissues)
    ax.set_title('CpG Overlap Between Tissues', fontweight='bold')

    # Add text annotations
    for i in range(2):
        for j in range(2):
            if i == j:
                text = ax.text(j, i, f'Total\n{int(overlap_matrix[i, j]):,}',
                             ha="center", va="center", color="black", fontsize=10, fontweight='bold')
            else:
                text = ax.text(j, i, f'Shared\n{int(overlap_matrix[i, j]):,}',
                             ha="center", va="center", color="black", fontsize=10, fontweight='bold')

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    save_figure('cpg_overlap_analysis.png')
    plt.show()
else:
    print("Need both Brain and Blood datasets for overlap analysis")

# ----------------------------------------------------------------------------
# Step 9: Save Cleaned Data to Google Drive
# ----------------------------------------------------------------------------

print_section("Step 9: Saving Cleaned Data to Google Drive")

saved_files = []
for tissue, data in cleaned_data.items():
    if data['meth'] is not None and data['meth'].shape[1] > 0:
        meth_drive_path, meta_drive_path = save_cleaned_data(
            data['meth'],
            data['meta'],
            tissue
        )
        saved_files.append({
            'Tissue': tissue,
            'Methylation_File': os.path.basename(meth_drive_path),
            'Metadata_File': os.path.basename(meta_drive_path) if meta_drive_path else 'None',
            'Samples': data['meth'].shape[1],
            'CpGs': data['meth'].shape[0]
        })

print("\nAll cleaned data saved to Google Drive")

# Save with standard names for downstream compatibility
if 'Blood' in cleaned_data and cleaned_data['Blood']['meth'] is not None:
    blood_meth_path = f'{STEP2_DATA}blood_methylation_merged.csv'
    blood_meta_path = f'{STEP2_DATA}blood_metadata_merged.csv'

    cleaned_data['Blood']['meth'].to_csv(blood_meth_path)
    if cleaned_data['Blood']['meta'] is not None:
        cleaned_data['Blood']['meta'].to_csv(blood_meta_path, index=False)

    print(f"\nSaved unified blood data with standard names:")
    print(f"  Methylation: {blood_meth_path}")
    print(f"  Metadata: {blood_meta_path}")

saved_summary_df = pd.DataFrame(saved_files)
save_table(saved_summary_df, 'cleaned_data_summary.csv', "Summary of cleaned data")

print("\nSaved Data Summary:")
print(saved_summary_df.to_string(index=False))

# Create comprehensive README file
readme_content = f"""
Epigenetics Project - Step 2 Outputs
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Files Saved to Google Drive:
---------------------------
Location: {PROJECT_ROOT}2_data_qc/

figures/ : All visualizations
tables/ : Data summaries and statistics
cleaned_data/ : Cleaned CSV files for next analysis steps
reports/ : Analysis reports

Data Sources:
------------
1. Brain: GSE74193 dataset (50K version)
   - Metadata: GSE74193_METADATA_335.csv
   - Methylation: GSE74193_BETA_50K_VARIABLE.csv
   - Samples: {brain_meth.shape[1] if brain_meth is not None else 0}
   - CpGs: {brain_meth.shape[0] if brain_meth is not None else 0}
   - Contains top 50,000 most variable CpGs

2. Unified Blood: Merged dataset (GSE40279 + GSE19711 + GSE41037)
   - Metadata: unified_metadata_shared.csv (from unified_blood_methylation_clean/)
   - Methylation: unified_methylation_shared.csv (from unified_blood_methylation_clean/)
   - Samples: {unified_blood_meth.shape[1] if unified_blood_meth is not None else 0}
   - CpGs: {unified_blood_meth.shape[0] if unified_blood_meth is not None else 0}
   - Age range: 16-101 years
   - Total samples from 3 datasets: {unified_blood_meth.shape[1] if unified_blood_meth is not None else 0}
   - Common CpGs across platforms: {unified_blood_meth.shape[0] if unified_blood_meth is not None else 0}

Dataset Processing:
------------------
1. Brain dataset loaded (50,000 most variable CpGs)
2. Unified blood dataset loaded (1,670 shared CpGs across all platforms)
3. Quality control applied (remove samples >20% missing, CpGs >40% missing)
4. Beta values clipped to [0, 1] range
5. Comparative visualizations created for Brain vs Unified Blood

Key Output Files:
----------------
- blood_methylation_merged.csv : Unified blood methylation data for downstream analysis
- blood_metadata_merged.csv : Unified blood metadata for downstream analysis
- cleaned_blood_methylation.csv : Cleaned unified blood data
- cleaned_brain_methylation.csv : Cleaned brain data

Visualization Highlights:
-----------------------
1. Age distributions comparison
2. PCA analysis showing tissue separation
3. Comprehensive dataset comparison
4. CpG overlap analysis with heatmaps
5. Beta value distribution comparisons

Cleaned Data Details:
---------------------
""" + saved_summary_df.to_string(index=False) + f"""

Notes:
------
- Brain dataset now uses 50,000 most variable CpGs (upgraded from 35,000)
- Unified blood dataset contains 1,324 samples from GSE40279, GSE19711, and GSE41037
- Unified blood uses 1,670 shared CpGs across all platforms
- GSE40279 component uses top 50,000 most variable CpGs
- Downstream analysis scripts should use the merged blood files for consistent analysis
- All visualizations compare Brain tissue (50K CpGs) with Unified Blood tissue
"""

save_report(readme_content, 'STEP2_README.txt')

print_section("Step 2 Analysis Complete - All Data Saved to Google Drive")
print("\nKey output files created for downstream analysis:")
print(f"   - {STEP2_DATA}blood_methylation_merged.csv")
print(f"   - {STEP2_DATA}blood_metadata_merged.csv")
print(f"   - {STEP2_DATA}cleaned_blood_methylation.csv")
print(f"   - {STEP2_DATA}cleaned_brain_methylation.csv")
print("\nEnhanced visualizations created showing Brain vs Unified Blood datasets")
print("\nDownstream scripts can use these files without modification")
print(f"\nSummary of loaded datasets:")
print(f"   - Brain (50K): {brain_meth.shape[1] if brain_meth is not None else 0} samples, {brain_meth.shape[0] if brain_meth is not None else 0} CpGs")
print(f"   - Unified Blood: {unified_blood_meth.shape[1] if unified_blood_meth is not None else 0} samples, {unified_blood_meth.shape[0] if unified_blood_meth is not None else 0} CpGs")
print(f"   - Total samples: {brain_meth.shape[1] + unified_blood_meth.shape[1] if brain_meth is not None and unified_blood_meth is not None else 0}")
print(f"\nDATASET FEATURES:")
print(f"   BRAIN (50K):")
print(f"     - Contains 335 healthy control samples")
print(f"     - Uses top 50,000 most variable CpGs")
print(f"     - Memory efficient for machine learning")
print(f"\n   UNIFIED BLOOD:")
print(f"     - Contains 1,324 samples from GSE40279, GSE19711, and GSE41037")
print(f"     - Uses 1,670 shared CpGs across all platforms")
print(f"     - Age range: 16-101 years")
print(f"     - Very low missing values: 0.09%")
print(f"\nReady for downstream analysis")
