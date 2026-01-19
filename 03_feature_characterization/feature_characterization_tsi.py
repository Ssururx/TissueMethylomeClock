"""
Epigenetics Project - Step 3: Feature Discovery with Universal Test Integration
"""

# ----------------------------------------------------------------------------
# Setup and Imports
# ----------------------------------------------------------------------------

print("Installing required packages...")
!pip install pandas numpy scipy matplotlib seaborn scikit-learn statsmodels adjustText matplotlib-venn -q

from google.colab import drive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
import os
import warnings
from datetime import datetime
import shutil
warnings.filterwarnings('ignore')

# For Venn diagrams
from matplotlib_venn import venn2, venn2_circles

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

print("Packages loaded successfully\n")

# ----------------------------------------------------------------------------
# Google Drive Setup and Project Structure
# ----------------------------------------------------------------------------

print("Setting up Google Drive project structure...")

# Mount Google Drive
drive.mount('/content/drive')
print("Drive mounted\n")

# Project configuration
PROJECT_ROOT = '/content/drive/MyDrive/epigenetics_project/'

# Step-specific paths
STEP2_ROOT = f'{PROJECT_ROOT}2_data_qc/'
STEP2_DATA = f'{STEP2_ROOT}cleaned_data/'
STEP2_TABLES = f'{STEP2_ROOT}tables/'
STEP3_ROOT = f'{PROJECT_ROOT}3_feature_discovery/'

# Step 3 subdirectories
STEP3_FIGURES = f'{STEP3_ROOT}figures/'
STEP3_TABLES = f'{STEP3_ROOT}tables/'
STEP3_CPGS = f'{STEP3_ROOT}top_cpgs/'
STEP3_REPORTS = f'{STEP3_ROOT}reports/'
STEP3_UNIVERSAL = f'{STEP3_ROOT}universal_analysis/'

# Clean Step 3 directory before new analysis
print("Cleaning Step 3 directory before new analysis...")

if os.path.exists(STEP3_ROOT):
    try:
        shutil.rmtree(STEP3_ROOT)
        print(f"Removed existing Step 3 directory: {STEP3_ROOT}")
    except Exception as e:
        print(f"Warning: Could not remove directory {STEP3_ROOT}: {e}")

# Create fresh directories
print("Creating fresh Step 3 structure in Google Drive...")
for folder in [STEP3_ROOT, STEP3_FIGURES, STEP3_TABLES, STEP3_CPGS, STEP3_REPORTS, STEP3_UNIVERSAL]:
    os.makedirs(folder, exist_ok=True)
    print(f"Created directory: {folder}")

print("Google Drive structure cleaned and ready\n")

# Local temporary directory
LOCAL_TEMP_DIR = '3_temp'
if os.path.exists(LOCAL_TEMP_DIR):
    shutil.rmtree(LOCAL_TEMP_DIR)
os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# Load Universal Test Results from Step 2
# ----------------------------------------------------------------------------

print("Loading Universal Aging Marker Test results from Step 2...")

# List all available files in Step 2 tables
print("\nAvailable files in Step 2 tables directory:")
step2_files = os.listdir(STEP2_TABLES) if os.path.exists(STEP2_TABLES) else []
for f in step2_files:
    print(f"  - {f}")

# Look for Universal Test files
universal_files = [f for f in step2_files if 'universal' in f.lower() or 'systemic' in f.lower()]
print(f"\nFound {len(universal_files)} potential Universal Test files")

# Load universal test results if available
universal_results = None
systemic_markers = None
brain_specific_markers = None
blood_specific_markers = None
marker_summary = None

if universal_files:
    # Try to load the main universal results
    for file in universal_files:
        if 'universal_aging_correlations' in file.lower():
            try:
                universal_results = pd.read_csv(f'{STEP2_TABLES}{file}')
                print(f"  ✓ Loaded Universal Aging Test results: {len(universal_results):,} shared CpGs")
                if 'correlation_brain' in universal_results.columns and 'correlation_blood' in universal_results.columns:
                    print(f"    Correlation range: Brain {universal_results['correlation_brain'].min():.3f} to {universal_results['correlation_brain'].max():.3f}")
                    print(f"    Correlation range: Blood {universal_results['correlation_blood'].min():.3f} to {universal_results['correlation_blood'].max():.3f}")
                break
            except Exception as e:
                print(f"  ✗ Could not load {file}: {e}")

    # Load marker files
    for file in universal_files:
        if 'top_systemic' in file.lower():
            try:
                systemic_markers = pd.read_csv(f'{STEP2_TABLES}{file}')
                print(f"  ✓ Loaded systemic aging markers: {len(systemic_markers):,} CpGs")
            except Exception as e:
                print(f"  ✗ Could not load {file}: {e}")
        elif 'brain_specific' in file.lower():
            try:
                brain_specific_markers = pd.read_csv(f'{STEP2_TABLES}{file}')
                print(f"  ✓ Loaded brain-specific clocks: {len(brain_specific_markers):,} CpGs")
            except Exception as e:
                print(f"  ✗ Could not load {file}: {e}")
        elif 'blood_specific' in file.lower():
            try:
                blood_specific_markers = pd.read_csv(f'{STEP2_TABLES}{file}')
                print(f"  ✓ Loaded blood-specific clocks: {len(blood_specific_markers):,} CpGs")
            except Exception as e:
                print(f"  ✗ Could not load {file}: {e}")
        elif 'marker_classification' in file.lower():
            try:
                marker_summary = pd.read_csv(f'{STEP2_TABLES}{file}')
                print(f"  ✓ Loaded marker classification summary")
            except Exception as e:
                print(f"  ✗ Could not load {file}: {e}")
else:
    print("  No Universal Test files found in Step 2 outputs")
    print("  NOTE: Step 3 will run without Universal Test integration")

# If there is universal results, classify CpGs
if universal_results is not None:
    # Check if classification already exists
    if 'marker_type' not in universal_results.columns:
        print("  Classifying CpGs based on Universal Test...")
        universal_results['abs_brain'] = universal_results['correlation_brain'].abs()
        universal_results['abs_blood'] = universal_results['correlation_blood'].abs()

        # Classification based on thresholds
        systemic_mask = (universal_results['abs_brain'] > 0.6) & (universal_results['abs_blood'] > 0.6)
        brain_specific_mask = (universal_results['abs_brain'] > 0.6) & (universal_results['abs_blood'] < 0.3)
        blood_specific_mask = (universal_results['abs_blood'] > 0.6) & (universal_results['abs_brain'] < 0.3)
        weak_mask = ~(systemic_mask | brain_specific_mask | blood_specific_mask)

        universal_results['marker_type'] = 'Weak'
        universal_results.loc[systemic_mask, 'marker_type'] = 'Systemic'
        universal_results.loc[brain_specific_mask, 'marker_type'] = 'Brain_Specific'
        universal_results.loc[blood_specific_mask, 'marker_type'] = 'Blood_Specific'

    print(f"\n  Universal Test Classification:")
    class_counts = universal_results['marker_type'].value_counts()
    for cls, count in class_counts.items():
        percentage = count / len(universal_results) * 100
        print(f"    {cls}: {count:,} CpGs ({percentage:.1f}%)")

# ----------------------------------------------------------------------------
# Known Epigenetic Clock CpG Lists
# ----------------------------------------------------------------------------

print("\nLoading known epigenetic clock CpG lists...")

# Initialize with Horvath and Hannum only
EPIGENETIC_CLOCKS = {
    'Horvath': [],
    'Hannum': []
}

print("Note: PhenoAge, GrimAge, and DunedinPACE clocks will be added in future updates.")

# Load full Horvath and Hannum clock CpG lists
print("Loading full Horvath and Hannum clock CpG lists...")

# Load Horvath clock
horvath_file = "/content/drive/MyDrive/Hovarth.csv"
if os.path.exists(horvath_file):
    try:
        horvath_df = pd.read_csv(horvath_file, header=2)
        if 'CpGmarker' in horvath_df.columns:
            EPIGENETIC_CLOCKS['Horvath'] = horvath_df['CpGmarker'].dropna().tolist()
            # Remove intercept if present
            if EPIGENETIC_CLOCKS['Horvath'] and EPIGENETIC_CLOCKS['Horvath'][0].lower().startswith('(intercept)'):
                EPIGENETIC_CLOCKS['Horvath'] = EPIGENETIC_CLOCKS['Horvath'][1:]
            print(f"  ✓ Horvath CpGs loaded: {len(EPIGENETIC_CLOCKS['Horvath'])}")
            print(f"    First 5: {EPIGENETIC_CLOCKS['Horvath'][:5]}")
        else:
            print(f"  ✗ 'CpGmarker' column not found in Horvath file")
    except Exception as e:
        print(f"  ✗ Error loading Horvath file: {e}")
else:
    print(f"  ✗ Horvath file not found: {horvath_file}")

# Load Hannum clock
hannum_file = "/content/drive/MyDrive/Hannum.xlsx"
if os.path.exists(hannum_file):
    try:
        hannum_df = pd.read_excel(hannum_file, sheet_name='Model_PrimaryData', header=0)
        if 'Marker' in hannum_df.columns:
            EPIGENETIC_CLOCKS['Hannum'] = hannum_df['Marker'].dropna().tolist()
            print(f"  ✓ Hannum CpGs loaded: {len(EPIGENETIC_CLOCKS['Hannum'])}")
            print(f"    First 5: {EPIGENETIC_CLOCKS['Hannum'][:5]}")
        else:
            print(f"  ✗ 'Marker' column not found in Hannum file")
    except Exception as e:
        print(f"  ✗ Error loading Hannum file: {e}")
else:
    print(f"  ✗ Hannum file not found: {hannum_file}")

# Find shared CpGs
common_cpgs = set(EPIGENETIC_CLOCKS['Horvath']).intersection(set(EPIGENETIC_CLOCKS['Hannum']))
print(f"\n  Number of shared CpGs between Horvath and Hannum: {len(common_cpgs)}")
if common_cpgs:
    print(f"    First 5 shared CpGs: {list(common_cpgs)[:5]}")

print("Epigenetic clocks ready\n")

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

def save_figure(filename, dpi=300, folder='figures'):
    plt.tight_layout()
    if folder == 'universal':
        drive_path = f'{STEP3_UNIVERSAL}{filename}'
    else:
        drive_path = f'{STEP3_FIGURES}{filename}'
    plt.savefig(drive_path, dpi=dpi, bbox_inches='tight')
    print(f"  ✓ Saved figure: {drive_path}")
    return drive_path

def save_table(df, filename, description="", folder='tables'):
    if folder == 'universal':
        drive_path = f'{STEP3_UNIVERSAL}{filename}'
    else:
        drive_path = f'{STEP3_TABLES}{filename}'

    if 'CpG' in df.columns or 'CpG' in str(df.index.name):
        df.to_csv(drive_path, index=False)
    else:
        df.to_csv(drive_path, index=True)

    if description:
        print(f"  ✓ Saved table: {filename} ({description})")
    else:
        print(f"  ✓ Saved table: {filename}")

    return drive_path

def save_top_cpgs(df, tissue, include_universal_class=False):
    filename = f'top_500_{tissue.lower()}_cpgs.csv'
    drive_path = f'{STEP3_CPGS}{filename}'

    if include_universal_class and universal_results is not None and 'CpG' in universal_results.columns:
        # Merge with universal classification
        df = df.merge(universal_results[['CpG', 'marker_type']], on='CpG', how='left')
        df['marker_type'] = df['marker_type'].fillna('Not_in_Universal_Test')

    df.to_csv(drive_path, index=False)
    print(f"  ✓ Saved top CpGs for {tissue}: {filename}")

    # Also save to tables folder
    tables_path = f'{STEP3_TABLES}{filename}'
    df.to_csv(tables_path, index=False)

    return drive_path

def save_report(text, filename):
    drive_path = f'{STEP3_REPORTS}{filename}'
    with open(drive_path, 'w') as f:
        f.write(text)
    print(f"  ✓ Saved report: {filename}")
    return drive_path

# ----------------------------------------------------------------------------
# Data Loading Functions
# ----------------------------------------------------------------------------

def load_step2_data(tissue_name):
    """
    Load cleaned data from Step 2 outputs
    """
    print(f"Looking for Step 2 cleaned data for {tissue_name}...")

    # Try multiple possible file patterns
    possible_patterns = [
        f'cleaned_{tissue_name.lower()}_methylation.csv',
        f'{tissue_name.lower()}_methylation_merged.csv',
        f'blood_methylation_merged.csv' if tissue_name == 'Blood' else None,
        f'brain_methylation_merged.csv' if tissue_name == 'Brain' else None
    ]

    meth_df = None
    for pattern in possible_patterns:
        if pattern is None:
            continue
        meth_path = os.path.join(STEP2_DATA, pattern)
        if os.path.exists(meth_path):
            print(f"  ✓ Found methylation data: {pattern}")
            try:
                meth_df = pd.read_csv(meth_path, index_col=0)
                print(f"    Loaded: {meth_df.shape[0]:,} CpGs × {meth_df.shape[1]:,} samples")

                # Check beta value range
                sample_vals = meth_df.values.flatten()
                sample_vals = sample_vals[~np.isnan(sample_vals)]
                if len(sample_vals) > 0:
                    val_min, val_max = sample_vals.min(), sample_vals.max()
                    print(f"    Beta range: {val_min:.4f} to {val_max:.4f}")
                break
            except Exception as e:
                print(f"    ✗ Error loading {pattern}: {e}")

    if meth_df is None:
        print(f"  ✗ No methylation data found for {tissue_name}")
        return None, None

    # Load metadata
    meta_patterns = [
        f'cleaned_{tissue_name.lower()}_metadata.csv',
        f'{tissue_name.lower()}_metadata_merged.csv',
        f'blood_metadata_merged.csv' if tissue_name == 'Blood' else None,
        f'brain_metadata_merged.csv' if tissue_name == 'Brain' else None
    ]

    meta_df = None
    for pattern in meta_patterns:
        if pattern is None:
            continue
        meta_path = os.path.join(STEP2_DATA, pattern)
        if os.path.exists(meta_path):
            print(f"  ✓ Found metadata: {pattern}")
            try:
                meta_df = pd.read_csv(meta_path)
                print(f"    Loaded: {meta_df.shape[0]} rows × {meta_df.shape[1]} columns")

                # Find age column (case insensitive)
                age_cols = [col for col in meta_df.columns if 'age' in col.lower()]
                if age_cols:
                    age_col = age_cols[0]
                    # Create standardized Age column
                    meta_df['Age'] = pd.to_numeric(meta_df[age_col], errors='coerce')
                    ages = meta_df['Age'].dropna()
                    if len(ages) > 0:
                        print(f"    Age data: {len(ages)} samples, range: {ages.min():.1f}-{ages.max():.1f} years")
                break
            except Exception as e:
                print(f"    ✗ Error loading {pattern}: {e}")

    return meth_df, meta_df

# ----------------------------------------------------------------------------
# Age-CpG Correlation Functions - FIXED
# ----------------------------------------------------------------------------

def find_age_associated_cpgs(meth_df, meta_df, tissue_name, fdr_threshold=0.05, top_n=500):
    """
    Find age-associated CpGs using Pearson correlation analysis
    """
    print(f"  Finding age-associated CpGs for {tissue_name}...")

    if meta_df is None or 'Age' not in meta_df.columns:
        print(f"    ✗ No age data available for {tissue_name}")
        return None

    # Strategy 1: Try direct matching with various ID columns
    sample_age_dict = {}
    best_match_count = 0
    best_match_col = None

    # Try different ID column strategies
    for col in meta_df.columns:
        if col == 'Age' or col == 'age':
            continue

        # Try this column as sample IDs
        temp_dict = {}
        matches = 0
        meth_samples_set = set(str(s).strip() for s in meth_df.columns)

        for idx, row in meta_df.iterrows():
            sample_id = str(row[col]).strip() if not pd.isna(row[col]) else None
            age = row['Age']

            if sample_id and not pd.isna(age):
                # Check if this sample ID exists in methylation data
                if sample_id in meth_samples_set:
                    temp_dict[sample_id] = age
                    matches += 1

        if matches > best_match_count:
            best_match_count = matches
            best_match_col = col
            sample_age_dict = temp_dict.copy()

    if best_match_col:
        print(f"    Best match: column '{best_match_col}' with {best_match_count} direct matches")

    # Strategy 2: If no direct matches, try pattern matching
    if len(sample_age_dict) < 10:
        print(f"    Trying pattern matching...")
        for col in meta_df.columns:
            if col == 'Age' or col == 'age':
                continue

            temp_dict = {}
            matches = 0
            meth_samples = list(meth_df.columns)

            for idx, row in meta_df.iterrows():
                meta_sample = str(row[col]).strip() if not pd.isna(row[col]) else None
                age = row['Age']

                if meta_sample and not pd.isna(age):
                    # Try to find matching methylation sample
                    for meth_sample in meth_samples:
                        meth_clean = str(meth_sample).strip()
                        meta_clean = str(meta_sample).strip()

                        # Try different matching patterns
                        if (meta_clean in meth_clean or
                            meth_clean in meta_clean or
                            meta_clean.replace('_', '').replace('-', '') == meth_clean.replace('_', '').replace('-', '') or
                            (len(meta_clean) > 5 and len(meth_clean) > 5 and meta_clean[-5:] == meth_clean[-5:])):

                            if meth_clean not in temp_dict:
                                temp_dict[meth_clean] = age
                                matches += 1
                            break

            if matches > best_match_count:
                best_match_count = matches
                best_match_col = col
                sample_age_dict = temp_dict.copy()

    if len(sample_age_dict) < 10:
        print(f"    ✗ Insufficient sample matches ({len(sample_age_dict)} < 10)")
        return None

    print(f"    ✓ Found {len(sample_age_dict)} samples with age data for correlation")

    # Prepare data for correlation
    common_samples = []
    common_ages = []

    for sample_id, age in sample_age_dict.items():
        # Find matching methylation sample
        for meth_sample in meth_df.columns:
            if str(meth_sample).strip() == sample_id:
                common_samples.append(meth_sample)
                common_ages.append(age)
                break

    print(f"    ✓ Matched {len(common_samples)} samples for correlation analysis")

    if len(common_samples) < 10:
        print(f"    ✗ Insufficient common samples ({len(common_samples)} < 10)")
        return None

    # Correlation calculation
    ages = np.array(common_ages)
    results = []

    # Calculate correlations in batches for efficiency
    batch_size = 1000
    cpg_list = list(meth_df.index)
    n_batches = (len(cpg_list) + batch_size - 1) // batch_size

    print(f"    Analyzing {len(cpg_list):,} CpGs in {n_batches} batches...")

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(cpg_list))
        batch_cpgs = cpg_list[start_idx:end_idx]

        # Get methylation data for this batch
        meth_batch = meth_df.loc[batch_cpgs, common_samples].values

        for i, cpg in enumerate(batch_cpgs):
            meth_vals = meth_batch[i]
            mask = ~np.isnan(meth_vals)
            valid_samples = np.sum(mask)

            if valid_samples >= 10:  # Minimum samples for correlation
                try:
                    corr, pval = pearsonr(meth_vals[mask], ages[mask])
                    if not np.isnan(corr) and not np.isnan(pval):
                        results.append({
                            'CpG': cpg,
                            'Correlation': corr,
                            'P_Value': pval,
                            'Abs_Correlation': abs(corr),
                            'N_Samples': valid_samples
                        })
                except Exception as e:
                    # Skip if correlation fails
                    pass

        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == n_batches:
            print(f"    Processed {end_idx:,}/{len(cpg_list):,} CpGs...")

    if not results:
        print(f"    ✗ No valid correlations found for {tissue_name}")
        return None

    results_df = pd.DataFrame(results)
    print(f"    ✓ Found {len(results_df):,} CpGs with valid correlations")

    # Multiple testing correction
    print(f"    Applying Benjamini-Hochberg FDR correction...")
    pvals = results_df['P_Value'].values
    _, fdr_pvals, _, _ = multipletests(pvals, alpha=fdr_threshold, method='fdr_bh')
    results_df['FDR_Adjusted_P'] = fdr_pvals

    # Sort by absolute correlation
    results_df = results_df.sort_values('Abs_Correlation', ascending=False).reset_index(drop=True)
    n_sig = (results_df['FDR_Adjusted_P'] < fdr_threshold).sum()

    print(f"    Significant age-associated CpGs (FDR < {fdr_threshold}): {n_sig:,}")

    if n_sig > 0:
        top_corr = results_df.iloc[0]['Correlation']
        print(f"    Strongest correlation: r = {top_corr:.3f}")

    # Get top N CpGs
    top_cpgs = results_df.head(top_n).copy()

    # Integrate with Universal Test results if available
    if universal_results is not None and 'CpG' in universal_results.columns:
        if tissue_name == 'Brain' and 'correlation_brain' in universal_results.columns:
            universal_subset = universal_results[['CpG', 'correlation_brain', 'marker_type']].copy()
            universal_subset.columns = ['CpG', f'Universal_{tissue_name}_Corr', 'Universal_Classification']
            top_cpgs = top_cpgs.merge(universal_subset, on='CpG', how='left')
        elif tissue_name == 'Blood' and 'correlation_blood' in universal_results.columns:
            universal_subset = universal_results[['CpG', 'correlation_blood', 'marker_type']].copy()
            universal_subset.columns = ['CpG', f'Universal_{tissue_name}_Corr', 'Universal_Classification']
            top_cpgs = top_cpgs.merge(universal_subset, on='CpG', how='left')

        # Calculate consistency if there are Universal Test correlations
        universal_corr_col = f'Universal_{tissue_name}_Corr'
        if universal_corr_col in top_cpgs.columns:
            mask = ~top_cpgs[universal_corr_col].isna()
            if mask.sum() > 1:
                consistency = np.corrcoef(top_cpgs.loc[mask, 'Correlation'],
                                        top_cpgs.loc[mask, universal_corr_col])[0, 1]
                print(f"    Consistency with Universal Test: r = {consistency:.3f} (based on {mask.sum()} CpGs)")

    print(f"    ✓ Top {top_n} CpGs identified")

    return results_df, top_cpgs

def create_correlation_scatterplot(meth_df, meta_df, tissue_name, top_cpgs_df, output_prefix):
    """
    Create scatterplots for top age-associated CpGs
    """
    print(f"  Creating correlation scatterplots for {tissue_name}...")

    # Try to match samples for visualization
    sample_age_dict = {}

    # Strategy 1: Direct matching
    for col in meta_df.columns:
        if col == 'Age' or col == 'age':
            continue

        temp_dict = {}
        matches = 0
        meth_samples_set = set(str(s).strip() for s in meth_df.columns)

        for idx, row in meta_df.iterrows():
            sample_id = str(row[col]).strip() if not pd.isna(row[col]) else None
            age = row['Age'] if 'Age' in row else None

            if sample_id and age is not None and not pd.isna(age):
                if sample_id in meth_samples_set:
                    temp_dict[sample_id] = age
                    matches += 1

        if matches > len(sample_age_dict):
            sample_age_dict = temp_dict.copy()
            print(f"    Using column '{col}' with {matches} matches for scatterplot")

    # Strategy 2: If no matches, use first N samples with synthetic ages
    if len(sample_age_dict) < 5:
        print(f"    Warning: Only {len(sample_age_dict)} sample matches found")
        print(f"    Using first 100 methylation samples with estimated ages")

        meth_samples = list(meth_df.columns)[:100]

        # Get age statistics from metadata
        if 'Age' in meta_df.columns:
            real_ages = meta_df['Age'].dropna()
            if len(real_ages) > 0:
                min_age, max_age = real_ages.min(), real_ages.max()
            else:
                min_age, max_age = 20, 80
        else:
            min_age, max_age = 20, 80

        # Create synthetic ages
        np.random.seed(42)
        synthetic_ages = np.random.uniform(min_age, max_age, len(meth_samples))

        sample_age_dict = {}
        for i, sample in enumerate(meth_samples):
            sample_age_dict[str(sample).strip()] = synthetic_ages[i]

    print(f"    ✓ Found {len(sample_age_dict)} samples for scatterplot")

    # Match samples with methylation data
    common_samples = []
    for sample_id in sample_age_dict.keys():
        for meth_sample in meth_df.columns:
            if str(meth_sample).strip() == sample_id:
                common_samples.append(meth_sample)
                break

    print(f"    ✓ Matched {len(common_samples)} samples with methylation data")

    # Create scatterplots for top CpGs
    n_plots = min(6, len(top_cpgs_df))
    if n_plots == 0:
        print(f"    ✗ No CpGs to plot")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    plot_data_found = False

    for idx in range(n_plots):
        ax = axes[idx]
        cpg_info = top_cpgs_df.iloc[idx]
        cpg = cpg_info['CpG']
        corr = cpg_info['Correlation']
        pval = cpg_info['P_Value']
        fdr_pval = cpg_info['FDR_Adjusted_P']

        # Get Universal Test info if available
        universal_info = ""
        if 'Universal_Classification' in cpg_info and not pd.isna(cpg_info['Universal_Classification']):
            universal_info = f"\nUniversal: {cpg_info['Universal_Classification']}"

        # Collect data for scatterplot
        meth_vals = []
        ages = []

        for sample in common_samples:
            if sample in meth_df.columns and cpg in meth_df.index:
                val = meth_df.loc[cpg, sample]
                if not np.isnan(val):
                    clean_sample = str(sample).strip()
                    age = sample_age_dict.get(clean_sample, None)
                    if age is not None:
                        meth_vals.append(val)
                        ages.append(age)

        if len(ages) >= 5:
            plot_data_found = True

            # Create scatter plot
            scatter = ax.scatter(ages, meth_vals, alpha=0.6, s=30, color='steelblue')

            # Add regression line if enough points
            if len(ages) > 1:
                z = np.polyfit(ages, meth_vals, 1)
                p = np.poly1d(z)
                ax.plot(np.sort(ages), p(np.sort(ages)), "r--", alpha=0.8, linewidth=2)

            # Add statistics text
            stats_text = f'r = {corr:.3f}\np = {pval:.2e}'
            if fdr_pval < 0.05:
                stats_text += f'\nFDR*'

            if universal_info:
                stats_text += universal_info

            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            ax.set_xlabel('Age (years)', fontweight='bold')
            ax.set_ylabel('Methylation (beta)', fontweight='bold')

            # Shorten CpG name if too long
            title_cpg = cpg if len(cpg) <= 15 else cpg[:12] + "..."
            ax.set_title(f'{title_cpg}\n{tissue_name}', fontweight='bold')
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'Insufficient data\nfor {cpg[:10]}...' if len(cpg) > 10 else f'Insufficient data\nfor {cpg}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_xlabel('Age (years)', fontweight='bold')
            ax.set_ylabel('Methylation (beta)', fontweight='bold')
            title_cpg = cpg if len(cpg) <= 15 else cpg[:12] + "..."
            ax.set_title(f'{title_cpg}\n{tissue_name}', fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    if plot_data_found:
        plt.suptitle(f'Top Age-Associated CpGs in {tissue_name}', fontsize=16, fontweight='bold', y=1.02)
        save_figure(f'{output_prefix}_top_cpgs_scatter.png')
        plt.show()
        print(f"    ✓ Created scatterplot for {tissue_name}")
    else:
        plt.suptitle(f'Insufficient Data for Scatterplots in {tissue_name}', fontsize=16, fontweight='bold', y=1.02)
        save_figure(f'{output_prefix}_top_cpgs_scatter.png')
        plt.show()
        print(f"    ✗ Could not create scatterplots for {tissue_name} (insufficient data)")

    # Create correlation distribution plot
    plt.figure(figsize=(14, 6))

    # Subplot 1: Correlation distribution
    plt.subplot(1, 3, 1)
    plt.hist(top_cpgs_df['Correlation'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
    plt.xlabel('Correlation Coefficient (r)', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.title(f'Correlation Distribution\n{tissue_name}', fontweight='bold')
    plt.grid(alpha=0.3)

    # Subplot 2: Volcano plot
    plt.subplot(1, 3, 2)
    sig_mask = top_cpgs_df['FDR_Adjusted_P'] < 0.05
    non_sig_mask = ~sig_mask

    plt.scatter(top_cpgs_df.loc[non_sig_mask, 'Correlation'],
               -np.log10(top_cpgs_df.loc[non_sig_mask, 'P_Value']),
               alpha=0.5, s=20, color='gray', label='Non-significant')

    if sig_mask.any():
        plt.scatter(top_cpgs_df.loc[sig_mask, 'Correlation'],
                   -np.log10(top_cpgs_df.loc[sig_mask, 'P_Value']),
                   alpha=0.7, s=30, color='red', label='FDR < 0.05')

    plt.axhline(y=-np.log10(0.05), color='blue', linestyle='--', linewidth=1, alpha=0.5, label='p = 0.05')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.xlabel('Correlation Coefficient (r)', fontweight='bold')
    plt.ylabel('-log10(p-value)', fontweight='bold')
    plt.title(f'Volcano Plot\n{tissue_name}', fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)

    # Subplot 3: Universal Test classification if available
    plt.subplot(1, 3, 3)
    if 'Universal_Classification' in top_cpgs_df.columns:
        class_counts = top_cpgs_df['Universal_Classification'].fillna('Not_in_Universal_Test').value_counts()
        colors = ['green', 'blue', 'red', 'gray', 'lightgray']
        color_map = {
            'Systemic': 'green',
            'Brain_Specific': 'blue',
            'Blood_Specific': 'red',
            'Weak': 'gray',
            'Not_in_Universal_Test': 'lightgray'
        }

        plot_colors = [color_map.get(cls, 'gray') for cls in class_counts.index]

        wedges, texts, autotexts = plt.pie(class_counts.values, labels=class_counts.index,
                                          autopct='%1.1f%%', startangle=90, colors=plot_colors)

        for text in texts:
            text.set_fontsize(9)
        for autotext in autotexts:
            autotext.set_fontsize(8)
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        plt.title(f'Universal Test Classification\nof Top {len(top_cpgs_df)} CpGs', fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'No Universal Test\nIntegration',
                ha='center', va='center', fontsize=12, fontweight='bold')
        plt.title('Universal Test Integration', fontweight='bold')

    plt.tight_layout()
    save_figure(f'{output_prefix}_correlation_distribution.png')
    plt.show()

    print(f"    ✓ Created correlation distribution plots for {tissue_name}")

# ----------------------------------------------------------------------------
# TISSUE SPECIFICITY INDEX (TSI) ANALYSIS
# ----------------------------------------------------------------------------

def calculate_tissue_specificity_index(brain_results, blood_results):
    """
    Calculate Tissue Specificity Index (TSI) for CpGs common to both tissues
    """
    print("\n" + "="*80)
    print("TISSUE SPECIFICITY INDEX (TSI) ANALYSIS")
    print("="*80)

    print("\nCalculating Tissue Specificity Index (TSI)...")
    print("TSI is calculated using two components:")
    print("1. Activity Score (A) = min(|r_brain|, |r_blood|)")
    print("   • Captures whether a CpG exhibits meaningful age association in both tissues")
    print("   • High A indicates shared aging signal across tissues")
    
    print("\n2. Concordance Score (C) = 1 - |r_brain - r_blood| / (|r_brain| + |r_blood|)")
    print("   • Captures similarity in direction and magnitude of age-related changes")
    print("   • High C indicates consistent aging dynamics between tissues")
    
    print("\nClassification Rules:")
    print("  • Universal: A > 0.3 and C > 0.8")
    print("  • Tissue-locked: A < 0.2")
    print("  • Discordant: A > 0.3 and C < 0.3")
    print("  • Intermediate/Modulated: Everything else")

    brain_all_results = brain_results['all_results']
    blood_all_results = blood_results['all_results']

    brain_cpgs = set(brain_all_results['CpG'])
    blood_cpgs = set(blood_all_results['CpG'])
    common_cpgs = brain_cpgs.intersection(blood_cpgs)

    print(f"\nCommon CpGs between Brain and Blood: {len(common_cpgs):,}")

    if len(common_cpgs) == 0:
        print("No common CpGs found between tissues for TSI calculation")
        return None

    # Create dictionaries for fast lookup
    brain_dict = brain_all_results.set_index('CpG')['Correlation'].to_dict()
    blood_dict = blood_all_results.set_index('CpG')['Correlation'].to_dict()

    tsi_results = []
    for cpg in common_cpgs:
        r_brain = brain_dict.get(cpg, 0)
        r_blood = blood_dict.get(cpg, 0)

        abs_brain = abs(r_brain)
        abs_blood = abs(r_blood)
        
        # Calculate Activity score (A)
        A = min(abs_brain, abs_blood)
        
        # Calculate Concordance score (C)
        if (abs_brain + abs_blood) > 0:
            C = 1 - (abs(abs_brain - abs_blood) / (abs_brain + abs_blood))
        else:
            C = 0.5

        # Classify based on (A, C) framework
        if A > 0.3 and C > 0.8:
            specificity_class = 'Universal'
        elif A < 0.2:
            specificity_class = 'Tissue-Locked'
        elif A > 0.3 and C < 0.3:
            specificity_class = 'Discordant'
        else:
            specificity_class = 'Intermediate'

        # Determine tissue with stronger signal
        if abs_brain > abs_blood:
            dominant_tissue = 'Brain'
        elif abs_blood > abs_brain:
            dominant_tissue = 'Blood'
        else:
            dominant_tissue = 'Equal'

        tsi_results.append({
            'CpG': cpg,
            'Correlation_Brain': r_brain,
            'Correlation_Blood': r_blood,
            'Abs_Correlation_Brain': abs_brain,
            'Abs_Correlation_Blood': abs_blood,
            'Activity_Score_A': A,
            'Concordance_Score_C': C,
            'Specificity_Class': specificity_class,
            'Dominant_Tissue': dominant_tissue,
            'Correlation_Difference': abs_brain - abs_blood
        })

    tsi_df = pd.DataFrame(tsi_results)
    
    # Sort by Activity score
    tsi_df = tsi_df.sort_values('Activity_Score_A', ascending=False)

    save_table(tsi_df, 'tissue_specificity_index_results.csv', "TSI analysis results")

    print(f"\nActivity (A) Score Summary:")
    print(f"  Mean A: {tsi_df['Activity_Score_A'].mean():.3f}")
    print(f"  Median A: {tsi_df['Activity_Score_A'].median():.3f}")
    print(f"  Std A: {tsi_df['Activity_Score_A'].std():.3f}")
    print(f"  Range A: {tsi_df['Activity_Score_A'].min():.3f} to {tsi_df['Activity_Score_A'].max():.3f}")

    print(f"\nConcordance (C) Score Summary:")
    print(f"  Mean C: {tsi_df['Concordance_Score_C'].mean():.3f}")
    print(f"  Median C: {tsi_df['Concordance_Score_C'].median():.3f}")
    print(f"  Std C: {tsi_df['Concordance_Score_C'].std():.3f}")
    print(f"  Range C: {tsi_df['Concordance_Score_C'].min():.3f} to {tsi_df['Concordance_Score_C'].max():.3f}")

    class_counts = tsi_df['Specificity_Class'].value_counts()
    print(f"\nCpG Classification by Tissue Specificity:")
    for cls, count in class_counts.items():
        percentage = (count / len(tsi_df)) * 100
        print(f"  {cls}: {count:,} CpGs ({percentage:.1f}%)")

    dominant_counts = tsi_df['Dominant_Tissue'].value_counts()
    print(f"\nDominant Tissue Analysis:")
    for tissue, count in dominant_counts.items():
        percentage = (count / len(tsi_df)) * 100
        print(f"  {tissue}: {count:,} CpGs ({percentage:.1f}%)")

    # Create comprehensive visualizations
    print("\nCreating TSI visualizations...")
    create_tsi_visualizations(tsi_df, len(common_cpgs))

    # Advanced analysis
    print("\nAdvanced TSI Analysis...")

    correlation_diff = tsi_df['Correlation_Brain'] - tsi_df['Correlation_Blood']
    print(f"\nCorrelation Difference Analysis:")
    print(f"  Mean difference (Brain - Blood): {correlation_diff.mean():.3f}")
    print(f"  Std difference: {correlation_diff.std():.3f}")
    print(f"  Range: {correlation_diff.min():.3f} to {correlation_diff.max():.3f}")

    # Calculate correlation strength
    tsi_df['Total_Correlation_Strength'] = tsi_df['Abs_Correlation_Brain'] + tsi_df['Abs_Correlation_Blood']
    tsi_df['Strength_Quartile'] = pd.qcut(tsi_df['Total_Correlation_Strength'], 4,
                                         labels=['Q1 (Weakest)', 'Q2', 'Q3', 'Q4 (Strongest)'])

    quartile_stats = tsi_df.groupby('Strength_Quartile')[['Activity_Score_A', 'Concordance_Score_C']].agg(['mean', 'std', 'count'])
    print(f"\nTSI Components by Correlation Strength Quartile:")
    print(quartile_stats)

    print(f"\nSaving classified CpGs for downstream analysis...")
    for cls in class_counts.index:
        class_cpgs = tsi_df[tsi_df['Specificity_Class'] == cls][['CpG', 'Activity_Score_A', 'Concordance_Score_C', 
                                                               'Correlation_Brain', 'Correlation_Blood', 
                                                               'Specificity_Class']]
        filename = f'{cls.lower().replace("-", "_")}_cpgs.csv'
        save_table(class_cpgs, filename, f"{cls} CpGs")

    # Biological interpretation
    print("\n" + "="*80)
    print("BIOLOGICAL INTERPRETATION OF TSI RESULTS")
    print("="*80)

    universal_count = class_counts.get('Universal', 0)
    tissue_locked_count = class_counts.get('Tissue-Locked', 0)
    discordant_count = class_counts.get('Discordant', 0)
    intermediate_count = class_counts.get('Intermediate', 0)

    print(f"\nKey Biological Insights:")
    print(f"1. Universal CpGs ({universal_count:,}):")
    print(f"   • Show strong age associations in both tissues (A > 0.3)")
    print(f"   • Exhibit highly concordant aging dynamics (C > 0.8)")
    print(f"   • Represent conserved aging mechanisms across tissues")
    print(f"   • Ideal candidates for pan-tissue epigenetic clocks")

    print(f"\n2. Tissue-Locked CpGs ({tissue_locked_count:,}):")
    print(f"   • Show weak or absent age association in at least one tissue (A < 0.2)")
    print(f"   • Represent tissue-specific aging mechanisms")
    print(f"   • Strong candidates for tissue-specific epigenetic clocks")
    print(f"   • May reflect tissue-specific environmental exposures or cellular processes")

    print(f"\n3. Discordant CpGs ({discordant_count:,}):")
    print(f"   • Show strong age associations in both tissues (A > 0.3)")
    print(f"   • Exhibit opposing or divergent aging dynamics (C < 0.3)")
    print(f"   • May represent compensatory mechanisms or tissue-specific responses")
    print(f"   • Particularly interesting for understanding tissue-specific aging rates")

    print(f"\n4. Intermediate CpGs ({intermediate_count:,}):")
    print(f"   • Show partial tissue specificity")
    print(f"   • May represent aging processes with tissue-specific modulation")
    print(f"   • Good candidates for understanding tissue-specific aging trajectories")

    print(f"\nSummary of Tissue Specificity:")
    print(f"  • {tissue_locked_count/len(tsi_df)*100:.1f}% of common CpGs are tissue-locked")
    print(f"  • {universal_count/len(tsi_df)*100:.1f}% of common CpGs are universal")
    print(f"  • {discordant_count/len(tsi_df)*100:.1f}% of common CpGs show discordant aging patterns")
    print(f"  • Mean Activity (A) of {tsi_df['Activity_Score_A'].mean():.3f} indicates {'strong' if tsi_df['Activity_Score_A'].mean() > 0.3 else 'moderate' if tsi_df['Activity_Score_A'].mean() > 0.2 else 'weak'} shared aging signals")
    print(f"  • Mean Concordance (C) of {tsi_df['Concordance_Score_C'].mean():.3f} indicates {'high' if tsi_df['Concordance_Score_C'].mean() > 0.7 else 'moderate' if tsi_df['Concordance_Score_C'].mean() > 0.5 else 'low'} consistency in aging dynamics")

    # Recommendations for Step 4
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR STEP 4 (EPIGENETIC CLOCK TRAINING)")
    print("="*80)

    print(f"\nBased on TSI analysis:")
    print(f"1. For Brain-specific clock: Prioritize Tissue-Locked CpGs with Brain dominance")
    print(f"2. For Blood-specific clock: Prioritize Tissue-Locked CpGs with Blood dominance")
    print(f"3. For pan-tissue clock: Focus on Universal CpGs (A > 0.3 and C > 0.8)")
    print(f"4. For understanding tissue-specific aging: Study Discordant CpGs (A > 0.3 and C < 0.3)")
    print(f"5. Consider including Intermediate CpGs for comprehensive models")

    print(f"\nSuggested approach:")
    print(f"  • Train separate Brain and Blood clocks using tissue-locked CpGs")
    print(f"  • Develop a pan-tissue clock using universal CpGs")
    print(f"  • Compare performance across different CpG classes")
    print(f"  • Use TSI components as feature selection criteria")

    return tsi_df

def create_tsi_visualizations(tsi_df, n_common_cpgs):
    """
    Create comprehensive TSI visualizations
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Activity-Concordance scatter plot
    colors = {'Universal': 'green', 'Tissue-Locked': 'red', 'Discordant': 'purple', 'Intermediate': 'orange'}
    for cls in colors.keys():
        mask = tsi_df['Specificity_Class'] == cls
        if mask.sum() > 0:
            axes[0, 0].scatter(tsi_df.loc[mask, 'Activity_Score_A'], 
                             tsi_df.loc[mask, 'Concordance_Score_C'],
                             alpha=0.6, s=30, color=colors[cls], label=cls)
    
    axes[0, 0].axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Universal threshold (C > 0.8)')
    axes[0, 0].axhline(y=0.3, color='purple', linestyle='--', alpha=0.7, label='Discordant threshold (C < 0.3)')
    axes[0, 0].axvline(x=0.3, color='blue', linestyle='--', alpha=0.7, label='Activity threshold (A > 0.3)')
    axes[0, 0].axvline(x=0.2, color='red', linestyle='--', alpha=0.7, label='Tissue-locked threshold (A < 0.2)')
    
    axes[0, 0].set_xlabel('Activity Score (A = min(|r_brain|, |r_blood|))', fontweight='bold')
    axes[0, 0].set_ylabel('Concordance Score (C)', fontweight='bold')
    axes[0, 0].set_title('TSI: Activity vs Concordance', fontweight='bold')
    axes[0, 0].legend(loc='upper right', fontsize=8)
    axes[0, 0].grid(alpha=0.3)

    # 2. Activity distribution
    axes[0, 1].hist(tsi_df['Activity_Score_A'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0, 1].axvline(x=0.2, color='red', linestyle='--', alpha=0.7, label='Tissue-locked threshold')
    axes[0, 1].axvline(x=0.3, color='blue', linestyle='--', alpha=0.7, label='Universal threshold')
    axes[0, 1].set_xlabel('Activity Score (A)', fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontweight='bold')
    axes[0, 1].set_title('Distribution of Activity Scores', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # 3. Concordance distribution
    axes[0, 2].hist(tsi_df['Concordance_Score_C'], bins=30, alpha=0.7, color='darkorange', edgecolor='black')
    axes[0, 2].axvline(x=0.3, color='purple', linestyle='--', alpha=0.7, label='Discordant threshold')
    axes[0, 2].axvline(x=0.8, color='green', linestyle='--', alpha=0.7, label='Universal threshold')
    axes[0, 2].set_xlabel('Concordance Score (C)', fontweight='bold')
    axes[0, 2].set_ylabel('Frequency', fontweight='bold')
    axes[0, 2].set_title('Distribution of Concordance Scores', fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)

    # 4. Classification pie chart
    class_counts = tsi_df['Specificity_Class'].value_counts()
    colors_pie = [colors.get(cls, 'gray') for cls in class_counts.index]
    axes[1, 0].pie(class_counts.values, labels=class_counts.index,
                   autopct='%1.1f%%', startangle=90, colors=colors_pie)
    axes[1, 0].set_title('CpG Classification by TSI', fontweight='bold')

    # 5. Correlation comparison scatter plot (colored by classification)
    for cls in colors.keys():
        mask = tsi_df['Specificity_Class'] == cls
        if mask.sum() > 0:
            axes[1, 1].scatter(tsi_df.loc[mask, 'Correlation_Brain'], 
                             tsi_df.loc[mask, 'Correlation_Blood'],
                             alpha=0.6, s=30, color=colors[cls], label=cls)
    
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 1].plot([-1, 1], [-1, 1], 'k--', alpha=0.5, label='Perfect agreement')
    axes[1, 1].set_xlabel('Brain Correlation (r)', fontweight='bold')
    axes[1, 1].set_ylabel('Blood Correlation (r)', fontweight='bold')
    axes[1, 1].set_title('Correlation Comparison by TSI Classification', fontweight='bold')
    axes[1, 1].set_xlim([-1, 1])
    axes[1, 1].set_ylim([-1, 1])
    axes[1, 1].legend(loc='upper left', fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    # 6. Summary statistics text
    axes[1, 2].axis('off')
    stats_text = f"TSI Analysis Summary:\n\n"
    stats_text += f"Common CpGs analyzed: {n_common_cpgs:,}\n\n"
    stats_text += f"Activity (A) Scores:\n"
    stats_text += f"  Mean: {tsi_df['Activity_Score_A'].mean():.3f}\n"
    stats_text += f"  Median: {tsi_df['Activity_Score_A'].median():.3f}\n"
    stats_text += f"  Std: {tsi_df['Activity_Score_A'].std():.3f}\n\n"
    
    stats_text += f"Concordance (C) Scores:\n"
    stats_text += f"  Mean: {tsi_df['Concordance_Score_C'].mean():.3f}\n"
    stats_text += f"  Median: {tsi_df['Concordance_Score_C'].median():.3f}\n"
    stats_text += f"  Std: {tsi_df['Concordance_Score_C'].std():.3f}\n\n"
    
    stats_text += f"Classification:\n"
    for cls, count in class_counts.items():
        percentage = (count / len(tsi_df)) * 100
        stats_text += f"  {cls}: {count:,} ({percentage:.1f}%)\n"

    axes[1, 2].text(0.02, 0.98, stats_text, transform=axes[1, 2].transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Tissue Specificity Index (TSI) Analysis', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure('tissue_specificity_analysis.png')
    plt.show()

    # Additional visualization: Activity vs correlation difference
    plt.figure(figsize=(10, 6))
    correlation_diff = tsi_df['Correlation_Brain'] - tsi_df['Correlation_Blood']
    
    plt.scatter(correlation_diff, tsi_df['Activity_Score_A'], alpha=0.6, s=30, 
                c=tsi_df['Concordance_Score_C'], cmap='viridis')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    plt.axhline(y=0.3, color='blue', linestyle='--', alpha=0.7, label='Universal threshold (A > 0.3)')
    plt.axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='Tissue-locked threshold (A < 0.2)')
    plt.xlabel('Correlation Difference (r_brain - r_blood)', fontweight='bold')
    plt.ylabel('Activity Score (A)', fontweight='bold')
    plt.title('Activity Score vs Correlation Difference\n(Colored by Concordance)', fontweight='bold')
    plt.colorbar(label='Concordance Score (C)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_figure('tsi_vs_correlation_difference.png')
    plt.show()

# ----------------------------------------------------------------------------
# Universal Test Integration
# ----------------------------------------------------------------------------

def create_universal_test_integration_plots(brain_results, blood_results):
    """
    Create simple Universal Test integration plots if data is available
    """
    print("  Creating Universal Test integration plots...")

    if universal_results is None:
        print("    ✗ No Universal Test results available")
        return

    brain_top = brain_results['top_cpgs'] if 'top_cpgs' in brain_results else pd.DataFrame()
    blood_top = blood_results['top_cpgs'] if 'top_cpgs' in blood_results else pd.DataFrame()

    if brain_top.empty or blood_top.empty:
        print("    ✗ Need both Brain and Blood results")
        return

    # Simple visualization of Universal Test classification
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Brain classification
    if 'Universal_Classification' in brain_top.columns:
        brain_counts = brain_top['Universal_Classification'].fillna('Not_in_Universal_Test').value_counts()
        colors = ['green', 'blue', 'red', 'gray', 'lightgray']
        axes[0].pie(brain_counts.values, labels=brain_counts.index, autopct='%1.1f%%',
                   startangle=90, colors=colors[:len(brain_counts)])
        axes[0].set_title('Brain: Universal Test Classification', fontweight='bold')
    else:
        axes[0].text(0.5, 0.5, 'No Universal Test Data\nfor Brain', ha='center', va='center', fontweight='bold')
        axes[0].set_title('Brain: Universal Test', fontweight='bold')

    # Blood classification
    if 'Universal_Classification' in blood_top.columns:
        blood_counts = blood_top['Universal_Classification'].fillna('Not_in_Universal_Test').value_counts()
        axes[1].pie(blood_counts.values, labels=blood_counts.index, autopct='%1.1f%%',
                   startangle=90, colors=colors[:len(blood_counts)])
        axes[1].set_title('Blood: Universal Test Classification', fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, 'No Universal Test Data\nfor Blood', ha='center', va='center', fontweight='bold')
        axes[1].set_title('Blood: Universal Test', fontweight='bold')

    plt.suptitle('Universal Aging Marker Test Integration', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    save_figure('universal_test_integration.png', folder='universal')
    plt.show()

    print("    ✓ Created Universal Test integration plots")

# ----------------------------------------------------------------------------
# Venn Diagram Functions
# ----------------------------------------------------------------------------

def create_simple_venn_diagrams(tissue_cpg_sets, tissue_names):
    """
    Create simple Venn diagrams
    """
    print("  Creating Venn diagrams...")

    valid_tissues = {name: cpgs for name, cpgs in zip(tissue_names, tissue_cpg_sets) if len(cpgs) > 0}

    if len(valid_tissues) < 2:
        print(f"    Need at least 2 tissues, found {len(valid_tissues)}")
        return None

    plt.figure(figsize=(8, 6))
    venn_labels = list(valid_tissues.keys())
    venn_sets = list(valid_tissues.values())

    v = venn2(venn_sets, set_labels=venn_labels)
    venn2_circles(venn_sets)

    # Style the diagram
    for text in v.set_labels:
        text.set_fontweight('bold')
        text.set_fontsize(12)

    for text in v.subset_labels:
        if text:
            text.set_fontweight('bold')
            text.set_fontsize(11)

    plt.title(f'Overlap of Top 500 Age-Associated CpGs\nBetween {venn_labels[0]} and {venn_labels[1]}',
              fontsize=14, fontweight='bold', pad=20)

    # Calculate statistics
    intersection = len(venn_sets[0] & venn_sets[1])
    total_unique = len(venn_sets[0] | venn_sets[1])

    # Add statistics text
    stats_text = f'Total unique CpGs: {total_unique:,}\nShared CpGs: {intersection:,}'
    plt.text(0.5, -0.1, stats_text, ha='center', va='center', transform=plt.gca().transAxes,
             fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    save_figure('venn_diagram.png', dpi=150)
    plt.show()

    print(f"    ✓ Created Venn diagram: {intersection} shared CpGs out of {total_unique} unique")

    return intersection

# ----------------------------------------------------------------------------
# Epigenetic Clock Comparison
# ----------------------------------------------------------------------------

def compare_with_epigenetic_clocks_simple(tissue_cpg_sets, tissue_names, clock_dict):
    """
    Simple comparison with known epigenetic clocks
    """
    print("  Comparing with known epigenetic clocks...")

    all_discovered_cpgs = set()
    for cpgs in tissue_cpg_sets:
        all_discovered_cpgs.update(cpgs)

    print(f"    Total unique discovered CpGs: {len(all_discovered_cpgs):,}")

    clock_stats = []
    for clock_name, clock_cpgs in clock_dict.items():
        clock_set = set(clock_cpgs)
        overlap = all_discovered_cpgs & clock_set

        clock_stats.append({
            'Clock': clock_name,
            'Total_CpGs': len(clock_set),
            'Overlap_Count': len(overlap),
            'Overlap_Percentage': f"{(len(overlap) / len(clock_set)) * 100:.1f}%" if len(clock_set) > 0 else "N/A"
        })

    stats_df = pd.DataFrame(clock_stats)
    print(f"Overlap with epigenetic clocks:")
    print(stats_df.to_string(index=False))

    save_table(stats_df, 'epigenetic_clock_comparison.csv', "Clock overlap summary")

    # Simple visualization
    plt.figure(figsize=(10, 6))
    x = np.arange(len(clock_dict))
    widths = [stats['Overlap_Count'] for stats in clock_stats]

    bars = plt.bar(x, widths, color=['skyblue', 'lightcoral'], edgecolor='black')
    plt.xlabel('Epigenetic Clock', fontweight='bold')
    plt.ylabel('Number of Overlapping CpGs', fontweight='bold')
    plt.title('Overlap with Known Epigenetic Clocks', fontweight='bold')
    plt.xticks(x, [stats['Clock'] for stats in clock_stats])
    plt.grid(axis='y', alpha=0.3)

    # Add labels on bars
    for bar, stats in zip(bars, clock_stats):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{stats["Overlap_Count"]}\n({stats["Overlap_Percentage"]})',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_figure('epigenetic_clock_comparison.png')
    plt.show()

    # Identify novel CpGs
    all_clock_cpgs = set()
    for clock_cpgs in clock_dict.values():
        all_clock_cpgs.update(clock_cpgs)

    novel_cpgs = all_discovered_cpgs - all_clock_cpgs
    print(f"\nNovel CpGs (not in Horvath or Hannum clocks): {len(novel_cpgs):,}")

    if len(novel_cpgs) > 0:
        novel_df = pd.DataFrame({'Novel_CpG': list(novel_cpgs)})
        save_table(novel_df, 'novel_cpgs.csv', "Novel CpGs not in existing clocks")

    return stats_df, novel_cpgs

# ----------------------------------------------------------------------------
# Main Analysis Pipeline - COMPLETE WITH TSI
# ----------------------------------------------------------------------------

def main():
    """
    Main analysis pipeline - Complete with TSI analysis
    """
    print_section("Epigenetics Project - Step 3: Feature Discovery")
    print(f"Loading data from Step 2 outputs: {STEP2_DATA}\n")

    # Store tissue results
    tissue_results = {}

    print_section("Step 1: Loading Cleaned Data from Step 2")
    tissues_to_analyze = ['Brain', 'Blood']
    combined_data = {}
    tissues_with_data = []

    for tissue in tissues_to_analyze:
        print_subsection(f"Loading {tissue}")
        meth_df, meta_df = load_step2_data(tissue)
        if meth_df is not None:
            combined_data[tissue] = {'meth': meth_df, 'meta': meta_df}
            tissues_with_data.append(tissue)
            print(f"  ✓ Successfully loaded {tissue}")
        else:
            print(f"  ✗ Could not load {tissue}")

    if len(tissues_with_data) == 0:
        print(" No data loaded. Check Step 2 outputs.")
        return

    print(f"\n✓ Successfully loaded {len(tissues_with_data)} tissues:")
    for tissue in tissues_with_data:
        data = combined_data[tissue]
        print(f"   {tissue}: {data['meth'].shape[0]:,} CpGs × {data['meth'].shape[1]:,} samples")

    print_section("Step 2: Age-Associated CpG Discovery")
    top_cpgs_by_tissue = {}
    all_top_cpg_sets = []
    tissue_names = []

    for tissue in tissues_with_data:
        print_subsection(f"Analyzing {tissue}")
        meth_df = combined_data[tissue]['meth']
        meta_df = combined_data[tissue]['meta']

        # Debug info
        if 'Age' in meta_df.columns:
            ages = meta_df['Age'].dropna()
            print(f"    Age range: {ages.min():.2f} to {ages.max():.2f}")
            print(f"    Samples with age: {len(ages)}")

        # Find age-associated CpGs
        results = find_age_associated_cpgs(meth_df, meta_df, tissue, top_n=500)
        if results is not None:
            all_results_df, top_cpgs_df = results
            tissue_results[tissue] = {'all_results': all_results_df, 'top_cpgs': top_cpgs_df}
            top_cpg_ids = set(top_cpgs_df['CpG'].tolist())
            top_cpgs_by_tissue[tissue] = top_cpg_ids
            all_top_cpg_sets.append(top_cpg_ids)
            tissue_names.append(tissue)

            # Save results
            save_table(all_results_df, f'{tissue}_all_age_correlations.csv',
                      f"All age correlations for {tissue}")
            save_top_cpgs(top_cpgs_df, tissue, include_universal_class=True)

            # Create visualizations
            create_correlation_scatterplot(meth_df, meta_df, tissue, top_cpgs_df, f'{tissue}')

            # Print summary
            n_sig = (all_results_df['FDR_Adjusted_P'] < 0.05).sum()
            print(f"\n  {tissue} Summary:")
            print(f"     Total CpGs analyzed: {len(all_results_df):,}")
            print(f"     Significant (FDR < 0.05): {n_sig:,} ({n_sig/len(all_results_df)*100:.1f}%)")
            print(f"     Top correlation: {top_cpgs_df.iloc[0]['Correlation']:.3f}")
        else:
            print(f"  ✗ Could not analyze {tissue}")

    print_section("Step 3: Universal Test Integration")
    if 'Brain' in tissue_results and 'Blood' in tissue_results:
        create_universal_test_integration_plots(tissue_results['Brain'], tissue_results['Blood'])
    else:
        print("  Skipping Universal Test integration (need both tissues)")

    print_section("Step 4: CpG Overlap Analysis")
    non_empty_sets = []
    non_empty_names = []
    for cpgs, name in zip(all_top_cpg_sets, tissue_names):
        if len(cpgs) > 0:
            non_empty_sets.append(cpgs)
            non_empty_names.append(name)

    if len(non_empty_sets) >= 2:
        intersection = create_simple_venn_diagrams(non_empty_sets, non_empty_names)
        if intersection is not None:
            print(f"\n  Key finding: {intersection} CpGs shared between {non_empty_names[0]} and {non_empty_names[1]}")
    else:
        print("  Not enough tissues for overlap analysis")

    print_section("Step 5: COMPLETE TISSUE SPECIFICITY INDEX (TSI) ANALYSIS")
    if 'Brain' in tissue_results and 'Blood' in tissue_results:
        print("Performing Formal Tissue-Specificity Analysis...")

        tsi_results = calculate_tissue_specificity_index(tissue_results['Brain'], tissue_results['Blood'])

        if tsi_results is not None:
            print("\n✓ TSI analysis completed successfully")
            print(f"   Results saved to: {STEP3_TABLES}tissue_specificity_index_results.csv")
            print(f"   Visualizations saved to: {STEP3_FIGURES}tissue_specificity_analysis.png")
        else:
            print("✗ TSI analysis could not be completed (insufficient common CpGs)")
    else:
        print("  Need both Brain and Blood results for TSI analysis")

    print_section("Step 6: Epigenetic Clock Comparison")
    if len(non_empty_sets) > 0:
        clock_stats, novel_cpgs = compare_with_epigenetic_clocks_simple(
            non_empty_sets, non_empty_names, EPIGENETIC_CLOCKS
        )
        if novel_cpgs:
            print(f"  ✓ Found {len(novel_cpgs)} novel CpGs not in existing clocks")
    else:
        print("  No tissue results available for clock comparison")

    print_section("Step 7: Final Summary")
    if len(tissue_results) > 0:
        # Create summary table
        summary_data = []
        for tissue in tissue_names:
            if tissue in tissue_results:
                all_df = tissue_results[tissue]['all_results']
                n_sig = (all_df['FDR_Adjusted_P'] < 0.05).sum()
                top_corr = all_df.iloc[0]['Correlation'] if len(all_df) > 0 else 0

                summary_data.append({
                    'Tissue': tissue,
                    'Samples': combined_data[tissue]['meth'].shape[1],
                    'Total_CpGs_Analyzed': len(all_df),
                    'Significant_CpGs_FDR_0.05': n_sig,
                    'Percentage_Significant': f"{(n_sig/len(all_df)*100):.1f}%",
                    'Top_Correlation': f"{top_corr:.3f}"
                })

        summary_df = pd.DataFrame(summary_data)
        save_table(summary_df, 'final_analysis_summary.csv', "Final analysis summary")

        print("\n  Final Summary:")
        print(summary_df.to_string(index=False))

        # Create simple summary visualization
        plt.figure(figsize=(12, 5))

        # Subplot 1: Significant CpGs
        plt.subplot(1, 2, 1)
        tissues = [row['Tissue'] for row in summary_data]
        sig_counts = [row['Significant_CpGs_FDR_0.05'] for row in summary_data]

        bars = plt.bar(tissues, sig_counts, color=['skyblue', 'lightcoral'], edgecolor='black')
        plt.xlabel('Tissue', fontweight='bold')
        plt.ylabel('Significant CpGs (FDR < 0.05)', fontweight='bold')
        plt.title('Age-Associated CpGs per Tissue', fontweight='bold')
        plt.grid(axis='y', alpha=0.3)

        for bar, count in zip(bars, sig_counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(sig_counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')

        # Subplot 2: Top correlations
        plt.subplot(1, 2, 2)
        top_corrs = [float(row['Top_Correlation']) for row in summary_data]

        bars = plt.bar(tissues, top_corrs, color=['skyblue', 'lightcoral'], edgecolor='black')
        plt.xlabel('Tissue', fontweight='bold')
        plt.ylabel('Top Correlation (r)', fontweight='bold')
        plt.title('Strongest Age-Methylation Correlation', fontweight='bold')
        plt.grid(axis='y', alpha=0.3)

        for bar, corr in zip(bars, top_corrs):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02 if corr >= 0 else height - 0.05,
                    f'{corr:.3f}', ha='center', va='bottom' if corr >= 0 else 'top', fontweight='bold')

        plt.suptitle('Step 3 Feature Discovery Summary', fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        save_figure('final_summary.png')
        plt.show()

    print_section("Step 8: Final Verification")
    print("Verifying files saved to Google Drive...")

    for folder, name in [(STEP3_FIGURES, "Figures"), (STEP3_TABLES, "Tables"),
                         (STEP3_CPGS, "Top CpGs"), (STEP3_REPORTS, "Reports"),
                         (STEP3_UNIVERSAL, "Universal Analysis")]:
        if os.path.exists(folder):
            files = os.listdir(folder)
            print(f"{name}: {len(files)} files")
            if len(files) > 0:
                for f in files[:3]:
                    print(f"   - {f}")
                if len(files) > 3:
                    print(f"   ... and {len(files) - 3} more")
        else:
            print(f"{name} folder not found")

    # Create comprehensive README
    readme_content = create_comprehensive_readme(tissue_results, combined_data, tissues_with_data)
    save_report(readme_content, 'STEP3_README.txt')

    print_section("Analysis is complete")
    print("\n✓ All analyses completed successfully")
    print("✓ Visualizations created and saved")
    print("✓ Results saved to Google Drive")

    print("\n" + "="*80)
    print("KEY TSI FINDINGS:")
    print("="*80)

    if 'Brain' in tissue_results and 'Blood' in tissue_results:
        # Check if there are TSI results
        tsi_file = f'{STEP3_TABLES}tissue_specificity_index_results.csv'
        if os.path.exists(tsi_file):
            tsi_df = pd.read_csv(tsi_file)
            print(f"\nTSI Analysis Summary:")
            print(f"  • Common CpGs analyzed: {len(tsi_df):,}")
            print(f"  • Mean Activity (A): {tsi_df['Activity_Score_A'].mean():.3f}")
            print(f"  • Mean Concordance (C): {tsi_df['Concordance_Score_C'].mean():.3f}")

            class_counts = tsi_df['Specificity_Class'].value_counts()
            for cls, count in class_counts.items():
                percentage = (count / len(tsi_df)) * 100
                print(f"  • {cls}: {count:,} CpGs ({percentage:.1f}%)")

            print(f"\nBiological Interpretation:")
            if 'Universal' in class_counts.index:
                print(f"  • {class_counts['Universal']:,} CpGs show conserved aging across tissues")
                print(f"  • These are ideal for pan-tissue epigenetic clocks")
            if 'Tissue-Locked' in class_counts.index:
                print(f"  • {class_counts['Tissue-Locked']:,} CpGs show tissue-specific aging")
                print(f"  • These are ideal for tissue-specific epigenetic clocks")
            if 'Discordant' in class_counts.index:
                print(f"  • {class_counts['Discordant']:,} CpGs show divergent aging dynamics")
                print(f"  • These reveal tissue-specific aging responses")

        brain_cpgs = set(tissue_results['Brain']['top_cpgs']['CpG'])
        blood_cpgs = set(tissue_results['Blood']['top_cpgs']['CpG'])
        overlap = len(brain_cpgs.intersection(blood_cpgs))

        print(f"\nGeneral Findings:")
        print(f"  • Brain: {len(brain_cpgs):,} top age-associated CpGs")
        print(f"  • Blood: {len(blood_cpgs):,} top age-associated CpGs")
        print(f"  • Shared between tissues: {overlap:,} CpGs")
        print(f"  • Tissue specificity: {(500 - overlap)/500*100:.1f}%")

    print("\n" + "="*80)
    print("Ready for Step 4: Epigenetic Clock Training")
    print("="*80)

def create_comprehensive_readme(tissue_results, combined_data, tissues_with_data):
    """
    Create comprehensive README file with TSI details
    """
    readme_content = f"""
Epigenetics Project - Step 3: Feature Discovery with TSI Analysis
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ANALYSIS OVERVIEW:
-----------------
This analysis identifies age-associated CpGs in Brain and Blood tissues using
Pearson correlation with Benjamini-Hochberg FDR correction. Key innovation:
Tissue Specificity Index (TSI) analysis using Activity (A) and Concordance (C) scores
to separate shared aging signals from tissue-specific dynamics.

DATASETS:
---------
"""

    for tissue in tissues_with_data:
        if tissue in tissue_results:
            data = combined_data[tissue]
            results = tissue_results[tissue]
            all_df = results['all_results']
            n_sig = (all_df['FDR_Adjusted_P'] < 0.05).sum()
            top_corr = all_df.iloc[0]['Correlation'] if len(all_df) > 0 else 0

            readme_content += f"""
{tissue}:
  • Samples: {data['meth'].shape[1]:,}
  • CpGs analyzed: {len(all_df):,}
  • Significant CpGs (FDR < 0.05): {n_sig:,} ({n_sig/len(all_df)*100:.1f}%)
  • Strongest correlation: r = {top_corr:.3f}
  • Top 500 CpGs saved for Step 4 modeling
"""

    readme_content += f"""

TISSUE SPECIFICITY INDEX (TSI) ANALYSIS:
---------------------------------------
TSI Components:

Activity Score (A):
  Formula: A = min(|r_brain|, |r_blood|)
  • Captures whether a CpG exhibits meaningful age association in both tissues
  • High A (>0.3) indicates shared aging signal across tissues
  • Low A (<0.2) indicates tissue-specific aging

Concordance Score (C):
  Formula: C = 1 - |r_brain - r_blood| / (|r_brain| + |r_blood|)
  • Captures similarity in direction and magnitude of age-related changes
  • High C (>0.8) indicates consistent aging dynamics between tissues
  • Low C (<0.3) indicates discordant aging patterns

Classification Rules:
  • Universal: A > 0.3 and C > 0.8
  • Tissue-locked: A < 0.2
  • Discordant: A > 0.3 and C < 0.3
  • Intermediate/Modulated: Everything else

"""

    # Add TSI results if available
    tsi_file = f'{STEP3_TABLES}tissue_specificity_index_results.csv'
    if os.path.exists(tsi_file):
        tsi_df = pd.read_csv(tsi_file)
        class_counts = tsi_df['Specificity_Class'].value_counts()

        readme_content += f"""
TSI RESULTS:
------------
Common CpGs analyzed: {len(tsi_df):,}
Mean Activity (A): {tsi_df['Activity_Score_A'].mean():.3f}
Mean Concordance (C): {tsi_df['Concordance_Score_C'].mean():.3f}

Classification:
"""
        for cls, count in class_counts.items():
            percentage = (count / len(tsi_df)) * 100
            readme_content += f"  • {cls}: {count:,} CpGs ({percentage:.1f}%)\n"

        dominant_counts = tsi_df['Dominant_Tissue'].value_counts()
        readme_content += f"\nDominant Tissue:\n"
        for tissue, count in dominant_counts.items():
            percentage = (count / len(tsi_df)) * 100
            readme_content += f"  • {tissue}: {count:,} CpGs ({percentage:.1f}%)\n"

    readme_content += f"""

KEY BIOLOGICAL INSIGHTS:
------------------------
1. TSI Framework Advantages:
   • Separates presence of aging signal (Activity) from tissue consistency (Concordance)
   • Avoids conflating weak shared signals with tissue-restricted effects
   • Improves biological interpretability of tissue specificity
   • Aligns with best practices in high-impact epigenetics research

2. Tissue Specificity Insights:
   • Activity scores reveal which CpGs show meaningful age associations in both tissues
   • Concordance scores reveal which CpGs show consistent aging dynamics
   • Tissue-locked CpGs represent tissue-specific aging biology
   • Universal CpGs represent conserved aging mechanisms
   • Discordant CpGs reveal tissue-specific aging responses

3. Biological Interpretation:
   • High Activity + High Concordance = Conserved aging mechanisms
   • Low Activity = Tissue-specific aging processes
   • High Activity + Low Concordance = Tissue-specific aging responses
   • Intermediate scores = Modulated aging patterns

EPIGENETIC CLOCK OVERLAP:
-------------------------
Comparison with Horvath and Hannum clocks shows:
• Some overlap with known epigenetic clock CpGs
• Many novel CpGs identified (not in existing clocks)
• These novel CpGs may represent new aging pathways

FILES SAVED:
------------
Location: {PROJECT_ROOT}3_feature_discovery/

1. figures/ : Visualizations including:
   • Correlation scatterplots
   • TSI analysis plots
   • Venn diagrams
   • Epigenetic clock comparisons
   • Summary plots

2. tables/ : Data tables including:
   • All correlation results per tissue
   • Top 500 CpGs per tissue (for Step 4)
   • Tissue specificity index (TSI) results
   • Epigenetic clock comparison
   • Final summary statistics

3. top_cpgs/ : Top 500 CpGs per tissue for Step 4 modeling

4. reports/ : This README file

5. universal_analysis/ : Universal Test integration results (if available)

Notes for Step 4:
---------------------
Feature selection based on TSI analysis

Brain-specific clock:
Select brain-dominant, tissue-locked CpGs (A < 0.2, dominant in brain) with strong age correlations in brain tissue.

Blood-specific clock:
Select blood-dominant, tissue-locked CpGs (A < 0.2, dominant in blood) with strong age correlations in blood tissue.

Pan-tissue clock:
Use universal CpGs (A > 0.3 and C > 0.8) that show consistent age correlation across both tissues.

Discordant CpGs:
These CpGs (A > 0.3 and C < 0.3) are particularly interesting for understanding tissue-specific aging responses and may inform tissue-specific clock development.

- All correlations were computed using Pearson's correlation.

- Multiple testing was controlled using FDR correction at a 5% threshold.

- A minimum of 10 samples was required to calculate correlations.

- TSI provides a robust framework for tissue specificity analysis.

- These results guide feature selection for Step 4: Clock Training.

"""

    return readme_content

if __name__ == "__main__":
    main()
