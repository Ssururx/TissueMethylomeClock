# Epigenetics Project - Step 11: Latent Aging Profile (LAP) Analysis 

"""
LAP analysis using hybrid Mathematical Component and Statistical Profile approaches.
First extracts Latent Cellular Profiles via NMF, then classifies them based on age-correlation behavior.
Organizes profiles into: Age-Associated (AAIP), Inverse-Aging, and Stable/Constitutive categories.
"""

# ============================================================================
# Section 1: Setup and Imports
# ============================================================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
from sklearn.impute import KNNImputer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

# ============================================================================
# Section 2: Path Configuration
# ============================================================================

base_dir = '/content/drive/MyDrive/epigenetics_project/'
methyl_dir = os.path.join(base_dir, '9_gene_mapping/top500_data')
step10_dir = os.path.join(base_dir, '10_comparative_analysis/results')
results_dir = os.path.join(base_dir, '11_latent_aging_profiles/results')
os.makedirs(results_dir, exist_ok=True)

print("=" * 80)
print("STEP 11: LATENT AGING PROFILE (LAP) ANALYSIS ")
print("=" * 80)
print("Hybrid Mathematical Component + Statistical Profile Approach")
print("1. Mathematical Component Phase: Extract Latent Cellular Profiles (LCPs)")
print("2. Statistical Profile Phase: Classify by age-correlation behavior")
print("3. Organized Output: AAIP, Inverse-Aging, and Stable categories")
print("=" * 80)

# ============================================================================
# Section 3:  Data Loading with Proper Orientation
# ============================================================================

print("\n DATA LOADING...")
print("=" * 80)

try:
  
    print("\n1. Reading methylation CSV files...")
    
    # Load with explicit handling
    blood_meth_raw = pd.read_csv(os.path.join(methyl_dir, 'blood_methylation_top500.csv'))
    brain_meth_raw = pd.read_csv(os.path.join(methyl_dir, 'brain_methylation_top500.csv'))
    
    print(f"   Raw Blood CSV shape: {blood_meth_raw.shape}")
    print(f"   Raw Brain CSV shape: {brain_meth_raw.shape}")
    print(f"   Blood CSV columns (first 3): {blood_meth_raw.columns[:3].tolist()}")
    print(f"   Brain CSV columns (first 3): {brain_meth_raw.columns[:3].tolist()}")
    
    # Check what the first column contains
    first_col = blood_meth_raw.columns[0]
    print(f"\n2. First column analysis:")
    print(f"   First column name: '{first_col}'")
    
    # Check if first column is an index column (contains non-CpG IDs)
    if first_col in ['index', 'Unnamed: 0', 'Sample', 'sample']:
        print(f"   '{first_col}' appears to be an index column, setting as index...")
        blood_meth = blood_meth_raw.set_index(first_col)
        brain_meth = brain_meth_raw.set_index(first_col)
        
        # Now check if index looks like samples or CpGs
        print(f"\n3. Checking index content:")
        index_sample = blood_meth.index[0]
        if index_sample.startswith('GSM') or index_sample.startswith('sample') or len(index_sample) > 10:
            print(f"   Index '{index_sample}' looks like a SAMPLE ID")
            print(f"   This means ROWS = SAMPLES, need to transpose...")
            
            # TRANSPOSE: Rows should be CpGs, Columns should be Samples
            blood_meth = blood_meth.T
            brain_meth = brain_meth.T
            print(f"   Transposed: Blood now {blood_meth.shape}, Brain now {brain_meth.shape}")
        else:
            print(f"   Index '{index_sample}' looks like a CpG ID")
            print(f"   Good ROWS = CpGs, no need to transpose")
    
    print(f"\n4. Final methylation data structure:")
    print(f"   Blood shape: {blood_meth.shape}")
    print(f"   Brain shape: {brain_meth.shape}")
    print(f"   Blood index (should be CpGs): {blood_meth.index[:3].tolist()}")
    print(f"   Blood columns (should be samples): {blood_meth.columns[:3].tolist()}")
    
    # Convert all values to numeric
    print(f"\n5. Converting to numeric and handling NaNs...")
    for col in blood_meth.columns:
        blood_meth[col] = pd.to_numeric(blood_meth[col], Errors='coerce')
    for col in brain_meth.columns:
        brain_meth[col] = pd.to_numeric(brain_meth[col], Errors='coerce')
    
    print(f"   NaN counts: Blood={blood_meth.isna().sum().sum()}, Brain={brain_meth.isna().sum().sum()}")
    
    # Load metadata
    print(f"\n6. Loading metadata...")
    blood_meta = pd.read_csv(os.path.join(methyl_dir, 'blood_metadata_top500.csv'))
    brain_meta = pd.read_csv(os.path.join(methyl_dir, 'brain_metadata_top500.csv'))
    
    print(f"   Blood metadata shape: {blood_meta.shape}")
    print(f"   Brain metadata shape: {brain_meta.shape}")
    print(f"   Blood metadata columns: {blood_meta.columns.tolist()}")
    
    # Show metadata structure
    print(f"\n7. Metadata sample:")
    print(blood_meta.head(3))
    
    # Get sample IDs from methylation data
    print(f"\n8. Aligning methylation data with age data...")
    
    # Get sample IDs from methylation columns
    blood_sample_ids = blood_meth.columns.tolist()
    brain_sample_ids = brain_meth.columns.tolist()
    
    print(f"   Blood methylation has {len(blood_sample_ids)} samples")
    print(f"   Brain methylation has {len(brain_sample_ids)} samples")
    print(f"   Sample IDs (first 3): {blood_sample_ids[:3]}")
    
    # Find age column in metadata
    def find_and_align_age(metadata, sample_ids, tissue_name):
        print(f"\n   {tissue_name}:")
        
        # Find age column
        age_col = None
        for col in metadata.columns:
            if 'age' in str(col).lower() and 'group' not in str(col).lower():
                age_col = col
                print(f"     Found age column: '{age_col}'")
                break
        
        if not age_col:
            print(f"     No age column found in metadata")
            return None
        
        # Find sample ID column
        sample_id_col = None
        possible_id_cols = ['sample_id', 'Sample_ID', 'sample', 'Sample', 'ID', 'id']
        for col in metadata.columns:
            if col in possible_id_cols:
                sample_id_col = col
                print(f"     Found sample ID column: '{sample_id_col}'")
                break
        
        if not sample_id_col:
            print(f"     No sample ID column found, using row index")
            # Assume metadata rows correspond to methylation columns
            if len(metadata) >= len(sample_ids):
                age_series = pd.Series(
                    pd.to_numeric(metadata[age_col], Errors='coerce').values[:len(sample_ids)],
                    index=sample_ids[:len(metadata)]
                )
                print(f"     Created age series with {len(age_series)} samples")
                return age_series
            else:
                print(f"     Error: Metadata has {len(metadata)} rows, but need {len(sample_ids)} samples")
                return None
        
        # Create mapping from sample ID to age
        print(f"     Creating age mapping from '{sample_id_col}' to '{age_col}'")
        
        # Clean sample IDs - remove any whitespace, convert to string
        metadata[sample_id_col] = metadata[sample_id_col].astype(str).str.strip()
        sample_ids_clean = [str(sid).strip() for sid in sample_ids]
        
        # Create dictionary mapping
        age_dict = {}
        for idx, row in metadata.iterrows():
            sample_id = str(row[sample_id_col]).strip()
            try:
                age_val = float(row[age_col])
                age_dict[sample_id] = age_val
            except:
                continue
        
        print(f"     Created age mapping for {len(age_dict)} samples")
        
        # Create age Series aligned with methylation samples
        age_series = pd.Series([age_dict.get(sid, np.nan) for sid in sample_ids_clean],
                              index=sample_ids_clean)
        
        valid_count = age_series.notna().sum()
        print(f"     Successfully aligned {valid_count}/{len(sample_ids_clean)} samples with age data")
        
        if valid_count > 0:
            print(f"     Age range: {age_series.min():.1f} to {age_series.max():.1f}")
        else:
            print(f"     WARNING: No age data aligned")
        
        return age_series
    
    # Get age data for both tissues
    blood_ages = find_and_align_age(blood_meta, blood_sample_ids, "Blood")
    brain_ages = find_and_align_age(brain_meta, brain_sample_ids, "Brain")
    
    # Load DMR data
    print(f"\n11. Loading DMR data...")
    blood_dmrs = pd.read_csv(os.path.join(step10_dir, 'blood_dmrs_detailed.csv'))
    brain_dmrs = pd.read_csv(os.path.join(step10_dir, 'brain_dmrs_detailed.csv'))
    
    print(f"    Blood DMRs: {len(blood_dmrs)}")
    print(f"    Brain DMRs: {len(brain_dmrs)}")
    
    print(f"\n12. FINAL DATA SUMMARY:")
    print(f"    Blood: {blood_meth.shape[0]} CpGs × {blood_meth.shape[1]} samples")
    print(f"    Brain: {brain_meth.shape[0]} CpGs × {brain_meth.shape[1]} samples")
    print(f"    Blood ages: {blood_ages.notna().sum()}/{len(blood_ages)} samples")
    print(f"    Brain ages: {brain_ages.notna().sum()}/{len(brain_ages)} samples")

except Exception as e:
    print(f"\n Error in data loading: {e}")
    import traceback
    traceback.print_exc()
    raise SystemExit("Data loading failed")

# ============================================================================
# Section 4: Data Cleaning and Preprocessing
# ============================================================================

print("\n" + "=" * 80)
print("DATA CLEANING AND PREPROCESSING")
print("=" * 80)

def prepare_methylation_matrix(meth_df, tissue_name):
    """Prepare methylation matrix for NMF decomposition"""
    print(f"\nPreparing {tissue_name} methylation matrix...")
    
    # Make a copy
    data = meth_df.copy()
    
    print(f"   Initial shape: {data.shape}")
    print(f"   Initial NaN count: {data.isna().sum().sum()}")
    
    # Ensure non-negative values for NMF
    min_val = data.min().min()
    if min_val < 0:
        print(f"   Shifting values by {abs(min_val):.4f} to ensure non-negativity")
        data = data - min_val
    
    # Remove CpGs with zero variance
    variances = data.var(axis=1)
    zero_var_mask = variances == 0
    if zero_var_mask.any():
        print(f"   Removing {zero_var_mask.sum()} CpGs with zero variance")
        data = data[~zero_var_mask]
    
    # Impute missing values using KNN
    nan_count = data.isna().sum().sum()
    if nan_count > 0:
        print(f"   Imputing {nan_count} missing values using KNN")
        imputer = KNNImputer(n_neighbors=5)
        # KNNImputer expects samples × features
        data_values = data.values.T  # Samples × CpGs
        imputed_data = imputer.fit_transform(data_values)
        data = pd.DataFrame(imputed_data.T, index=data.index, columns=data.columns)
    
    print(f"   Final matrix: {data.shape[0]} CpGs × {data.shape[1]} samples")
    print(f"   Value range: [{data.min().min():.3f}, {data.max().max():.3f}]")
    print(f"   Final NaN count: {data.isna().sum().sum()}")
    
    return data

blood_meth_clean = prepare_methylation_matrix(blood_meth, "Blood")
brain_meth_clean = prepare_methylation_matrix(brain_meth, "Brain")

# ============================================================================
# Section 5: MATHEMATICAL COMPONENT PHASE - NMF Extraction
# ============================================================================

print("\n" + "=" * 80)
print("MATHEMATICAL COMPONENT PHASE: NMF EXTRACTION")
print("=" * 80)
print("The code treats the methylation data as a composite signal.")
print("It uses Non-negative Matrix Factorization (NMF) to decompose the")
print("high-dimensional CpG matrix into two lower-dimensional sub-matrices.")
print("=" * 80)

def extract_latent_profiles(data, tissue_name, n_components):
    """
    Extract Latent Cellular Profiles (LCPs) using NMF
    
    MATRIX W (The Basis): Defines the "Latent Cellular Profiles" (LCPs)
    MATRIX H (The Coefficients): Provides relative proportion of each LCP within every sample
    
    At this stage, "cells" are organized strictly as mathematical factors
    (LCP₁, LCP₂, ... LCPₙ) based on reconstruction Error and stability.
    """
    print(f"\n{tissue_name.upper()} - MATHEMATICAL COMPONENT EXTRACTION")
    print("-" * 50)
    
    print(f"   Input data: {data.shape[0]} CpGs × {data.shape[1]} samples")
    print(f"   Extracting {n_components} Latent Cellular Profiles (LCPs)...")
    
    # Prepare matrix for NMF (samples × features)
    X = data.values.T  # Samples × CpGs
    
    print(f"   NMF input matrix: {X.shape[0]} samples × {X.shape[1]} CpGs")
    print(f"   Running NMF decomposition...")
    
    try:
        # Perform NMF decomposition
        model = NMF(n_components=n_components, init='nndsvda', random_state=42, 
                    max_iter=2000, tol=1e-4, verbose=0)
        
        # W: Samples × Components (Proportions)
        # H: Components × CpGs (Profiles)
        W = model.fit_transform(X)
        H = model.components_
        
        print(f"   ✓ NMF completed successfully")
        print(f"   Reconstruction Error: {model.reconstruction_err_:.6f}")
        print(f"   Iterations: {model.n_iter_}")
        
    except Exception as e:
        print(f"   ✗ NMF failed: {e}")
        return None, None, None
    
    # Create profiles and proportions dataframes
    profiles_df = pd.DataFrame(H, 
                             index=[f'LCP_{i+1}' for i in range(n_components)],
                             columns=data.index)  # CpGs as columns
    
    proportions_df = pd.DataFrame(W,
                                index=data.columns,  # Sample IDs as index
                                columns=[f'LCP_{i+1}' for i in range(n_components)])
    
    # Normalize proportions to sum to 1 per sample
    proportions_df = proportions_df.div(proportions_df.sum(axis=1), axis=0)
    
    print(f"\n   LCP MATHEMATICAL STATISTICS:")
    print(f"   {'LCP':<10} {'Mean Proportion':<15} {'Std':<10} {'Min':<10} {'Max':<10}")
    for lcp in proportions_df.columns:
        mean_prop = proportions_df[lcp].mean()
        std_prop = proportions_df[lcp].std()
        min_prop = proportions_df[lcp].min()
        max_prop = proportions_df[lcp].max()
        print(f"   {lcp:<10} {mean_prop:<15.4f} {std_prop:<10.4f} {min_prop:<10.4f} {max_prop:<10.4f}")
    
    return profiles_df, proportions_df, model

# Extract LCPs for both tissues
print("\nBLOOD - Mathematical Component Extraction:")
blood_profiles, blood_proportions, blood_model = extract_latent_profiles(
    blood_meth_clean, "Blood", n_components=5
)

print("\nBRAIN - Mathematical Component Extraction:")
brain_profiles, brain_proportions, brain_model = extract_latent_profiles(
    brain_meth_clean, "Brain", n_components=6
)

# Check if extraction succeeded
if blood_profiles is None or brain_profiles is None:
    print("\n Error: NMF extraction failed. Exiting...")
    raise SystemExit("NMF extraction failed")

# Save mathematical component results
print(f"\nSaving Mathematical Component results...")
blood_profiles.to_csv(os.path.join(results_dir, 'blood_latent_profiles.csv'))
blood_proportions.to_csv(os.path.join(results_dir, 'blood_latent_proportions.csv'))
brain_profiles.to_csv(os.path.join(results_dir, 'brain_latent_profiles.csv'))
brain_proportions.to_csv(os.path.join(results_dir, 'brain_latent_proportions.csv'))

print(f"\n MATHEMATICAL COMPONENT PHASE COMPLETED:")
print(f"   Blood: {blood_profiles.shape[0]} LCPs extracted")
print(f"   Brain: {brain_profiles.shape[0]} LCPs extracted")
print(f"   LCPs are now mathematical factors (LCP₁, LCP₂, ... LCPₙ)")

# ============================================================================
# Section 6: STATISTICAL PROFILE PHASE - Age Correlation Classification
# ============================================================================

print("\n" + "=" * 80)
print("STATISTICAL PROFILE PHASE: AGE CORRELATION CLASSIFICATION")
print("=" * 80)
print("The code moves from mathematical abstraction to biological relevance")
print("by running Pearson correlation analyses between each LCP proportion and age.")
print("\nORGANIZATION SHIFTS TO FUNCTIONAL HIERARCHY:")
print("1. Age-Associated Inferred Profiles (AAIP): Positive age correlations")
print("2. Inverse-Aging/Declining Profiles: Negative age correlations")
print("3. Stable/Constitutive Profiles: Near-zero correlations")
print("=" * 80)

def classify_latent_profiles(proportions_df, age_series, tissue_name):
    """
    Classify LCPs based on correlation with chronological age
    
    Creates functional hierarchy:
    - Age-Associated Inferred Profiles (AAIP): Primary drivers of "Aging Signal"
    - Inverse-Aging/Declining Profiles: Populations that diminish over time
    - Stable/Constitutive Profiles: Tissue's "Baseline Structure"
    """
    print(f"\n{tissue_name.upper()} - STATISTICAL PROFILE CLASSIFICATION")
    print("-" * 50)
    
    if age_series is None:
        print("   ✗ No age data available for classification")
        # Create dummy classification
        classification_df = pd.DataFrame({
            'LCP': proportions_df.columns,
            'Correlation': 0.0,
            'p_value': 1.0,
            'Category': 'Unclassified (No Age Data)'
        })
        return classification_df, {}, {}
    
    # Check alignment
    common_samples = proportions_df.index.intersection(age_series.index)
    print(f"   Samples with both proportions and age: {len(common_samples)}")
    
    if len(common_samples) < 10:
        print(f"   ✗ Insufficient samples ({len(common_samples)}) for age correlation")
        classification_df = pd.DataFrame({
            'LCP': proportions_df.columns,
            'Correlation': 0.0,
            'p_value': 1.0,
            'Category': 'Insufficient Age Data'
        })
        return classification_df, {}, {}
    
    print(f"   Calculating Pearson correlations for {len(common_samples)} samples...")
    
    # Calculate correlations
    correlations = {}
    p_values = {}
    
    for lcp in proportions_df.columns:
        lcp_data = proportions_df.loc[common_samples, lcp]
        age_data = age_series.loc[common_samples]
        
        # Remove any NaN
        valid_mask = ~lcp_data.isna() & ~age_data.isna()
        if valid_mask.sum() < 10:
            correlations[lcp] = 0
            p_values[lcp] = 1
            continue
            
        try:
            corr, p_val = stats.pearsonr(lcp_data[valid_mask], age_data[valid_mask])
            correlations[lcp] = corr
            p_values[lcp] = p_val
        except:
            correlations[lcp] = 0
            p_values[lcp] = 1
    
    # Classify LCPs into functional hierarchy
    print(f"\n   CLASSIFYING INTO FUNCTIONAL HIERARCHY:")
    
    classification_data = []
    for lcp in proportions_df.columns:
        corr = correlations[lcp]
        p_val = p_values[lcp]
        
        if p_val < 0.05:  # Statistically significant
            if corr > 0.3:  # Strong positive association
                category = 'AAIP (Strong Positive)'
            elif corr < -0.3:  # Strong negative association
                category = 'Inverse-Aging (Strong Negative)'
            elif corr > 0:  # Weak positive
                category = 'AAIP (Weak Positive)'
            else:  # Weak negative
                category = 'Inverse-Aging (Weak Negative)'
        else:  # Not significant
            if abs(corr) < 0.1:  # Near-zero correlation
                category = 'Stable/Constitutive'
            elif corr > 0:
                category = 'Trend Positive (NS)'
            else:
                category = 'Trend Negative (NS)'
        
        classification_data.append({
            'LCP': lcp,
            'Correlation': corr,
            'p_value': p_val,
            'Category': category
        })
    
    classification_df = pd.DataFrame(classification_data)
    
    # Print classification summary
    print(f"\n   CLASSIFICATION SUMMARY:")
    category_counts = classification_df['Category'].value_counts()
    for category, count in category_counts.items():
        print(f"   {category}: {count} LCPs")
    
    # Show top correlations
    if len(classification_df) > 0:
        print(f"\n   TOP AGE CORRELATIONS:")
        # Sort by absolute correlation
        sorted_df = classification_df.sort_values('Correlation', key=abs, ascending=False)
        for _, row in sorted_df.head(5).iterrows():
            sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
            direction = "↑" if row['Correlation'] > 0 else "↓"
            print(f"   {row['LCP']}: r = {row['Correlation']:.3f}{sig} ({row['Category']}) {direction}")
    
    return classification_df, correlations, p_values

print("\nBLOOD - Statistical Profile Classification:")
blood_class_df, blood_corrs, blood_pvals = classify_latent_profiles(
    blood_proportions, blood_ages, "Blood"
)

print("\nBRAIN - Statistical Profile Classification:")
brain_class_df, brain_corrs, brain_pvals = classify_latent_profiles(
    brain_proportions, brain_ages, "Brain"
)

# Save classification results
print(f"\nSaving Statistical Profile results...")
blood_class_df.to_csv(os.path.join(results_dir, 'blood_lcp_classification.csv'), index=False)
brain_class_df.to_csv(os.path.join(results_dir, 'brain_lcp_classification.csv'), index=False)

print(f"\n STATISTICAL PROFILE PHASE COMPLETED:")
print(f"   LCPs now organized by impact on biological age")
print(f"   Mathematical components tagged with statistical descriptors")

# ============================================================================
# Section 7: GENOMIC DASHBOARD CREATION
# ============================================================================

print("\n" + "=" * 80)
print("GENOMIC DASHBOARD CREATION")
print("=" * 80)
print("Creating comprehensive visualization that combines")
print("mathematical components with statistical descriptors.")
print("\nThe dashboard presents finalized results where 'cells' are")
print("organized by their impact on biological age.")
print("=" * 80)

def create_genomic_dashboard(profiles_df, proportions_df, classification_df, 
                           correlations, tissue_name, age_series):
    """
    Create Genomic Dashboard visualization
    
    Combines mathematical components (NMF) with statistical descriptors (Correlation)
    to present results where "cells" are organized by impact on biological age.
    """
    print(f"\n{tissue_name.upper()} - CREATING GENOMIC DASHBOARD")
    print("-" * 50)
    
    # Create figure
    fig = plt.figure(figsize=(20, 16))
    
    # Define colors for categories
    category_colors = {
        'AAIP': 'firebrick',
        'Inverse-Aging': 'steelblue',
        'Stable/Constitutive': 'gray',
        'Trend Positive': 'lightcoral',
        'Trend Negative': 'lightblue',
        'Unclassified': 'lightgray',
        'Insufficient': 'lightgray'
    }
    
    # PANEL 1: Age Correlation Bar Chart
    print("   Creating Panel 1: Age Correlation Dashboard...")
    ax1 = plt.subplot(3, 3, 1)
    
    # Sort LCPs by correlation
    if correlations:
        sorted_items = sorted(correlations.items(), key=lambda x: x[1])
        lcps = [x[0] for x in sorted_items]
        corr_vals = [x[1] for x in sorted_items]
    else:
        lcps = classification_df['LCP'].tolist()
        corr_vals = [0] * len(lcps)
    
    # Assign colors based on category
    colors = []
    for lcp in lcps:
        cat_row = classification_df[classification_df['LCP'] == lcp]
        if not cat_row.empty:
            category = cat_row['Category'].iloc[0]
            if 'AAIP' in category:
                colors.append(category_colors['AAIP'])
            elif 'Inverse' in category:
                colors.append(category_colors['Inverse-Aging'])
            elif 'Stable' in category:
                colors.append(category_colors['Stable/Constitutive'])
            elif 'Trend Positive' in category:
                colors.append(category_colors['Trend Positive'])
            elif 'Trend Negative' in category:
                colors.append(category_colors['Trend Negative'])
            else:
                colors.append('lightgray')
        else:
            colors.append('lightgray')
    
    bars = ax1.bar(range(len(lcps)), corr_vals, color=colors, edgecolor='black', alpha=0.8)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax1.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Strong correlation')
    ax1.axhline(y=-0.3, color='blue', linestyle='--', alpha=0.5)
    
    # Add significance markers
    for i, (bar, lcp) in enumerate(zip(bars, lcps)):
        cat_row = classification_df[classification_df['LCP'] == lcp]
        if not cat_row.empty:
            p_val = cat_row['p_value'].iloc[0]
            if p_val < 0.001:
                marker = '***'
            elif p_val < 0.01:
                marker = '**'
            elif p_val < 0.05:
                marker = '*'
            else:
                continue
                
            y_pos = bar.get_height() + (0.02 if bar.get_height() > 0 else -0.03)
            ax1.text(i, y_pos, marker, ha='center', 
                    va='bottom' if bar.get_height() > 0 else 'top', fontsize=8)
    
    ax1.set_xlabel('Latent Cellular Profile')
    ax1.set_ylabel('Correlation with Age (r)')
    ax1.set_title('Age Correlation Dashboard')
    ax1.set_xticks(range(len(lcps)))
    ax1.set_xticklabels(lcps, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(fontsize=9)
    
    # PANEL 2: Category Distribution
    print("   Creating Panel 2: Category Distribution...")
    ax2 = plt.subplot(3, 3, 2)
    
    # Simplify categories
    simple_cats = []
    for cat in classification_df['Category']:
        if 'AAIP' in cat:
            simple_cats.append('AAIP')
        elif 'Inverse' in cat:
            simple_cats.append('Inverse-Aging')
        elif 'Stable' in cat:
            simple_cats.append('Stable/Constitutive')
        elif 'Trend' in cat:
            if 'Positive' in cat:
                simple_cats.append('Trend Positive')
            else:
                simple_cats.append('Trend Negative')
        else:
            simple_cats.append('Other')
    
    cat_counts = pd.Series(simple_cats).value_counts()
    
    # Get colors
    pie_colors = []
    for cat in cat_counts.index:
        if cat in category_colors:
            pie_colors.append(category_colors[cat])
        else:
            pie_colors.append('lightgray')
    
    wedges, texts, autotexts = ax2.pie(cat_counts.values, labels=cat_counts.index,
                                       colors=pie_colors, autopct='%1.1f%%',
                                       startangle=90, textprops={'fontsize': 9})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax2.set_title('LAP Category Distribution')
    
    # PANEL 3: Mean Proportions
    print("   Creating Panel 3: Mean Proportions...")
    ax3 = plt.subplot(3, 3, 3)
    
    mean_props = proportions_df.mean().sort_values(ascending=False)
    
    # Color bars by category
    mean_colors = []
    for lcp in mean_props.index:
        cat_row = classification_df[classification_df['LCP'] == lcp]
        if not cat_row.empty:
            category = cat_row['Category'].iloc[0]
            if 'AAIP' in category:
                mean_colors.append(category_colors['AAIP'])
            elif 'Inverse' in category:
                mean_colors.append(category_colors['Inverse-Aging'])
            elif 'Stable' in category:
                mean_colors.append(category_colors['Stable/Constitutive'])
            else:
                mean_colors.append('lightgray')
        else:
            mean_colors.append('lightgray')
    
    bars = ax3.bar(range(len(mean_props)), mean_props.values, color=mean_colors, 
                   edgecolor='black', alpha=0.8)
    ax3.set_xlabel('LCP')
    ax3.set_ylabel('Mean Proportion')
    ax3.set_title('Mean LCP Proportions')
    ax3.set_xticks(range(len(mean_props)))
    ax3.set_xticklabels(mean_props.index, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, mean_props.values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # PANEL 4: Proportion Distribution Box Plot
    print("   Creating Panel 4: Proportion Distribution...")
    ax4 = plt.subplot(3, 3, 4)
    
    box_data = [proportions_df[col] for col in proportions_df.columns]
    
    # Color boxes by category
    box_colors = []
    for col in proportions_df.columns:
        cat_row = classification_df[classification_df['LCP'] == col]
        if not cat_row.empty:
            category = cat_row['Category'].iloc[0]
            if 'AAIP' in category:
                box_colors.append(category_colors['AAIP'])
            elif 'Inverse' in category:
                box_colors.append(category_colors['Inverse-Aging'])
            elif 'Stable' in category:
                box_colors.append(category_colors['Stable/Constitutive'])
            else:
                box_colors.append('lightgray')
        else:
            box_colors.append('lightgray')
    
    box = ax4.boxplot(box_data, labels=proportions_df.columns, patch_artist=True)
    for patch, color in zip(box['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_xlabel('LCP')
    ax4.set_ylabel('Proportion')
    ax4.set_title('LCP Proportion Distribution')
    ax4.set_xticklabels(proportions_df.columns, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # PANEL 5: Top AAIP Age Trajectory
    print("   Creating Panel 5: AAIP Age Trajectory...")
    ax5 = plt.subplot(3, 3, 5)
    
    aaip_lcps = classification_df[classification_df['Category'].str.contains('AAIP')]
    if not aaip_lcps.empty and age_series is not None:
        top_aaip = aaip_lcps.sort_values('Correlation', ascending=False).iloc[0]['LCP']
        
        valid_data = pd.DataFrame({
            'Age': age_series,
            'Proportion': proportions_df[top_aaip]
        }).dropna()
        
        if len(valid_data) > 5:
            ax5.scatter(valid_data['Age'], valid_data['Proportion'], 
                       alpha=0.6, s=30, color='firebrick', label=top_aaip)
            
            # Add trendline
            z = np.polyfit(valid_data['Age'], valid_data['Proportion'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(valid_data['Age'].min(), valid_data['Age'].max(), 100)
            ax5.plot(x_range, p(x_range), 'r-', linewidth=2, alpha=0.8)
            
            # Add correlation info
            corr, p_val = stats.pearsonr(valid_data['Age'], valid_data['Proportion'])
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            info_text = f'{top_aaip}\nr = {corr:.3f}{sig}\nn = {len(valid_data)}'
            ax5.text(0.05, 0.95, info_text, transform=ax5.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax5.set_xlabel('Age')
            ax5.set_ylabel('Proportion')
            ax5.set_title(f'Top AAIP: {top_aaip}')
            ax5.grid(True, alpha=0.3)
            ax5.legend()
        else:
            ax5.text(0.5, 0.5, 'Insufficient data\nfor AAIP trajectory', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('AAIP Age Trajectory')
    else:
        ax5.text(0.5, 0.5, 'No AAIP profiles\nor age data available', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('AAIP Age Trajectory')
    
    # PANEL 6: Top Inverse-Aging Age Trajectory
    print("   Creating Panel 6: Inverse-Aging Age Trajectory...")
    ax6 = plt.subplot(3, 3, 6)
    
    inverse_lcps = classification_df[classification_df['Category'].str.contains('Inverse')]
    if not inverse_lcps.empty and age_series is not None:
        top_inverse = inverse_lcps.sort_values('Correlation', ascending=True).iloc[0]['LCP']
        
        valid_data = pd.DataFrame({
            'Age': age_series,
            'Proportion': proportions_df[top_inverse]
        }).dropna()
        
        if len(valid_data) > 5:
            ax6.scatter(valid_data['Age'], valid_data['Proportion'], 
                       alpha=0.6, s=30, color='steelblue', label=top_inverse)
            
            # Add trendline
            z = np.polyfit(valid_data['Age'], valid_data['Proportion'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(valid_data['Age'].min(), valid_data['Age'].max(), 100)
            ax6.plot(x_range, p(x_range), 'b-', linewidth=2, alpha=0.8)
            
            # Add correlation info
            corr, p_val = stats.pearsonr(valid_data['Age'], valid_data['Proportion'])
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            info_text = f'{top_inverse}\nr = {corr:.3f}{sig}\nn = {len(valid_data)}'
            ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax6.set_xlabel('Age')
            ax6.set_ylabel('Proportion')
            ax6.set_title(f'Top Inverse-Aging: {top_inverse}')
            ax6.grid(True, alpha=0.3)
            ax6.legend()
        else:
            ax6.text(0.5, 0.5, 'Insufficient data\nfor Inverse-Aging trajectory', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Inverse-Aging Trajectory')
    else:
        ax6.text(0.5, 0.5, 'No Inverse-Aging profiles\nor age data available', 
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Inverse-Aging Trajectory')
    
    # PANEL 7: Sample Proportions Heatmap
    print("   Creating Panel 7: Sample Proportions Heatmap...")
    ax7 = plt.subplot(3, 3, 7)
    
    # Sort samples by age if available
    if age_series is not None and len(age_series) > 0:
        valid_ages = age_series.dropna()
        if len(valid_ages) > 0:
            samples_sorted = valid_ages.sort_values().index
            # Filter to samples that exist in proportions
            samples_sorted = [s for s in samples_sorted if s in proportions_df.index]
            if len(samples_sorted) > 50:
                samples_sorted = samples_sorted[:50]
            
            if len(samples_sorted) > 0:
                prop_heatmap = proportions_df.loc[samples_sorted].T
            else:
                prop_heatmap = proportions_df.iloc[:50].T
        else:
            prop_heatmap = proportions_df.iloc[:50].T
    else:
        prop_heatmap = proportions_df.iloc[:50].T
    
    im = ax7.imshow(prop_heatmap.values, aspect='auto', cmap='YlOrRd',
                    interpolation='nearest', vmin=0, vmax=1)
    ax7.set_xlabel('Samples (sorted by age)')
    ax7.set_ylabel('LCPs')
    ax7.set_title('LCP Proportions Across Samples')
    ax7.set_yticks(range(len(prop_heatmap.index)))
    ax7.set_yticklabels(prop_heatmap.index)
    plt.colorbar(im, ax=ax7, label='Proportion', fraction=0.046, pad=0.04)
    
    # PANEL 8: LCP Methylation Profiles
    print("   Creating Panel 8: LCP Methylation Profiles...")
    ax8 = plt.subplot(3, 3, 8)
    
    # Get top markers for each LCP
    top_markers_all = []
    for lcp in profiles_df.index:
        markers = profiles_df.loc[lcp].abs().sort_values(ascending=False).head(3).index.tolist()
        top_markers_all.extend(markers)
    
    top_markers_all = list(set(top_markers_all))[:15]  # Limit to top 15
    
    if top_markers_all and len(top_markers_all) > 0:
        profile_subset = profiles_df[top_markers_all]
        im2 = ax8.imshow(profile_subset.values, aspect='auto', cmap='RdBu_r',
                        interpolation='nearest', 
                        vmin=-np.abs(profile_subset.values).max(),
                        vmax=np.abs(profile_subset.values).max())
        ax8.set_xlabel('Top CpG Markers')
        ax8.set_ylabel('LCPs')
        ax8.set_title('LCP Methylation Profiles\n(Top Markers)')
        ax8.set_yticks(range(len(profiles_df.index)))
        ax8.set_yticklabels(profiles_df.index)
        ax8.set_xticks(range(len(top_markers_all)))
        marker_labels = [f"M{i+1}" for i in range(len(top_markers_all))]
        ax8.set_xticklabels(marker_labels, rotation=90, fontsize=8)
        plt.colorbar(im2, ax=ax8, label='Methylation Weight', fraction=0.046, pad=0.04)
    else:
        ax8.text(0.5, 0.5, 'No marker data\navailable', 
                ha='center', va='center', transform=ax8.transAxes)
        ax8.set_title('LCP Methylation Profiles')
    
    # PANEL 9: Summary
    print("   Creating Panel 9: Summary and Legend...")
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Create summary text
    summary_text = []
    summary_text.append(f"{tissue_name} LAP ANALYSIS")
    summary_text.append("=" * 30)
    summary_text.append(f"Total LCPs: {len(profiles_df)}")
    
    # Count by category
    aaip_count = sum('AAIP' in cat for cat in classification_df['Category'])
    inverse_count = sum('Inverse' in cat for cat in classification_df['Category'])
    stable_count = sum('Stable' in cat for cat in classification_df['Category'])
    
    summary_text.append(f"AAIP Profiles: {aaip_count}")
    summary_text.append(f"Inverse-Aging: {inverse_count}")
    summary_text.append(f"Stable: {stable_count}")
    
    # Add strongest correlations
    if correlations and len(correlations) > 0:
        sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        if len(sorted_corrs) > 0:
            summary_text.append("\nStrongest Correlations:")
            for lcp, corr in sorted_corrs[:3]:
                p_val = classification_df[classification_df['LCP'] == lcp]['p_value'].iloc[0]
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                summary_text.append(f"  {lcp}: r = {corr:.3f}{sig}")
    
    summary_str = "\n".join(summary_text)
    ax9.text(0.05, 0.95, summary_str, transform=ax9.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Add legend
    legend_text = []
    legend_text.append("CATEGORY LEGEND:")
    legend_text.append("AAIP: Age-associated profiles")
    legend_text.append("  (Primary aging drivers)")
    legend_text.append("Inverse-Aging: Declining profiles")
    legend_text.append("  (Diminish with age)")
    legend_text.append("Stable: Constitutive profiles")
    legend_text.append("  (Baseline tissue structure)")
    
    legend_str = "\n".join(legend_text)
    ax9.text(0.05, 0.4, legend_str, transform=ax9.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # FINAL FIGURE SETUP
    plt.suptitle(f'{tissue_name} - LATENT AGING PROFILE GENOMIC DASHBOARD\n'
                f'Mathematical Component × Statistical Profile Classification', 
                fontsize=18, y=1.02, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    dashboard_path = os.path.join(results_dir, f'{tissue_name.lower()}_genomic_dashboard.png')
    plt.savefig(dashboard_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"    Genomic Dashboard saved: {dashboard_path}")
    print(f"   Dashboard contains 9 panels showing:")
    print(f"     1. Age correlations with significance")
    print(f"     2. Category distribution")
    print(f"     3. Mean proportions by category")
    print(f"     4. Proportion distribution box plot")
    print(f"     5. Top AAIP age trajectory")
    print(f"     6. Top Inverse-Aging age trajectory")
    print(f"     7. Sample proportions heatmap")
    print(f"     8. LCP methylation profiles")
    print(f"     9. Summary and legend")
    
    return fig

# Create dashboards for both tissues
print("\n" + "=" * 80)
print("CREATING GENOMIC DASHBOARDS")
print("=" * 80)

print("\nBLOOD - Creating Genomic Dashboard...")
blood_dashboard = create_genomic_dashboard(
    blood_profiles, blood_proportions, blood_class_df, 
    blood_corrs, "Blood", blood_ages
)

print("\nBRAIN - Creating Genomic Dashboard...")
brain_dashboard = create_genomic_dashboard(
    brain_profiles, brain_proportions, brain_class_df,
    brain_corrs, "Brain", brain_ages
)

# ============================================================================
# Section 8: FINAL SUMMARY REPORT
# ============================================================================

print("\n" + "=" * 80)
print("FINAL SUMMARY REPORT")
print("=" * 80)

# Generate comprehensive summary
summary_report = f"""
LATENT AGING PROFILE (LAP) ANALYSIS -  REPORT
===========================================================
Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY:
------------------
This analysis successfully implemented the hybrid Mathematical Component × 
Statistical Profile approach to organize cellular signatures by their 
impact on biological aging.

FRAMEWORK SUCCESSFULLY IMPLEMENTED:
-----------------------------------
1.  MATHEMATICAL COMPONENT PHASE:
   - Non-negative Matrix Factorization (NMF) decomposition
   - Extraction of Latent Cellular Profiles (LCPs) as mathematical factors
   - Matrix W: LCP methylation profiles (Components × CpGs)
   - Matrix H: LCP proportions across samples (Samples × Components)
   - LCPs organized as mathematical factors: LCP₁, LCP₂, ... LCPₙ

2.  STATISTICAL PROFILE PHASE:
   - Pearson correlation of LCP proportions with chronological age
   - Functional classification hierarchy:
     * Age-Associated Inferred Profiles (AAIP): Positive correlations
       (Primary drivers of "Aging Signal")
     * Inverse-Aging/Declining Profiles: Negative correlations
       (Populations that diminish over time)
     * Stable/Constitutive Profiles: Near-zero correlations
       (Tissue's "Baseline Structure")

3.  GENOMIC DASHBOARD:
   - 9-panel visualization combining mathematical and statistical components
   - Presents "cells" organized by impact on biological age
   - Mathematical components tagged with statistical descriptors

RESULTS SUMMARY:
----------------
BLOOD ANALYSIS:
---------------
- LCPs Extracted: {blood_profiles.shape[0]}
- Samples Analyzed: {blood_proportions.shape[0]}
- CpGs in Analysis: {blood_profiles.shape[1]}
- Age Data: {blood_ages.notna().sum()}/{len(blood_ages)} samples
- Age Range: {blood_ages.min():.1f} to {blood_ages.max():.1f} years

LCP Classification (Blood):
{blood_class_df[['LCP', 'Category', 'Correlation']].to_string(index=False)}

BRAIN ANALYSIS:
---------------
- LCPs Extracted: {brain_profiles.shape[0]}
- Samples Analyzed: {brain_proportions.shape[0]}
- CpGs in Analysis: {brain_profiles.shape[1]}
- Age Data: {brain_ages.notna().sum()}/{len(brain_ages)} samples
- Age Range: {brain_ages.min():.1f} to {brain_ages.max():.1f} years

LCP Classification (Brain):
{brain_class_df[['LCP', 'Category', 'Correlation']].to_string(index=False)}

NMF PERFORMANCE:
----------------
Blood:
- Reconstruction Error: {blood_model.reconstruction_err_:.6f}
- Iterations: {blood_model.n_iter_}
- Components: {blood_profiles.shape[0]}

Brain:
- Reconstruction Error: {brain_model.reconstruction_err_:.6f}
- Iterations: {brain_model.n_iter_}
- Components: {brain_profiles.shape[0]}

KEY INSIGHTS FROM LAP FRAMEWORK:
--------------------------------
1. The framework successfully moves from mathematical abstraction to biological relevance
2. Mathematical components (LCPs) have been extracted via NMF
3. Each LCP has been tagged with statistical descriptors based on aging impact
4. The Genomic Dashboard visualizes this hybrid approach
5. Cells are now organized by their impact on biological age

OUTPUT FILES GENERATED:
-----------------------
1. blood_latent_profiles.csv - LCP methylation profiles (Matrix W)
2. blood_latent_proportions.csv - LCP sample proportions (Matrix H)
3. blood_lcp_classification.csv - LCP age correlation classification
4. blood_genomic_dashboard.png - 9-panel visualization
5. brain_latent_profiles.csv - LCP methylation profiles
6. brain_latent_proportions.csv - LCP sample proportions
7. brain_lcp_classification.csv - LCP age correlation classification
8. brain_genomic_dashboard.png - 9-panel visualization

ANALYSIS COMPLETED SUCCESSFULLY:
--------------------------------
The Latent Aging Profile framework has been fully implemented and debugged.
Mathematical components have been extracted and tagged with statistical 
descriptors, creating a functional organization of cellular signatures 
by their impact on biological aging.

Ready for biological interpretation and validation with real age data.
=====================================================================
"""

# Save report
report_path = os.path.join(results_dir, 'lap_final_debugged_report.txt')
with open(report_path, 'w') as f:
    f.write(summary_report)

print(summary_report)

# Final statistics
print("\n" + "=" * 80)
print("ANALYSIS COMPLETION - FINAL STATISTICS")
print("=" * 80)

# Create final summary table
stats_data = {
    'Metric': ['LCPs Extracted', 'Samples', 'CpGs', 'Age Samples', 
               'AAIP Profiles', 'Inverse-Aging Profiles', 'Stable Profiles',
               'NMF Reconstruction Error', 'Dashboard Panels'],
    'Blood': [
        blood_profiles.shape[0],
        blood_proportions.shape[0],
        blood_profiles.shape[1],
        blood_ages.notna().sum(),
        sum('AAIP' in cat for cat in blood_class_df['Category']),
        sum('Inverse' in cat for cat in blood_class_df['Category']),
        sum('Stable' in cat for cat in blood_class_df['Category']),
        f"{blood_model.reconstruction_err_:.4f}",
        9
    ],
    'Brain': [
        brain_profiles.shape[0],
        brain_proportions.shape[0],
        brain_profiles.shape[1],
        brain_ages.notna().sum(),
        sum('AAIP' in cat for cat in brain_class_df['Category']),
        sum('Inverse' in cat for cat in brain_class_df['Category']),
        sum('Stable' in cat for cat in brain_class_df['Category']),
        f"{brain_model.reconstruction_err_:.4f}",
        9
    ]
}

stats_df = pd.DataFrame(stats_data)
print(stats_df.to_string(index=False))

print(f"\nAll results saved to: {results_dir}")

print("\n" + "=" * 80)

print("=" * 80)
print("\nThe code has successfully:")
print("1. Treated methylation data as composite signal")
print("2. Decomposed it into Latent Cellular Profiles (LCPs)")
print("3. Classified LCPs by aging impact using Pearson correlations")
print("4. Created Genomic Dashboard where cells are organized by biological age impact")
print("=" * 80)
