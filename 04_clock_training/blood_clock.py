# ----------------------------------------------------------------------
# Epigenetic Aging Project - Step 4: Blood Epigenetic Clock Training
# Comprehensive training with systematic hyperparameter optimization
# Tissue: Blood 
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Setup and Imports
# ----------------------------------------------------------------------

print("Installing packages...")
!pip install pandas numpy scipy matplotlib seaborn scikit-learn joblib -q

from google.colab import drive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Machine learning imports
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer

# For data handling
import os
from datetime import datetime
import joblib
import re
import time
from collections import Counter

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

print("Packages loaded")

# ----------------------------------------------------------------------
# Google Drive Setup and Project Structure
# ----------------------------------------------------------------------

print("Setting up Google Drive project structure...")

# Mount Google Drive
try:
    drive.mount('/content/drive')
    print("Drive mounted")
except:
    print("Could not mount Google Drive. Using local storage.")
    PROJECT_ROOT = '/content/'
else:
    PROJECT_ROOT = '/content/drive/MyDrive/epigenetics_project/'

# Previous steps' data locations
STEP2_DATA = f'{PROJECT_ROOT}2_data_qc/cleaned_data/'
STEP3_CPGS = f'{PROJECT_ROOT}3_feature_discovery/top_cpgs/'

# Step 4 - Create output directories
STEP4_ROOT = f'{PROJECT_ROOT}4_model_training/'
STEP4_FIGURES = f'{STEP4_ROOT}figures/'
STEP4_TABLES = f'{STEP4_ROOT}tables/'
STEP4_MODELS = f'{STEP4_ROOT}models/'
STEP4_REPORTS = f'{STEP4_ROOT}reports/'

# Create directories
print("Creating step 4 directory structure...")
for folder in [STEP4_ROOT, STEP4_FIGURES, STEP4_TABLES, STEP4_MODELS, STEP4_REPORTS]:
    os.makedirs(folder, exist_ok=True)
    print(f"   Created: {folder}")

print("Directory structure ready")

# ----------------------------------------------------------------------
# Configuration Parameters
# ----------------------------------------------------------------------

CONFIG = {
    # Data splitting
    'test_size': 0.2,
    'validation_size': 0.2,
    'random_state': 42,

    # Feature selection options to test
    'feature_sizes': [300, 400, 500, 600, 700],

    # Hyperparameter search spaces
    'elasticnet_params': {
        'alpha': np.logspace(-4, 1, 20),  # Wide range for discovery
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],  # Various ratios
        'max_iter': [10000]
    },

    # Ensemble configurations to test
    'ensemble_configs': [
        {'n_estimators': 30, 'max_samples': 0.7},
        {'n_estimators': 50, 'max_samples': 0.8},
        {'n_estimators': 75, 'max_samples': 0.8},
        {'n_estimators': 100, 'max_samples': 0.9}
    ],

    # Cross-validation
    'cv_folds': 5,
    'n_random_search_iter': 30
}

RANDOM_STATE = 42
TISSUE = 'Blood'

# ----------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------

def print_section(title):
    width = 80
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)

def print_subsection(title):
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)

def save_figure(filename, dpi=300):
    plt.tight_layout()
    drive_path = f'{STEP4_FIGURES}{filename}'
    plt.savefig(drive_path, dpi=dpi, bbox_inches='tight')
    print(f"   Saved figure: {drive_path}")
    return drive_path

def save_table(df, filename, description=""):
    drive_path = f'{STEP4_TABLES}{filename}'
    df.to_csv(drive_path, index=False)
    print(f"   Saved table: {filename} ({description})")
    return drive_path

def save_model(model, filename, description=""):
    drive_path = f'{STEP4_MODELS}{filename}'
    joblib.dump(model, drive_path)
    print(f"   Saved model: {filename} ({description})")
    return drive_path

def save_report(content, filename):
    drive_path = f'{STEP4_REPORTS}{filename}'
    with open(drive_path, 'w') as f:
        f.write(content)
    print(f"   Saved report: {filename}")

def assess_generalization(train_metric, test_metric, metric_name="MAE"):
    """Assess generalization performance"""
    if metric_name == "MAE":
        gap = train_metric - test_metric
        ratio = train_metric / test_metric if test_metric > 0 else float('inf')
        if ratio < 1.05:
            return "excellent - minimal overfitting", "good", gap, ratio
        elif ratio < 1.15:
            return "good - slight overfitting", "good", gap, ratio
        elif ratio < 1.3:
            return "moderate - noticeable overfitting", "moderate", gap, ratio
        elif ratio < 1.5:
            return "poor - significant overfitting", "poor", gap, ratio
        else:
            return "bad - severe overfitting", "bad", gap, ratio
    elif metric_name == "R2":
        gap = train_metric - test_metric
        if gap < 0.05:
            return "excellent - minimal generalization gap", "good", gap, None
        elif gap < 0.1:
            return "good - small generalization gap", "good", gap, None
        elif gap < 0.2:
            return "moderate - moderate generalization gap", "moderate", gap, None
        elif gap < 0.3:
            return "poor - large generalization gap", "poor", gap, None
        else:
            return "bad - very large generalization gap", "bad", gap, None

# ----------------------------------------------------------------------
# Data Loading Function
# ----------------------------------------------------------------------

def load_blood_data():
    """Load blood methylation data"""
    print_section(f"Loading {TISSUE} Data")

    try:
        # Load top CpGs from Step 3
        print("1. Loading top CpGs from Step 3...")
        cpgs_path = os.path.join(STEP3_CPGS, 'top_500_blood_cpgs.csv')
        if not os.path.exists(cpgs_path):
            raise FileNotFoundError(f"Top CpGs file not found: {cpgs_path}")

        top_cpgs_df = pd.read_csv(cpgs_path)

        # Identify CpG column
        cpg_col = None
        for col in top_cpgs_df.columns:
            if 'cpg' in col.lower() or col.lower() == 'index':
                cpg_col = col
                break
        if cpg_col is None:
            cpg_col = top_cpgs_df.columns[0]

        top_cpgs = top_cpgs_df[cpg_col].astype(str).tolist()
        print(f"  Loaded {len(top_cpgs)} top CpGs")

        # Load methylation data
        print("\n2. Loading methylation data...")
        meth_path = os.path.join(STEP2_DATA, 'cleaned_blood_methylation.csv')
        if not os.path.exists(meth_path):
            meth_path = os.path.join(STEP2_DATA, 'blood_methylation_merged.csv')

        meth_data = pd.read_csv(meth_path, index_col=0)
        print(f"  Methylation data shape: {meth_data.shape}")

        # Load metadata
        print("\n3. Loading metadata...")
        meta_path = os.path.join(STEP2_DATA, 'cleaned_blood_metadata.csv')
        if not os.path.exists(meta_path):
            meta_path = os.path.join(STEP2_DATA, 'blood_metadata_merged.csv')

        meta_data = pd.read_csv(meta_path)
        print(f"  Metadata shape: {meta_data.shape}")

        # Find age column
        age_col = None
        for col in meta_data.columns:
            if col.lower() == 'age' or 'age' in col.lower():
                age_col = col
                break

        print(f"  Age column: '{age_col}'")

        # Find sample ID column
        sample_id_col = None
        for col in meta_data.columns:
            if 'sample' in col.lower() or 'id' in col.lower():
                sample_id_col = col
                break

        print(f"  Sample ID column: '{sample_id_col}'")

        # Create age dictionary
        ages = pd.to_numeric(meta_data[age_col], errors='coerce').values
        age_dict = {}

        for idx, row in meta_data.iterrows():
            sample_id = str(row[sample_id_col]).strip()
            age = ages[idx]
            if not pd.isna(age):
                clean_id = sample_id.upper().replace(' ', '')
                age_dict[clean_id] = age

        print(f"  Created age dictionary with {len(age_dict)} entries")

        # Match samples
        print("\n4. Matching samples...")
        common_samples = []
        age_list = []

        for meth_sample in meth_data.columns:
            clean_meth_sample = str(meth_sample).strip().upper().replace(' ', '')

            if clean_meth_sample in age_dict:
                common_samples.append(meth_sample)
                age_list.append(age_dict[clean_meth_sample])
            else:
                gsm_match = re.search(r'GSM\d+', clean_meth_sample)
                if gsm_match:
                    gsm_id = gsm_match.group(0)
                    if gsm_id in age_dict:
                        common_samples.append(meth_sample)
                        age_list.append(age_dict[gsm_id])

        print(f"  Matched samples: {len(common_samples)}/{len(meth_data.columns)}")

        # Prepare feature matrix
        print("\n5. Preparing feature matrix...")
        X = meth_data[common_samples].T  # Samples × CpGs
        print(f"  X shape: {X.shape}")

        # Filter to available CpGs
        available_cpgs = [cpg for cpg in top_cpgs if cpg in X.columns]
        print(f"  Available top CpGs: {len(available_cpgs)}/{len(top_cpgs)}")

        if len(available_cpgs) == 0:
            print("  Using all available CpGs")
            available_cpgs = X.columns.tolist()

        X = X[available_cpgs].copy()
        y = np.array(age_list)

        # Apply quality control
        print("\n6. Applying quality control...")
        original_shape = X.shape

        # Remove samples with too many missing values
        sample_missing = X.isna().mean(axis=1)
        keep_samples = sample_missing < 0.2
        X = X[keep_samples]
        y = y[keep_samples.values]

        # Remove CpGs with too many missing values
        cpg_missing = X.isna().mean(axis=0)
        keep_cpgs = cpg_missing < 0.3
        X = X.loc[:, keep_cpgs]

        print(f"  Removed {original_shape[0] - X.shape[0]} samples, {original_shape[1] - X.shape[1]} CpGs")
        print(f"  Final dataset: {X.shape[0]} samples × {X.shape[1]} CpGs")

        # Age statistics
        print(f"\n7. Age statistics:")
        print(f"   Min: {y.min():.1f} years")
        print(f"   Max: {y.max():.1f} years")
        print(f"   Mean: {y.mean():.1f} ± {y.std():.1f} years")
        print(f"   Median: {np.median(y):.1f} years")

        print_section("Data Loading Complete")
        print(f"✓ Samples: {X.shape[0]}")
        print(f"✓ Features: {X.shape[1]}")
        print(f"✓ Age range: {y.min():.1f} - {y.max():.1f} years")

        return X, y, available_cpgs

    except Exception as e:
        print(f"\nERROR in load_blood_data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

# ----------------------------------------------------------------------
# Feature Selection
# ----------------------------------------------------------------------

def select_features_correlation(X, y, n_features):
    """Select features based on correlation with age"""
    print(f"  Selecting {n_features} features by correlation...")

    if n_features > X.shape[1]:
        n_features = X.shape[1]
        print(f"  Adjusted to {n_features} features (max available)")

    correlations = []
    for col in X.columns:
        valid_idx = ~X[col].isna()
        if sum(valid_idx) > 10:
            try:
                corr = abs(pearsonr(X.loc[valid_idx, col], y[valid_idx])[0])
                correlations.append((col, corr))
            except:
                correlations.append((col, 0))
        else:
            correlations.append((col, 0))

    correlations.sort(key=lambda x: x[1], reverse=True)
    selected = [cpg for cpg, _ in correlations[:n_features]]

    return selected

# ----------------------------------------------------------------------
# Feature Size Optimization
# ----------------------------------------------------------------------

def optimize_feature_size(X_train, y_train, X_val, y_val):
    """Find optimal number of features"""
    print_section("Feature Size Optimization")
    print("Testing different numbers of features to find optimal performance...")

    results = []

    for n_features in CONFIG['feature_sizes']:
        if n_features > X_train.shape[1]:
            print(f"  Skipping {n_features} features (more than available)")
            continue

        print(f"\n  Testing {n_features} features...")
        selected_features = select_features_correlation(X_train, y_train, n_features)

        # Preprocessing
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()

        X_train_sel = X_train[selected_features]
        X_val_sel = X_val[selected_features]

        X_train_imp = imputer.fit_transform(X_train_sel)
        X_val_imp = imputer.transform(X_val_sel)

        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_val_scaled = scaler.transform(X_val_imp)

        # Train simple ElasticNet with reasonable defaults
        model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=RANDOM_STATE)
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_val_pred = model.predict(X_val_scaled)
        r2 = r2_score(y_val, y_val_pred)
        mae = mean_absolute_error(y_val, y_val_pred)

        results.append({
            'n_features': n_features,
            'validation_r2': r2,
            'validation_mae': mae,
            'selected_features': selected_features
        })

        print(f"    Validation R²: {r2:.4f}")
        print(f"    Validation MAE: {mae:.2f} years")

    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("  No valid feature sizes tested, using default 400")
        selected_features = select_features_correlation(X_train, y_train, 400)
        return selected_features, 400, pd.DataFrame()

    results_df = results_df.sort_values('validation_r2', ascending=False)

    # Select best
    best_result = results_df.iloc[0]
    best_features = best_result['selected_features']
    best_size = best_result['n_features']

    print(f"\n✓ Optimal feature size: {best_size} features (R²: {best_result['validation_r2']:.4f})")

    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['n_features'], results_df['validation_r2'], 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Features')
    plt.ylabel('Validation R²')
    plt.title('Feature Size Optimization')
    plt.grid(True, alpha=0.3)

    # Mark best point
    best_idx = results_df['validation_r2'].idxmax()
    plt.plot(results_df.loc[best_idx, 'n_features'], results_df.loc[best_idx, 'validation_r2'],
             'r*', markersize=15, label=f'Best: {best_size} features')
    plt.legend()

    save_figure('feature_size_optimization.png')
    plt.show()

    # Save results
    save_table(results_df, 'feature_size_optimization_results.csv', 'Feature size optimization')

    return best_features, best_size, results_df

# ----------------------------------------------------------------------
# Cross-Validation with Feature Stability
# ----------------------------------------------------------------------

def cross_validation_stability(X_train, y_train, n_features):
    """Perform CV to assess feature stability"""
    print(f"\nPerforming cross-validation with {n_features} features...")

    if len(y_train) >= 20:
        try:
            age_bins = pd.qcut(y_train, q=min(5, len(y_train)//4), labels=False, duplicates='drop')
            skf = StratifiedKFold(n_splits=CONFIG['cv_folds'], shuffle=True, random_state=RANDOM_STATE)
            print(f"  Using stratified {CONFIG['cv_folds']}-fold CV")
        except:
            skf = KFold(n_splits=CONFIG['cv_folds'], shuffle=True, random_state=RANDOM_STATE)
            print(f"  Using {CONFIG['cv_folds']}-fold CV")
    else:
        skf = KFold(n_splits=min(3, len(y_train)//3), shuffle=True, random_state=RANDOM_STATE)
        print(f"  Using {skf.n_splits}-fold CV (small dataset)")

    cv_maes = []
    cv_r2s = []
    all_selected_features = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train[val_idx]

        # Feature selection per fold
        selected_features = select_features_correlation(X_fold_train, y_fold_train, n_features)
        all_selected_features.extend(selected_features)

        # Preprocessing
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()

        X_train_sel = X_fold_train[selected_features]
        X_val_sel = X_fold_val[selected_features]

        X_train_imp = imputer.fit_transform(X_train_sel)
        X_val_imp = imputer.transform(X_val_sel)

        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_val_scaled = scaler.transform(X_val_imp)

        # Train with reasonable defaults
        model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=RANDOM_STATE)
        model.fit(X_train_scaled, y_fold_train)

        # Evaluate
        y_val_pred = model.predict(X_val_scaled)
        mae = mean_absolute_error(y_fold_val, y_val_pred)
        r2 = r2_score(y_fold_val, y_val_pred)

        cv_maes.append(mae)
        cv_r2s.append(r2)

        print(f"    Fold {fold}: MAE={mae:.2f}, R²={r2:.3f}")

    # Feature stability analysis
    feature_counts = Counter(all_selected_features)
    n_folds = skf.n_splits

    total_unique = len(feature_counts)
    in_all_folds = sum(1 for count in feature_counts.values() if count == n_folds)
    in_80pct_folds = sum(1 for count in feature_counts.values() if count >= 0.8 * n_folds)

    print(f"\n  Feature stability:")
    print(f"     Total unique features selected: {total_unique}")
    print(f"     Features selected in all folds: {in_all_folds}")
    print(f"     Features selected in ≥80% folds: {in_80pct_folds}")

    # Create stability dataframe
    stability_data = []
    for feature, count in feature_counts.most_common():
        frequency = count / n_folds
        stability_data.append({
            'CpG': feature,
            'Frequency': frequency,
            'Count': count,
            'Stability': 'High' if frequency >= 0.8 else 'Medium' if frequency >= 0.6 else 'Low'
        })

    stability_df = pd.DataFrame(stability_data)
    stability_df = stability_df.sort_values('Frequency', ascending=False)

    # Select stable features
    stable_features = stability_df[stability_df['Frequency'] >= 0.7]['CpG'].tolist()
    if len(stable_features) >= n_features:
        final_features = stable_features[:n_features]
        print(f"  Selected {len(final_features)} stable features (≥70% frequency)")
    else:
        final_features = stability_df['CpG'].head(n_features).tolist()
        stable_count = len([f for f in final_features if f in stable_features])
        print(f"  Selected {len(final_features)} features ({stable_count} with ≥70% stability)")

    # CV summary
    mean_mae = np.mean(cv_maes)
    mean_r2 = np.mean(cv_r2s)

    print(f"\n  Cross-validation summary:")
    print(f"     Mean MAE: {mean_mae:.2f} ± {np.std(cv_maes):.2f}")
    print(f"     Mean R²: {mean_r2:.3f} ± {np.std(cv_r2s):.3f}")

    return final_features, stability_df, {
        'mean_mae': mean_mae,
        'mean_r2': mean_r2,
        'std_mae': np.std(cv_maes),
        'std_r2': np.std(cv_r2s),
        'n_stable_features': in_all_folds
    }

# ----------------------------------------------------------------------
# Hyperparameter Optimization
# ----------------------------------------------------------------------

def optimize_hyperparameters(X_train, y_train, X_val, y_val, features):
    """Find optimal ElasticNet hyperparameters"""
    print_section("Hyperparameter Optimization")
    print("Finding optimal alpha and l1_ratio for ElasticNet...")

    if len(features) == 0 or len(y_train) < 10:
        print("  Not enough data for optimization, using defaults")
        return {'alpha': 0.1, 'l1_ratio': 0.5}, None

    # Preprocessing
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_train_sel = X_train[features]
    X_val_sel = X_val[features]

    X_train_imp = imputer.fit_transform(X_train_sel)
    X_val_imp = imputer.transform(X_val_sel)

    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_val_scaled = scaler.transform(X_val_imp)

    # Parameter grid
    param_grid = CONFIG['elasticnet_params']

    print(f"  Searching {len(param_grid['alpha'])} alpha values × {len(param_grid['l1_ratio'])} l1 ratios")
    print(f"  Testing {CONFIG['n_random_search_iter']} random combinations...")

    try:
        search = RandomizedSearchCV(
            ElasticNet(max_iter=10000, random_state=RANDOM_STATE),
            param_grid,
            n_iter=CONFIG['n_random_search_iter'],
            cv=min(3, len(y_train)//5),
            scoring='r2',
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=0
        )

        search.fit(X_train_scaled, y_train)

        # Evaluate best model
        best_model = search.best_estimator_
        y_val_pred = best_model.predict(X_val_scaled)
        val_r2 = r2_score(y_val, y_val_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)

        print(f"\n  Optimal parameters found:")
        print(f"     Alpha: {search.best_params_['alpha']:.6f}")
        print(f"     L1 ratio: {search.best_params_['l1_ratio']:.3f}")
        print(f"     Validation R²: {val_r2:.4f}")
        print(f"     Validation MAE: {val_mae:.2f} years")

        # Save search results
        cv_results = pd.DataFrame(search.cv_results_)
        cv_results = cv_results.sort_values('mean_test_score', ascending=False)

        # Visualize parameter effects
        visualize_hyperparameter_effects(cv_results)

        return search.best_params_, cv_results

    except Exception as e:
        print(f"  Optimization failed: {e}")
        print("  Using default parameters")
        return {'alpha': 0.1, 'l1_ratio': 0.5}, None

def visualize_hyperparameter_effects(cv_results):
    """Visualize hyperparameter effects on performance"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Alpha vs performance
    axes[0].scatter(cv_results['param_alpha'], cv_results['mean_test_score'], alpha=0.6)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Alpha (log scale)')
    axes[0].set_ylabel('CV R² Score')
    axes[0].set_title('Effect of Alpha on Performance')
    axes[0].grid(True, alpha=0.3)

    # Mark best alpha
    best_idx = cv_results['mean_test_score'].idxmax()
    best_alpha = cv_results.loc[best_idx, 'param_alpha']
    best_score = cv_results.loc[best_idx, 'mean_test_score']
    axes[0].plot(best_alpha, best_score, 'r*', markersize=12, label=f'Best: α={best_alpha:.4f}')
    axes[0].legend()

    # L1 ratio vs performance
    axes[1].scatter(cv_results['param_l1_ratio'], cv_results['mean_test_score'], alpha=0.6, color='orange')
    axes[1].set_xlabel('L1 Ratio')
    axes[1].set_ylabel('CV R² Score')
    axes[1].set_title('Effect of L1 Ratio on Performance')
    axes[1].grid(True, alpha=0.3)

    # Mark best L1 ratio
    best_l1 = cv_results.loc[best_idx, 'param_l1_ratio']
    axes[1].plot(best_l1, best_score, 'r*', markersize=12, label=f'Best: L1={best_l1:.3f}')
    axes[1].legend()

    plt.tight_layout()
    save_figure('hyperparameter_optimization_results.png')
    plt.show()

# ----------------------------------------------------------------------
# Ensemble Configuration Optimization
# ----------------------------------------------------------------------

def optimize_ensemble_config(X_train, y_train, X_val, y_val, features, best_params):
    """Find optimal ensemble configuration"""
    print_section("Ensemble Configuration Optimization")
    print("Testing different ensemble configurations...")

    results = []

    for config in CONFIG['ensemble_configs']:
        print(f"\n  Testing: {config['n_estimators']} estimators, {config['max_samples']*100:.0f}% subsample")

        # Preprocessing
        imputer = SimpleImputer(strategy='median')
        scaler = RobustScaler()

        X_train_sel = X_train[features]
        X_val_sel = X_val[features]

        X_train_imp = imputer.fit_transform(X_train_sel)
        X_val_imp = imputer.transform(X_val_sel)

        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_val_scaled = scaler.transform(X_val_imp)

        # Base model with optimal hyperparameters
        base_model = ElasticNet(
            alpha=best_params['alpha'],
            l1_ratio=best_params['l1_ratio'],
            max_iter=10000,
            random_state=RANDOM_STATE
        )

        # Create ensemble
        ensemble = BaggingRegressor(
            estimator=base_model,
            n_estimators=config['n_estimators'],
            max_samples=config['max_samples'],
            bootstrap=True,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        # Train and evaluate
        start_time = time.time()
        ensemble.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time

        y_val_pred = ensemble.predict(X_val_scaled)
        r2 = r2_score(y_val, y_val_pred)
        mae = mean_absolute_error(y_val, y_val_pred)

        results.append({
            'n_estimators': config['n_estimators'],
            'max_samples': config['max_samples'],
            'validation_r2': r2,
            'validation_mae': mae,
            'training_time': training_time
        })

        print(f"    Validation R²: {r2:.4f}")
        print(f"    Validation MAE: {mae:.2f} years")
        print(f"    Training time: {training_time:.1f}s")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('validation_r2', ascending=False)

    best_config = results_df.iloc[0]

    print(f"\n✓ Optimal ensemble configuration:")
    print(f"   Estimators: {best_config['n_estimators']}")
    print(f"   Max samples: {best_config['max_samples']*100:.0f}%")
    print(f"   Validation R²: {best_config['validation_r2']:.4f}")

    # Save results
    save_table(results_df, 'ensemble_configuration_results.csv', 'Ensemble configuration optimization')

    return {
        'n_estimators': int(best_config['n_estimators']),
        'max_samples': best_config['max_samples']
    }, results_df

# ----------------------------------------------------------------------
# Final Model Training
# ----------------------------------------------------------------------

def train_final_model(X_train, y_train, X_test, y_test, features, best_params, ensemble_config):
    """Train final model with optimized configuration"""
    print_section("Final Model Training")

    # Preprocessing
    imputer = SimpleImputer(strategy='median')
    scaler = RobustScaler()

    X_train_sel = X_train[features]
    X_test_sel = X_test[features]

    X_train_imp = imputer.fit_transform(X_train_sel)
    X_test_imp = imputer.transform(X_test_sel)

    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    # Base model with optimized hyperparameters
    base_model = ElasticNet(
        alpha=best_params['alpha'],
        l1_ratio=best_params['l1_ratio'],
        max_iter=10000,
        random_state=RANDOM_STATE
    )

    # Ensemble with optimized configuration
    ensemble = BaggingRegressor(
        estimator=base_model,
        n_estimators=ensemble_config['n_estimators'],
        max_samples=ensemble_config['max_samples'],
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    print(f"  Training final ensemble...")
    print(f"     Features: {len(features)}")
    print(f"     Alpha: {best_params['alpha']:.6f}")
    print(f"     L1 ratio: {best_params['l1_ratio']:.3f}")
    print(f"     Ensemble: {ensemble_config['n_estimators']} estimators")
    print(f"     Subsample: {ensemble_config['max_samples']*100:.0f}%")

    start_time = time.time()
    ensemble.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time

    # Predictions
    y_train_pred = ensemble.predict(X_train_scaled)
    y_test_pred = ensemble.predict(X_test_scaled)

    # Calculate metrics
    metrics = {
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_corr': pearsonr(y_train, y_train_pred)[0],
        'test_corr': pearsonr(y_test, y_test_pred)[0],
        'training_time': training_time
    }

    # Generalization metrics
    metrics['mae_gap'] = metrics['train_mae'] - metrics['test_mae']
    metrics['r2_gap'] = metrics['train_r2'] - metrics['test_r2']
    metrics['mae_ratio'] = metrics['train_mae'] / metrics['test_mae'] if metrics['test_mae'] > 0 else np.nan
    metrics['r2_ratio'] = metrics['test_r2'] / metrics['train_r2'] if metrics['train_r2'] > 0 else np.nan

    # Feature importance
    feature_importance = pd.DataFrame()
    if hasattr(ensemble, 'estimators_') and len(ensemble.estimators_) > 0:
        all_coefs = []
        for estimator in ensemble.estimators_:
            if hasattr(estimator, 'coef_'):
                all_coefs.append(estimator.coef_)
        if all_coefs:
            mean_coefs = np.mean(all_coefs, axis=0)
            std_coefs = np.std(all_coefs, axis=0)
            metrics['n_nonzero'] = np.sum(np.abs(mean_coefs) > 1e-5)
            feature_importance = pd.DataFrame({
                'CpG': features,
                'Coefficient': mean_coefs,
                'Coefficient_Std': std_coefs,
                'Abs_Coefficient': np.abs(mean_coefs)
            }).sort_values('Abs_Coefficient', ascending=False)
        else:
            metrics['n_nonzero'] = 0
    else:
        metrics['n_nonzero'] = 0

    # Generalization assessment
    mae_assessment, mae_severity, _, _ = assess_generalization(
        metrics['train_mae'], metrics['test_mae'], "MAE"
    )
    r2_assessment, r2_severity, _, _ = assess_generalization(
        metrics['train_r2'], metrics['test_r2'], "R2"
    )

    metrics['mae_assessment'] = mae_assessment
    metrics['mae_severity'] = mae_severity
    metrics['r2_assessment'] = r2_assessment
    metrics['r2_severity'] = r2_severity

    # Print results
    print(f"\n  Final performance:")
    print(f"     Training set: R²={metrics['train_r2']:.4f}, MAE={metrics['train_mae']:.2f} years")
    print(f"     Test set:     R²={metrics['test_r2']:.4f}, MAE={metrics['test_mae']:.2f} years")
    print(f"     Generalization: {mae_assessment}")
    print(f"     Non-zero features: {metrics.get('n_nonzero', 0)}/{len(features)}")
    print(f"     Training time: {training_time:.1f} seconds")

    return ensemble, metrics, y_train_pred, y_test_pred, feature_importance

# ----------------------------------------------------------------------
# Visualization Functions
# ----------------------------------------------------------------------

def create_comprehensive_visualizations(y_train_true, y_train_pred, y_test_true, y_test_pred, metrics):
    """Create comprehensive visualizations"""
    fig = plt.figure(figsize=(20, 12))

    # Training set
    ax1 = plt.subplot(2, 3, 1)
    mask_train = ~(np.isnan(y_train_true) | np.isnan(y_train_pred))
    y_train_true_clean = y_train_true[mask_train]
    y_train_pred_clean = y_train_pred[mask_train]

    ax1.scatter(y_train_true_clean, y_train_pred_clean, alpha=0.6, s=50, color='blue', edgecolor='black')
    min_age = min(min(y_train_true_clean), min(y_train_pred_clean))
    max_age = max(max(y_train_true_clean), max(y_train_pred_clean))
    ax1.plot([min_age, max_age], [min_age, max_age], 'r--', linewidth=2, label='Perfect prediction')

    if len(y_train_true_clean) > 1:
        z = np.polyfit(y_train_true_clean, y_train_pred_clean, 1)
        p = np.poly1d(z)
        ax1.plot(np.sort(y_train_true_clean), p(np.sort(y_train_true_clean)), 'g-', linewidth=2, label='Regression line')

    ax1.set_xlabel('Actual Age (years)')
    ax1.set_ylabel('Predicted Age (years)')
    ax1.set_title(f'Training Set (n={len(y_train_true_clean)})\nR²={metrics["train_r2"]:.4f}, MAE={metrics["train_mae"]:.2f} years')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Test set
    ax2 = plt.subplot(2, 3, 2)
    mask_test = ~(np.isnan(y_test_true) | np.isnan(y_test_pred))
    y_test_true_clean = y_test_true[mask_test]
    y_test_pred_clean = y_test_pred[mask_test]

    ax2.scatter(y_test_true_clean, y_test_pred_clean, alpha=0.6, s=50, color='red', edgecolor='black')
    min_age = min(min(y_test_true_clean), min(y_test_pred_clean))
    max_age = max(max(y_test_true_clean), max(y_test_pred_clean))
    ax2.plot([min_age, max_age], [min_age, max_age], 'r--', linewidth=2, label='Perfect prediction')

    if len(y_test_true_clean) > 1:
        z = np.polyfit(y_test_true_clean, y_test_pred_clean, 1)
        p = np.poly1d(z)
        ax2.plot(np.sort(y_test_true_clean), p(np.sort(y_test_true_clean)), 'g-', linewidth=2, label='Regression line')

    ax2.set_xlabel('Actual Age (years)')
    ax2.set_ylabel('Predicted Age (years)')
    ax2.set_title(f'Test Set (n={len(y_test_true_clean)})\nR²={metrics["test_r2"]:.4f}, MAE={metrics["test_mae"]:.2f} years')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Residuals
    ax3 = plt.subplot(2, 3, 3)
    train_residuals = y_train_pred_clean - y_train_true_clean
    test_residuals = y_test_pred_clean - y_test_true_clean

    bp = ax3.boxplot([train_residuals, test_residuals], positions=[1, 2], patch_artist=True, widths=0.6,
                    labels=['Training', 'Test'], showfliers=True)
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax3.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax3.set_ylabel('Residual (Predicted - Actual)')
    ax3.set_title('Residual Distributions')
    ax3.grid(True, alpha=0.3, axis='y')

    # Metrics comparison
    ax4 = plt.subplot(2, 3, 4)
    metrics_to_plot = ['MAE', 'R²', 'Correlation']
    train_values = [metrics['train_mae'], metrics['train_r2'], metrics['train_corr']]
    test_values = [metrics['test_mae'], metrics['test_r2'], metrics['test_corr']]
    x = np.arange(len(metrics_to_plot))
    width = 0.35

    ax4.bar(x - width/2, train_values, width, label='Training', color='blue', alpha=0.7)
    ax4.bar(x + width/2, test_values, width, label='Test', color='red', alpha=0.7)

    ax4.set_xlabel('Metric')
    ax4.set_ylabel('Value')
    ax4.set_title('Generalization Gap Analysis')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics_to_plot)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Age distributions
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(y_train_true_clean, bins=20, alpha=0.5, label='Training', color='blue', density=True)
    ax6.hist(y_test_true_clean, bins=20, alpha=0.5, label='Test', color='red', density=True)
    ax6.set_xlabel('Age (years)')
    ax6.set_ylabel('Density')
    ax6.set_title('Age Distribution Comparison')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure('Blood_comprehensive_performance_analysis.png')
    plt.show()

# ----------------------------------------------------------------------
# Main Pipeline
# ----------------------------------------------------------------------

def main():
    print_section("Blood Epigenetic Clock - Complete Training Pipeline")

    try:
        # Load data
        X, y, all_cpgs = load_blood_data()

        if X is None or y is None:
            print("ERROR: Failed to load data. Exiting.")
            return

        print(f"\nData loaded successfully:")
        print(f"  Samples: {X.shape[0]}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Age range: {y.min():.1f} to {y.max():.1f} years")

        # Split data into train/validation/test
        print("\nSplitting data...")
        if len(y) >= 20:
            try:
                age_bins = pd.qcut(y, q=min(5, len(y)//4), labels=False, duplicates='drop')
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=CONFIG['test_size'], stratify=age_bins, random_state=RANDOM_STATE
                )
                # Further split train into train/validation
                train_bins = pd.qcut(y_train, q=min(5, len(y_train)//4), labels=False, duplicates='drop')
                X_train_final, X_val, y_train_final, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, stratify=train_bins, random_state=RANDOM_STATE
                )
                print(f"  Training set: {len(y_train_final)} samples")
                print(f"  Validation set: {len(y_val)} samples (for optimization)")
                print(f"  Test set: {len(y_test)} samples (final evaluation)")
            except:
                X_train_final, X_test, y_train_final, y_test = train_test_split(
                    X, y, test_size=CONFIG['test_size'], random_state=RANDOM_STATE
                )
                X_val, y_val = X_test.copy(), y_test.copy()
        else:
            X_train_final, X_test, y_train_final, y_test = train_test_split(
                X, y, test_size=min(CONFIG['test_size'], 5/len(y)), random_state=RANDOM_STATE
            )
            X_val, y_val = X_test.copy(), y_test.copy()

        # STEP 1: Feature size optimization
        print_section("STEP 1: Feature Size Optimization")
        best_features, best_size, feature_results = optimize_feature_size(
            X_train_final, y_train_final, X_val, y_val
        )

        # STEP 2: Cross-validation for feature stability
        print_section("STEP 2: Cross-Validation with Feature Stability")
        final_features, stability_df, cv_metrics = cross_validation_stability(
            X_train_final, y_train_final, best_size
        )

        # Save stability results
        save_table(stability_df, 'Blood_feature_stability.csv', 'Feature stability analysis')

        # STEP 3: Hyperparameter optimization
        print_section("STEP 3: Hyperparameter Optimization")
        best_params, hp_results = optimize_hyperparameters(
            X_train_final, y_train_final, X_val, y_val, final_features
        )

        if hp_results is not None:
            save_table(hp_results, 'hyperparameter_optimization_details.csv', 'Hyperparameter optimization details')

        # STEP 4: Ensemble configuration optimization
        print_section("STEP 4: Ensemble Configuration Optimization")
        best_ensemble_config, ensemble_results = optimize_ensemble_config(
            X_train_final, y_train_final, X_val, y_val, final_features, best_params
        )

        # STEP 5: Final model training
        print_section("STEP 5: Final Model Training")
        final_model, final_metrics, y_train_pred, y_test_pred, feature_importance = train_final_model(
            X_train_final, y_train_final, X_test, y_test, final_features,
            best_params, best_ensemble_config
        )

        # Create visualizations
        print("\nCreating comprehensive visualizations...")
        create_comprehensive_visualizations(
            y_train_final, y_train_pred, y_test, y_test_pred, final_metrics
        )

        # Save model and results
        print("\nSaving results...")

        # Prepare model artifact
        model_artifact = {
            'model': final_model,
            'features': final_features,
            'feature_importance': feature_importance,
            'optimized_parameters': {
                'alpha': best_params['alpha'],
                'l1_ratio': best_params['l1_ratio'],
                'n_features': best_size,
                'n_estimators': best_ensemble_config['n_estimators'],
                'max_samples': best_ensemble_config['max_samples']
            },
            'performance_metrics': final_metrics,
            'optimization_results': {
                'feature_size_results': feature_results.to_dict('records') if not feature_results.empty else [],
                'cv_metrics': cv_metrics,
                'ensemble_results': ensemble_results.to_dict('records') if not ensemble_results.empty else []
            },
            'data_info': {
                'n_samples_total': len(y),
                'n_samples_train': len(y_train_final),
                'n_samples_test': len(y_test),
                'age_range': f"{y.min():.1f}-{y.max():.1f}",
                'mean_age': f"{y.mean():.1f} ± {y.std():.1f}"
            }
        }

        # Save files
        save_model(model_artifact, 'Blood_epigenetic_clock.pkl', 'Optimized blood epigenetic clock')

        if not feature_importance.empty:
            save_table(feature_importance, 'Blood_feature_importance.csv', 'Feature importance scores')

        # Performance summary
        perf_summary = pd.DataFrame([{
            'Model': 'Optimized_ElasticNet_Ensemble',
            'Test_R2': final_metrics['test_r2'],
            'Test_MAE': final_metrics['test_mae'],
            'Test_Correlation': final_metrics['test_corr'],
            'Test_RMSE': final_metrics['test_rmse'],
            'Train_R2': final_metrics['train_r2'],
            'Train_MAE': final_metrics['train_mae'],
            'Train_Correlation': final_metrics['train_corr'],
            'MAE_Gap': final_metrics['mae_gap'],
            'MAE_Ratio': final_metrics['mae_ratio'],
            'R2_Gap': final_metrics['r2_gap'],
            'R2_Ratio': final_metrics['r2_ratio'],
            'MAE_Generalization_Assessment': final_metrics['mae_assessment'],
            'R2_Generalization_Assessment': final_metrics['r2_assessment'],
            'Non_Zero_Features': final_metrics.get('n_nonzero', 0),
            'Total_Features': len(final_features),
            'Alpha': best_params['alpha'],
            'L1_Ratio': best_params['l1_ratio'],
            'Ensemble_Estimators': best_ensemble_config['n_estimators'],
            'Ensemble_MaxSamples': best_ensemble_config['max_samples'],
            'Training_Time_Seconds': final_metrics['training_time']
        }])

        save_table(perf_summary, 'Blood_clock_performance.csv', 'Blood clock performance')

        # Create optimization summary
        opt_summary = pd.DataFrame([{
            'Optimization_Step': 'Feature_Size',
            'Optimal_Value': best_size,
            'Performance_Metric': 'Validation_R2',
            'Optimal_Score': feature_results['validation_r2'].max() if not feature_results.empty else 'N/A'
        }, {
            'Optimization_Step': 'Hyperparameters',
            'Optimal_Value': f"alpha={best_params['alpha']:.6f}, l1_ratio={best_params['l1_ratio']:.3f}",
            'Performance_Metric': 'Validation_R2',
            'Optimal_Score': hp_results['mean_test_score'].max() if hp_results is not None else 'N/A'
        }, {
            'Optimization_Step': 'Ensemble_Configuration',
            'Optimal_Value': f"n_estimators={best_ensemble_config['n_estimators']}, max_samples={best_ensemble_config['max_samples']}",
            'Performance_Metric': 'Validation_R2',
            'Optimal_Score': ensemble_results['validation_r2'].max() if not ensemble_results.empty else 'N/A'
        }])

        save_table(opt_summary, 'optimization_summary.csv', 'Optimization process summary')

        # Final report
        print_section("TRAINING COMPLETE - FINAL RESULTS")

        print(f"\n FINAL MODEL PERFORMANCE:")
        print(f"   Test R²: {final_metrics['test_r2']:.4f}")
        print(f"   Test MAE: {final_metrics['test_mae']:.2f} years")
        print(f"   Test Correlation: {final_metrics['test_corr']:.3f}")
        print(f"   Test RMSE: {final_metrics['test_rmse']:.2f} years")

        print(f"\n  OPTIMIZED CONFIGURATION:")
        print(f"   Features: {len(final_features)}")
        print(f"   Alpha: {best_params['alpha']:.6f}")
        print(f"   L1 Ratio: {best_params['l1_ratio']:.3f}")
        print(f"   Ensemble: {best_ensemble_config['n_estimators']} estimators")
        print(f"   Subsample: {best_ensemble_config['max_samples']*100:.0f}%")
        print(f"   Non-zero coefficients: {final_metrics.get('n_nonzero', 0)}")

        print(f"\n MODEL CHARACTERISTICS:")
        print(f"   Training R²: {final_metrics['train_r2']:.4f}")
        print(f"   Training MAE: {final_metrics['train_mae']:.2f} years")
        print(f"   Generalization assessment: {final_metrics['mae_assessment']}")
        print(f"   MAE ratio (train/test): {final_metrics['mae_ratio']:.3f}")
        print(f"   R² gap (train-test): {final_metrics['r2_gap']:.4f}")

        print(f"\n OUTPUT FILES SAVED:")
        print(f"   Model: Blood_epigenetic_clock.pkl")
        print(f"   Performance: Blood_clock_performance.csv")
        print(f"   Feature importance: Blood_feature_importance.csv")
        print(f"   Feature stability: Blood_feature_stability.csv")
        print(f"   Optimization summary: optimization_summary.csv")
        print(f"   Feature size results: feature_size_optimization_results.csv")
        print(f"   Hyperparameter results: hyperparameter_optimization_details.csv")
        print(f"   Ensemble results: ensemble_configuration_results.csv")
        print(f"   Visualizations: See figures directory")

        # Create comprehensive README
        readme = f"""
Blood Epigenetic Clock - Optimized ElasticNet Ensemble
=======================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OPTIMIZATION PROCESS:
This model was developed through systematic optimization:
1. Feature Size Optimization: Tested {len(CONFIG['feature_sizes'])} different feature counts
2. Hyperparameter Tuning: Searched {CONFIG['n_random_search_iter']} combinations of alpha and l1_ratio
3. Ensemble Configuration: Tested {len(CONFIG['ensemble_configs'])} ensemble configurations

OPTIMAL PARAMETERS FOUND:
- Alpha: {best_params['alpha']:.6f}
- L1 Ratio: {best_params['l1_ratio']:.3f}
- Number of Features: {len(final_features)}
- Ensemble Size: {best_ensemble_config['n_estimators']} estimators
- Subsample Ratio: {best_ensemble_config['max_samples']*100:.0f}%

PERFORMANCE:
- Test R²: {final_metrics['test_r2']:.4f}
- Test MAE: {final_metrics['test_mae']:.2f} years
- Test Correlation: {final_metrics['test_corr']:.3f}
- Non-zero Features: {final_metrics.get('n_nonzero', 0)}/{len(final_features)}
- Generalization: {final_metrics['mae_assessment']}

DATA SUMMARY:
- Total Samples: {len(y)}
- Training Samples: {len(y_train_final)}
- Test Samples: {len(y_test)}
- Age Range: {y.min():.1f} - {y.max():.1f} years
- Mean Age: {y.mean():.1f} ± {y.std():.1f} years

FILES:
- Model: Blood_epigenetic_clock.pkl (contains model, features, and metadata)
- Performance: Blood_clock_performance.csv
- Feature Importance: Blood_feature_importance.csv
- Feature Stability: Blood_feature_stability.csv
- Optimization Summary: optimization_summary.csv
- Visualizations: PNG files in figures directory

Note: This model was optimized using systematic search to find the best hyperparameters
for the given dataset, ensuring robust performance and good generalization.
"""

        save_report(readme, 'Blood_clock_README.txt')

        print_section("✓ TRAINING PIPELINE COMPLETE ✓")

    except Exception as e:
        print(f"\n Error in training pipeline: {str(e)}")
        import traceback
        traceback.print_exc()

        error_report = f"""
Error report - Blood Epigenetic Clock Training
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Error: {str(e)}
"""
        save_report(error_report, 'Blood_training_error.txt')

if __name__ == "__main__":
    main()
