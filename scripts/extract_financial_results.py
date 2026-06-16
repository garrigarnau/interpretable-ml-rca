"""
Script to extract financial results from EBM model without Streamlit UI
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from interpret.glassbox import ExplainableBoostingRegressor
import os
from collections import defaultdict

# Import data cleaning functions
from scripts.data_cleaning import (
    load_data,
    load_variable_descriptions,
    classify_batch_quality,
    get_basic_stats,
    apply_column_removal,
    get_equipment_columns,
    get_lot_number_columns,
    get_manual_categorical_columns,
    get_time_columns,
)

print("="*80)
print("FINANCIAL IMPACT ANALYSIS - EBM MODEL")
print("="*80)

# ============================================================================
# DATA LOADING
# ============================================================================
print("\n[1] Loading data...")
data_file_path = "data/fractionation_data.csv"
current_data_mtime = os.path.getmtime(data_file_path) if os.path.exists(data_file_path) else None

df = load_data(file_path=data_file_path, cache_buster=current_data_mtime)
var_descriptions = load_variable_descriptions()

if df is None:
    print("ERROR: Failed to load data")
    exit(1)

# Drop batch identifier column (A1)
if 'A1' in df.columns:
    df = df.drop(columns=['A1'])

basic_stats = get_basic_stats(df)
print(f"   Total batches loaded: {len(df)}")
print(f"   Total variables: {len(df.columns)}")

# ============================================================================
# DATA FILTERING
# ============================================================================
print("\n[2] Filtering data...")
df, thresholds, quality_counts = classify_batch_quality(df, target_col='D49')

# Apply date filter: from 2024-01-08 onwards (production method change)
df_filtered = df.copy()
if 'B1' in df_filtered.columns:
    df_filtered['B1_datetime'] = pd.to_datetime(df_filtered['B1'], errors='coerce')
    start_date = pd.to_datetime('2024-01-08', utc=True)
    # Convert to same timezone for comparison
    if df_filtered['B1_datetime'].dt.tz is not None:
        start_date = start_date.tz_convert(df_filtered['B1_datetime'].dt.tz)
    date_mask = df_filtered['B1_datetime'] >= start_date
    df_filtered = df_filtered[date_mask.fillna(False)]
    df_filtered = df_filtered.drop(columns=['B1_datetime'])
    print(f"   Applied date filter: >= 2024-01-08 (production method change)")
else:
    print(f"   WARNING: B1 column not found, no date filter applied")

# Apply quality filter (all qualities)
if 'Batch_Quality' in df_filtered.columns:
    quality_options = df_filtered['Batch_Quality'].dropna().unique().tolist()
    df_filtered = df_filtered[df_filtered['Batch_Quality'].isin(quality_options)]

print(f"   Batches after filtering: {len(df_filtered)}")
print(f"   Quality distribution: {quality_counts}")

# ============================================================================
# AUTOMATIC FEATURE CLEANING
# ============================================================================
print("\n[3] Feature cleaning...")

# Step 1: High missing columns (threshold 35%)
missing_threshold = 35
missing_pct = (df_filtered.isnull().sum() / len(df_filtered)) * 100
cols_high_missing = missing_pct[missing_pct > missing_threshold].index.tolist()
protected_cols = ['D49', 'Batch_Quality']
cols_high_missing = [col for col in cols_high_missing if col not in protected_cols]
df_after_missing = apply_column_removal(df_filtered, cols_high_missing)
print(f"   Removed {len(cols_high_missing)} high-missing columns")

# Step 2: Time and lot columns
time_cols = get_time_columns()
lot_cols = get_lot_number_columns()
time_cols_in_df = [col for col in time_cols if col in df_after_missing.columns]
lot_cols_in_df = [col for col in lot_cols if col in df_after_missing.columns]
cols_time_and_lot = sorted(set(time_cols_in_df + lot_cols_in_df))
df_after_time_lot = apply_column_removal(df_after_missing, cols_time_and_lot)
print(f"   Removed {len(cols_time_and_lot)} time/lot columns")

# Step 3: Low variance columns
variance_threshold = 0.10
categorical_cols = (get_equipment_columns() + get_manual_categorical_columns())
categorical_cols = [col for col in categorical_cols if col in df_after_time_lot.columns]
numeric_cols = [
    col for col in df_after_time_lot.select_dtypes(include=[np.number]).columns
    if col not in protected_cols + categorical_cols
]
if numeric_cols:
    variance_series = df_after_time_lot[numeric_cols].var()
    low_variance_cols = variance_series[variance_series < variance_threshold].index.tolist()
    df_after_variance = apply_column_removal(df_after_time_lot, low_variance_cols)
else:
    low_variance_cols = []
    df_after_variance = df_after_time_lot.copy()
print(f"   Removed {len(low_variance_cols)} low-variance columns")

# Step 4: Highly correlated columns
correlation_threshold = 0.90
numeric_after_var = [
    col for col in df_after_variance.select_dtypes(include=[np.number]).columns
    if col not in protected_cols + categorical_cols
]
cols_to_drop_corr = []
if len(numeric_after_var) > 1:
    corr_matrix = df_after_variance[numeric_after_var].corr(method='pearson', min_periods=1)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col_a = corr_matrix.columns[i]
            col_b = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > correlation_threshold:
                missing_a = df_after_variance[col_a].isnull().sum()
                missing_b = df_after_variance[col_b].isnull().sum()
                if missing_a > missing_b:
                    drop_col = col_a
                elif missing_b > missing_a:
                    drop_col = col_b
                else:
                    drop_col = col_a if col_a > col_b else col_b
                high_corr_pairs.append(drop_col)
    cols_to_drop_corr = sorted(set(high_corr_pairs))
df_after_correlation = apply_column_removal(df_after_variance, cols_to_drop_corr)
print(f"   Removed {len(cols_to_drop_corr)} correlated columns")

total_removed = len(cols_high_missing) + len(cols_time_and_lot) + len(low_variance_cols) + len(cols_to_drop_corr)
print(f"   Total columns removed: {total_removed}")
print(f"   Final column count: {len(df_after_correlation.columns)}")

# ============================================================================
# DATA PREPARATION
# ============================================================================
print("\n[4] Data preparation...")
df_model = df_after_correlation[df_after_correlation['D49'].notna()].copy()
print(f"   Batches with valid D49: {len(df_model)}")

# Save D48 for financial calculations
if 'D48' in df_model.columns:
    d48_saved = df_model['D48'].copy()
    df_model = df_model.drop(columns=['D48'])
else:
    d48_saved = None

# Separate features and target
d49_position = df_model.columns.get_loc('D49')
feature_cols = [col for col in df_model.columns[:d49_position] if col != 'Batch_Quality']
X_raw = df_model[feature_cols].copy()
y = df_model['D49'].copy()

# Identify categorical columns
known_categorical_cols = set(categorical_cols)
categorical_in_features = [
    col for col in X_raw.columns
    if X_raw[col].dtype == object or col in known_categorical_cols
]
numeric_in_features = [col for col in X_raw.columns if col not in categorical_in_features]
print(f"   Categorical features: {len(categorical_in_features)}")
print(f"   Numeric features: {len(numeric_in_features)}")

# Prepare data for EBM
X_prepared = X_raw.copy()
for col in categorical_in_features:
    X_prepared[col] = X_prepared[col].fillna('Unknown').astype(str)
for col in numeric_in_features:
    X_prepared[col] = X_prepared[col].fillna(X_prepared[col].mean())

print(f"   Total samples for CV: {len(X_prepared)}")
print(f"   Will use 5-fold cross-validation")

# ============================================================================
# EBM TRAINING WITH 5-FOLD CROSS-VALIDATION
# ============================================================================
print("\n[5] Training EBM models with 5-Fold Cross-Validation...")

# Configuration (using best config from notebook CV-5: top_n=15, interactions=10, lr=0.05, min_samples_leaf=10)
top_n_features = min(15, X_prepared.shape[1])
n_interactions = 10
learning_rate = 0.05  # Best from CV-5
max_bins = 256
max_rounds = 5000  # Default from best config
min_samples_leaf = 10
n_jobs = 1
random_state = 42

# Phase 1: Feature ranking with 5-fold CV
print(f"   Phase 1: 5-Fold CV with all {X_prepared.shape[1]} features (interactions=0)...")

kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
cv_scores_phase1 = {'mae': [], 'rmse': [], 'r2': []}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_prepared), 1):
    X_train_cv = X_prepared.iloc[train_idx]
    X_val_cv = X_prepared.iloc[val_idx]
    y_train_cv = y.iloc[train_idx]
    y_val_cv = y.iloc[val_idx]

    ebm_cv = ExplainableBoostingRegressor(
        interactions=0,
        learning_rate=learning_rate,
        max_bins=max_bins,
        max_rounds=max_rounds,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
        random_state=random_state
    )
    ebm_cv.fit(X_train_cv, y_train_cv)
    y_pred_cv = ebm_cv.predict(X_val_cv)

    mae_cv = mean_absolute_error(y_val_cv, y_pred_cv)
    rmse_cv = np.sqrt(mean_squared_error(y_val_cv, y_pred_cv))
    r2_cv = r2_score(y_val_cv, y_pred_cv)

    cv_scores_phase1['mae'].append(mae_cv)
    cv_scores_phase1['rmse'].append(rmse_cv)
    cv_scores_phase1['r2'].append(r2_cv)

mae_phase1 = np.mean(cv_scores_phase1['mae'])
mae_std_phase1 = np.std(cv_scores_phase1['mae'])
rmse_phase1 = np.mean(cv_scores_phase1['rmse'])
r2_phase1 = np.mean(cv_scores_phase1['r2'])

print(f"      MAE={mae_phase1:.4f}±{mae_std_phase1:.4f}  RMSE={rmse_phase1:.4f}  R²={r2_phase1:.4f}")

# Train final model on full dataset for feature importance extraction
ebm_phase1 = ExplainableBoostingRegressor(
    interactions=0,
    learning_rate=learning_rate,
    max_bins=max_bins,
    max_rounds=max_rounds,
    min_samples_leaf=min_samples_leaf,
    n_jobs=n_jobs,
    random_state=random_state
)
ebm_phase1.fit(X_prepared, y)

# Extract feature importance
global_exp_phase1 = ebm_phase1.explain_global(name='Phase 1 Global')
global_data_phase1 = global_exp_phase1.data()
importance_phase1 = pd.DataFrame({
    'Feature': global_data_phase1.get('names', []),
    'Importance': global_data_phase1.get('scores', [])
}).sort_values('Importance', ascending=False)

# Phase 2: Feature pruning
top_features = importance_phase1.head(top_n_features)['Feature'].tolist()
X_prepared_pruned = X_prepared[top_features].copy()
print(f"   Phase 2: Pruned to top-{top_n_features} features")

# Phase 3: Interaction hunting with 5-fold CV
print(f"   Phase 3: 5-Fold CV with top-{top_n_features} features + {n_interactions} interactions...")

cv_scores_phase3 = {'mae': [], 'rmse': [], 'r2': []}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_prepared_pruned), 1):
    X_train_cv = X_prepared_pruned.iloc[train_idx]
    X_val_cv = X_prepared_pruned.iloc[val_idx]
    y_train_cv = y.iloc[train_idx]
    y_val_cv = y.iloc[val_idx]

    ebm_cv = ExplainableBoostingRegressor(
        interactions=n_interactions,
        learning_rate=learning_rate,
        max_bins=max_bins,
        max_rounds=max_rounds,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
        random_state=random_state
    )
    ebm_cv.fit(X_train_cv, y_train_cv)
    y_pred_cv = ebm_cv.predict(X_val_cv)

    mae_cv = mean_absolute_error(y_val_cv, y_pred_cv)
    rmse_cv = np.sqrt(mean_squared_error(y_val_cv, y_pred_cv))
    r2_cv = r2_score(y_val_cv, y_pred_cv)

    cv_scores_phase3['mae'].append(mae_cv)
    cv_scores_phase3['rmse'].append(rmse_cv)
    cv_scores_phase3['r2'].append(r2_cv)

mae_phase3 = np.mean(cv_scores_phase3['mae'])
mae_std_phase3 = np.std(cv_scores_phase3['mae'])
rmse_phase3 = np.mean(cv_scores_phase3['rmse'])
r2_phase3 = np.mean(cv_scores_phase3['r2'])

print(f"      MAE={mae_phase3:.4f}±{mae_std_phase3:.4f}  RMSE={rmse_phase3:.4f}  R²={r2_phase3:.4f}")

# Train final model on full dataset for predictions
ebm_phase3 = ExplainableBoostingRegressor(
    interactions=n_interactions,
    learning_rate=learning_rate,
    max_bins=max_bins,
    max_rounds=max_rounds,
    min_samples_leaf=min_samples_leaf,
    n_jobs=n_jobs,
    random_state=random_state
)
ebm_phase3.fit(X_prepared_pruned, y)

# ============================================================================
# PORTFOLIO ANALYSIS FOR FINANCIAL IMPACT
# ============================================================================
print("\n" + "="*80)
print("FINANCIAL IMPACT ANALYSIS")
print("="*80)

# Get predictions for all batches (using full dataset)
X_all_pruned = X_prepared_pruned.copy()
y_all = y.copy()
y_pred_all = ebm_phase3.predict(X_all_pruned)

# Analyze all batches for optimization potential
all_batch_analysis = []
print(f"\n[6] Analyzing {len(X_all_pruned)} batches for optimization potential...")

for idx in X_all_pruned.index:
    single_batch = X_all_pruned.loc[[idx]].copy()
    try:
        local_exp = ebm_phase3.explain_local(single_batch, name=f"Batch {idx}")
        local_data = local_exp.data(0)
        contribution_scores = [float(v) for v in local_data.get('scores', [])]
        contribution_feature_names = local_data.get('names', list(single_batch.columns))

        pred = float(ebm_phase3.predict(single_batch)[0])
        actual = float(y_all.loc[idx])

        # Get D48 weight for this batch
        d48_weight = d48_saved.loc[idx] if d48_saved is not None and idx in d48_saved.index else np.nan

        # Identify negative contributors
        negative_contribs = []
        negative_features = []
        for fname, score in zip(contribution_feature_names, contribution_scores):
            if score < -0.1:
                negative_contribs.append(score)
                negative_features.append(fname)

        negative_sum = sum(negative_contribs)
        conservative_gain = -negative_sum * 0.7  # 70% recovery
        optimized_potential = pred + conservative_gain

        all_batch_analysis.append({
            'batch_index': idx,
            'actual': actual,
            'predicted': pred,
            'd48_weight': d48_weight,
            'negative_sum': negative_sum,
            'conservative_gain': conservative_gain,
            'optimized_potential': optimized_potential,
            'num_negative_features': len(negative_features),
            'negative_features': negative_features
        })
    except Exception:
        continue

portfolio_df = pd.DataFrame(all_batch_analysis)

# Calculate aggregate metrics
total_batches = len(portfolio_df)

# Filter batches with valid D48 weights for weighted average
portfolio_with_d48 = portfolio_df[portfolio_df['d48_weight'].notna() & (portfolio_df['d48_weight'] > 0)].copy()

if len(portfolio_with_d48) > 0:
    # Weighted averages (weighted by D48 kg)
    total_weight = portfolio_with_d48['d48_weight'].sum()
    weighted_current_pred = (portfolio_with_d48['predicted'] * portfolio_with_d48['d48_weight']).sum() / total_weight
    weighted_optimized = (portfolio_with_d48['optimized_potential'] * portfolio_with_d48['d48_weight']).sum() / total_weight
    weighted_gain = weighted_optimized - weighted_current_pred
    weighted_gain_pct = (weighted_gain / weighted_current_pred) * 100 if weighted_current_pred > 0 else 0

    # Simple averages for comparison
    simple_current_pred = portfolio_df['predicted'].mean()
    simple_optimized = portfolio_df['optimized_potential'].mean()
    simple_gain = simple_optimized - simple_current_pred
    simple_gain_pct = (simple_gain / simple_current_pred) * 100 if simple_current_pred > 0 else 0

    print(f"\nPORTFOLIO-LEVEL METRICS (WEIGHTED BY D48 kg):")
    print(f"   Total Batches Analyzed: {total_batches}")
    print(f"   Batches with Valid D48: {len(portfolio_with_d48)}")
    print(f"   Total Weight (D48): {total_weight:,.1f} kg")
    print(f"")
    print(f"   WEIGHTED AVERAGES (by production volume):")
    print(f"     Avg Current Prediction: {weighted_current_pred:.2f}%")
    print(f"     Avg Optimized Potential: {weighted_optimized:.2f}%")
    print(f"     Portfolio Improvement Potential: +{weighted_gain:.2f} yield points ({weighted_gain_pct:.1f}%)")
    print(f"")
    print(f"   SIMPLE AVERAGES (for comparison):")
    print(f"     Avg Current Prediction: {simple_current_pred:.2f}%")
    print(f"     Avg Optimized Potential: {simple_optimized:.2f}%")
    print(f"     Portfolio Improvement Potential: +{simple_gain:.2f} yield points ({simple_gain_pct:.1f}%)")

    # Use weighted values for downstream calculations
    avg_current_pred = weighted_current_pred
    avg_optimized = weighted_optimized
    total_gain = weighted_gain
    total_gain_pct = weighted_gain_pct
else:
    # Fallback to simple average if no D48 data
    avg_current_pred = portfolio_df['predicted'].mean()
    avg_optimized = portfolio_df['optimized_potential'].mean()
    total_gain = avg_optimized - avg_current_pred
    total_gain_pct = (total_gain / avg_current_pred) * 100 if avg_current_pred > 0 else 0

    print(f"\nPORTFOLIO-LEVEL METRICS (SIMPLE AVERAGES - no D48 data):")
    print(f"   Total Batches Analyzed: {total_batches}")
    print(f"   Avg Current Prediction: {avg_current_pred:.2f}%")
    print(f"   Avg Optimized Potential: {avg_optimized:.2f}%")
    print(f"   Portfolio Improvement Potential: +{total_gain:.2f} yield points ({total_gain_pct:.1f}%)")

batches_with_issues = (portfolio_df['num_negative_features'] > 0).sum()
pct_batches_with_issues = (batches_with_issues / total_batches) * 100
print(f"")
print(f"   Batches with Negative Drivers: {batches_with_issues} ({pct_batches_with_issues:.1f}%)")

# Most common problematic features
feature_problem_count = {}
for batch in all_batch_analysis:
    for fname in batch['negative_features']:
        feature_problem_count[fname] = feature_problem_count.get(fname, 0) + 1

if feature_problem_count:
    problem_features_df = pd.DataFrame([
        {'Feature': fname, 'Batches Affected': count, '% of Batches': (count / total_batches) * 100}
        for fname, count in feature_problem_count.items()
    ]).sort_values('Batches Affected', ascending=False)

    print(f"\nMOST COMMON PROBLEMATIC FEATURES (Top 10):")
    for i, row in problem_features_df.head(10).iterrows():
        print(f"   {row['Feature']}: {row['Batches Affected']} batches ({row['% of Batches']:.1f}%)")

    # Top 3 focused optimization (weighted by D48)
    top_3_features = problem_features_df.head(3)['Feature'].tolist()
    focused_gain_weighted = 0
    total_weight_focused = 0

    for batch in all_batch_analysis:
        single_batch = X_all_pruned.loc[[batch['batch_index']]].copy()
        d48_weight = batch['d48_weight']

        # Skip if no valid D48
        if pd.isna(d48_weight) or d48_weight <= 0:
            continue

        try:
            local_exp = ebm_phase3.explain_local(single_batch)
            local_data = local_exp.data(0)
            contribution_scores = [float(v) for v in local_data.get('scores', [])]
            contribution_feature_names = local_data.get('names', [])

            top_3_negative = sum([
                score for fname, score in zip(contribution_feature_names, contribution_scores)
                if score < -0.1 and fname in top_3_features
            ])
            batch_focused_gain = -top_3_negative * 0.7

            # Weight by D48
            focused_gain_weighted += batch_focused_gain * d48_weight
            total_weight_focused += d48_weight
        except Exception:
            continue

    if total_weight_focused > 0:
        avg_focused_gain = focused_gain_weighted / total_weight_focused
        focused_pct = (avg_focused_gain / avg_current_pred) * 100 if avg_current_pred > 0 else 0

        print(f"\nFOCUSED OPTIMIZATION (Top 3 Features, weighted by D48):")
        print(f"   Features to Optimize: 3 ({', '.join(top_3_features)})")
        print(f"   Expected Gain (weighted): +{avg_focused_gain:.2f} yield points")
        print(f"   Improvement %: +{focused_pct:.1f}%")
    else:
        print(f"\nFOCUSED OPTIMIZATION: No valid D48 data for calculation")

# ============================================================================
# FINANCIAL IMPACT CALCULATION
# ============================================================================
print(f"\n" + "="*80)
print("FINANCIAL IMPACT CALCULATION")
print("="*80)

if d48_saved is not None:
    price_min = 50  # $/g
    price_max = 60  # $/g
    price_mid = 55  # $/g (midpoint)

    # Apply R² adjustment
    r2_adjustment_portfolio = max(0.3, r2_phase3)

    total_current_revenue_mid = 0
    total_optimized_revenue_mid = 0
    total_optimized_revenue_best = 0
    total_optimized_revenue_worst = 0
    batches_with_d48 = 0

    for batch in all_batch_analysis:
        batch_idx = batch['batch_index']

        if batch_idx in d48_saved.index:
            d48_value = d48_saved.loc[batch_idx]

            if pd.notna(d48_value) and d48_value > 0:
                batches_with_d48 += 1

                # Current revenue
                current_quantity_g = d48_value * 1000
                current_revenue = current_quantity_g * price_mid

                # Optimized revenue scenario
                current_d49 = batch['actual']
                predicted_d49 = batch['predicted']
                optimized_d49_pred = batch['optimized_potential']

                # Yield gain from optimization (already has 70% * R² adjustment)
                yield_gain_expected = optimized_d49_pred - predicted_d49
                yield_gain_best = yield_gain_expected + mae_phase3
                yield_gain_worst = max(0, yield_gain_expected - mae_phase3)

                # Calculate revenues for three scenarios
                def calc_revenue(yield_gain):
                    scenario_d49 = current_d49 + yield_gain
                    if current_d49 > 0:
                        factor = scenario_d49 / current_d49
                        return current_quantity_g * factor * price_mid
                    else:
                        return current_revenue

                optimized_revenue_expected = calc_revenue(yield_gain_expected)
                optimized_revenue_best = calc_revenue(yield_gain_best)
                optimized_revenue_worst = calc_revenue(yield_gain_worst)

                total_current_revenue_mid += current_revenue
                total_optimized_revenue_mid += optimized_revenue_expected
                total_optimized_revenue_best += optimized_revenue_best
                total_optimized_revenue_worst += optimized_revenue_worst

    if batches_with_d48 > 0:
        # Calculate revenue gains
        total_revenue_gain_mid = total_optimized_revenue_mid - total_current_revenue_mid
        total_revenue_gain_min = total_revenue_gain_mid * (price_min / price_mid)
        total_revenue_gain_max = total_revenue_gain_mid * (price_max / price_mid)

        # Per-batch averages
        avg_current_revenue = total_current_revenue_mid / batches_with_d48
        avg_optimized_revenue = total_optimized_revenue_mid / batches_with_d48
        avg_revenue_gain = total_revenue_gain_mid / batches_with_d48

        # Calculate quantity gains
        total_quantity_gain_kg = ((total_optimized_revenue_mid - total_current_revenue_mid) / price_mid) / 1000

        print(f"\nAnalysis based on {batches_with_d48} batches with D48 data")
        print(f"\nCURRENT PORTFOLIO REVENUE (@ $55/g):")
        print(f"   Total Revenue: ${total_current_revenue_mid:,.0f}")
        print(f"   Avg Revenue per Batch: ${avg_current_revenue:,.0f}")

        print(f"\nOPTIMIZED PORTFOLIO REVENUE (@ $55/g):")
        print(f"   Total Revenue: ${total_optimized_revenue_mid:,.0f} (+${total_revenue_gain_mid:,.0f})")
        print(f"   Avg Revenue per Batch: ${avg_optimized_revenue:,.0f} (+${avg_revenue_gain:,.0f})")

        print(f"\nTOTAL REVENUE OPPORTUNITY (Price Range):")
        print(f"   @ $50/g: +${total_revenue_gain_min:,.0f}")
        print(f"   @ $55/g: +${total_revenue_gain_mid:,.0f}")
        print(f"   @ $60/g: +${total_revenue_gain_max:,.0f}")

        print(f"\nADDITIONAL PRODUCT OUTPUT:")
        print(f"   Portfolio Total: +{total_quantity_gain_kg:,.1f} kg")

        # Annual projection
        batches_per_year = 100  # Assumption
        annual_revenue_gain_mid = avg_revenue_gain * batches_per_year
        annual_revenue_gain_min = annual_revenue_gain_mid * (price_min / price_mid)
        annual_revenue_gain_max = annual_revenue_gain_mid * (price_max / price_mid)

        print(f"\nANNUAL PROJECTION (assuming {batches_per_year} batches/year):")
        print(f"   Annual Gain @ $50/g: ${annual_revenue_gain_min:,.0f}")
        print(f"   Annual Gain @ $55/g: ${annual_revenue_gain_mid:,.0f}")
        print(f"   Annual Gain @ $60/g: ${annual_revenue_gain_max:,.0f}")

        print(f"\n" + "="*80)
        print("BUSINESS CASE SUMMARY")
        print("="*80)
        print(f"Total Revenue Opportunity: ${total_revenue_gain_min:,.0f} - ${total_revenue_gain_max:,.0f}")
        print(f"Per Batch Average: ${avg_revenue_gain:,.0f} additional revenue")
        print(f"Total Additional Product: {total_quantity_gain_kg:,.1f} kg")
        print(f"Model Uncertainty: MAE={mae_phase3:.2f}pp, R²={r2_phase3:.2f}")
        print(f"Conservative Adjustment: 70% × {r2_adjustment_portfolio:.2f} = {0.7*r2_adjustment_portfolio:.1%}")
        print("="*80)
    else:
        print("\nNo batches with valid D48 data found")
else:
    print("\nD48 data not available - cannot calculate financial impact")

print("\nAnalysis complete!")
