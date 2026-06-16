import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from interpret.glassbox import ExplainableBoostingRegressor
import os

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
    get_time_columns_in_dataframe
)

# Import dataset overview functions
from scripts.dataset_overview import (
    display_overview_metrics,
    display_target_variable_visualization,
    display_target_by_quality
)

# Page configuration
st.set_page_config(page_title="EBM Filter-Then-Interact Pipeline", layout="wide")

st.title("EBM Filter-Then-Interact Pipeline")
st.markdown("""
This app implements an **automated data cleaning pipeline** followed by the **EBM filter-then-interact approach**:
1. Automatic removal of high-missing, low-variance, and highly-correlated features
2. EBM Phase 1: Feature ranking with interactions=0
3. EBM Phase 2: Prune to top-N features
4. EBM Phase 3: Re-train with interaction detection
""")

# ============================================================================
# PHASE 1: DATA LOADING & INITIAL SETUP
# ============================================================================
st.header("1. Data Loading")

data_file_path = "data/fractionation_data.csv"
current_data_mtime = os.path.getmtime(data_file_path) if os.path.exists(data_file_path) else None

df = load_data(file_path=data_file_path, cache_buster=current_data_mtime)
var_descriptions = load_variable_descriptions()

if df is None:
    st.error("Failed to load data. Please ensure 'data/fractionation_data.csv' exists.")
    st.stop()

# Drop batch identifier column (A1) - tracked via row index instead
if 'A1' in df.columns:
    df = df.drop(columns=['A1'])
    st.caption("Removed A1 (batch identifier)")

# Get basic statistics
basic_stats = get_basic_stats(df)
display_overview_metrics(basic_stats)

st.divider()

# ============================================================================
# PHASE 2: DATA FILTERING
# ============================================================================
st.header("2. Data Filtering")

# Display target variable
display_target_variable_visualization(df, target_col='D49', var_descriptions=var_descriptions)

# Batch quality classification
df, thresholds, quality_counts = classify_batch_quality(df, target_col='D49')

if thresholds is not None:
    st.subheader("Batch Quality Classification")
    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    col_q1.metric("Good Batches", quality_counts.get('Good', 0))
    col_q2.metric("Average Batches", quality_counts.get('Average', 0))
    col_q3.metric("Bad Batches", quality_counts.get('Bad', 0))
    col_q4.metric("Unknown", quality_counts.get('Unknown', 0))

    # Show distribution colored by quality
    display_target_by_quality(df, target_col='D49', quality_col='Batch_Quality')

st.subheader("Filter Batches")

col_filter1, col_filter2 = st.columns(2)

with col_filter1:
    quality_options = sorted(df['Batch_Quality'].dropna().unique().tolist()) if 'Batch_Quality' in df.columns else []
    selected_qualities = st.multiselect(
        "Filter by Batch Quality",
        options=quality_options,
        default=quality_options,
        help="Select quality levels to include"
    )

with col_filter2:
    if 'B1' in df.columns:
        b1_datetime = pd.to_datetime(df['B1'], errors='coerce')
        valid_b1 = b1_datetime.dropna()

        if not valid_b1.empty:
            min_date = valid_b1.min().date()
            max_date = valid_b1.max().date()

            # Default date range from notebook configuration
            default_start = pd.to_datetime("2024-01-08").date()
            default_end = pd.to_datetime("2025-12-13").date()

            # Ensure defaults are within the available data range
            if default_start < min_date:
                default_start = min_date
            if default_end > max_date:
                default_end = max_date

            selected_date_range = st.date_input(
                "Filter by Date Range (B1)",
                value=(default_start, default_end),
                min_value=min_date,
                max_value=max_date,
                help="Default: 2024-01-08 to 2025-12-13 (from notebook config)"
            )

            if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
                start_date, end_date = selected_date_range
            else:
                start_date = default_start
                end_date = default_end
        else:
            start_date = None
            end_date = None
            st.info("B1 column exists but cannot be parsed as datetime")
    else:
        start_date = None
        end_date = None
        st.info("B1 column not found")

# Apply filters
df_filtered = df.copy()

if 'Batch_Quality' in df_filtered.columns and selected_qualities:
    df_filtered = df_filtered[df_filtered['Batch_Quality'].isin(selected_qualities)]

if 'B1' in df_filtered.columns and start_date is not None and end_date is not None:
    b1_filtered = pd.to_datetime(df_filtered['B1'], errors='coerce')
    date_mask = (b1_filtered.dt.date >= start_date) & (b1_filtered.dt.date <= end_date)
    df_filtered = df_filtered[date_mask.fillna(False)]

col_f1, col_f2, col_f3 = st.columns(3)
col_f1.metric("Original Batches", len(df))
col_f2.metric("Filtered Batches", len(df_filtered))
col_f3.metric("Batches Removed", len(df) - len(df_filtered))

if len(df_filtered) == 0:
    st.error("No batches match the filters. Please adjust.")
    st.stop()

st.success(f"Using {len(df_filtered)} batches for analysis")

st.divider()

# ============================================================================
# PHASE 3: AUTOMATIC FEATURE CLEANING
# ============================================================================
st.header("3. Automatic Feature Cleaning")

st.subheader("Step 1: Remove High-Missing Columns")

missing_threshold = st.slider(
    "Missing Data Threshold (%)",
    min_value=0,
    max_value=100,
    value=35,
    step=5,
    help="Columns with missing data above this threshold will be removed"
)

# Calculate missing percentages
missing_pct = (df_filtered.isnull().sum() / len(df_filtered)) * 100
cols_high_missing = missing_pct[missing_pct > missing_threshold].index.tolist()

# Remove target and quality from removal list
protected_cols = ['D49', 'Batch_Quality']
cols_high_missing = [col for col in cols_high_missing if col not in protected_cols]

st.info(f"Found {len(cols_high_missing)} columns with >{missing_threshold}% missing data")

if cols_high_missing:
    with st.expander("View columns to be removed"):
        missing_df = pd.DataFrame({
            'Column': cols_high_missing,
            'Missing %': [missing_pct[col] for col in cols_high_missing]
        }).sort_values('Missing %', ascending=False)
        st.dataframe(missing_df, use_container_width=True, hide_index=True)

# Apply missing data removal
df_after_missing = apply_column_removal(df_filtered, cols_high_missing)

col_m1, col_m2 = st.columns(2)
col_m1.metric("Columns Before", len(df_filtered.columns))
col_m2.metric("Columns After Missing Filter", len(df_after_missing.columns))

st.divider()

st.subheader("Step 2: Remove Time and Lot Number Columns")

st.write("""
**Time columns** (start/end timestamps) and **lot number columns** are removed as they:
- Don't directly contribute to predictive modeling
- May introduce temporal dependencies
- Add noise without interpretable physical meaning for the process
""")

# Get time and lot columns that exist in the dataset
time_cols = get_time_columns()
lot_cols = get_lot_number_columns()

time_cols_in_df = [col for col in time_cols if col in df_after_missing.columns]
lot_cols_in_df = [col for col in lot_cols if col in df_after_missing.columns]

cols_time_and_lot = sorted(set(time_cols_in_df + lot_cols_in_df))

st.info(f"Found {len(time_cols_in_df)} time columns and {len(lot_cols_in_df)} lot number columns")

if cols_time_and_lot:
    with st.expander("View time and lot columns to be removed"):
        removal_df = pd.DataFrame({
            'Column': cols_time_and_lot,
            'Type': ['Time' if col in time_cols else 'Lot Number' for col in cols_time_and_lot],
            'Description': [var_descriptions.get(col, 'N/A')[:80] for col in cols_time_and_lot]
        })
        st.dataframe(removal_df, use_container_width=True, hide_index=True)

    # Apply removal
    df_after_time_lot = apply_column_removal(df_after_missing, cols_time_and_lot)
else:
    cols_time_and_lot = []
    df_after_time_lot = df_after_missing.copy()
    st.info("No time or lot columns found to remove")

col_tl1, col_tl2 = st.columns(2)
col_tl1.metric("Columns After Missing Filter", len(df_after_missing.columns))
col_tl2.metric("Columns After Time/Lot Removal", len(df_after_time_lot.columns))

st.divider()

st.subheader("Step 3: Remove Low-Variance Columns")

variance_threshold = st.number_input(
    "Variance Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.10,
    step=0.01,
    format="%.4f",
    help="Numeric columns with variance below this threshold will be removed"
)

# Get numeric columns (exclude target, quality, and categorical)
categorical_cols = (
    get_equipment_columns() +
    get_manual_categorical_columns()
)
categorical_cols = [col for col in categorical_cols if col in df_after_time_lot.columns]

numeric_cols = [
    col for col in df_after_time_lot.select_dtypes(include=[np.number]).columns
    if col not in protected_cols + categorical_cols
]

# Calculate variance
if numeric_cols:
    variance_series = df_after_time_lot[numeric_cols].var()
    low_variance_cols = variance_series[variance_series < variance_threshold].index.tolist()

    st.info(f"Found {len(low_variance_cols)} low-variance columns (< {variance_threshold})")

    if low_variance_cols:
        with st.expander("View low-variance columns"):
            var_df = pd.DataFrame({
                'Column': low_variance_cols,
                'Variance': [variance_series[col] for col in low_variance_cols]
            }).sort_values('Variance')
            st.dataframe(var_df, use_container_width=True, hide_index=True)

    # Apply variance removal
    df_after_variance = apply_column_removal(df_after_time_lot, low_variance_cols)
else:
    low_variance_cols = []
    df_after_variance = df_after_time_lot.copy()
    st.info("No numeric columns to filter by variance")

col_v1, col_v2 = st.columns(2)
col_v1.metric("Columns After Time/Lot Removal", len(df_after_time_lot.columns))
col_v2.metric("Columns After Variance Filter", len(df_after_variance.columns))

st.divider()

st.subheader("Step 4: Remove Highly-Correlated Columns")

correlation_threshold = st.slider(
    "Correlation Threshold",
    min_value=0.5,
    max_value=1.0,
    value=0.90,
    step=0.05,
    help="For pairs with |correlation| above this, keep the one with fewer missing values"
)

# Get numeric columns after variance filter
numeric_after_var = [
    col for col in df_after_variance.select_dtypes(include=[np.number]).columns
    if col not in protected_cols + categorical_cols
]

cols_to_drop_corr = []

if len(numeric_after_var) > 1:
    corr_matrix = df_after_variance[numeric_after_var].corr(method='pearson', min_periods=1)

    # Find high-correlation pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col_a = corr_matrix.columns[i]
            col_b = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]

            if abs(corr_value) > correlation_threshold:
                missing_a = df_after_variance[col_a].isnull().sum()
                missing_b = df_after_variance[col_b].isnull().sum()

                # Drop the one with more missing values
                if missing_a > missing_b:
                    drop_col = col_a
                    keep_col = col_b
                elif missing_b > missing_a:
                    drop_col = col_b
                    keep_col = col_a
                else:
                    # Tie-breaker: alphabetical
                    drop_col = col_a if col_a > col_b else col_b
                    keep_col = col_b if col_a > col_b else col_a

                high_corr_pairs.append({
                    'Feature A': col_a,
                    'Feature B': col_b,
                    'Correlation': corr_value,
                    'Keep': keep_col,
                    'Drop': drop_col
                })

    if high_corr_pairs:
        cols_to_drop_corr = sorted(set([pair['Drop'] for pair in high_corr_pairs]))

        st.warning(f"Found {len(high_corr_pairs)} high-correlation pairs (|r| > {correlation_threshold})")

        with st.expander("View high-correlation pairs"):
            corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', key=lambda x: x.abs(), ascending=False)
            st.dataframe(corr_df, use_container_width=True, hide_index=True)

        st.info(f"Will remove {len(cols_to_drop_corr)} features based on higher missingness")
    else:
        st.success(f"No feature pairs with |correlation| > {correlation_threshold}")
else:
    st.info("Need at least 2 numeric features to compute correlation")

# Apply correlation removal
df_after_correlation = apply_column_removal(df_after_variance, cols_to_drop_corr)

col_c1, col_c2 = st.columns(2)
col_c1.metric("Columns After Variance Filter", len(df_after_variance.columns))
col_c2.metric("Columns After Correlation Filter", len(df_after_correlation.columns))

st.divider()

# Summary of cleaning
st.subheader("Cleaning Summary")

summary_col1, summary_col2, summary_col3, summary_col4, summary_col5 = st.columns(5)
summary_col1.metric("Original Columns", len(df_filtered.columns))
summary_col2.metric("Removed (Missing)", len(cols_high_missing))
summary_col3.metric("Removed (Time/Lot)", len(cols_time_and_lot))
summary_col4.metric("Removed (Variance)", len(low_variance_cols))
summary_col5.metric("Removed (Correlation)", len(cols_to_drop_corr))

total_removed = len(cols_high_missing) + len(cols_time_and_lot) + len(low_variance_cols) + len(cols_to_drop_corr)
st.metric("Total Columns Removed", total_removed)
st.metric("Final Column Count", len(df_after_correlation.columns))

st.success("Automatic feature cleaning complete!")

# Preview cleaned dataset
with st.expander("Preview Cleaned Dataset"):
    st.dataframe(df_after_correlation.head(20), use_container_width=True)

st.divider()

# ============================================================================
# PHASE 4: DATA PREPARATION FOR MODELING
# ============================================================================
st.header("4. Data Preparation")

# Check if target exists
if 'D49' not in df_after_correlation.columns:
    st.error("Target column D49 not found in cleaned dataset")
    st.stop()

# Remove rows with missing target
df_model = df_after_correlation[df_after_correlation['D49'].notna()].copy()

if len(df_model) < 10:
    st.error(f"Not enough rows with valid target values ({len(df_model)} rows)")
    st.stop()

st.info(f"Using {len(df_model)} batches with valid D49 values")

# Save D48 for financial calculations before dropping (to avoid leakage in modeling)
if 'D48' in df_model.columns:
    d48_saved = df_model['D48'].copy()
    df_model = df_model.drop(columns=['D48'])
    st.caption("Removed D48 from features (potential data leakage, but saved for financial analysis)")
else:
    d48_saved = None

# Separate features and target
# Use all columns before D49 as features (except Batch_Quality)
d49_position = df_model.columns.get_loc('D49')
feature_cols = [col for col in df_model.columns[:d49_position] if col != 'Batch_Quality']

X_raw = df_model[feature_cols].copy()
y = df_model['D49'].copy()

st.subheader("EBM-Native Categorical Handling")

st.write("""
**EBM handles categorical variables natively** — no one-hot encoding needed.
Categorical columns are converted to string dtype and EBM detects them automatically.
""")

# Identify categorical columns in features using domain knowledge
# Combine dtype detection with domain-knowledge list (selected_categorical_cols)
known_categorical_cols = set(categorical_cols)
categorical_in_features = [
    col for col in X_raw.columns
    if X_raw[col].dtype == object or col in known_categorical_cols
]
numeric_in_features = [col for col in X_raw.columns if col not in categorical_in_features]

st.info(f"Found {len(categorical_in_features)} categorical and {len(numeric_in_features)} numeric features")

with st.expander("View categorical columns"):
    if categorical_in_features:
        cat_df = pd.DataFrame({
            'Column': categorical_in_features,
            'Description': [var_descriptions.get(col, 'N/A')[:80] for col in categorical_in_features],
            'Unique Values': [X_raw[col].nunique() for col in categorical_in_features]
        })
        st.dataframe(cat_df, use_container_width=True, hide_index=True)
    else:
        st.write("No categorical columns")

# Prepare data for EBM (no encoding, just proper dtype handling)
X_prepared = X_raw.copy()

# Convert categorical columns to string dtype (so EBM recognizes them)
for col in categorical_in_features:
    X_prepared[col] = X_prepared[col].fillna('Unknown').astype(str)

# Fill missing values in numeric columns with mean
for col in numeric_in_features:
    X_prepared[col] = X_prepared[col].fillna(X_prepared[col].mean())

col_enc1, col_enc2, col_enc3 = st.columns(3)
col_enc1.metric("Total Features", X_prepared.shape[1])
col_enc2.metric("Categorical Features", len(categorical_in_features))
col_enc3.metric("Numeric Features", len(numeric_in_features))

st.subheader("Train-Test Split")

test_size = st.slider(
    "Test Size",
    min_value=0.1,
    max_value=0.4,
    value=0.25,
    step=0.05
)

X_train, X_test, y_train, y_test = train_test_split(
    X_prepared, y, test_size=test_size, random_state=42
)

col_split1, col_split2 = st.columns(2)
col_split1.metric("Training Samples", len(X_train))
col_split2.metric("Test Samples", len(X_test))

st.success("Data preparation complete!")

st.divider()

# ============================================================================
# PHASE 5: FILTER-THEN-INTERACT EBM
# ============================================================================
st.header("5. EBM Filter-Then-Interact Pipeline")

st.markdown("""
**Three-phase approach:**
1. **Phase 1**: Train EBM with `interactions=0` to rank all features by main effects
2. **Phase 2**: Prune to top-N most important features
3. **Phase 3**: Re-train EBM on pruned features with interaction detection enabled
""")

st.subheader("Configuration")

st.write("**EBM Hyperparameters** (from notebook configuration)")

col_config1, col_config2, col_config3 = st.columns(3)

with col_config1:
    top_n_features = st.slider(
        "Top-N Features to Keep (Phase 2)",
        min_value=5,
        max_value=min(30, X_prepared.shape[1]),
        value=min(15, X_prepared.shape[1]),
        help="Number of top-ranked features to keep after Phase 1"
    )

with col_config2:
    n_interactions = st.slider(
        "Max Interaction Pairs (Phase 3)",
        min_value=0,
        max_value=20,
        value=10,
        help="Maximum number of pairwise interactions to detect in Phase 3"
    )

with col_config3:
    learning_rate = st.number_input(
        "Learning Rate",
        min_value=0.001,
        max_value=0.5,
        value=0.05,
        step=0.01,
        format="%.3f",
        help="Step size for gradient boosting"
    )

col_config4, col_config5, col_config6 = st.columns(3)

with col_config4:
    max_bins = st.number_input(
        "Max Bins",
        min_value=16,
        max_value=512,
        value=256,
        step=16,
        help="Maximum number of bins for continuous features"
    )

with col_config5:
    max_rounds = st.number_input(
        "Max Rounds",
        min_value=1000,
        max_value=10000,
        value=5000,
        step=500,
        help="Maximum boosting rounds"
    )

with col_config6:
    min_samples_leaf = st.number_input(
        "Min Samples Leaf",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
        help="Minimum samples required in a leaf node"
    )

col_config7, col_config8 = st.columns(2)

with col_config7:
    n_jobs = st.number_input(
        "Number of Jobs",
        min_value=1,
        max_value=16,
        value=1,
        step=1,
        help="Number of parallel jobs (use 1 to avoid pickling errors in Streamlit)"
    )

with col_config8:
    random_state = st.number_input(
        "Random State",
        value=42,
        step=1,
        help="Seed for reproducibility"
    )

# Train button
train_clicked = st.button("Train EBM Pipeline", type="primary", key="train_ebm")

if train_clicked:
    with st.spinner("Training EBM models..."):

        # ====================================================================
        # PHASE 1: Feature Ranking (interactions=0)
        # ====================================================================
        st.subheader("Phase 1: Feature Ranking (interactions=0)")

        st.write(f"**Input:** {X_train.shape[0]} training samples × {X_train.shape[1]} features "
                f"({len(categorical_in_features)} categorical, {len(numeric_in_features)} numeric)")

        ebm_phase1 = ExplainableBoostingRegressor(
            interactions=0,
            learning_rate=learning_rate,
            max_bins=max_bins,
            max_rounds=max_rounds,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state
        )

        ebm_phase1.fit(X_train, y_train)
        y_pred_phase1 = ebm_phase1.predict(X_test)

        mae_phase1 = mean_absolute_error(y_test, y_pred_phase1)
        rmse_phase1 = np.sqrt(mean_squared_error(y_test, y_pred_phase1))
        r2_phase1 = r2_score(y_test, y_pred_phase1)

        st.write("**Phase 1 Metrics (All Features, No Interactions)**")
        col_p1_1, col_p1_2, col_p1_3 = st.columns(3)
        col_p1_1.metric("MAE", f"{mae_phase1:.4f}")
        col_p1_2.metric("RMSE", f"{rmse_phase1:.4f}")
        col_p1_3.metric("R²", f"{r2_phase1:.4f}")

        # Extract feature importance
        try:
            global_exp_phase1 = ebm_phase1.explain_global(name='Phase 1 Global')
            global_data_phase1 = global_exp_phase1.data()

            importance_phase1 = pd.DataFrame({
                'Feature': global_data_phase1.get('names', []),
                'Importance': global_data_phase1.get('scores', [])
            }).sort_values('Importance', ascending=False)

            # Add feature descriptions
            importance_phase1['Description'] = importance_phase1['Feature'].apply(
                lambda f: var_descriptions.get(f, var_descriptions.get(f.split('_')[0], 'N/A'))[:80]
            )

            st.write("**Feature Importance Ranking (Top 20)**")
            st.dataframe(
                importance_phase1.head(20)[['Feature', 'Description', 'Importance']],
                use_container_width=True,
                hide_index=True
            )

            # Importance bar chart
            fig_imp_phase1 = px.bar(
                importance_phase1.head(20),
                x='Importance',
                y='Feature',
                orientation='h',
                title='Phase 1: Feature Importance (Top 20)'
            )
            fig_imp_phase1.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
            st.plotly_chart(fig_imp_phase1, use_container_width=True)

            # Elbow detection
            importances_sorted = importance_phase1['Importance'].values
            if len(importances_sorted) > 1:
                diffs = np.abs(np.diff(importances_sorted))
                elbow_rank = int(np.argmax(diffs)) + 1

                st.info(f"Elbow criterion suggests keeping top **{elbow_rank}** features (largest drop in importance)")
                st.caption(f"Current selection: Top **{top_n_features}** features")

        except Exception as e:
            st.error(f"Failed to extract Phase 1 importance: {str(e)}")
            importance_phase1 = pd.DataFrame()

        st.divider()

        # ====================================================================
        # PHASE 2: Feature Pruning
        # ====================================================================
        st.subheader(f"Phase 2: Feature Pruning (Top-{top_n_features})")

        if not importance_phase1.empty:
            top_features = importance_phase1.head(top_n_features)['Feature'].tolist()

            st.write(f"**Selected Top-{top_n_features} Features:**")
            st.dataframe(
                importance_phase1.head(top_n_features)[['Feature', 'Description', 'Importance']],
                use_container_width=True,
                hide_index=True
            )

            # Filter datasets to top features
            X_train_pruned = X_train[top_features].copy()
            X_test_pruned = X_test[top_features].copy()

            st.success(f"Pruned from {X_train.shape[1]} to {len(top_features)} features")

        else:
            st.error("Cannot prune features without Phase 1 importance data")
            st.stop()

        st.divider()

        # ====================================================================
        # PHASE 3: Interaction Hunting
        # ====================================================================
        st.subheader(f"Phase 3: Interaction Hunting ({n_interactions} pairs)")

        ebm_phase3 = ExplainableBoostingRegressor(
            interactions=n_interactions,
            learning_rate=learning_rate,
            max_bins=max_bins,
            max_rounds=max_rounds,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state
        )

        ebm_phase3.fit(X_train_pruned, y_train)
        y_pred_phase3 = ebm_phase3.predict(X_test_pruned)

        mae_phase3 = mean_absolute_error(y_test, y_pred_phase3)
        rmse_phase3 = np.sqrt(mean_squared_error(y_test, y_pred_phase3))
        r2_phase3 = r2_score(y_test, y_pred_phase3)

        st.write(f"**Phase 3 Metrics (Top-{top_n_features} Features + {n_interactions} Interactions)**")
        col_p3_1, col_p3_2, col_p3_3 = st.columns(3)
        col_p3_1.metric("MAE", f"{mae_phase3:.4f}")
        col_p3_2.metric("RMSE", f"{rmse_phase3:.4f}")
        col_p3_3.metric("R²", f"{r2_phase3:.4f}")

        # Extract Phase 3 importance
        try:
            global_exp_phase3 = ebm_phase3.explain_global(name='Phase 3 Global')
            global_data_phase3 = global_exp_phase3.data()

            importance_phase3 = pd.DataFrame({
                'Feature': global_data_phase3.get('names', []),
                'Importance': global_data_phase3.get('scores', [])
            }).sort_values('Importance', ascending=False)

            st.write("**Feature + Interaction Importance (Top 20)**")
            st.dataframe(
                importance_phase3.head(20),
                use_container_width=True,
                hide_index=True
            )

            # Importance bar chart
            fig_imp_phase3 = px.bar(
                importance_phase3.head(20),
                x='Importance',
                y='Feature',
                orientation='h',
                title='Phase 3: Feature + Interaction Importance (Top 20)'
            )
            fig_imp_phase3.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
            st.plotly_chart(fig_imp_phase3, use_container_width=True)

        except Exception as e:
            st.error(f"Failed to extract Phase 3 importance: {str(e)}")
            importance_phase3 = pd.DataFrame()

        st.divider()

        # ====================================================================
        # COMPARISON
        # ====================================================================
        st.subheader("Phase Comparison")

        comparison_df = pd.DataFrame({
            'Phase': ['Phase 1 (All Features, No Interactions)',
                     f'Phase 3 (Top-{top_n_features} + {n_interactions} Interactions)'],
            'Total Features': [X_train.shape[1], len(top_features)],
            'Categorical': [len(categorical_in_features), len([f for f in top_features if f in categorical_in_features])],
            'Numeric': [len(numeric_in_features), len([f for f in top_features if f in numeric_in_features])],
            'MAE': [mae_phase1, mae_phase3],
            'RMSE': [rmse_phase1, rmse_phase3],
            'R²': [r2_phase1, r2_phase3]
        })

        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        # Improvement metrics
        mae_improvement = ((mae_phase1 - mae_phase3) / mae_phase1) * 100
        r2_improvement = ((r2_phase3 - r2_phase1) / abs(r2_phase1)) * 100 if r2_phase1 != 0 else 0

        col_imp1, col_imp2 = st.columns(2)
        col_imp1.metric("MAE Improvement", f"{mae_improvement:.2f}%")
        col_imp2.metric("R² Improvement", f"{r2_improvement:.2f}%")

        # Actual vs Predicted plots
        st.subheader("Predictions: Actual vs Predicted")

        col_plot1, col_plot2 = st.columns(2)

        with col_plot1:
            pred_df_phase1 = pd.DataFrame({
                'Actual': y_test.values,
                'Predicted': y_pred_phase1
            })

            fig_phase1 = px.scatter(
                pred_df_phase1,
                x='Actual',
                y='Predicted',
                title='Phase 1: Actual vs Predicted'
            )
            axis_min = min(pred_df_phase1['Actual'].min(), pred_df_phase1['Predicted'].min())
            axis_max = max(pred_df_phase1['Actual'].max(), pred_df_phase1['Predicted'].max())
            fig_phase1.add_trace(
                go.Scatter(
                    x=[axis_min, axis_max],
                    y=[axis_min, axis_max],
                    mode='lines',
                    name='Ideal',
                    line=dict(color='red', dash='dash')
                )
            )
            st.plotly_chart(fig_phase1, use_container_width=True)

        with col_plot2:
            pred_df_phase3 = pd.DataFrame({
                'Actual': y_test.values,
                'Predicted': y_pred_phase3
            })

            fig_phase3 = px.scatter(
                pred_df_phase3,
                x='Actual',
                y='Predicted',
                title='Phase 3: Actual vs Predicted'
            )
            fig_phase3.add_trace(
                go.Scatter(
                    x=[axis_min, axis_max],
                    y=[axis_min, axis_max],
                    mode='lines',
                    name='Ideal',
                    line=dict(color='red', dash='dash')
                )
            )
            st.plotly_chart(fig_phase3, use_container_width=True)

        st.success("EBM Filter-Then-Interact pipeline complete!")

        # Store all results in session state for persistence across reruns
        st.session_state.ebm_trained = True
        st.session_state.ebm_phase1 = ebm_phase1
        st.session_state.ebm_phase3 = ebm_phase3
        st.session_state.top_features = top_features
        st.session_state.mae_phase1 = mae_phase1
        st.session_state.rmse_phase1 = rmse_phase1
        st.session_state.r2_phase1 = r2_phase1
        st.session_state.mae_phase3 = mae_phase3
        st.session_state.rmse_phase3 = rmse_phase3
        st.session_state.r2_phase3 = r2_phase3
        st.session_state.importance_phase1 = importance_phase1
        st.session_state.importance_phase3 = importance_phase3
        st.session_state.y_pred_phase1 = y_pred_phase1
        st.session_state.y_pred_phase3 = y_pred_phase3
        st.session_state.comparison_df = comparison_df
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.categorical_in_features = categorical_in_features
        st.session_state.numeric_in_features = numeric_in_features

# Check if EBM has been trained (either just now or in a previous run)
if st.session_state.get('ebm_trained', False):

    # Retrieve results from session state
    ebm_phase1 = st.session_state.ebm_phase1
    ebm_phase3 = st.session_state.ebm_phase3
    top_features = st.session_state.top_features
    mae_phase1 = st.session_state.mae_phase1
    rmse_phase1 = st.session_state.rmse_phase1
    r2_phase1 = st.session_state.r2_phase1
    mae_phase3 = st.session_state.mae_phase3
    rmse_phase3 = st.session_state.rmse_phase3
    r2_phase3 = st.session_state.r2_phase3
    importance_phase1 = st.session_state.importance_phase1
    importance_phase3 = st.session_state.importance_phase3
    y_pred_phase1 = st.session_state.y_pred_phase1
    y_pred_phase3 = st.session_state.y_pred_phase3
    comparison_df = st.session_state.comparison_df
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    categorical_in_features = st.session_state.categorical_in_features
    numeric_in_features = st.session_state.numeric_in_features

    st.divider()

    # ====================================================================
    # SHAPE FUNCTIONS (Feature Effect Curves)
    # ====================================================================
    st.subheader("Shape Functions: Feature Effect on D49 Yield")

    st.write("""
    **Shape functions** show the relationship between each feature's value and its impact on D49 yield.
    These curves reveal the optimal operating ranges for each parameter.
    """)

    try:
        # Extract shape functions from Phase 3 model
        global_exp_shapes = ebm_phase3.explain_global(name='Shape Functions')
        global_data_shapes = global_exp_shapes.data()

        feature_names_shapes = global_data_shapes.get('names', [])
        feature_scores_shapes = global_data_shapes.get('scores', [])

        # Build shape function map
        shape_function_map = {}
        for idx, feature_name in enumerate(feature_names_shapes):
            try:
                feature_data = global_exp_shapes.data(idx)
                shape_function_map[feature_name] = {
                    'x': list(feature_data.get('names', [])),
                    'y': list(feature_data.get('scores', []))
                }
            except Exception:
                continue

        if shape_function_map:
            # Sort features by importance for selector
            importance_sorted_shapes = sorted(
                zip(feature_names_shapes, feature_scores_shapes),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            shape_feature_options = [f[0] for f in importance_sorted_shapes]

            # Feature selector
            selected_shape_feature = st.selectbox(
                "Select a feature to inspect its shape function:",
                options=shape_feature_options,
                help="Shows how different values of this feature affect D49 prediction"
            )

            if selected_shape_feature in shape_function_map:
                shape_data = shape_function_map[selected_shape_feature]
                shape_x = shape_data['x']
                shape_y = shape_data['y']

                # Ensure equal length by truncating to minimum
                min_len = min(len(shape_x), len(shape_y))
                if min_len == 0:
                    st.warning(f"No data available for {selected_shape_feature}")
                else:
                    shape_x = shape_x[:min_len]
                    shape_y = shape_y[:min_len]

                    # Get feature description
                    feature_desc = var_descriptions.get(
                        selected_shape_feature,
                        var_descriptions.get(selected_shape_feature.split('_')[0], 'N/A')
                    )

                    # Check if feature is in categorical list (domain knowledge)
                    # Strip interaction suffix if present (e.g., "D1 x D24" -> "D1")
                    base_feature_name = selected_shape_feature.split(' x ')[0] if ' x ' in selected_shape_feature else selected_shape_feature
                    is_categorical = base_feature_name in categorical_in_features

                    # Also check if values are numeric
                    shape_x_numeric = pd.to_numeric(pd.Series(shape_x), errors='coerce')

                    if not is_categorical and shape_x_numeric.notna().all() and len(shape_x_numeric) > 0:
                        # Numeric feature - create line plot
                        shape_df = pd.DataFrame({
                            'Feature Value': shape_x_numeric.astype(float),
                            'Impact on D49': shape_y
                        }).sort_values('Feature Value')

                        # Find optimal range
                        max_impact_idx = shape_df['Impact on D49'].idxmax()
                        min_impact_idx = shape_df['Impact on D49'].idxmin()
                        optimal_value = shape_df.loc[max_impact_idx, 'Feature Value']
                        optimal_impact = shape_df.loc[max_impact_idx, 'Impact on D49']

                        fig_shape = px.line(
                            shape_df,
                            x='Feature Value',
                            y='Impact on D49',
                            markers=True,
                            title=f'Shape Function: {selected_shape_feature}'
                        )

                        # Add horizontal line at y=0
                        fig_shape.add_hline(
                            y=0,
                            line_dash="dash",
                            line_color="gray",
                            annotation_text="Neutral Impact"
                        )

                        # Highlight optimal point
                        fig_shape.add_scatter(
                            x=[optimal_value],
                            y=[optimal_impact],
                            mode='markers',
                            marker=dict(size=15, color='red', symbol='star'),
                            name='Optimal Point',
                            showlegend=True
                        )

                        fig_shape.update_layout(height=500)
                        st.plotly_chart(fig_shape, use_container_width=True)

                        # Show insights
                        col_shape1, col_shape2, col_shape3 = st.columns(3)
                        col_shape1.metric("Optimal Value", f"{optimal_value:.2f}")
                        col_shape2.metric("Peak Impact", f"{optimal_impact:.3f}")

                        # Identify safe zone (positive impact region)
                        safe_zone = shape_df[shape_df['Impact on D49'] > 0]
                        if len(safe_zone) > 0:
                            safe_min = safe_zone['Feature Value'].min()
                            safe_max = safe_zone['Feature Value'].max()
                            col_shape3.metric("Safe Zone Range", f"{safe_min:.1f} - {safe_max:.1f}")
                        else:
                            col_shape3.metric("Safe Zone Range", "None positive")

                        st.success(f"""
                        **Interpretation:**
                        - **{feature_desc}**
                        - Peak performance occurs around **{optimal_value:.2f}**
                        - Values in the Safe Zone have positive impact on yield
                        - Keep this parameter in the highlighted optimal range for best results
                        """)

                    else:
                        # Categorical feature - create bar plot
                        shape_df = pd.DataFrame({
                            'Category': [str(x) for x in shape_x],
                            'Impact on D49': shape_y
                        })

                        # Sort by impact
                        shape_df = shape_df.sort_values('Impact on D49', ascending=True)

                        # Color by positive/negative
                        colors = ['#2ca02c' if imp > 0 else '#d62728' for imp in shape_df['Impact on D49']]

                        fig_shape = go.Figure(data=[
                            go.Bar(
                                x=shape_df['Impact on D49'],
                                y=shape_df['Category'],
                                orientation='h',
                                marker=dict(color=colors),
                                text=[f"{imp:+.3f}" for imp in shape_df['Impact on D49']],
                                textposition='auto'
                            )
                        ])

                        fig_shape.update_layout(
                            title=f'Shape Function: {selected_shape_feature} (Categorical)',
                            xaxis_title='Impact on D49',
                            yaxis_title='Category',
                            height=max(400, len(shape_df) * 40)
                        )

                        st.plotly_chart(fig_shape, use_container_width=True)

                        # Show best category
                        best_idx = shape_df['Impact on D49'].idxmax()
                        best_category = shape_df.loc[best_idx, 'Category']
                        best_impact = shape_df.loc[best_idx, 'Impact on D49']

                        worst_idx = shape_df['Impact on D49'].idxmin()
                        worst_category = shape_df.loc[worst_idx, 'Category']
                        worst_impact = shape_df.loc[worst_idx, 'Impact on D49']

                        col_cat1, col_cat2 = st.columns(2)
                        col_cat1.metric("Best Choice", f"{best_category} (+{best_impact:.3f})")
                        col_cat2.metric("Worst Choice", f"{worst_category} ({worst_impact:.3f})")

                        st.success(f"""
                        **Interpretation:**
                        - **{feature_desc}**
                        - Best category: **{best_category}** (green bar)
                        - Worst category: **{worst_category}** (red bar)
                        - Choose the category with the highest positive impact
                        """)

        else:
            st.warning("No shape functions available")

    except Exception as e:
        st.error(f"Failed to generate shape functions: {str(e)}")

    st.divider()

    # ====================================================================
    # LOCAL EXPLANATIONS (Individual Batch Predictions)
    # ====================================================================
    st.subheader("Local Explanations: Individual Batch Analysis")

    st.write("""
    **Waterfall charts** show how each feature contributes to the prediction for a specific batch.
    Select a batch to see which features drove its D49 prediction up or down.
    """)

    # Get predictions for all batches (full dataset)
    X_train_pruned_full = X_train[top_features].copy()
    X_test_pruned_full = X_test[top_features].copy()
    X_all_pruned = pd.concat([X_train_pruned_full, X_test_pruned_full])
    y_all = pd.concat([y_train, y_test])

    y_pred_all = ebm_phase3.predict(X_all_pruned)

    # Get batch quality for filtering
    if 'Batch_Quality' in df_model.columns:
        batch_quality_all = df_model.loc[X_all_pruned.index, 'Batch_Quality'].fillna('Unknown').astype(str).values
    else:
        batch_quality_all = np.array(['Unknown'] * len(X_all_pruned))

    # Create batch selector dataframe
    batch_selector_df = pd.DataFrame({
        'Batch': [f"Batch {idx}" for idx in X_all_pruned.index],
        'Batch Index': list(X_all_pruned.index),
        'Batch Quality': batch_quality_all,
        'Actual D49': y_all.values,
        'Predicted D49': y_pred_all
    }).reset_index(drop=True)

    batch_selector_df['Residual'] = batch_selector_df['Actual D49'] - batch_selector_df['Predicted D49']
    batch_selector_df['Abs Residual'] = batch_selector_df['Residual'].abs()

    # Quality filter
    quality_options = sorted(batch_selector_df['Batch Quality'].unique().tolist())

    st.write("**Filter batches by quality:**")
    selected_quality_filter = st.multiselect(
        "Batch Quality Filter",
        options=quality_options,
        default=quality_options,
        help="Select one or more quality levels to view"
    )

    # Filter batches by selected quality
    if selected_quality_filter:
        filtered_batch_df = batch_selector_df[
            batch_selector_df['Batch Quality'].isin(selected_quality_filter)
        ]
    else:
        filtered_batch_df = batch_selector_df.iloc[0:0]

    if filtered_batch_df.empty:
        st.warning("No batches match the selected quality filter.")
    else:
        col_batch1, col_batch2, col_batch3 = st.columns(3)
        col_batch1.metric("Batches in Selection", len(filtered_batch_df))
        col_batch2.metric("Average |Residual|", f"{filtered_batch_df['Abs Residual'].mean():.4f}")
        col_batch3.metric("Max |Residual|", f"{filtered_batch_df['Abs Residual'].max():.4f}")

        # Batch selector
        batch_options = filtered_batch_df['Batch'].tolist()
        selected_batch_label = st.selectbox(
            "Select a batch to explain:",
            options=batch_options,
            help="Choose a batch to see its local explanation waterfall"
        )

        selected_batch_row = filtered_batch_df[
            filtered_batch_df['Batch'] == selected_batch_label
        ].iloc[0]
        selected_batch_index = selected_batch_row['Batch Index']

        # Get single batch data
        single_batch_X = X_all_pruned.loc[[selected_batch_index]].copy()

        # Generate local explanation
        try:
            local_explanation = ebm_phase3.explain_local(
                single_batch_X,
                name=f"Local explanation for {selected_batch_label}"
            )
            local_data = local_explanation.data(0)

            # Extract base value and contributions
            base_value = float(local_data.get('extra', {}).get('scores', [0.0])[0])
            contribution_scores = [float(v) for v in local_data.get('scores', [])]
            contribution_feature_names = local_data.get('names', list(single_batch_X.columns))

            predicted_from_model = float(ebm_phase3.predict(single_batch_X)[0])
            actual_value = float(selected_batch_row['Actual D49'])

            # Create horizontal bar chart for contributions
            # Build feature labels with values
            feature_labels = []
            feature_contributions = []

            # Add intercept/baseline first
            feature_labels.append('Intercept')
            feature_contributions.append(base_value)

            # Add individual features
            for i, feature_name in enumerate(contribution_feature_names):
                feature_value = single_batch_X.iloc[0].get(feature_name, np.nan)
                if isinstance(feature_value, (int, float, np.integer, np.floating)) and pd.notna(feature_value):
                    value_str = f"({feature_value:.2f})"
                else:
                    value_str = f"({feature_value})"

                feature_labels.append(f"{feature_name} {value_str}")
                feature_contributions.append(contribution_scores[i])

            # Reverse order for plotting (top to bottom)
            feature_labels = feature_labels[::-1]
            feature_contributions = feature_contributions[::-1]

            # Determine colors: blue for negative (left), orange for positive (right)
            colors = ['#1f77b4' if contrib < 0 else '#ff7f0e' for contrib in feature_contributions]
            # Make intercept gray
            colors[0] = '#7f7f7f'

            # Create horizontal bar chart
            fig_local = go.Figure()

            fig_local.add_trace(go.Bar(
                x=feature_contributions,
                y=feature_labels,
                orientation='h',
                marker=dict(color=colors),
                text=[f"{contrib:+.2f}" for contrib in feature_contributions],
                textposition='auto',
                showlegend=False
            ))

            fig_local.update_layout(
                title=f"Local Explanation (Actual: {actual_value:.1f} | Predicted: {predicted_from_model:.1f})",
                xaxis_title='Contribution to Prediction',
                yaxis_title='',
                height=max(400, len(feature_labels) * 35),
                xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='lightgray'),
                bargap=0.15
            )

            st.plotly_chart(fig_local, use_container_width=True)

            # Show metrics
            wf_m1, wf_m2, wf_m3, wf_m4 = st.columns(4)
            wf_m1.metric("Actual D49", f"{selected_batch_row['Actual D49']:.4f}")
            wf_m2.metric("Predicted D49", f"{predicted_from_model:.4f}")
            wf_m3.metric("Residual", f"{selected_batch_row['Residual']:.4f}")
            wf_m4.metric("Model Baseline", f"{base_value:.4f}")

            # Batch explanation confidence alarm
            st.write("**Batch Explanation Confidence**")
            batch_abs_residual = float(selected_batch_row['Abs Residual'])
            batch_quality = str(selected_batch_row['Batch Quality'])
            quality_group_size = int((batch_selector_df['Batch Quality'] == batch_quality).sum())

            total_contrib_abs = float(np.sum(np.abs(contribution_scores)))
            max_contrib_abs = float(np.max(np.abs(contribution_scores))) if len(contribution_scores) > 0 else 0.0
            dominant_contrib_ratio = (max_contrib_abs / total_contrib_abs) if total_contrib_abs > 0 else 0.0

            # Simple alarm logic
            batch_alarm_level = 'good'
            batch_alarm_reasons = []

            if batch_abs_residual > 2 * mae_phase3:
                batch_alarm_level = 'high_risk'
                batch_alarm_reasons.append("Residual is > 2× model MAE (prediction unusually uncertain).")
            elif batch_abs_residual > mae_phase3:
                batch_alarm_level = 'caution'
                batch_alarm_reasons.append("Residual is above model MAE (moderate uncertainty).")

            if dominant_contrib_ratio > 0.70:
                batch_alarm_level = 'high_risk'
                batch_alarm_reasons.append("One variable dominates the explanation (>70% of total contribution).")
            elif dominant_contrib_ratio > 0.55 and batch_alarm_level != 'high_risk':
                batch_alarm_level = 'caution'
                batch_alarm_reasons.append("Explanation is concentrated in few variables (>55% dominance).")

            if quality_group_size < 5 and batch_alarm_level != 'high_risk':
                batch_alarm_level = 'caution'
                batch_alarm_reasons.append("Selected quality category has low sample support (<5 batches).")

            if batch_alarm_level == 'good':
                st.success("✅ Selected batch explanation confidence is GOOD.")
            elif batch_alarm_level == 'caution':
                st.warning("⚠️ Selected batch explanation confidence is MODERATE.")
            else:
                st.error("🚨 Selected batch explanation confidence is LOW.")

            ba1, ba2, ba3 = st.columns(3)
            ba1.metric("Batch |Residual|", f"{batch_abs_residual:.4f}")
            ba2.metric("Dominant Contribution Ratio", f"{dominant_contrib_ratio:.2f}")
            ba3.metric(f"Batches in '{batch_quality}'", quality_group_size)

            if batch_alarm_reasons:
                st.markdown("**Alarm reasons:**\n- " + "\n- ".join(batch_alarm_reasons))
            else:
                st.caption("No risk flags detected for this selected batch.")

            st.caption(
                "Horizontal bars show how each feature contributes to the prediction. "
                "Orange bars increase D49 (push right), blue bars decrease D49 (push left). "
                "The intercept (gray) is the model's baseline prediction."
            )

            # Financial Impact Analysis
            st.write("**💰 Financial Impact Analysis**")
            st.caption("Revenue calculation based on D48 (quantity in kg) and D49 (yield %)")

            # Show model uncertainty disclaimer
            with st.expander("⚠️ Model Uncertainty Disclaimer"):
                st.write(f"""
                **Model Performance Metrics:**
                - MAE (Mean Absolute Error): {mae_phase3:.2f} percentage points
                - R² Score: {r2_phase3:.2f}
                - Model explains {r2_phase3*100:.1f}% of yield variance

                **Implications for Financial Estimates:**
                - Predictions have ~±{mae_phase3:.2f}pp uncertainty on average
                - With avg yield {y_test.mean():.2f}% (std {y_test.std():.2f}pp), this represents ~{(mae_phase3/y_test.std())*100:.0f}% of typical variation
                - Financial gains shown are **estimates with significant uncertainty**
                - Conservative scenarios account for this by using 70% recovery and showing ranges

                **Recommendation:** Use these estimates for **directional guidance** and **prioritization**,
                not as precise financial guarantees. Validate improvements through controlled experiments.
                """)

            # Get D48 value for this batch (quantity in kg)
            if d48_saved is not None and selected_batch_index in d48_saved.index:
                d48_value = d48_saved.loc[selected_batch_index]

                if pd.notna(d48_value) and d48_value > 0:
                    # Price range
                    price_min = 50  # $/g
                    price_max = 60  # $/g
                    price_mid = 55  # $/g (midpoint)

                    # Current metrics
                    current_d49 = float(selected_batch_row['Actual D49'])
                    predicted_d49 = predicted_from_model
                    current_quantity_kg = float(d48_value)
                    current_quantity_g = current_quantity_kg * 1000

                    # Current revenue (at midpoint price)
                    current_revenue_mid = current_quantity_g * price_mid
                    current_revenue_min = current_quantity_g * price_min
                    current_revenue_max = current_quantity_g * price_max

                    st.write(f"**Current Batch Financials** (D48 = {current_quantity_kg:.2f} kg)")
                    col_fin1, col_fin2, col_fin3 = st.columns(3)
                    col_fin1.metric("Revenue @ $50/g", f"${current_revenue_min:,.0f}")
                    col_fin2.metric("Revenue @ $55/g", f"${current_revenue_mid:,.0f}")
                    col_fin3.metric("Revenue @ $60/g", f"${current_revenue_max:,.0f}")

                    # Check if there are negative contributors (optimization potential)
                    if contribution_scores and len(contribution_scores) > 0:
                        negative_sum_local = sum([score for score in contribution_scores if score < -0.1])

                        if negative_sum_local < -0.01:  # Has meaningful negative impact
                            # Calculate yield gain potential with uncertainty adjustment
                            # Use R² score to adjust confidence in optimization potential
                            r2_adjustment = max(0.3, r2_phase3)  # Floor at 30% to remain conservative
                            conservative_gain_local = -negative_sum_local * 0.7 * r2_adjustment
                            optimized_d49_pred = predicted_d49 + conservative_gain_local

                            # For "what-if" scenario: if this batch had been run with optimal parameters
                            # Assume the yield gain transfers from prediction space to actual space
                            yield_gain_points = optimized_d49_pred - predicted_d49

                            # Create confidence bounds using MAE
                            yield_gain_best = yield_gain_points + mae_phase3  # Optimistic
                            yield_gain_expected = yield_gain_points  # Expected
                            yield_gain_worst = max(0, yield_gain_points - mae_phase3)  # Pessimistic (floor at 0)

                            optimized_d49_scenario = current_d49 + yield_gain_expected

                            # Calculate scenarios with uncertainty bounds
                            def calc_scenario(yield_gain):
                                scenario_d49 = current_d49 + yield_gain
                                if current_d49 > 0:
                                    factor = scenario_d49 / current_d49
                                else:
                                    factor = 1.0
                                scenario_kg = current_quantity_kg * factor
                                scenario_g = scenario_kg * 1000
                                return {
                                    'd49': scenario_d49,
                                    'kg': scenario_kg,
                                    'revenue_min': scenario_g * price_min,
                                    'revenue_mid': scenario_g * price_mid,
                                    'revenue_max': scenario_g * price_max
                                }

                            best_case = calc_scenario(yield_gain_best)
                            expected_case = calc_scenario(yield_gain_expected)
                            worst_case = calc_scenario(yield_gain_worst)

                            # Revenue gains for expected case
                            revenue_gain_mid = expected_case['revenue_mid'] - current_revenue_mid
                            revenue_gain_min = expected_case['revenue_min'] - current_revenue_min
                            revenue_gain_max = expected_case['revenue_max'] - current_revenue_max

                            # Quantity gain range
                            quantity_gain_best = best_case['kg'] - current_quantity_kg
                            quantity_gain_expected = expected_case['kg'] - current_quantity_kg
                            quantity_gain_worst = worst_case['kg'] - current_quantity_kg

                            st.write("**What-If Scenario: If This Batch Had Been Optimized**")
                            st.caption(f"Expected yield gain: +{yield_gain_expected:.2f}pp (adjusted for R²={r2_phase3:.2f} and 70% recovery)")

                            # Show three scenarios
                            st.write("**Yield Gain Scenarios** (accounting for ±MAE uncertainty)")
                            col_scen1, col_scen2, col_scen3 = st.columns(3)
                            col_scen1.metric(
                                "Pessimistic",
                                f"+{yield_gain_worst:.2f}pp",
                                help=f"Expected gain - MAE ({mae_phase3:.2f})"
                            )
                            col_scen2.metric(
                                "Expected",
                                f"+{yield_gain_expected:.2f}pp",
                                help="Conservative estimate with R² and recovery adjustments"
                            )
                            col_scen3.metric(
                                "Optimistic",
                                f"+{yield_gain_best:.2f}pp",
                                help=f"Expected gain + MAE ({mae_phase3:.2f})"
                            )

                            # Expected scenario details
                            st.write("**Expected Scenario Impact**")
                            col_what_if1, col_what_if2 = st.columns(2)
                            col_what_if1.metric(
                                "D49 Yield",
                                f"{expected_case['d49']:.2f}%",
                                delta=f"+{yield_gain_expected:.2f}pp"
                            )
                            col_what_if2.metric(
                                "D48 Quantity",
                                f"{expected_case['kg']:.2f} kg",
                                delta=f"+{quantity_gain_expected:.2f} kg"
                            )

                            # Revenue ranges
                            st.write("**Revenue Gain Estimates** (@ $55/g, Expected Scenario)")
                            col_fin_opt1, col_fin_opt2 = st.columns(2)

                            # Calculate revenue gains for all scenarios at mid price
                            revenue_gain_worst_case = (worst_case['revenue_mid'] - current_revenue_mid)
                            revenue_gain_best_case = (best_case['revenue_mid'] - current_revenue_mid)

                            col_fin_opt1.metric(
                                "Expected Revenue Gain",
                                f"+${revenue_gain_mid:,.0f}"
                            )
                            col_fin_opt2.metric(
                                "Uncertainty Range",
                                f"${revenue_gain_worst_case:,.0f} to ${revenue_gain_best_case:,.0f}",
                                help="Range based on ±MAE uncertainty"
                            )

                            # Price sensitivity for expected scenario
                            st.write("**Price Sensitivity** (Expected Scenario)")
                            col_price1, col_price2, col_price3 = st.columns(3)
                            col_price1.metric("@ $50/g", f"+${revenue_gain_min:,.0f}")
                            col_price2.metric("@ $55/g", f"+${revenue_gain_mid:,.0f}")
                            col_price3.metric("@ $60/g", f"+${revenue_gain_max:,.0f}")

                            st.info(f"""
                            **Interpretation:**
                            - **Expected gain**: +{yield_gain_expected:.2f} percentage points → +${revenue_gain_mid:,.0f}
                            - **Uncertainty range**: ${revenue_gain_worst_case:,.0f} to ${revenue_gain_best_case:,.0f} (±{mae_phase3:.2f}pp)
                            - **Quantity gain**: +{quantity_gain_expected:.2f} kg (expected case)

                            These estimates account for model uncertainty (MAE={mae_phase3:.2f}, R²={r2_phase3:.2f})
                            and use conservative assumptions (70% × R² recovery rate = {0.7*r2_adjustment:.0%}).
                            """)
                        else:
                            st.info("✅ This batch is already operating optimally (no negative drivers to fix)")
                    else:
                        st.info("No optimization potential calculated for this batch")

                else:
                    st.warning("D48 value is missing or zero for this batch - cannot calculate revenue")
            else:
                st.warning("D48 data not available - cannot calculate revenue impact")

        except Exception as e:
            st.error(f"Failed to generate local explanation: {str(e)}")

    st.divider()

    # ====================================================================
    # LOCAL VIEW: ACTIONABLE CHECKLIST (Problem Solving)
    # ====================================================================
    st.subheader("Local View: Actionable Improvement Checklist")

    st.write("""
    **Action Plan** for the selected batch: identifies negative drivers and recommends specific adjustments
    based on the global optimal ranges to maximize yield.
    """)

    if not filtered_batch_df.empty and 'selected_batch_label' in locals():
        try:
            # Get the selected batch's contributions
            local_explanation = ebm_phase3.explain_local(single_batch_X, name=f"Local for {selected_batch_label}")
            local_data = local_explanation.data(0)

            base_value = float(local_data.get('extra', {}).get('scores', [0.0])[0])
            contribution_scores = [float(v) for v in local_data.get('scores', [])]
            contribution_feature_names = local_data.get('names', list(single_batch_X.columns))

            # Identify negative contributors (problems)
            negative_contributors = []
            for i, (fname, score) in enumerate(zip(contribution_feature_names, contribution_scores)):
                if score < -0.1:  # Threshold for "significant" negative impact
                    current_value = single_batch_X.iloc[0].get(fname, np.nan)
                    negative_contributors.append({
                        'feature': fname,
                        'contribution': score,
                        'current_value': current_value
                    })

            # Sort by magnitude of negative impact
            negative_contributors.sort(key=lambda x: x['contribution'])

            if negative_contributors:
                st.write(f"**🔴 Found {len(negative_contributors)} parameter(s) reducing yield:**")

                # For each negative contributor, recommend action
                action_plan = []

                for item in negative_contributors:
                    feature_name = item['feature']
                    current_val = item['current_value']
                    contribution = item['contribution']

                    # Get optimal range from global explanation
                    try:
                        global_exp = ebm_phase3.explain_global()
                        global_data = global_exp.data()
                        feature_names_global = global_data.get('names', [])

                        if feature_name in feature_names_global:
                            feature_idx = feature_names_global.index(feature_name)
                            feature_data = global_exp.data(feature_idx)

                            x_vals = feature_data.get('names', [])
                            y_vals = feature_data.get('scores', [])

                            # Find the optimal range (highest positive contribution)
                            if len(y_vals) > 0:
                                max_idx = np.argmax(y_vals)
                                optimal_range = str(x_vals[max_idx])
                                optimal_contribution = y_vals[max_idx]

                                action_plan.append({
                                    'Parameter': feature_name,
                                    'Current Value': str(current_val) if not isinstance(current_val, float) else f"{current_val:.2f}",
                                    'Current Impact': f"{contribution:.3f}",
                                    'Recommended Range': optimal_range,
                                    'Potential Gain': f"+{optimal_contribution:.3f}"
                                })
                    except Exception:
                        action_plan.append({
                            'Parameter': feature_name,
                            'Current Value': str(current_val) if not isinstance(current_val, float) else f"{current_val:.2f}",
                            'Current Impact': f"{contribution:.3f}",
                            'Recommended Range': "See heatmap above",
                            'Potential Gain': "N/A"
                        })

                if action_plan:
                    action_df = pd.DataFrame(action_plan)
                    st.dataframe(action_df, use_container_width=True, hide_index=True)

                    # Calculate optimized potential
                    current_prediction = predicted_from_model
                    negative_sum = sum(item['contribution'] for item in negative_contributors)

                    # Estimate: if we fix all negatives to neutral (0), how much gain?
                    conservative_gain = -negative_sum * 0.7  # 70% recovery (conservative)
                    optimized_potential = current_prediction + conservative_gain

                    st.write("**📊 Optimization Potential:**")
                    col_opt1, col_opt2, col_opt3 = st.columns(3)
                    col_opt1.metric("Current Prediction", f"{current_prediction:.2f}")
                    col_opt2.metric("Negative Impact", f"{negative_sum:.2f}", delta=f"{negative_sum:.2f}", delta_color="inverse")
                    col_opt3.metric("Optimized Potential", f"{optimized_potential:.2f}", delta=f"+{conservative_gain:.2f}")

                    st.success("""
                    ✅ **Action:** Adjust the parameters above to their recommended ranges.
                    This could improve D49 yield and move this batch toward optimal performance.
                    """)
                else:
                    st.info("Unable to generate specific recommendations. Refer to the global heatmap above.")

            else:
                st.success("✅ This batch has no significant negative drivers. It's operating in the optimal zone!")

        except Exception as e:
            st.error(f"Failed to generate actionable checklist: {str(e)}")
    else:
        st.info("Select a batch above to see its actionable improvement checklist.")

    st.divider()

    # ====================================================================
    # PORTFOLIO VIEW: AGGREGATE OPTIMIZATION POTENTIAL
    # ====================================================================
    st.subheader("Portfolio View: Aggregate Optimization Potential")

    st.write("""
    **What if we optimized all batches?** This analysis shows the aggregate improvement potential
    if we applied the model's recommendations across the entire dataset.
    """)

    try:
        # Analyze all batches for optimization potential
        all_batch_analysis = []

        for idx in X_all_pruned.index:
            single_batch = X_all_pruned.loc[[idx]].copy()

            try:
                # Get local explanation
                local_exp = ebm_phase3.explain_local(single_batch, name=f"Batch {idx}")
                local_data = local_exp.data(0)

                base_value = float(local_data.get('extra', {}).get('scores', [0.0])[0])
                contribution_scores = [float(v) for v in local_data.get('scores', [])]
                contribution_feature_names = local_data.get('names', list(single_batch.columns))

                # Get prediction and actual
                pred = float(ebm_phase3.predict(single_batch)[0])
                actual = float(y_all.loc[idx])

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
                    'negative_sum': negative_sum,
                    'conservative_gain': conservative_gain,
                    'optimized_potential': optimized_potential,
                    'num_negative_features': len(negative_features),
                    'negative_features': negative_features
                })
            except Exception:
                continue

        if all_batch_analysis:
            portfolio_df = pd.DataFrame(all_batch_analysis)

            # Calculate aggregate metrics
            total_batches = len(portfolio_df)
            avg_current_pred = portfolio_df['predicted'].mean()
            avg_optimized = portfolio_df['optimized_potential'].mean()
            total_gain = avg_optimized - avg_current_pred
            total_gain_pct = (total_gain / avg_current_pred) * 100 if avg_current_pred > 0 else 0

            batches_with_issues = (portfolio_df['num_negative_features'] > 0).sum()
            pct_batches_with_issues = (batches_with_issues / total_batches) * 100

            st.write("**Portfolio-Level Metrics**")

            col_port1, col_port2, col_port3, col_port4 = st.columns(4)
            col_port1.metric("Total Batches Analyzed", total_batches)
            col_port2.metric("Batches w/ Negative Drivers", f"{batches_with_issues} ({pct_batches_with_issues:.1f}%)")
            col_port3.metric("Avg Current Prediction", f"{avg_current_pred:.2f}")
            col_port4.metric("Avg Optimized Potential", f"{avg_optimized:.2f}", delta=f"+{total_gain:.2f}")

            st.metric(
                "Portfolio Improvement Potential",
                f"+{total_gain:.2f} yield points ({total_gain_pct:.1f}%)",
                help="Average improvement if all negative drivers are addressed with 70% efficiency"
            )

            # Visualization: Current vs Optimized Distribution
            st.write("**Distribution: Current Predictions vs Optimized Potential**")

            fig_portfolio = go.Figure()

            # Current predictions histogram
            fig_portfolio.add_trace(go.Histogram(
                x=portfolio_df['predicted'],
                name='Current Predictions',
                opacity=0.7,
                marker_color='#1f77b4',
                nbinsx=30
            ))

            # Optimized potential histogram
            fig_portfolio.add_trace(go.Histogram(
                x=portfolio_df['optimized_potential'],
                name='Optimized Potential',
                opacity=0.7,
                marker_color='#2ca02c',
                nbinsx=30
            ))

            fig_portfolio.update_layout(
                barmode='overlay',
                xaxis_title='D49 Yield',
                yaxis_title='Number of Batches',
                title='Yield Distribution: Current vs Optimized',
                height=400,
                legend=dict(x=0.02, y=0.98)
            )

            st.plotly_chart(fig_portfolio, use_container_width=True)

            # Feature-level analysis: Which features cause the most problems?
            st.write("**Most Common Problematic Features**")
            st.caption("Features that most frequently have negative impact across batches")

            # Count how often each feature appears as a negative driver
            feature_problem_count = {}
            feature_total_negative_impact = {}

            for batch in all_batch_analysis:
                for fname in batch['negative_features']:
                    feature_problem_count[fname] = feature_problem_count.get(fname, 0) + 1

            if feature_problem_count:
                problem_features_df = pd.DataFrame([
                    {'Feature': fname, 'Batches Affected': count, '% of Batches': (count / total_batches) * 100}
                    for fname, count in feature_problem_count.items()
                ]).sort_values('Batches Affected', ascending=False)

                st.dataframe(
                    problem_features_df.head(10),
                    use_container_width=True,
                    hide_index=True
                )

                # Bar chart of top problematic features
                fig_problems = px.bar(
                    problem_features_df.head(10),
                    x='Batches Affected',
                    y='Feature',
                    orientation='h',
                    title='Top 10 Features with Most Negative Impact Across Batches',
                    labels={'Batches Affected': 'Number of Batches'},
                    color='% of Batches',
                    color_continuous_scale='Reds'
                )
                fig_problems.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
                st.plotly_chart(fig_problems, use_container_width=True)

                # Strategic recommendations
                st.write("**Strategic Recommendations**")

                top_problem = problem_features_df.iloc[0]
                top_feature_name = top_problem['Feature']
                top_feature_pct = top_problem['% of Batches']

                st.info(f"""
                **Priority Action:** Focus on optimizing **{top_feature_name}**, which negatively affects
                **{top_feature_pct:.1f}%** of batches. Fixing this single parameter could yield the highest
                portfolio-wide improvement.
                """)

                # ROI estimation
                st.write("**Return on Investment Estimation**")
                st.caption("If you could optimize the top 3 most problematic features")

                top_3_features = problem_features_df.head(3)['Feature'].tolist()

                # Calculate potential gain if we only fix top 3 features
                focused_gain = 0
                for batch in all_batch_analysis:
                    # Get local explanation again to find which features are in top 3
                    single_batch = X_all_pruned.loc[[batch['batch_index']]].copy()
                    try:
                        local_exp = ebm_phase3.explain_local(single_batch)
                        local_data = local_exp.data(0)
                        contribution_scores = [float(v) for v in local_data.get('scores', [])]
                        contribution_feature_names = local_data.get('names', [])

                        # Sum negative contributions from top 3 features only
                        top_3_negative = sum([
                            score for fname, score in zip(contribution_feature_names, contribution_scores)
                            if score < -0.1 and fname in top_3_features
                        ])
                        focused_gain += -top_3_negative * 0.7
                    except Exception:
                        continue

                avg_focused_gain = focused_gain / total_batches
                focused_pct = (avg_focused_gain / avg_current_pred) * 100 if avg_current_pred > 0 else 0

                col_roi1, col_roi2, col_roi3 = st.columns(3)
                col_roi1.metric("Features to Optimize", "3 (Top Problems)")
                col_roi2.metric("Expected Gain", f"+{avg_focused_gain:.2f} yield points")
                col_roi3.metric("Improvement %", f"+{focused_pct:.1f}%")

                st.success("""
                **Takeaway:** By focusing on just 3 key parameters, you can capture a significant
                portion of the total optimization potential with minimal operational changes.
                """)

                # Financial Impact Analysis - Portfolio Level
                st.write("**💰 Portfolio-Level Financial Impact**")
                st.caption("Aggregate revenue analysis across all batches using D48 (quantity) and D49 (yield)")

                st.info(f"""
                **Model Uncertainty Notice:** These portfolio estimates account for prediction uncertainty:
                - MAE = {mae_phase3:.2f}pp, R² = {r2_phase3:.2f}
                - Conservative adjustments applied: 70% recovery × {max(0.3, r2_phase3):.2f} R²-adjustment = {0.7*max(0.3, r2_phase3):.1%}
                - Ranges shown reflect ±MAE uncertainty bounds
                """)

                # Check if D48 data is available
                if d48_saved is not None:
                    # Calculate financial metrics for each batch in portfolio
                    price_min = 50  # $/g
                    price_max = 60  # $/g
                    price_mid = 55  # $/g

                    # Apply R² adjustment to portfolio calculations
                    r2_adjustment_portfolio = max(0.3, r2_phase3)

                    total_current_revenue_mid = 0
                    total_optimized_revenue_mid = 0
                    total_optimized_revenue_best = 0  # Optimistic (+ MAE)
                    total_optimized_revenue_worst = 0  # Pessimistic (- MAE)
                    batches_with_d48 = 0

                    for batch in all_batch_analysis:
                        batch_idx = batch['batch_index']

                        # Get D48 value from saved data
                        if batch_idx in d48_saved.index:
                            d48_value = d48_saved.loc[batch_idx]

                            if pd.notna(d48_value) and d48_value > 0:
                                batches_with_d48 += 1

                                # Current revenue
                                current_quantity_g = d48_value * 1000
                                current_revenue = current_quantity_g * price_mid

                                # Optimized revenue scenario with uncertainty
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
                        # Calculate revenue gain
                        total_revenue_gain_mid = total_optimized_revenue_mid - total_current_revenue_mid
                        total_revenue_gain_min = total_revenue_gain_mid * (price_min / price_mid)
                        total_revenue_gain_max = total_revenue_gain_mid * (price_max / price_mid)

                        # Per-batch averages
                        avg_current_revenue = total_current_revenue_mid / batches_with_d48
                        avg_optimized_revenue = total_optimized_revenue_mid / batches_with_d48
                        avg_revenue_gain = total_revenue_gain_mid / batches_with_d48

                        st.write(f"**Analysis based on {batches_with_d48} batches with D48 data**")

                        # Current portfolio value
                        st.write("**Current Portfolio Revenue** (@ $55/g)")
                        col_port_fin1, col_port_fin2 = st.columns(2)
                        col_port_fin1.metric("Total Revenue", f"${total_current_revenue_mid:,.0f}")
                        col_port_fin2.metric("Avg Revenue per Batch", f"${avg_current_revenue:,.0f}")

                        # Optimized portfolio value
                        st.write("**Optimized Portfolio Revenue** (@ $55/g)")
                        col_port_fin3, col_port_fin4 = st.columns(2)
                        col_port_fin3.metric(
                            "Total Revenue",
                            f"${total_optimized_revenue_mid:,.0f}",
                            delta=f"+${total_revenue_gain_mid:,.0f}"
                        )
                        col_port_fin4.metric(
                            "Avg Revenue per Batch",
                            f"${avg_optimized_revenue:,.0f}",
                            delta=f"+${avg_revenue_gain:,.0f}"
                        )

                        # Revenue gain range (price sensitivity)
                        st.write("**Total Revenue Opportunity (Price Range)**")
                        col_port_fin5, col_port_fin6, col_port_fin7 = st.columns(3)
                        col_port_fin5.metric("@ $50/g", f"+${total_revenue_gain_min:,.0f}")
                        col_port_fin6.metric("@ $55/g", f"+${total_revenue_gain_mid:,.0f}")
                        col_port_fin7.metric("@ $60/g", f"+${total_revenue_gain_max:,.0f}")

                        # Calculate quantity gains
                        total_quantity_gain_kg = ((total_optimized_revenue_mid - total_current_revenue_mid) / price_mid) / 1000

                        st.metric(
                            "Additional Product Output (Portfolio)",
                            f"+{total_quantity_gain_kg:,.1f} kg",
                            help="Total estimated additional quantity from optimizing all batches"
                        )

                        # Business case summary
                        st.success(f"""
                        **Business Case Summary:**
                        - **Total Revenue Opportunity**: ${total_revenue_gain_min:,.0f} - ${total_revenue_gain_max:,.0f}
                        - **Per Batch Average**: ${avg_revenue_gain:,.0f} additional revenue
                        - **Total Additional Product**: {total_quantity_gain_kg:,.1f} kg

                        By implementing the model's recommendations across {batches_with_d48} batches,
                        you could capture this additional value with targeted parameter optimization.
                        """)

                        # Annual projection (optional - if user wants to scale)
                        st.write("**Annual Projection** (assuming current batch volume)")
                        batches_per_year = st.number_input(
                            "Estimated batches per year",
                            min_value=1,
                            max_value=10000,
                            value=100,
                            step=10,
                            help="Enter typical annual batch volume for projection"
                        )

                        annual_revenue_gain_mid = avg_revenue_gain * batches_per_year
                        annual_revenue_gain_min = annual_revenue_gain_mid * (price_min / price_mid)
                        annual_revenue_gain_max = annual_revenue_gain_mid * (price_max / price_mid)

                        col_annual1, col_annual2, col_annual3 = st.columns(3)
                        col_annual1.metric("Annual Gain @ $50/g", f"${annual_revenue_gain_min:,.0f}")
                        col_annual2.metric("Annual Gain @ $55/g", f"${annual_revenue_gain_mid:,.0f}")
                        col_annual3.metric("Annual Gain @ $60/g", f"${annual_revenue_gain_max:,.0f}")

                        st.info(f"""
                        **Projected Annual Impact:** With {batches_per_year} batches per year,
                        optimization could generate **${annual_revenue_gain_min:,.0f} - ${annual_revenue_gain_max:,.0f}**
                        in additional annual revenue.
                        """)

                    else:
                        st.warning("No batches with valid D48 data found - cannot calculate financial impact")
                else:
                    st.warning("D48 data not available - cannot calculate portfolio financial impact")

            else:
                st.info("No problematic features found across batches (all batches operating optimally)")

        else:
            st.warning("Could not analyze batches for portfolio optimization")

    except Exception as e:
        st.error(f"Failed to generate portfolio analysis: {str(e)}")

    st.divider()

    # ====================================================================
    # DOWNLOADS
    # ====================================================================
    st.subheader("Download Results")

    col_dl1, col_dl2, col_dl3 = st.columns(3)

    with col_dl1:
        # Download cleaned dataset
        csv_cleaned = df_after_correlation.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Cleaned Dataset",
            data=csv_cleaned,
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )

    with col_dl2:
        # Download selected features
        if not importance_phase1.empty:
            features_csv = importance_phase1.head(len(top_features)).to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"Download Top-{len(top_features)} Features",
                data=features_csv,
                file_name="top_features.csv",
                mime="text/csv"
            )

    with col_dl3:
        # Download metrics
        metrics_csv = comparison_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Metrics",
            data=metrics_csv,
            file_name="ebm_metrics.csv",
            mime="text/csv"
        )

else:
    st.info("Click 'Train EBM Pipeline' above to start the analysis.")

st.divider()
st.caption("EBM Filter-Then-Interact Pipeline - Fractionation Dataset Analysis")
