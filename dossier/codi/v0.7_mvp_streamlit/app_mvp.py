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
st.set_page_config(page_title="EBM Root Cause Analysis - MVP", layout="wide")

st.title("EBM Root Cause Analysis - MVP")
st.markdown("""
This app implements the **EBM filter-then-interact approach** for Root Cause Analysis
in pharmaceutical plasma fractionation:
1. Automatic data cleaning pipeline
2. EBM Phase 1: Feature ranking with interactions=0
3. EBM Phase 2: Prune to top-N features
4. EBM Phase 3: Re-train with interaction detection
5. Global and local explanations + What-If simulator
""")

# ============================================================================
# DATA LOADING
# ============================================================================
st.header("1. Data Loading")

data_file_path = "data/fractionation_data.csv"
current_data_mtime = os.path.getmtime(data_file_path) if os.path.exists(data_file_path) else None

df = load_data(file_path=data_file_path, cache_buster=current_data_mtime)
var_descriptions = load_variable_descriptions()

if df is None:
    st.error("Failed to load data. Please ensure 'data/fractionation_data.csv' exists.")
    st.stop()

if 'A1' in df.columns:
    df = df.drop(columns=['A1'])

basic_stats = get_basic_stats(df)
display_overview_metrics(basic_stats)

st.divider()

# ============================================================================
# DATA FILTERING
# ============================================================================
st.header("2. Data Filtering")

display_target_variable_visualization(df, target_col='D49', var_descriptions=var_descriptions)

df, thresholds, quality_counts = classify_batch_quality(df, target_col='D49')

if thresholds is not None:
    st.subheader("Batch Quality Classification")
    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    col_q1.metric("Good Batches", quality_counts.get('Good', 0))
    col_q2.metric("Average Batches", quality_counts.get('Average', 0))
    col_q3.metric("Bad Batches", quality_counts.get('Bad', 0))
    col_q4.metric("Unknown", quality_counts.get('Unknown', 0))

    display_target_by_quality(df, target_col='D49', quality_col='Batch_Quality')

# Use all batches (no quality filtering in MVP)
df_filtered = df.copy()
st.success(f"Using {len(df_filtered)} batches for analysis")

st.divider()

# ============================================================================
# AUTOMATIC FEATURE CLEANING
# ============================================================================
st.header("3. Automatic Feature Cleaning")

missing_threshold = st.slider("Missing Data Threshold (%)", 0, 100, 35, 5)
missing_pct = (df_filtered.isnull().sum() / len(df_filtered)) * 100
cols_high_missing = missing_pct[missing_pct > missing_threshold].index.tolist()
protected_cols = ['D49', 'Batch_Quality']
cols_high_missing = [col for col in cols_high_missing if col not in protected_cols]
df_after_missing = apply_column_removal(df_filtered, cols_high_missing)

st.info(f"Removed {len(cols_high_missing)} columns with >{missing_threshold}% missing")

# Remove time and lot columns
time_cols = get_time_columns()
lot_cols = get_lot_number_columns()
time_cols_in_df = [col for col in time_cols if col in df_after_missing.columns]
lot_cols_in_df = [col for col in lot_cols if col in df_after_missing.columns]
cols_time_and_lot = sorted(set(time_cols_in_df + lot_cols_in_df))
df_after_time_lot = apply_column_removal(df_after_missing, cols_time_and_lot)

# Remove low variance
variance_threshold = 0.10
categorical_cols = get_equipment_columns() + get_manual_categorical_columns()
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

# Remove high correlation
correlation_threshold = 0.90
numeric_after_var = [
    col for col in df_after_variance.select_dtypes(include=[np.number]).columns
    if col not in protected_cols + categorical_cols
]
cols_to_drop_corr = []

if len(numeric_after_var) > 1:
    corr_matrix = df_after_variance[numeric_after_var].corr(method='pearson', min_periods=1)
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
                col_a = corr_matrix.columns[i]
                col_b = corr_matrix.columns[j]
                missing_a = df_after_variance[col_a].isnull().sum()
                missing_b = df_after_variance[col_b].isnull().sum()
                drop_col = col_a if missing_a > missing_b else col_b
                cols_to_drop_corr.append(drop_col)
    cols_to_drop_corr = sorted(set(cols_to_drop_corr))

df_after_correlation = apply_column_removal(df_after_variance, cols_to_drop_corr)

total_removed = len(cols_high_missing) + len(cols_time_and_lot) + len(low_variance_cols) + len(cols_to_drop_corr)
st.metric("Final Column Count", len(df_after_correlation.columns))
st.success(f"Removed {total_removed} columns total. Feature cleaning complete!")

st.divider()

# ============================================================================
# DATA PREPARATION
# ============================================================================
st.header("4. Data Preparation")

if 'D49' not in df_after_correlation.columns:
    st.error("Target column D49 not found")
    st.stop()

df_model = df_after_correlation[df_after_correlation['D49'].notna()].copy()

if 'D48' in df_model.columns:
    df_model = df_model.drop(columns=['D48'])

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

# Prepare data for EBM
X_prepared = X_raw.copy()
for col in categorical_in_features:
    X_prepared[col] = X_prepared[col].fillna('Unknown').astype(str)
for col in numeric_in_features:
    X_prepared[col] = X_prepared[col].fillna(X_prepared[col].mean())

X_train, X_test, y_train, y_test = train_test_split(X_prepared, y, test_size=0.25, random_state=42)

col_split1, col_split2 = st.columns(2)
col_split1.metric("Training Samples", len(X_train))
col_split2.metric("Test Samples", len(X_test))

st.divider()

# ============================================================================
# EBM FILTER-THEN-INTERACT PIPELINE
# ============================================================================
st.header("5. EBM Filter-Then-Interact Pipeline")

col_config1, col_config2 = st.columns(2)
with col_config1:
    top_n_features = st.slider("Top-N Features", 5, min(30, X_prepared.shape[1]), min(15, X_prepared.shape[1]))
with col_config2:
    n_interactions = st.slider("Max Interactions", 0, 20, 10)

train_clicked = st.button("Train EBM Pipeline", type="primary")

if train_clicked:
    with st.spinner("Training EBM models..."):
        # Phase 1
        st.subheader("Phase 1: Feature Ranking")
        ebm_phase1 = ExplainableBoostingRegressor(
            interactions=0, learning_rate=0.05, max_bins=256,
            max_rounds=5000, min_samples_leaf=10, n_jobs=1, random_state=42
        )
        ebm_phase1.fit(X_train, y_train)
        y_pred_phase1 = ebm_phase1.predict(X_test)

        # Extract importance
        global_exp_phase1 = ebm_phase1.explain_global(name='Phase 1')
        global_data_phase1 = global_exp_phase1.data()
        importance_phase1 = pd.DataFrame({
            'Feature': global_data_phase1.get('names', []),
            'Importance': global_data_phase1.get('scores', [])
        }).sort_values('Importance', ascending=False)

        top_features = importance_phase1.head(top_n_features)['Feature'].tolist()

        # Phase 3
        st.subheader(f"Phase 3: Interaction Hunting (Top-{top_n_features})")
        X_train_pruned = X_train[top_features].copy()
        X_test_pruned = X_test[top_features].copy()

        ebm_phase3 = ExplainableBoostingRegressor(
            interactions=n_interactions, learning_rate=0.05, max_bins=256,
            max_rounds=5000, min_samples_leaf=10, n_jobs=1, random_state=42
        )
        ebm_phase3.fit(X_train_pruned, y_train)
        y_pred_phase3 = ebm_phase3.predict(X_test_pruned)

        mae_phase3 = mean_absolute_error(y_test, y_pred_phase3)
        rmse_phase3 = np.sqrt(mean_squared_error(y_test, y_pred_phase3))
        r2_phase3 = r2_score(y_test, y_pred_phase3)

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("MAE", f"{mae_phase3:.4f}")
        col_m2.metric("RMSE", f"{rmse_phase3:.4f}")
        col_m3.metric("R2", f"{r2_phase3:.4f}")

        # Store in session state
        st.session_state.ebm_trained = True
        st.session_state.ebm_phase3 = ebm_phase3
        st.session_state.top_features = top_features
        st.session_state.mae_phase3 = mae_phase3
        st.session_state.r2_phase3 = r2_phase3
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.categorical_in_features = categorical_in_features

# ============================================================================
# EXPLANATIONS (after training)
# ============================================================================
if st.session_state.get('ebm_trained', False):
    ebm_phase3 = st.session_state.ebm_phase3
    top_features = st.session_state.top_features
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    categorical_in_features = st.session_state.categorical_in_features

    st.divider()

    # ====================================================================
    # GLOBAL EXPLANATIONS
    # ====================================================================
    st.header("6. Global Explanations")

    st.subheader("Feature Importance")
    global_exp = ebm_phase3.explain_global(name='Final EBM')
    global_data = global_exp.data()

    importance_final = pd.DataFrame({
        'Feature': global_data['names'],
        'Importance': global_data['scores']
    }).sort_values('Importance', ascending=False)

    fig_imp = px.bar(importance_final, x='Importance', y='Feature', orientation='h',
                     title='EBM Global Feature Importances')
    fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
    st.plotly_chart(fig_imp, use_container_width=True)

    # Shape Functions
    st.subheader("Shape Functions")

    feature_names = global_data.get('names', [])
    feature_scores = global_data.get('scores', [])

    shape_feature_options = [f[0] for f in sorted(
        zip(feature_names, feature_scores), key=lambda x: abs(x[1]), reverse=True
    )]

    selected_shape_feature = st.selectbox("Select feature:", options=shape_feature_options)

    if selected_shape_feature in feature_names:
        feat_idx = list(feature_names).index(selected_shape_feature)
        feat_data = global_exp.data(feat_idx)
        shape_x = list(feat_data.get('names', []))
        shape_y = list(feat_data.get('scores', []))

        min_len = min(len(shape_x), len(shape_y))
        shape_x = shape_x[:min_len]
        shape_y = shape_y[:min_len]

        if min_len > 0:
            x_numeric = pd.to_numeric(pd.Series(shape_x), errors='coerce')
            if x_numeric.notna().all():
                shape_df = pd.DataFrame({'Feature Value': x_numeric.astype(float), 'Impact on D49': shape_y})
                shape_df = shape_df.sort_values('Feature Value')
                fig_shape = px.line(shape_df, x='Feature Value', y='Impact on D49',
                                    markers=True, title=f'Shape Function: {selected_shape_feature}')
                fig_shape.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_shape, use_container_width=True)
            else:
                shape_df = pd.DataFrame({'Category': [str(x) for x in shape_x], 'Impact on D49': shape_y})
                fig_shape = px.bar(shape_df, x='Impact on D49', y='Category', orientation='h',
                                   title=f'Shape Function: {selected_shape_feature}')
                st.plotly_chart(fig_shape, use_container_width=True)

    st.divider()

    # ====================================================================
    # LOCAL EXPLANATIONS (Batch Waterfall)
    # ====================================================================
    st.header("7. Local Explanations (Batch Waterfall)")

    X_train_pruned = X_train[top_features].copy()
    X_test_pruned = X_test[top_features].copy()
    X_all_pruned = pd.concat([X_train_pruned, X_test_pruned])
    y_all = pd.concat([y_train, y_test])

    batch_options = [f"Batch {idx}" for idx in X_all_pruned.index[:50]]
    selected_batch_label = st.selectbox("Select batch:", options=batch_options)

    if selected_batch_label:
        batch_idx = int(selected_batch_label.split(" ")[1])
        single_batch_X = X_all_pruned.loc[[batch_idx]].copy()

        try:
            local_exp = ebm_phase3.explain_local(single_batch_X)
            local_data = local_exp.data(0)

            base_value = float(local_data.get('extra', {}).get('scores', [0.0])[0])
            scores = [float(v) for v in local_data.get('scores', [])]
            names = local_data.get('names', list(single_batch_X.columns))

            predicted = float(ebm_phase3.predict(single_batch_X)[0])
            actual = float(y_all.loc[batch_idx])

            # Create waterfall bar chart
            labels = ['Intercept'] + [f"{n} ({single_batch_X.iloc[0].get(n, '?')})" for n in names]
            contributions = [base_value] + scores
            labels = labels[::-1]
            contributions = contributions[::-1]
            colors = ['#7f7f7f'] + ['#1f77b4' if c < 0 else '#ff7f0e' for c in contributions[1:]]
            colors = colors[::-1]

            fig_local = go.Figure(go.Bar(
                x=contributions, y=labels, orientation='h',
                marker=dict(color=colors),
                text=[f"{c:+.3f}" for c in contributions], textposition='auto'
            ))
            fig_local.update_layout(
                title=f"Waterfall (Actual: {actual:.2f} | Predicted: {predicted:.2f})",
                xaxis_title='Contribution', height=max(400, len(labels) * 35)
            )
            st.plotly_chart(fig_local, use_container_width=True)

            col_w1, col_w2, col_w3 = st.columns(3)
            col_w1.metric("Actual D49", f"{actual:.4f}")
            col_w2.metric("Predicted D49", f"{predicted:.4f}")
            col_w3.metric("Residual", f"{actual - predicted:.4f}")

        except Exception as e:
            st.error(f"Failed to generate local explanation: {str(e)}")

    st.divider()

    # ====================================================================
    # WHAT-IF SIMULATOR
    # ====================================================================
    st.header("8. What-If Simulator")

    st.write("Adjust feature values to see predicted D49 yield change.")

    # Use median values as defaults
    X_median = X_all_pruned.median()

    what_if_values = {}
    cols = st.columns(3)
    for i, feat in enumerate(top_features):
        with cols[i % 3]:
            if feat in categorical_in_features:
                unique_vals = X_all_pruned[feat].unique().tolist()
                what_if_values[feat] = st.selectbox(f"{feat}", options=unique_vals, key=f"wi_{feat}")
            else:
                feat_min = float(X_all_pruned[feat].min())
                feat_max = float(X_all_pruned[feat].max())
                feat_med = float(X_median.get(feat, (feat_min + feat_max) / 2))
                what_if_values[feat] = st.slider(
                    f"{feat}", min_value=feat_min, max_value=feat_max,
                    value=feat_med, key=f"wi_{feat}"
                )

    if st.button("Predict", key="what_if_predict"):
        what_if_df = pd.DataFrame([what_if_values])
        prediction = ebm_phase3.predict(what_if_df)[0]
        st.metric("Predicted D49 Yield", f"{prediction:.4f}")

        # Show contributions
        try:
            local_exp = ebm_phase3.explain_local(what_if_df)
            local_data = local_exp.data(0)
            scores = [float(v) for v in local_data.get('scores', [])]
            names = local_data.get('names', top_features)

            contrib_df = pd.DataFrame({'Feature': names, 'Contribution': scores})
            contrib_df = contrib_df.sort_values('Contribution', ascending=False)
            st.dataframe(contrib_df, use_container_width=True, hide_index=True)
        except Exception:
            pass

else:
    st.info("Click 'Train EBM Pipeline' above to start the analysis.")

st.divider()
st.caption("EBM Root Cause Analysis MVP - Plasma Fractionation")
