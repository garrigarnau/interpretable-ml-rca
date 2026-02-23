import streamlit as st
import pandas as pd
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from interpret.glassbox import ExplainableBoostingRegressor
from interpret import show
import tempfile
import os

# Import data cleaning functions
from data_cleaning import (
    load_data,
    load_variable_descriptions,
    classify_batch_quality,
    get_basic_stats,
    get_column_type_analysis,
    get_missing_summary,
    filter_columns_by_missing_threshold,
    get_equipment_columns,
    get_lot_number_columns,
    get_manual_categorical_columns,
    analyze_categorical_columns,
    get_categorical_unique_values,
    get_time_columns,
    create_column_selection_dataframe,
    get_columns_above_missing_threshold,
    apply_column_removal,
    get_time_columns_in_dataframe,
    merge_column_selections
)

# Import dataset overview functions
from dataset_overview import (
    display_overview_metrics,
    display_target_variable_visualization,
    display_target_by_quality,
    display_missing_data_timeline
)

# Page configuration
st.set_page_config(page_title="Fractionation Data Analysis", layout="wide")

st.title("🧪 Fractionation Dataset: Data Analysis & Quality Assessment")
st.markdown("""
This app provides comprehensive data analysis for the `fractionation_data.csv` dataset,
including data cleaning, feature engineering, missing value analysis, and predictive modeling.
""")

# Data loading functions are now imported from data_cleaning module

# Initialize session state for removed columns
if 'removed_redundant_cols' not in st.session_state:
    st.session_state.removed_redundant_cols = []

if 'columns_to_remove' not in st.session_state:
    st.session_state.columns_to_remove = []

data_file_path = "data/fractionation_data.csv"
current_data_mtime = os.path.getmtime(data_file_path) if os.path.exists(data_file_path) else None

# Reset stale derived state if source CSV changed
if st.session_state.get('data_file_mtime') != current_data_mtime:
    for key in ['df_filtered', 'cols_removed', 'last_missing_threshold', 'scaler_applied', 'df_before_scaling_for_ebm']:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.removed_redundant_cols = []
    st.session_state.columns_to_remove = []
    st.session_state.data_file_mtime = current_data_mtime

df = load_data(file_path=data_file_path, cache_buster=current_data_mtime)
var_descriptions = load_variable_descriptions()

if df is not None:
    # Get basic statistics for use throughout the app
    basic_stats = get_basic_stats(df)
    
    # --- DATASET OVERVIEW SECTION ---
    # Display overview metrics
    display_overview_metrics(basic_stats)
    
    # Display target variable visualization
    display_target_variable_visualization(df, target_col='D49', var_descriptions=var_descriptions)
    
    # --- DATA MASSAGING & CLEANING SECTION ---
    st.header("🔧 Data Massaging & Cleaning")
    
    # --- BASIC STATISTICS ---
    st.subheader("📊 Basic Dataset Statistics")
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    col_stat1.metric("Total Rows (Batches)", basic_stats['total_rows'])
    col_stat2.metric("Total Columns", basic_stats['total_columns'])
    col_stat3.metric("Numeric Columns", basic_stats['numeric_columns'])
    col_stat4.metric("Categorical Columns", basic_stats['categorical_columns'])
    
    # --- COLUMN TYPE ANALYSIS ---
    st.subheader("🏷️ Column Type Analysis")
    
    # Get column type analysis using the modular function
    numeric_summary, categorical_summary = get_column_type_analysis(df, var_descriptions)
    
    col_type1, col_type2 = st.columns(2)
    
    with col_type1:
        st.write(f"**Numeric Columns ({len(numeric_summary) if not numeric_summary.empty else 0})**")
        if not numeric_summary.empty:
            st.dataframe(numeric_summary, use_container_width=True, hide_index=True)
        else:
            st.info("No numeric columns found.")
    
    with col_type2:
        st.write(f"**Categorical Columns ({len(categorical_summary) if not categorical_summary.empty else 0})**")
        if not categorical_summary.empty:
            st.dataframe(categorical_summary, use_container_width=True, hide_index=True)
        else:
            st.info("No categorical columns found.")
    
    # --- CATEGORICAL VARIABLES ANALYSIS ---
    st.subheader("📋 Categorical Variables Analysis")
    st.write("These columns represent equipment choices and lot numbers that should be treated as categorical variables.")
    
    # Get categorical column groups from data_cleaning module
    equipment_cols = get_equipment_columns()
    lot_number_cols = get_lot_number_columns()

    # Track selected categorical columns (equipment + lot numbers)
    selected_categorical_cols = sorted(list(set(
        [col for col in equipment_cols if col in df.columns] +
        [col for col in lot_number_cols if col in df.columns] +
        [col for col in get_manual_categorical_columns() if col in df.columns]
    )))
    
    # Create tabs for the two categories
    equip_tab, lot_tab = st.tabs(["🔧 Equipment Columns", "🏷️ Lot Number Columns"])
    
    with equip_tab:
        st.write("**Equipment columns** (BKA bowls, tanks, filters, separation lines):")
        
        # Use modular function to analyze equipment columns
        equip_df = analyze_categorical_columns(df, equipment_cols, var_descriptions)
        
        if not equip_df.empty:
            st.dataframe(equip_df, use_container_width=True, hide_index=True)
            
            # Detailed view in expander
            with st.expander("🔍 View detailed unique values for each equipment column"):
                existing_equip = [col for col in equipment_cols if col in df.columns]
                unique_values = get_categorical_unique_values(df, equipment_cols)
                
                for col in existing_equip:
                    desc = var_descriptions.get(col, 'N/A')
                    n_unique = len(unique_values.get(col, []))
                    non_null = df[col].notna().sum()
                    
                    st.write(f"**{col}** - {desc}")
                    st.caption(f"{non_null} non-null values, {n_unique} unique values")
                    
                    if n_unique > 0:
                        st.write(unique_values[col])
                    else:
                        st.write("No values")
                    st.divider()
        else:
            st.warning("⚠️ No equipment columns found in the dataset.")
    
    with lot_tab:
        st.write("**Lot number columns** (buffer lots, ethanol lots, paper lots, etc.):")
        
        # Use modular function to analyze lot number columns
        lot_df = analyze_categorical_columns(df, lot_number_cols, var_descriptions)
        
        if not lot_df.empty:
            st.dataframe(lot_df, use_container_width=True, hide_index=True)
            
            # Detailed view in expander
            with st.expander("🔍 View detailed unique values for each lot number column"):
                existing_lots = [col for col in lot_number_cols if col in df.columns]
                unique_values = get_categorical_unique_values(df, lot_number_cols)
                
                for col in existing_lots:
                    desc = var_descriptions.get(col, 'N/A')
                    n_unique = len(unique_values.get(col, []))
                    non_null = df[col].notna().sum()
                    
                    st.write(f"**{col}** - {desc}")
                    st.caption(f"{non_null} non-null values, {n_unique} unique values")
                    
                    if n_unique > 0:
                        st.write(unique_values[col])
                    else:
                        st.write("No values")
                    st.divider()
        else:
            st.warning("⚠️ No lot number columns found in the dataset.")
    
    # Summary metrics
    col_cat1, col_cat2, col_cat3 = st.columns(3)
    existing_equip = [col for col in equipment_cols if col in df.columns]
    existing_lots = [col for col in lot_number_cols if col in df.columns]
    col_cat1.metric("Equipment Columns", len(existing_equip))
    col_cat2.metric("Lot Number Columns", len(existing_lots))
    col_cat3.metric("Total Categorical", len(existing_equip) + len(existing_lots))
    
    st.info("💡 **Note**: These columns will be encoded appropriately for modeling (e.g., one-hot encoding or target encoding).")
    
    # --- FEATURE ENGINEERING: BATCH QUALITY CLASSIFICATION ---
    st.subheader("⚙️ Feature Engineering: Batch Quality Classification")
    
    # Use the modular function to classify batch quality
    df, thresholds, quality_counts = classify_batch_quality(df, target_col='D49')
    
    if thresholds is not None:
        st.write(f"""
        Creating **Batch_Quality** feature based on D49 (FII+III PPT Yield):
        - **Good**: D49 > {thresholds['good']:.2f} (above median + 0.5×std)
        - **Average**: {thresholds['bad']:.2f} ≤ D49 ≤ {thresholds['good']:.2f} (within ±0.5×std of median)
        - **Bad**: D49 < {thresholds['bad']:.2f} (below median - 0.5×std)
        """)
        
        # Show distribution
        col_q1, col_q2, col_q3, col_q4 = st.columns(4)
        
        col_q1.metric("Good Batches", quality_counts.get('Good', 0))
        col_q2.metric("Average Batches", quality_counts.get('Average', 0))
        col_q3.metric("Bad Batches", quality_counts.get('Bad', 0))
        col_q4.metric("Unknown (Missing D49)", quality_counts.get('Unknown', 0))
        
        # Visualize distribution
        fig_quality = px.bar(
            x=quality_counts.index,
            y=quality_counts.values,
            labels={'x': 'Batch Quality', 'y': 'Count'},
            title='Distribution of Batch Quality',
            color=quality_counts.index,
            color_discrete_map={'Good': '#28a745', 'Average': '#ffc107', 'Bad': '#dc3545', 'Unknown': '#6c757d'}
        )
        fig_quality.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_quality, use_container_width=True)
        
        st.success("✅ **Batch_Quality** feature created and added to the dataset!")
        
        # Show D49 distribution colored by quality
        st.write("**D49 Distribution Across Batches (Colored by Quality)**")
        display_target_by_quality(df, target_col='D49', quality_col='Batch_Quality')

    # --- BATCH SELECTION FILTER ---
    st.subheader("🧪 Batch Selection")
    st.write("Select which batches to include in the dataset using time start (`B1`), quality (`Batch_Quality`), or both.")

    df_original_batches = df.copy()
    df_filtered_batches = df.copy()

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        quality_options = sorted(df['Batch_Quality'].dropna().unique().tolist()) if 'Batch_Quality' in df.columns else []
        selected_qualities = st.multiselect(
            "Filter by Batch Quality",
            options=quality_options,
            default=quality_options,
            help="Choose one or more quality classes to keep"
        )

    with filter_col2:
        b1_datetime = None
        if 'B1' in df.columns:
            b1_datetime = pd.to_datetime(df['B1'], errors='coerce')
            valid_b1 = b1_datetime.dropna()

            if not valid_b1.empty:
                min_date = valid_b1.min().date()
                max_date = valid_b1.max().date()

                selected_date_range = st.date_input(
                    "Filter by Time Start (B1)",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    help="Keep batches whose B1 timestamp falls within the selected date range"
                )

                if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
                    start_date, end_date = selected_date_range
                else:
                    start_date = min_date
                    end_date = max_date
            else:
                start_date = None
                end_date = None
                st.info("`B1` exists but cannot be parsed as datetime for time-based filtering.")
        else:
            start_date = None
            end_date = None
            st.info("`B1` column not found. Time-based filtering is unavailable.")

    # Apply quality filter
    if 'Batch_Quality' in df_filtered_batches.columns and selected_qualities:
        df_filtered_batches = df_filtered_batches[df_filtered_batches['Batch_Quality'].isin(selected_qualities)]

    # Apply B1 date filter
    if 'B1' in df_filtered_batches.columns and b1_datetime is not None and start_date is not None and end_date is not None:
        b1_datetime_filtered = pd.to_datetime(df_filtered_batches['B1'], errors='coerce')
        date_mask = (b1_datetime_filtered.dt.date >= start_date) & (b1_datetime_filtered.dt.date <= end_date)
        df_filtered_batches = df_filtered_batches[date_mask.fillna(False)]

    # Show filtering summary
    col_f1, col_f2, col_f3 = st.columns(3)
    col_f1.metric("Batches Before Filter", len(df_original_batches))
    col_f2.metric("Batches After Filter", len(df_filtered_batches))
    col_f3.metric("Batches Removed", len(df_original_batches) - len(df_filtered_batches))

    if len(df_filtered_batches) == 0:
        st.error("❌ No batches match the selected filters. Please adjust the filters to continue.")
        st.stop()

    # Use selected batches for all subsequent analysis
    df = df_filtered_batches.copy()
    st.success(f"✅ Using {len(df)} selected batches for the remaining workflow.")
    
    st.divider()
    
    # --- MISSING DATA THRESHOLD CONTROL ---
    st.subheader("🎚️ Missing Data Threshold")
    st.write("Adjust the threshold to control which columns are considered for removal based on missing data percentage.")
    
    # Add slider for missing data threshold
    missing_threshold = st.slider(
        "Missing Data Threshold (%)",
        min_value=0,
        max_value=100,
        value=35,
        step=5,
        help="Columns with missing data above this threshold will be analyzed and pre-selected for removal"
    )
    
    st.divider()
    
    # --- MISSING DATA TIMELINE ---
    st.subheader("📅 Missing Data Timeline Analysis")
    st.write(f"Visualize where missing data occurs over time for columns above {missing_threshold}% missing threshold.")
    
    # Get columns above threshold using modular function
    cols_to_remove_preview = get_columns_above_missing_threshold(df, threshold=missing_threshold)
    
    if cols_to_remove_preview:
        display_missing_data_timeline(df, cols_to_remove_preview, time_col='B1', var_descriptions=var_descriptions)
    else:
        st.success(f"✅ No columns with >{missing_threshold}% missing data found!")
    
    st.divider()
    
    # --- COLUMN REMOVAL SELECTION ---
    st.subheader("🎯 Column Removal Selection")
    st.write(f"Select which columns you want to remove from the dataset. Columns with >{missing_threshold}% missing data are pre-selected.")
    
    # Get time columns and columns above threshold using modular functions
    time_columns = get_time_columns()
    # Use the same threshold from the slider above
    
    # Get all columns with missing data
    missing_pct_per_col_all = (df.isnull().sum() / len(df)) * 100
    cols_with_missing = missing_pct_per_col_all[missing_pct_per_col_all > 0].sort_values(ascending=False)
    
    if len(cols_with_missing) > 0:
        # Create selection dataframe using modular function
        selection_df = create_column_selection_dataframe(df, var_descriptions)
        
        # Display summary
        col_sel1, col_sel2, col_sel3, col_sel4 = st.columns(4)
        cols_above_threshold = len(cols_with_missing[cols_with_missing > missing_threshold])
        existing_time_cols_in_df = get_time_columns_in_dataframe(df)
        col_sel1.metric("Total Columns Available", len(selection_df))
        col_sel2.metric(f"Columns with >{missing_threshold}% Missing", cols_above_threshold)
        col_sel3.metric("Time Parameters", len(existing_time_cols_in_df))
        col_sel4.metric("Currently Selected for Removal", len(st.session_state.columns_to_remove))
        
        # Show the selection table
        st.write("**Column Selection Table** (all columns, sorted by % missing)")
        st.dataframe(selection_df, use_container_width=True, hide_index=True)
        
        # Column selection interface
        st.write("**Select columns to remove:**")
        
        col_btn_row1, col_btn_row2 = st.columns(2)
        
        with col_btn_row1:
            if st.button(f"✅ Select All with >{missing_threshold}% Missing", use_container_width=True):
                st.session_state.columns_to_remove = cols_to_remove_preview
                st.rerun()
        
        with col_btn_row2:
            if st.button("❌ Clear Selection", use_container_width=True):
                st.session_state.columns_to_remove = []
                st.rerun()
        
        # Additional button row for time columns
        col_btn_row3, col_btn_row4 = st.columns(2)
        
        with col_btn_row3:
            # Get time columns that exist in the dataset
            existing_time_cols = get_time_columns_in_dataframe(df)
            if st.button(f"🕐 Select All Time Parameters ({len(existing_time_cols)} columns)", use_container_width=True):
                # Add time parameters to existing selection using merge function
                st.session_state.columns_to_remove = merge_column_selections(
                    st.session_state.columns_to_remove, 
                    existing_time_cols
                )
                st.rerun()
        
        with col_btn_row4:
            # Keep this column empty or add another quick action if needed
            st.write("")  # Placeholder for visual balance
        
        # Info about time columns
        existing_time_cols = get_time_columns_in_dataframe(df)
        if existing_time_cols:
            with st.expander("ℹ️ View Time Parameters (Start/End Times)"):
                time_col_info = pd.DataFrame({
                    'Column': existing_time_cols,
                    'Description': [var_descriptions.get(col, 'N/A')[:150] for col in existing_time_cols],
                    '% Missing': [missing_pct_per_col_all.get(col, 0) for col in existing_time_cols]
                })
                st.dataframe(time_col_info, use_container_width=True, hide_index=True)
                st.caption("These are all start/end time parameters from Phases B, C, and D")
        
        # Multiselect for custom selection
        # Include all columns as options (exclude target and derived features)
        excluded_from_removal = ['D49', 'Batch_Quality'] + selected_categorical_cols  # Keep selected categorical columns
        all_removable_cols = [col for col in df.columns if col not in excluded_from_removal]

        # Keep selection aligned with current threshold and available parameters
        if 'last_missing_threshold' not in st.session_state:
            st.session_state.last_missing_threshold = missing_threshold

        threshold_based_selection = [col for col in cols_to_remove_preview if col in all_removable_cols]

        if st.session_state.last_missing_threshold != missing_threshold:
            st.session_state.columns_to_remove = threshold_based_selection
            st.session_state.last_missing_threshold = missing_threshold
        else:
            st.session_state.columns_to_remove = [
                col for col in st.session_state.columns_to_remove if col in all_removable_cols
            ]

        default_selection = st.session_state.columns_to_remove if st.session_state.columns_to_remove else threshold_based_selection
        
        selected_columns = st.multiselect(
            "Choose columns to remove:",
            options=all_removable_cols,
            default=[col for col in default_selection if col in all_removable_cols],
            help="Select columns you want to remove from the dataset. Use the buttons above for quick selection."
        )
        
        # Update session state
        st.session_state.columns_to_remove = selected_columns
        
        # Apply removal button
        if selected_columns:
            st.info(f"💡 **{len(selected_columns)} columns selected for removal.** Click the button below to apply changes.")
            
            if st.button("🗑️ Apply Column Removal", type="primary", use_container_width=True):
                # Apply column removal using modular function
                df_filtered = apply_column_removal(df, selected_columns)

                # Safety: ensure categorical columns use object dtype for Streamlit/PyArrow compatibility
                for col in selected_categorical_cols:
                    if col in df_filtered.columns and pd.api.types.is_categorical_dtype(df_filtered[col]):
                        df_filtered[col] = df_filtered[col].astype('object')
                
                # Store in session state for use in later sections
                st.session_state.df_filtered = df_filtered
                st.session_state.cols_removed = selected_columns
                # Reset scaler flag since dataset changed
                if 'scaler_applied' in st.session_state:
                    del st.session_state['scaler_applied']
                
                st.success(f"✅ Successfully removed {len(selected_columns)} columns!")
                
                # Show summary
                col_sum1, col_sum2, col_sum3 = st.columns(3)
                col_sum1.metric("Original Columns", len(df.columns))
                col_sum2.metric("Columns Removed", len(selected_columns))
                col_sum3.metric("Columns Remaining", len(df_filtered.columns))

            # Persist filtered dataset preview across reruns (e.g., when calculating variance)
            if 'df_filtered' in st.session_state:
                df_filtered_preview = st.session_state.df_filtered
                cols_removed_preview = st.session_state.get('cols_removed', [])

                col_sum1, col_sum2, col_sum3 = st.columns(3)
                col_sum1.metric("Original Columns", len(df.columns))
                col_sum2.metric("Columns Removed", len(cols_removed_preview))
                col_sum3.metric("Columns Remaining", len(df_filtered_preview.columns))

                st.write(f"**Filtered Dataset Preview** ({len(df_filtered_preview)} rows × {len(df_filtered_preview.columns)} columns)")
                st.dataframe(df_filtered_preview.head(10), use_container_width=True)
                csv = df_filtered_preview.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Filtered Dataset",
                    data=csv,
                    file_name="fractionation_data_filtered.csv",
                    mime="text/csv",
                    key="download_filtered_dataset"
                )

            # Variance-based feature removal, then scaling
            if 'df_filtered' in st.session_state:
                st.subheader("📈 Variance-Based Feature Selection")
                st.caption("Step 1: Remove low-variance features. Step 2: Remove highly correlated features (>0.90). Step 3: Apply StandardScaler.")

                df_for_variance = st.session_state.df_filtered
                # Never scale target/quality/categorical features
                scaling_excluded_cols = set(selected_categorical_cols + ['D49', 'Batch_Quality'])
                remaining_numeric_cols = [
                    col for col in df_for_variance.select_dtypes(include=[np.number]).columns.tolist()
                    if col not in scaling_excluded_cols
                ]

                if remaining_numeric_cols:
                    variance_df = pd.DataFrame({
                        'Column': remaining_numeric_cols,
                        'Description': [var_descriptions.get(col, 'N/A') for col in remaining_numeric_cols],
                        'Variance': [df_for_variance[col].var() for col in remaining_numeric_cols],
                        'Non-Null Values': [df_for_variance[col].notna().sum() for col in remaining_numeric_cols],
                        '% Missing': [(df_for_variance[col].isnull().sum() / len(df_for_variance)) * 100 for col in remaining_numeric_cols]
                    }).sort_values(by='Variance', ascending=True)

                    variance_threshold = st.number_input(
                        "Low-variance threshold",
                        min_value=0.0,
                        value=0.10,
                        step=0.001,
                        format="%.4f",
                        key="variance_threshold"
                    )

                    low_variance_default = variance_df[variance_df['Variance'] < variance_threshold]['Column'].tolist()

                    st.dataframe(variance_df, use_container_width=True, hide_index=True)

                    variance_cols_to_remove = st.multiselect(
                        "Columns to remove by variance",
                        options=remaining_numeric_cols,
                        default=low_variance_default,
                        help="Columns with variance below threshold are preselected. You can adjust selection manually.",
                        key="variance_cols_to_remove"
                    )

                    if st.button("🗑️ Remove Low-Variance Columns", key="remove_low_variance"):
                        if variance_cols_to_remove:
                            updated_df = apply_column_removal(df_for_variance, variance_cols_to_remove)
                            st.session_state.df_filtered = updated_df
                            existing_removed = st.session_state.get('cols_removed', [])
                            st.session_state.cols_removed = list(dict.fromkeys(existing_removed + variance_cols_to_remove))
                            # Reset scaler flag since dataset changed
                            if 'scaler_applied' in st.session_state:
                                del st.session_state['scaler_applied']
                            st.success(f"✅ Removed {len(variance_cols_to_remove)} low-variance columns.")
                            st.rerun()
                        else:
                            st.info("No low-variance columns selected for removal.")

                    st.divider()

                    # Step 2: Correlation matrix and high-correlation removal
                    st.subheader("🔗 Step 2: Correlation Matrix & Redundancy Removal")
                    st.write("Identify pairs with |correlation| > 0.90 and remove the feature with more missing data.")

                    df_after_variance = st.session_state.df_filtered
                    numeric_after_variance = [
                        col for col in df_after_variance.select_dtypes(include=[np.number]).columns.tolist()
                        if col not in scaling_excluded_cols
                    ]

                    if len(numeric_after_variance) > 1:
                        corr_matrix = df_after_variance[numeric_after_variance].corr(method='pearson', min_periods=1)

                        fig_corr = go.Figure(data=go.Heatmap(
                            z=corr_matrix.values,
                            x=corr_matrix.columns,
                            y=corr_matrix.columns,
                            colorscale='RdBu',
                            zmid=0,
                            zmin=-1,
                            zmax=1,
                            colorbar=dict(title="Correlation")
                        ))
                        fig_corr.update_layout(
                            title="Correlation Matrix (Numeric Features after Variance Filter)",
                            height=600
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)

                        high_corr_pairs = []
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i + 1, len(corr_matrix.columns)):
                                col_a = corr_matrix.columns[i]
                                col_b = corr_matrix.columns[j]
                                corr_value = corr_matrix.iloc[i, j]

                                if abs(corr_value) > 0.90:
                                    missing_a = df_after_variance[col_a].isnull().sum()
                                    missing_b = df_after_variance[col_b].isnull().sum()

                                    if missing_a > missing_b:
                                        drop_col = col_a
                                        keep_col = col_b
                                    else:
                                        drop_col = col_b
                                        keep_col = col_a

                                    high_corr_pairs.append({
                                        'Feature A': col_a,
                                        'Missing A': missing_a,
                                        'Feature B': col_b,
                                        'Missing B': missing_b,
                                        'Correlation': corr_value,
                                        'Keep': keep_col,
                                        'Drop': drop_col
                                    })

                        if high_corr_pairs:
                            high_corr_df = pd.DataFrame(high_corr_pairs).sort_values(
                                by='Correlation', key=lambda s: s.abs(), ascending=False
                            )
                            st.warning(f"⚠️ Found {len(high_corr_df)} high-correlation pair(s) with |r| > 0.90.")
                            st.dataframe(high_corr_df, use_container_width=True, hide_index=True)

                            cols_to_drop_corr = sorted(set(high_corr_df['Drop'].tolist()))
                            st.info(f"{len(cols_to_drop_corr)} feature(s) will be removed based on higher missingness in correlated pairs.")

                            if st.button("🗑️ Remove Highly Correlated Features (>0.90)", key="remove_high_corr"):
                                df_after_corr = apply_column_removal(df_after_variance, cols_to_drop_corr)
                                st.session_state.df_filtered = df_after_corr
                                existing_removed = st.session_state.get('cols_removed', [])
                                st.session_state.cols_removed = list(dict.fromkeys(existing_removed + cols_to_drop_corr))
                                if 'scaler_applied' in st.session_state:
                                    del st.session_state['scaler_applied']
                                st.success(f"✅ Removed {len(cols_to_drop_corr)} highly correlated feature(s).")
                                st.rerun()
                        else:
                            st.success("✅ No feature pairs with |correlation| > 0.90.")
                    else:
                        st.info("Need at least 2 numeric features to compute correlation matrix.")

                    st.divider()

                    # Step 3: StandardScaler after variance and correlation-based removal
                    st.subheader("⚖️ Step 3: Apply StandardScaler")
                    st.write("Standardize the remaining numeric features (mean=0, std=1) for modeling.")

                    df_after_correlation = st.session_state.df_filtered
                    st.info(f"ℹ️ Using dataset after variance/correlation removal ({len(df_after_correlation)} batches)")
                    if selected_categorical_cols:
                        st.caption(f"Categorical features excluded from scaling: {', '.join([col for col in selected_categorical_cols if col in df_after_correlation.columns])}")

                    numeric_after_variance = [
                        col for col in df_after_correlation.select_dtypes(include=[np.number]).columns.tolist()
                        if col not in scaling_excluded_cols
                    ]

                    if numeric_after_variance:
                        if st.button("⚖️ Apply StandardScaler", key="apply_standard_scaler"):
                            scaled_df = df_after_correlation.copy()
                            st.session_state.df_before_scaling_for_ebm = df_after_correlation.copy()
                            scaler = StandardScaler()
                            numeric_for_scaling = scaled_df[numeric_after_variance].copy()
                            numeric_for_scaling = numeric_for_scaling.fillna(numeric_for_scaling.mean())
                            scaled_df[numeric_after_variance] = scaler.fit_transform(numeric_for_scaling)
                            st.session_state.df_filtered = scaled_df
                            st.session_state.scaler_applied = True
                            st.success(f"✅ StandardScaler applied to {len(numeric_after_variance)} numeric columns.")
                            st.rerun()
                        
                        # Show preview of scaled dataset if scaler has been applied
                        if st.session_state.get('scaler_applied', False):
                            st.subheader("📋 Final Cleaned & Scaled Dataset Preview")
                            final_df = st.session_state.df_filtered
                            
                            col_final1, col_final2, col_final3 = st.columns(3)
                            col_final1.metric("Total Rows", len(final_df))
                            col_final2.metric("Total Columns", len(final_df.columns))
                            col_final3.metric("Numeric Columns (Scaled)", len(numeric_after_variance))
                            
                            st.write("**First 10 rows of the final dataset:**")
                            st.dataframe(final_df.head(10), use_container_width=True)
                            
                            # Download button for final dataset
                            csv_final = final_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Final Cleaned & Scaled Dataset",
                                data=csv_final,
                                file_name="fractionation_data_final_cleaned_scaled.csv",
                                mime="text/csv",
                                key="download_final_scaled"
                            )
                else:
                    st.info("No numeric feature columns remaining for variance-based selection.")
        else:
            st.warning("⚠️ No columns selected for removal.")
    else:
        st.success("✅ No columns with missing data found!")

    st.divider()

    # --- BASELINE MODEL: LASSO REGRESSION ---
    st.header("📉 Baseline Model: Lasso Regression")
    st.write("Train a basic Lasso regression model on the cleaned dataset to establish a baseline.")

    if 'df_filtered' in st.session_state:
        model_df = st.session_state.df_filtered.copy()
        st.info(f"ℹ️ Using processed dataset ({len(model_df)} rows × {len(model_df.columns)} columns).")
    else:
        model_df = df.copy()
        st.info(f"ℹ️ Using current filtered batches dataset ({len(model_df)} rows × {len(model_df.columns)} columns).")

    if 'D48' in model_df.columns:
        model_df = model_df.drop(columns=['D48'])
        st.caption("Removed D48 before Lasso training.")

    if 'D49' not in model_df.columns:
        st.warning("⚠️ Target column `D49` not found. Lasso baseline cannot be trained.")
    else:
        model_df = model_df[model_df['D49'].notna()].copy()

        if len(model_df) < 10:
            st.warning("⚠️ Not enough rows with target values to train a baseline model.")
        else:
            d49_position = model_df.columns.get_loc('D49')
            candidate_cols_before_d49 = model_df.columns[:d49_position].tolist()
            feature_cols = [col for col in candidate_cols_before_d49 if col != 'Batch_Quality']

            cols_after_d49 = model_df.columns[d49_position + 1:].tolist()
            if cols_after_d49:
                st.caption(f"Excluded {len(cols_after_d49)} column(s) after D49 from Lasso features.")

            X_raw = model_df[feature_cols].copy()
            y = model_df['D49'].copy()

            X_encoded = pd.get_dummies(X_raw, drop_first=True)
            X_encoded = X_encoded.fillna(0)

            duplicate_feature_names = X_encoded.columns[X_encoded.columns.duplicated()].tolist()
            if duplicate_feature_names:
                unique_duplicates = sorted(set(duplicate_feature_names))
                X_encoded = X_encoded.loc[:, ~X_encoded.columns.duplicated()].copy()
                st.warning(
                    f"⚠️ Removed {len(unique_duplicates)} duplicated feature name(s) before training: "
                    f"{', '.join(unique_duplicates[:10])}" + ("..." if len(unique_duplicates) > 10 else "")
                )

            if X_encoded.shape[1] == 0:
                st.warning("⚠️ No usable feature columns available for Lasso baseline.")
            else:
                col_lasso1, col_lasso2 = st.columns(2)
                with col_lasso1:
                    alpha = st.number_input(
                        "Lasso alpha (regularization)",
                        min_value=0.0001,
                        max_value=100.0,
                        value=0.1,
                        step=0.01,
                        format="%.4f",
                        key="lasso_alpha"
                    )
                with col_lasso2:
                    test_size = st.slider(
                        "Test size",
                        min_value=0.1,
                        max_value=0.4,
                        value=0.2,
                        step=0.05,
                        key="lasso_test_size"
                    )

                max_cv_folds = min(10, len(X_encoded))
                cv_folds = st.number_input(
                    "Cross-validation folds (K)",
                    min_value=3,
                    max_value=max_cv_folds,
                    value=min(5, max_cv_folds),
                    step=1,
                    key="lasso_cv_folds"
                )

                current_training_signature = {
                    'alpha': float(alpha),
                    'test_size': float(test_size),
                    'cv_folds': int(cv_folds),
                    'n_rows': int(len(X_encoded)),
                    'n_features': int(X_encoded.shape[1]),
                    'feature_cols': tuple(X_encoded.columns.tolist())
                }

                if st.session_state.get('lasso_training_signature') != current_training_signature:
                    st.session_state.pop('lasso_training_output', None)
                    st.session_state.lasso_training_signature = current_training_signature

                if st.button("🚀 Train Lasso Baseline", type="primary", key="train_lasso_baseline"):
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_encoded,
                        y,
                        test_size=test_size,
                        random_state=42
                    )

                    lasso_model = Lasso(alpha=alpha, random_state=42, max_iter=10000)
                    lasso_model.fit(X_train, y_train)
                    y_pred = lasso_model.predict(X_test)

                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    r2 = r2_score(y_test, y_pred)

                    cv_model = Lasso(alpha=alpha, random_state=42, max_iter=10000)
                    cv = KFold(n_splits=int(cv_folds), shuffle=True, random_state=42)
                    cv_r2 = cross_val_score(cv_model, X_encoded, y, cv=cv, scoring='r2')
                    cv_mae = -cross_val_score(cv_model, X_encoded, y, cv=cv, scoring='neg_mean_absolute_error')
                    cv_mse = -cross_val_score(cv_model, X_encoded, y, cv=cv, scoring='neg_mean_squared_error')
                    cv_rmse = np.sqrt(cv_mse)

                    pred_df = pd.DataFrame({
                        'Actual D49': y_test.values,
                        'Predicted D49': y_pred
                    })

                    coef_df = pd.DataFrame({
                        'Feature': X_encoded.columns,
                        'Coefficient': lasso_model.coef_,
                        'Abs Coefficient': np.abs(lasso_model.coef_)
                    }).sort_values(by='Abs Coefficient', ascending=False)

                    top_10_features = coef_df.head(10).copy()
                    top_10_features['Feature Name'] = top_10_features['Feature'].apply(
                        lambda feature: var_descriptions.get(feature, var_descriptions.get(feature.split('_')[0], 'N/A'))
                    )

                    ebm_results = []
                    ebm_warning = None
                    ebm_quality_assessment = None
                    ebm_pred_df = pd.DataFrame()
                    importance_df = pd.DataFrame(columns=['Feature', 'Importance'])
                    shape_function_map = {}
                    ebm_top10_model = None
                    X_top10_for_local = pd.DataFrame()
                    local_batch_selector_df = pd.DataFrame()

                    if st.session_state.get('scaler_applied', False) and 'df_before_scaling_for_ebm' in st.session_state:
                        ebm_source_df = st.session_state.df_before_scaling_for_ebm.copy()
                        ebm_source_caption = "EBM uses the pre-standardization dataset (numeric variables are not scaled)."
                    else:
                        ebm_source_df = model_df.copy()
                        ebm_source_caption = "EBM uses current dataset values without additional standardization."

                    if 'D48' in ebm_source_df.columns:
                        ebm_source_df = ebm_source_df.drop(columns=['D48'])

                    if 'D49' in ebm_source_df.columns:
                        ebm_source_df = ebm_source_df[ebm_source_df['D49'].notna()].copy()
                    else:
                        ebm_source_df = pd.DataFrame()

                    ebm_feature_cols = [col for col in feature_cols if col in ebm_source_df.columns]
                    X_raw_ebm = ebm_source_df[ebm_feature_cols].copy() if not ebm_source_df.empty else pd.DataFrame()
                    y_ebm = ebm_source_df['D49'].copy() if not ebm_source_df.empty and 'D49' in ebm_source_df.columns else pd.Series(dtype=float)

                    X_encoded_ebm = pd.get_dummies(X_raw_ebm, drop_first=True)
                    X_encoded_ebm = X_encoded_ebm.fillna(0)

                    duplicate_ebm_feature_names = X_encoded_ebm.columns[X_encoded_ebm.columns.duplicated()].tolist()
                    ebm_duplicates_removed = bool(duplicate_ebm_feature_names)
                    if ebm_duplicates_removed:
                        X_encoded_ebm = X_encoded_ebm.loc[:, ~X_encoded_ebm.columns.duplicated()].copy()

                    if X_encoded_ebm.shape[1] == 0 or len(y_ebm) < 10:
                        ebm_warning = "⚠️ Not enough usable unscaled features/rows to train EBM models."
                    else:
                        top_10_feature_cols = top_10_features['Feature'].tolist()
                        top_10_feature_cols_ebm = [col for col in top_10_feature_cols if col in X_encoded_ebm.columns]

                        if top_10_feature_cols_ebm:
                            X_top10 = X_encoded_ebm[top_10_feature_cols_ebm].copy()
                            X_top10_for_local = X_top10.copy()

                            X_train_top10_ebm, X_test_top10_ebm, y_train_top10_ebm, y_test_top10_ebm = train_test_split(
                                X_top10,
                                y_ebm,
                                test_size=test_size,
                                random_state=42
                            )

                            ebm_top10 = ExplainableBoostingRegressor(random_state=42, n_jobs=1, interactions=10)
                            ebm_top10.fit(X_train_top10_ebm, y_train_top10_ebm)
                            ebm_top10_model = ebm_top10
                            y_pred_top10_ebm = ebm_top10.predict(X_test_top10_ebm)

                            y_pred_top10_all = ebm_top10.predict(X_top10)
                            if 'Batch_Quality' in ebm_source_df.columns:
                                batch_quality_values = ebm_source_df.loc[X_top10.index, 'Batch_Quality'].fillna('Unknown').astype(str).values
                            else:
                                batch_quality_values = np.array(['Unknown'] * len(X_top10))

                            local_batch_selector_df = pd.DataFrame({
                                'Batch': [f"Batch {idx}" for idx in X_top10.index],
                                'Batch Index': list(X_top10.index),
                                'Batch Quality': batch_quality_values,
                                'Actual D49': y_ebm.values,
                                'Predicted D49': y_pred_top10_all
                            }).reset_index(drop=True)
                            local_batch_selector_df['Residual'] = (
                                local_batch_selector_df['Actual D49'] - local_batch_selector_df['Predicted D49']
                            )
                            local_batch_selector_df['Abs Residual'] = local_batch_selector_df['Residual'].abs()

                            ebm_top10_mae = mean_absolute_error(y_test_top10_ebm, y_pred_top10_ebm)
                            ebm_top10_rmse = np.sqrt(mean_squared_error(y_test_top10_ebm, y_pred_top10_ebm))
                            ebm_top10_r2 = r2_score(y_test_top10_ebm, y_pred_top10_ebm)

                            target_std_test = float(np.std(y_test_top10_ebm))
                            rmse_to_std = (ebm_top10_rmse / target_std_test) if target_std_test > 0 else np.nan

                            if ebm_top10_r2 >= 0.50 and (np.isnan(rmse_to_std) or rmse_to_std <= 0.80):
                                ebm_alarm_level = 'good'
                                ebm_alarm_message = "✅ EBM reliability is GOOD for operational explanation use."
                            elif ebm_top10_r2 >= 0.25 and (np.isnan(rmse_to_std) or rmse_to_std <= 1.10):
                                ebm_alarm_level = 'caution'
                                ebm_alarm_message = "⚠️ EBM reliability is MODERATE. Use explanations with engineering validation."
                            else:
                                ebm_alarm_level = 'high_risk'
                                ebm_alarm_message = "🚨 EBM reliability is LOW. Do not use explanations alone for decisions."

                            ebm_quality_assessment = {
                                'alarm_level': ebm_alarm_level,
                                'alarm_message': ebm_alarm_message,
                                'r2': float(ebm_top10_r2),
                                'mae': float(ebm_top10_mae),
                                'rmse': float(ebm_top10_rmse),
                                'target_std_test': float(target_std_test),
                                'rmse_to_std': float(rmse_to_std) if not np.isnan(rmse_to_std) else np.nan,
                                'test_rows': int(len(y_test_top10_ebm))
                            }

                            ebm_results.append({
                                'Model': 'EBM (Top 10 Lasso Features)',
                                'MAE': ebm_top10_mae,
                                'RMSE': ebm_top10_rmse,
                                'R²': ebm_top10_r2
                            })

                            ebm_pred_df = pd.DataFrame({
                                'Actual D49': y_test_top10_ebm.values,
                                'Predicted D49 (EBM Top 10)': y_pred_top10_ebm
                            })

                            ebm_global = ebm_top10.explain_global(name='EBM Top 10 Global Explanation')
                            global_data = ebm_global.data()

                            importance_df = pd.DataFrame({
                                'Feature': global_data.get('names', []),
                                'Importance': global_data.get('scores', [])
                            }).sort_values(by='Importance', ascending=False)

                            for idx, feature_name in enumerate(global_data.get('names', [])):
                                feature_data = ebm_global.data(idx)
                                shape_function_map[feature_name] = {
                                    'x': list(feature_data.get('names', [])),
                                    'y': list(feature_data.get('scores', []))
                                }
                        else:
                            ebm_warning = "No overlapping top-10 Lasso features available in unscaled EBM matrix. Skipping first EBM model."

                    st.session_state.lasso_training_output = {
                        'train_rows': len(X_train),
                        'test_rows': len(X_test),
                        'mae': mae,
                        'rmse': rmse,
                        'r2': r2,
                        'cv_mae_mean': cv_mae.mean(),
                        'cv_mae_std': cv_mae.std(),
                        'cv_rmse_mean': cv_rmse.mean(),
                        'cv_rmse_std': cv_rmse.std(),
                        'cv_r2_mean': cv_r2.mean(),
                        'cv_r2_std': cv_r2.std(),
                        'pred_df': pred_df,
                        'coef_df': coef_df,
                        'top_10_features': top_10_features,
                        'ebm_source_caption': ebm_source_caption,
                        'ebm_duplicates_removed': ebm_duplicates_removed,
                        'ebm_warning': ebm_warning,
                        'ebm_results_df': pd.DataFrame(ebm_results),
                        'ebm_pred_df': ebm_pred_df,
                        'importance_df': importance_df,
                        'shape_function_map': shape_function_map,
                        'default_shape_feature': importance_df.iloc[0]['Feature'] if not importance_df.empty else None,
                        'ebm_top10_model': ebm_top10_model,
                        'X_top10_for_local': X_top10_for_local,
                        'local_batch_selector_df': local_batch_selector_df,
                        'ebm_quality_assessment': ebm_quality_assessment
                    }

                if 'lasso_training_output' in st.session_state:
                    training_output = st.session_state.lasso_training_output

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Train Rows", training_output['train_rows'])
                    m2.metric("Test Rows", training_output['test_rows'])
                    m3.metric("MAE", f"{training_output['mae']:.4f}")
                    m4.metric("RMSE", f"{training_output['rmse']:.4f}")
                    st.metric("R²", f"{training_output['r2']:.4f}")

                    st.subheader("🔁 Cross-Validation Results")
                    cv1, cv2, cv3 = st.columns(3)
                    cv1.metric("CV MAE (mean ± std)", f"{training_output['cv_mae_mean']:.4f} ± {training_output['cv_mae_std']:.4f}")
                    cv2.metric("CV RMSE (mean ± std)", f"{training_output['cv_rmse_mean']:.4f} ± {training_output['cv_rmse_std']:.4f}")
                    cv3.metric("CV R² (mean ± std)", f"{training_output['cv_r2_mean']:.4f} ± {training_output['cv_r2_std']:.4f}")

                    pred_df = training_output['pred_df']
                    fig_pred = px.scatter(
                        pred_df,
                        x='Actual D49',
                        y='Predicted D49',
                        title='Lasso Baseline: Actual vs Predicted (Test Set)'
                    )
                    actual_col = "Actual D49"
                    predicted_col = "Predicted D49"
                    axis_min = min(pred_df[actual_col].min(), pred_df[predicted_col].min())
                    axis_max = max(pred_df[actual_col].max(), pred_df[predicted_col].max())
                    fig_pred.add_trace(
                        go.Scatter(
                            x=[axis_min, axis_max],
                            y=[axis_min, axis_max],
                            mode='lines',
                            name='Ideal (y = x)',
                            line=dict(color='red', dash='dash')
                        )
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)

                    coef_df = training_output['coef_df']
                    non_zero_coef = coef_df[coef_df['Coefficient'] != 0]
                    st.caption(f"Non-zero coefficients: {len(non_zero_coef)} / {len(coef_df)}")

                    top_10_features = training_output['top_10_features']
                    st.subheader("🏆 Top 10 Lasso Features")
                    st.dataframe(
                        top_10_features[['Feature', 'Feature Name', 'Coefficient', 'Abs Coefficient']],
                        use_container_width=True,
                        hide_index=True
                    )

                    st.dataframe(coef_df.head(20), use_container_width=True, hide_index=True)

                    st.subheader("🤖 EBM Models")
                    st.caption(training_output['ebm_source_caption'])

                    if training_output.get('ebm_duplicates_removed', False):
                        st.warning("⚠️ Removed duplicated EBM feature names before training.")

                    ebm_warning = training_output.get('ebm_warning')
                    if ebm_warning:
                        st.warning(ebm_warning)
                    else:
                        ebm_results_df = training_output['ebm_results_df']
                        if not ebm_results_df.empty:
                            st.write("**EBM (Top 10 Lasso Features) Metrics**")
                            ebm_top10_m1, ebm_top10_m2, ebm_top10_m3 = st.columns(3)
                            ebm_top10_m1.metric("MAE", f"{ebm_results_df.iloc[0]['MAE']:.4f}")
                            ebm_top10_m2.metric("RMSE", f"{ebm_results_df.iloc[0]['RMSE']:.4f}")
                            ebm_top10_m3.metric("R²", f"{ebm_results_df.iloc[0]['R²']:.4f}")

                        st.write("**🚨 EBM Reliability Alarm**")
                        ebm_quality_assessment = training_output.get('ebm_quality_assessment')
                        if ebm_quality_assessment:
                            if ebm_quality_assessment['alarm_level'] == 'good':
                                st.success(ebm_quality_assessment['alarm_message'])
                            elif ebm_quality_assessment['alarm_level'] == 'caution':
                                st.warning(ebm_quality_assessment['alarm_message'])
                            else:
                                st.error(ebm_quality_assessment['alarm_message'])

                            qa1, qa2, qa3, qa4 = st.columns(4)
                            qa1.metric("R²", f"{ebm_quality_assessment['r2']:.4f}")
                            qa2.metric("MAE", f"{ebm_quality_assessment['mae']:.4f}")
                            qa3.metric("RMSE", f"{ebm_quality_assessment['rmse']:.4f}")
                            qa4.metric("Test Rows", f"{ebm_quality_assessment['test_rows']}")

                            if np.isnan(ebm_quality_assessment['rmse_to_std']):
                                st.info("RMSE/target-std ratio unavailable (low target variance).")
                            else:
                                st.caption(
                                    f"RMSE / target std = {ebm_quality_assessment['rmse_to_std']:.3f}. "
                                    "Lower is better (<0.8 preferred for explanation confidence)."
                                )

                            st.markdown(
                                "**What to take into account before trusting explanations:**\n"
                                "- Explanations are most reliable when global model alarm is GOOD.\n"
                                "- Use CAUTION/HIGH-RISK alarms as a trigger for process review, not automatic action.\n"
                                "- Always cross-check top contributing variables with plant context and sensor health.\n"
                                "- Prefer decisions supported by repeated behavior across batches, not a single point."
                            )
                        else:
                            st.warning("EBM reliability alarm is unavailable for this run.")

                        ebm_pred_df = training_output['ebm_pred_df']
                        if not ebm_pred_df.empty:
                            ebm_actual_col = "Actual D49"
                            ebm_predicted_col = "Predicted D49 (EBM Top 10)"
                            fig_ebm = px.scatter(
                                ebm_pred_df,
                                x=ebm_actual_col,
                                y=ebm_predicted_col,
                                title='EBM (Top 10 Lasso Features): Actual vs Predicted'
                            )
                            ebm_axis_min = min(ebm_pred_df[ebm_actual_col].min(), ebm_pred_df[ebm_predicted_col].min())
                            ebm_axis_max = max(ebm_pred_df[ebm_actual_col].max(), ebm_pred_df[ebm_predicted_col].max())
                            fig_ebm.add_trace(
                                go.Scatter(
                                    x=[ebm_axis_min, ebm_axis_max],
                                    y=[ebm_axis_min, ebm_axis_max],
                                    mode='lines',
                                    name='Ideal (y = x)',
                                    line=dict(color='green', dash='dash')
                                )
                            )
                            st.plotly_chart(fig_ebm, use_container_width=True)

                        st.subheader("📊 EBM Dashboard")
                        importance_df = training_output['importance_df']

                        st.write("**Summary Graph (Feature Importance)**")
                        fig_importance = px.bar(
                            importance_df,
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title='EBM Feature Importance Ranking (Top Drivers of D49)'
                        )
                        fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig_importance, use_container_width=True)

                        st.write("**Shape Functions (Feature Effect on D49)**")
                        shape_function_map = training_output.get('shape_function_map', {})
                        shape_feature_options = list(importance_df['Feature'])

                        if shape_feature_options:
                            default_shape_feature = training_output.get('default_shape_feature') or shape_feature_options[0]
                            if (
                                st.session_state.get('ebm_shape_selector_source_signature')
                                != st.session_state.get('lasso_training_signature')
                            ):
                                st.session_state.ebm_shape_feature_selector = default_shape_feature
                                st.session_state.ebm_shape_selector_source_signature = st.session_state.get('lasso_training_signature')
                            elif (
                                'ebm_shape_feature_selector' not in st.session_state
                                or st.session_state.ebm_shape_feature_selector not in shape_feature_options
                            ):
                                st.session_state.ebm_shape_feature_selector = default_shape_feature

                            selected_shape_feature = st.selectbox(
                                "Select a feature to inspect its shape function",
                                options=shape_feature_options,
                                key="ebm_shape_feature_selector"
                            )

                            selected_shape_data = shape_function_map.get(selected_shape_feature, {'x': [], 'y': []})
                            shape_x_raw = selected_shape_data.get('x', [])
                            shape_y = selected_shape_data.get('y', [])

                            aligned_len = min(len(shape_x_raw), len(shape_y))
                            if aligned_len == 0:
                                st.warning("No shape-function data available for this feature.")
                                shape_x_aligned = []
                                shape_y_aligned = []
                            else:
                                if len(shape_x_raw) != len(shape_y):
                                    st.warning(
                                        f"Shape-function point mismatch detected for {selected_shape_feature} "
                                        f"(x={len(shape_x_raw)}, y={len(shape_y)}). Using first {aligned_len} aligned points."
                                    )
                                shape_x_aligned = list(shape_x_raw)[:aligned_len]
                                shape_y_aligned = list(shape_y)[:aligned_len]

                            shape_x_numeric = pd.to_numeric(pd.Series(shape_x_aligned), errors='coerce')
                            if shape_x_numeric.notna().all() and len(shape_x_numeric) > 0:
                                shape_df = pd.DataFrame({
                                    'Sensor Reading': shape_x_numeric.astype(float),
                                    'Impact on D49 Yield': shape_y_aligned
                                }).sort_values('Sensor Reading')

                                fig_shape = px.line(
                                    shape_df,
                                    x='Sensor Reading',
                                    y='Impact on D49 Yield',
                                    markers=True,
                                    title=f'Shape Function: {selected_shape_feature}'
                                )
                            else:
                                shape_df = pd.DataFrame({
                                    'Sensor Reading': [str(x) for x in shape_x_aligned],
                                    'Impact on D49 Yield': shape_y_aligned
                                })

                                fig_shape = px.line(
                                    shape_df,
                                    x='Sensor Reading',
                                    y='Impact on D49 Yield',
                                    markers=True,
                                    title=f'Shape Function: {selected_shape_feature}'
                                )

                            st.plotly_chart(fig_shape, use_container_width=True)

                            if not shape_df.empty:
                                peak_row = shape_df.loc[shape_df['Impact on D49 Yield'].idxmax()]
                                st.success(
                                    f"Peak impact for {selected_shape_feature} occurs around "
                                    f"{peak_row['Sensor Reading']} with an effect of {peak_row['Impact on D49 Yield']:.4f} on D49 yield."
                                )
                                st.info("Use this peak region to guide control-limit discussions and SOP updates with plant managers.")

                        st.write("**Local Waterfall (Single Batch Explanation)**")
                        local_selector_df = training_output.get('local_batch_selector_df', pd.DataFrame())
                        ebm_model_local = training_output.get('ebm_top10_model')
                        X_top10_local = training_output.get('X_top10_for_local', pd.DataFrame())

                        if ebm_model_local is not None and not X_top10_local.empty and not local_selector_df.empty:
                            quality_options = sorted(local_selector_df['Batch Quality'].dropna().unique().tolist())
                            if 'ebm_local_quality_filter' not in st.session_state:
                                st.session_state.ebm_local_quality_filter = quality_options

                            selected_quality_filter = st.multiselect(
                                "Filter batches by quality category",
                                options=quality_options,
                                default=[q for q in st.session_state.ebm_local_quality_filter if q in quality_options],
                                key="ebm_local_quality_filter"
                            )

                            filtered_local_selector_df = local_selector_df[
                                local_selector_df['Batch Quality'].isin(selected_quality_filter)
                            ] if selected_quality_filter else local_selector_df.iloc[0:0]

                            if filtered_local_selector_df.empty:
                                st.warning("No batches available for the selected quality category filter.")
                            else:
                                batch_options = filtered_local_selector_df['Batch'].tolist()
                                default_batch = batch_options[0]
                                if (
                                    'ebm_local_batch_selector' not in st.session_state
                                    or st.session_state.ebm_local_batch_selector not in batch_options
                                ):
                                    st.session_state.ebm_local_batch_selector = default_batch

                                selected_batch_label = st.selectbox(
                                    "Select batch for local explanation",
                                    options=batch_options,
                                    key="ebm_local_batch_selector"
                                )

                                selected_batch_row = filtered_local_selector_df[
                                    filtered_local_selector_df['Batch'] == selected_batch_label
                                ].iloc[0]
                                selected_batch_index = selected_batch_row['Batch Index']
                                single_batch_X = X_top10_local.loc[[selected_batch_index]].copy()

                                local_explanation = ebm_model_local.explain_local(single_batch_X, name=f"Local explanation for {selected_batch_label}")
                                local_data = local_explanation.data(0)

                                base_value = float(local_data.get('extra', {}).get('scores', [0.0])[0])
                                contribution_scores = [float(v) for v in local_data.get('scores', [])]
                                contribution_feature_names = local_data.get('names', list(single_batch_X.columns))

                                step_labels = []
                                for feature_name in contribution_feature_names:
                                    feature_value = single_batch_X.iloc[0].get(feature_name, np.nan)
                                    if isinstance(feature_value, (int, float, np.integer, np.floating)) and pd.notna(feature_value):
                                        value_str = f"{feature_value:.4f}"
                                    else:
                                        value_str = str(feature_value)
                                    step_labels.append(f"{feature_name}: {value_str}")

                                predicted_from_components = base_value + float(np.sum(contribution_scores))
                                predicted_from_model = float(ebm_model_local.predict(single_batch_X)[0])

                                wf_x = ['Historical Average Yield'] + step_labels + ['Predicted D49 Yield']
                                wf_y = [base_value] + contribution_scores + [predicted_from_components]
                                wf_measure = ['absolute'] + ['relative'] * len(contribution_scores) + ['total']

                                fig_local_waterfall = go.Figure(
                                    go.Waterfall(
                                        orientation='v',
                                        measure=wf_measure,
                                        x=wf_x,
                                        y=wf_y,
                                        increasing={'marker': {'color': '#2ca02c'}},
                                        decreasing={'marker': {'color': '#d62728'}},
                                        totals={'marker': {'color': '#1f77b4'}},
                                        connector={'line': {'color': 'rgb(63, 63, 63)'}}
                                    )
                                )
                                fig_local_waterfall.update_layout(
                                    title=f"EBM Local Explanation Waterfall — {selected_batch_label}",
                                    yaxis_title='D49 Yield Contribution',
                                    height=550
                                )
                                st.plotly_chart(fig_local_waterfall, use_container_width=True)

                                wf_m1, wf_m2, wf_m3 = st.columns(3)
                                wf_m1.metric("Historical Average Yield", f"{base_value:.4f}")
                                wf_m2.metric("Predicted D49 (Waterfall Total)", f"{predicted_from_components:.4f}")
                                wf_m3.metric("Predicted D49 (EBM)", f"{predicted_from_model:.4f}")

                                st.write("**🚨 Selected Batch Alarm**")
                                batch_abs_residual = float(selected_batch_row.get('Abs Residual', np.nan))
                                batch_quality = str(selected_batch_row.get('Batch Quality', 'Unknown'))
                                quality_group_size = int((local_selector_df['Batch Quality'] == batch_quality).sum())
                                total_contrib_abs = float(np.sum(np.abs(contribution_scores)))
                                max_contrib_abs = float(np.max(np.abs(contribution_scores))) if len(contribution_scores) > 0 else 0.0
                                dominant_contrib_ratio = (max_contrib_abs / total_contrib_abs) if total_contrib_abs > 0 else 0.0

                                model_mae_reference = training_output.get('ebm_quality_assessment', {}).get('mae', np.nan)

                                batch_alarm_level = 'good'
                                batch_alarm_reasons = []

                                if not np.isnan(batch_abs_residual) and not np.isnan(model_mae_reference):
                                    if batch_abs_residual > 2 * model_mae_reference:
                                        batch_alarm_level = 'high_risk'
                                        batch_alarm_reasons.append("Residual is > 2× model MAE (prediction unusually uncertain).")
                                    elif batch_abs_residual > model_mae_reference and batch_alarm_level != 'high_risk':
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
                                ba1.metric("Batch |Residual|", f"{batch_abs_residual:.4f}" if not np.isnan(batch_abs_residual) else "N/A")
                                ba2.metric("Dominant Contribution Ratio", f"{dominant_contrib_ratio:.2f}")
                                ba3.metric(f"Batches in '{batch_quality}'", quality_group_size)

                                if batch_alarm_reasons:
                                    st.markdown("**Alarm reasons:**\n- " + "\n- ".join(batch_alarm_reasons))
                                else:
                                    st.caption("No risk flags detected for this selected batch.")

                                st.caption(
                                    "Waterfall total is calculated as base value + sum of local feature contributions "
                                    "for the selected batch."
                                )
                        else:
                            st.info("Local waterfall is available after successful EBM training data is generated.")

                        st.success("✅ EBM training complete.")
                        st.dataframe(ebm_results_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Click '🚀 Train Lasso Baseline' to generate Lasso and EBM results.")

    st.success("✅ Data cleaning workflow complete up to StandardScaler.")

else:
    st.info("💡 Please place your `fractionation_data.csv` file in the same folder as this script.")