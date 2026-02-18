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
    for key in ['df_filtered', 'cols_removed', 'last_missing_threshold']:
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
                st.caption("Step 1: Remove low-variance features. Step 2: Apply StandardScaler on remaining numeric features.")

                df_for_variance = st.session_state.df_filtered
                remaining_numeric_cols = [
                    col for col in df_for_variance.select_dtypes(include=[np.number]).columns.tolist()
                    if col not in selected_categorical_cols and col != 'D49'
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
                        value=0.01,
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
                            st.success(f"✅ Removed {len(variance_cols_to_remove)} low-variance columns.")
                            st.rerun()
                        else:
                            st.info("No low-variance columns selected for removal.")

                    # Step 2: StandardScaler after variance-based removal
                    df_after_variance = st.session_state.df_filtered
                    numeric_after_variance = [
                        col for col in df_after_variance.select_dtypes(include=[np.number]).columns.tolist()
                        if col not in selected_categorical_cols and col != 'D49'
                    ]

                    if numeric_after_variance:
                        if st.button("⚖️ Apply StandardScaler", key="apply_standard_scaler"):
                            scaled_df = df_after_variance.copy()
                            scaler = StandardScaler()
                            numeric_for_scaling = scaled_df[numeric_after_variance].copy()
                            numeric_for_scaling = numeric_for_scaling.fillna(numeric_for_scaling.mean())
                            scaled_df[numeric_after_variance] = scaler.fit_transform(numeric_for_scaling)
                            st.session_state.df_filtered = scaled_df
                            st.success(f"✅ StandardScaler applied to {len(numeric_after_variance)} numeric columns.")
                            st.rerun()
                else:
                    st.info("No numeric feature columns remaining for variance-based selection.")
        else:
            st.warning("⚠️ No columns selected for removal.")
    else:
        st.success("✅ No columns with missing data found!")

    st.divider()
    
    # --- USE FILTERED DATASET FOR SUBSEQUENT ANALYSIS ---
    # Check if filtered dataset exists in session state, otherwise use original
    if 'df_filtered' in st.session_state:
        df_filtered = st.session_state.df_filtered
        cols_removed = st.session_state.cols_removed
        st.info(f"ℹ️ Using filtered dataset with {len(cols_removed)} columns removed for subsequent analysis.")
    else:
        df_filtered = df.copy()
        cols_removed = []
    
    # Apply redundant column removal if any were marked
    if st.session_state.removed_redundant_cols:
        # Apply column removal using modular function
        cols_to_remove = [col for col in st.session_state.removed_redundant_cols if col in df_filtered.columns]
        if cols_to_remove:
            df_filtered = apply_column_removal(df_filtered, cols_to_remove)
            st.info(f"ℹ️ {len(cols_to_remove)} redundant features have been removed from this dataset.")
    
    st.divider()
    
    # Show status if redundant features were removed
    if st.session_state.removed_redundant_cols:
        st.success(f"✅ {len(st.session_state.removed_redundant_cols)} redundant features have been removed from the analysis")

    st.divider()

    # --- PHASE 1: SIGNAL PRESERVATION ---
    st.header("📊 Phase 1: Signal Preservation")
    
    # --- STEP 2: TARGET CORRELATION ANALYSIS ---
    st.subheader("🎯 Step 2: Target Correlation Analysis (D49)")
    st.write("**Identify VIP (Very Important Predictor) features BEFORE removing anything.**")
    
    # Get numeric columns
    numeric_cols_for_target = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'D49' in numeric_cols_for_target and len(numeric_cols_for_target) > 1:
        # Calculate correlation with D49 using pairwise deletion
        target_correlations = []
        for col in numeric_cols_for_target:
            if col != 'D49':
                # Get valid pairs (where both values exist)
                valid_mask = df_filtered[col].notna() & df_filtered['D49'].notna()
                if valid_mask.sum() > 1:  # Need at least 2 points
                    corr_val = df_filtered.loc[valid_mask, [col, 'D49']].corr().iloc[0, 1]
                    target_correlations.append({
                        'Feature': col,
                        'Description': var_descriptions.get(col, 'N/A'),
                        'Correlation with D49': corr_val,
                        'Abs Correlation': abs(corr_val),
                        'Valid Pairs': valid_mask.sum()
                    })
        
        target_corr_df = pd.DataFrame(target_correlations)
        target_corr_df = target_corr_df.sort_values(by='Abs Correlation', ascending=False)
        
        # Show top correlations
        col_tc1, col_tc2, col_tc3 = st.columns(3)
        strong_corr = target_corr_df[target_corr_df['Abs Correlation'] > 0.5]
        moderate_corr = target_corr_df[(target_corr_df['Abs Correlation'] >= 0.3) & (target_corr_df['Abs Correlation'] <= 0.5)]
        weak_corr = target_corr_df[target_corr_df['Abs Correlation'] < 0.3]
        
        col_tc1.metric("Strong Correlations (|r| > 0.5)", len(strong_corr))
        col_tc2.metric("Moderate Correlations (0.3-0.5)", len(moderate_corr))
        col_tc3.metric("Weak Correlations (|r| < 0.3)", len(weak_corr))
        
        # Visualization
        top_20_target = target_corr_df.head(20)
        fig_target = px.bar(
            top_20_target,
            x='Correlation with D49',
            y='Feature',
            orientation='h',
            title='Top 20 Features by Correlation with D49 Target (Pre-Redundancy Removal)',
            color='Correlation with D49',
            color_continuous_scale='RdBu',
            range_color=[-1, 1],
            hover_data=['Description', 'Valid Pairs']
        )
        fig_target.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_target, use_container_width=True)
        
        # Show table
        with st.expander("📋 View All Target Correlations"):
            display_target_corr = target_corr_df[['Feature', 'Description', 'Correlation with D49', 'Valid Pairs']].copy()
            st.dataframe(display_target_corr, use_container_width=True, hide_index=True)
        
        st.success("✅ Target correlation analysis complete. These VIP features will guide redundancy removal.")
    else:
        st.warning("⚠️ Target variable D49 not found or insufficient numeric columns.")
        target_corr_df = pd.DataFrame()  # Empty dataframe for later use
    
    st.divider()

    # --- STEP 3: SMART REDUNDANCY REMOVAL ---
    st.subheader("✂️ Step 3: Smart Redundancy Removal (Feature-to-Feature Analysis)")
    
    if st.session_state.removed_redundant_cols:
        st.write("Showing correlations **after smart redundancy removal**. Features with higher D49 correlation were preserved.")
    else:
        st.write("**New Strategy**: When removing redundant pairs, we keep the feature with HIGHER correlation to D49, not just fewer NaNs.")
    
    # Get numeric columns for correlation
    numeric_cols_raw = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols_raw and len(numeric_cols_raw) > 1:
        # Calculate correlation matrix using pairwise deletion (default behavior)
        corr_matrix_raw = df_filtered[numeric_cols_raw].corr(method='pearson', min_periods=1)
        
        # Create correlation heatmap
        st.write("**Correlation Heatmap (Pairwise Deletion)**")
        
        heatmap_title = "Feature Correlation Matrix"
        if st.session_state.removed_redundant_cols:
            heatmap_title += " (After Redundancy Removal)"
        else:
            heatmap_title += " (Before Redundancy Removal)"
        
        fig_corr_raw = go.Figure(data=go.Heatmap(
            z=corr_matrix_raw.values,
            x=corr_matrix_raw.columns,
            y=corr_matrix_raw.columns,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=corr_matrix_raw.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 8},
            colorbar=dict(title="Correlation")
        ))
        
        fig_corr_raw.update_layout(
            title=heatmap_title,
            xaxis_title="Features",
            yaxis_title="Features",
            height=600,
            xaxis={'side': 'bottom'}
        )
        
        st.plotly_chart(fig_corr_raw, use_container_width=True)
        
        # Only show redundancy detection if not already removed
        if not st.session_state.removed_redundant_cols:
            # Identify highly correlated pairs (>0.90)
            st.write("**High Correlation Pairs (|r| > 0.90)**")
            
            redundant_pairs = []
            for i in range(len(corr_matrix_raw.columns)):
                for j in range(i+1, len(corr_matrix_raw.columns)):
                    var1 = corr_matrix_raw.columns[i]
                    var2 = corr_matrix_raw.columns[j]
                    corr_value = corr_matrix_raw.iloc[i, j]
                    
                    # SAFETY: Never drop the target variable D49
                    if var1 == 'D49' or var2 == 'D49':
                        continue
                    
                    if abs(corr_value) > 0.90:
                        # Calculate missing counts
                        missing_var1 = df_filtered[var1].isnull().sum()
                        missing_var2 = df_filtered[var2].isnull().sum()
                        
                        # SMART LOGIC: Check correlation with target D49
                        # Get correlation with D49 for both variables
                        if not target_corr_df.empty and 'D49' in df_filtered.columns:
                            d49_corr_var1 = target_corr_df[target_corr_df['Feature'] == var1]['Abs Correlation'].values
                            d49_corr_var2 = target_corr_df[target_corr_df['Feature'] == var2]['Abs Correlation'].values
                            
                            d49_corr_var1 = d49_corr_var1[0] if len(d49_corr_var1) > 0 else 0
                            d49_corr_var2 = d49_corr_var2[0] if len(d49_corr_var2) > 0 else 0
                            
                            # Keep the one with HIGHER D49 correlation
                            if d49_corr_var1 >= d49_corr_var2:
                                keep_var = var1
                                drop_var = var2
                                keep_missing = missing_var1
                                drop_missing = missing_var2
                                keep_d49_corr = d49_corr_var1
                                drop_d49_corr = d49_corr_var2
                            else:
                                keep_var = var2
                                drop_var = var1
                                keep_missing = missing_var2
                                drop_missing = missing_var1
                                keep_d49_corr = d49_corr_var2
                                drop_d49_corr = d49_corr_var1
                        else:
                            # Fallback to old logic if D49 not available
                            if missing_var1 <= missing_var2:
                                keep_var = var1
                                drop_var = var2
                                keep_missing = missing_var1
                                drop_missing = missing_var2
                            else:
                                keep_var = var2
                                drop_var = var1
                                keep_missing = missing_var2
                                drop_missing = missing_var1
                            keep_d49_corr = 0
                            drop_d49_corr = 0
                        
                        redundant_pairs.append({
                            'Variable 1': var1,
                            'Variable 2': var2,
                            'Correlation': corr_value,
                            'Abs Correlation': abs(corr_value),
                            'Var1 Missing': missing_var1,
                            'Var2 Missing': missing_var2,
                            'Var1 D49 Corr': keep_d49_corr if keep_var == var1 else drop_d49_corr,
                            'Var2 D49 Corr': keep_d49_corr if keep_var == var2 else drop_d49_corr,
                            'Recommended to Keep': keep_var,
                            'Recommended to Drop': drop_var,
                            'Reason': f"Higher D49 corr ({keep_d49_corr:.3f} vs {drop_d49_corr:.3f})" if not target_corr_df.empty else f"Fewer NaNs ({keep_missing} vs {drop_missing})",
                            'Description Keep': var_descriptions.get(keep_var, 'N/A'),
                            'Description Drop': var_descriptions.get(drop_var, 'N/A')
                        })
            
            if redundant_pairs:
                redundant_df = pd.DataFrame(redundant_pairs)
                redundant_df = redundant_df.sort_values(by='Abs Correlation', ascending=False)
                
                st.warning(f"⚠️ Found {len(redundant_df)} highly correlated pairs (|r| > 0.90)")
                st.info("💡 **Smart Strategy**: Keeping features with higher D49 correlation to preserve target signal.")
                
                # Show redundant pairs with D49 correlation info
                display_redundant = redundant_df[[
                    'Variable 1', 'Var1 D49 Corr', 'Var1 Missing',
                    'Variable 2', 'Var2 D49 Corr', 'Var2 Missing',
                    'Correlation', 'Recommended to Keep', 'Reason'
                ]].copy()
                st.dataframe(display_redundant, use_container_width=True, hide_index=True)
                
                # Determine columns to drop
                cols_to_drop = list(set(redundant_df['Recommended to Drop'].tolist()))
                
                col_red1, col_red2 = st.columns(2)
                col_red1.metric("Redundant Features to Remove", len(cols_to_drop))
                col_red2.metric("Features Remaining After Removal", len(numeric_cols_raw) - len(cols_to_drop))
                
                # Show detailed information about columns to drop
                with st.expander(f"View {len(cols_to_drop)} features recommended for removal"):
                    drop_info = pd.DataFrame({
                        'Feature': cols_to_drop,
                        'Description': [var_descriptions.get(col, 'N/A') for col in cols_to_drop],
                        'Missing Values': [df_filtered[col].isnull().sum() for col in cols_to_drop]
                    })
                    st.dataframe(drop_info, use_container_width=True, hide_index=True)
                
                # Button to apply redundancy removal
                col_btn1, col_btn2 = st.columns([1, 1])
                with col_btn1:
                    if st.button("✂️ Remove Redundant Features", type="primary"):
                        # Store columns to drop in session state
                        st.session_state.removed_redundant_cols = cols_to_drop
                        st.rerun()
                with col_btn2:
                    if st.session_state.removed_redundant_cols:
                        if st.button("🔄 Reset Redundancy Removal"):
                            st.session_state.removed_redundant_cols = []
                            st.rerun()
            else:
                st.success("✅ No highly correlated pairs (|r| > 0.90) found. Data has low redundancy.")
        else:
            # Show summary of what was removed
            st.success(f"✅ Redundancy removal applied: {len(st.session_state.removed_redundant_cols)} features removed")
            
            with st.expander(f"View removed features"):
                removed_info = pd.DataFrame({
                    'Removed Feature': st.session_state.removed_redundant_cols,
                    'Description': [var_descriptions.get(col, 'N/A') for col in st.session_state.removed_redundant_cols]
                })
                st.dataframe(removed_info, use_container_width=True, hide_index=True)
            
            if st.button("🔄 Reset Redundancy Removal"):
                st.session_state.removed_redundant_cols = []
                st.rerun()
    else:
        st.warning("Correlation analysis requires at least 2 numeric columns.")
    
    st.divider()

    # --- PHASE 2: THE FORK IN THE ROAD ---
    st.header("🔀 Phase 2: The Fork in the Road")
    st.write("""
    PCA and EBM handle data differently:
    - **Path A (PCA)**: Requires complete data → Use Iterative Imputation to preserve variance
    - **Path B (EBM)**: Handles NaNs natively → Skip imputation to let model learn from missing patterns
    """)
    
    # Create tabs for two paths
    path_a_tab, path_b_tab = st.tabs(["📈 Path A: PCA (Visualization)", "🤖 Path B: EBM (Prediction)"])
    
    with path_a_tab:
        st.subheader("🔧 Path A - Step 1: Iterative Imputation")
        st.write("Iterative imputation preserves relationships between variables, leading to better PCA variance explained.")
    
        # Calculate remaining missing values in filtered dataset
        remaining_missing = df_filtered.isnull().sum().sum()
        remaining_missing_pct = (remaining_missing / df_filtered.size) * 100
        
        st.write(f"Remaining missing values in filtered dataset: **{remaining_missing:,}** ({remaining_missing_pct:.2f}%)")
        
        if remaining_missing > 0:
            # Imputation method selection - default to Iterative
            imputation_method = st.radio(
                "Select imputation method:",
                ["Iterative (Recommended)", "Mean", "Median"],
                help="Iterative: Preserves relationships between features for better variance. Mean/Median: Faster but may reduce variance."
            )
        
            if st.button("Apply Imputation for PCA Path", type="primary"):
                with st.spinner("Applying imputation..."):
                    # Separate numeric and non-numeric columns
                    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
                    non_numeric_cols = df_filtered.select_dtypes(exclude=[np.number]).columns.tolist()
                    
                    df_imputed = df_filtered.copy()
                    
                    if numeric_cols:
                        if imputation_method == "Mean":
                            imputer = SimpleImputer(strategy='mean')
                            df_imputed[numeric_cols] = imputer.fit_transform(df_filtered[numeric_cols])
                        elif imputation_method == "Median":
                            imputer = SimpleImputer(strategy='median')
                            df_imputed[numeric_cols] = imputer.fit_transform(df_filtered[numeric_cols])
                        else:  # Iterative
                            imputer = IterativeImputer(random_state=42, max_iter=10)
                            df_imputed[numeric_cols] = imputer.fit_transform(df_filtered[numeric_cols])
                    
                    # For non-numeric columns, use most frequent
                    if non_numeric_cols:
                        imputer_non_numeric = SimpleImputer(strategy='most_frequent')
                        df_imputed[non_numeric_cols] = imputer_non_numeric.fit_transform(df_filtered[non_numeric_cols])
                    
                    st.success(f"✅ Imputation complete using {imputation_method} method!")
                
                    # Show comparison metrics
                    col_x, col_y = st.columns(2)
                    col_x.metric("Missing Values Before", f"{remaining_missing:,}")
                    col_y.metric("Missing Values After", f"{df_imputed.isnull().sum().sum():,}")
                
                    # --- VARIANCE ANALYSIS ---
                    st.subheader("📈 Variance Analysis of Numeric Columns")
                
                    if numeric_cols:
                        variance_df = pd.DataFrame({
                            'Column Name': numeric_cols,
                            'Description': [var_descriptions.get(col, 'N/A') for col in numeric_cols],
                            'Variance (Before)': [df_filtered[col].var() for col in numeric_cols],
                            'Variance (After)': [df_imputed[col].var() for col in numeric_cols]
                        })
                        
                        variance_df['Variance Change'] = variance_df['Variance (After)'] - variance_df['Variance (Before)']
                        variance_df['% Change'] = ((variance_df['Variance (After)'] - variance_df['Variance (Before)']) / 
                                                   variance_df['Variance (Before)']) * 100
                        
                        # Sort by variance (after imputation)
                        variance_df = variance_df.sort_values(by='Variance (After)', ascending=False)
                        
                        st.dataframe(variance_df, use_container_width=True)
                    else:
                        st.info("No numeric columns found for variance analysis.")
                
                    # Show imputed dataset
                    st.subheader("📊 Imputed Dataset")
                    st.write(f"**Dataset after imputation** ({len(df_imputed)} rows × {len(df_imputed.columns)} columns)")
                    st.dataframe(df_imputed, use_container_width=True)
                    
                    st.divider()
                    
                    # --- PCA ANALYSIS ---
                    st.subheader("🔬 Path A - Step 2: Principal Component Analysis")
                    st.write("PCA with iterative imputation preserves variance and reveals patterns better than mean/median imputation.")
                
                    if numeric_cols and len(numeric_cols) > 1:
                        st.success("✨ PCA on cleaned data (smart redundancy removal + iterative imputation) for optimal variance")
                        # Standardize the data
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(df_imputed[numeric_cols])
                    
                        # Apply PCA
                        n_components = min(10, len(numeric_cols), len(df_imputed))
                        pca = PCA(n_components=n_components)
                        X_pca = pca.fit_transform(X_scaled)
                    
                        # Create DataFrame with principal components
                        pca_df = pd.DataFrame(
                            X_pca,
                            columns=[f'PC{i+1}' for i in range(n_components)]
                        )
                        
                        # Add target variable if available
                        if 'D49' in df_imputed.columns:
                            pca_df['D49_Target'] = df_imputed['D49'].values
                    
                        # Chart 1: Explained Variance Ratio (Scree Plot)
                        col_pca1, col_pca2 = st.columns(2)
                    
                        with col_pca1:
                            st.write("**Explained Variance by Component**")
                            variance_data = pd.DataFrame({
                                'Component': [f'PC{i+1}' for i in range(n_components)],
                                'Explained Variance (%)': pca.explained_variance_ratio_ * 100,
                                'Cumulative Variance (%)': np.cumsum(pca.explained_variance_ratio_) * 100
                            })
                        
                        fig_variance = go.Figure()
                        
                        # Bar chart for individual variance
                        fig_variance.add_trace(go.Bar(
                            x=variance_data['Component'],
                            y=variance_data['Explained Variance (%)'],
                            name='Individual',
                            marker_color='steelblue'
                        ))
                        
                        # Line chart for cumulative variance
                        fig_variance.add_trace(go.Scatter(
                            x=variance_data['Component'],
                            y=variance_data['Cumulative Variance (%)'],
                            name='Cumulative',
                            mode='lines+markers',
                            marker=dict(color='red', size=8),
                            line=dict(color='red', width=2),
                            yaxis='y2'
                        ))
                        
                        fig_variance.update_layout(
                            title="Scree Plot - Variance Explained",
                            xaxis_title="Principal Component",
                            yaxis_title="Individual Variance (%)",
                            yaxis2=dict(
                                title="Cumulative Variance (%)",
                                overlaying='y',
                                side='right'
                            ),
                            height=400,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_variance, use_container_width=True)
                        
                        # Show variance table
                        st.dataframe(variance_data, use_container_width=True, hide_index=True)
                    
                        with col_pca2:
                            st.write("**PCA 2D Projection (First Two Components)**")
                            
                            # Chart 2: Scatter plot of first two principal components
                            if 'D49_Target' in pca_df.columns:
                                fig_pca_scatter = px.scatter(
                                    pca_df,
                                    x='PC1',
                                    y='PC2',
                                    color='D49_Target',
                                    title="Samples in PC1-PC2 Space",
                                    labels={
                                        'PC1': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                                        'PC2': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                                        'D49_Target': 'D49 Yield'
                                    },
                                    color_continuous_scale='Viridis',
                                    height=400
                                )
                            else:
                                fig_pca_scatter = px.scatter(
                                    pca_df,
                                    x='PC1',
                                    y='PC2',
                                    title="Samples in PC1-PC2 Space",
                                    labels={
                                        'PC1': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                                        'PC2': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'
                                    },
                                    height=400
                                )
                            
                            fig_pca_scatter.update_traces(marker=dict(size=8))
                            st.plotly_chart(fig_pca_scatter, use_container_width=True)
                            
                            # Summary metrics
                            st.metric(
                                "Total Variance Explained (First 2 PCs)",
                                f"{sum(pca.explained_variance_ratio_[:2])*100:.2f}%"
                            )
                            st.info(f"ℹ️ The first two components capture {sum(pca.explained_variance_ratio_[:2])*100:.1f}% of the total variance in {len(numeric_cols)} features.")
                    else:
                        st.warning("PCA requires at least 2 numeric columns and sufficient data.")
                
                st.divider()
                
                # Download button for imputed dataset
                csv = df_imputed.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Imputed Dataset as CSV",
                    data=csv,
                    file_name="fractionation_data_imputed.csv",
                    mime="text/csv"
                )
        else:
            st.success("✅ No missing values to impute!")

else:
    st.info("💡 Please place your `fractionation_data.csv` file in the same folder as this script.")