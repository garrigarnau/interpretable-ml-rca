"""
Data Cleaning and Massaging Module
This module contains functions for loading, cleaning, and preprocessing the fractionation dataset.
"""

import pandas as pd
import json
import streamlit as st


@st.cache_data
def load_data(file_path="data/fractionation_data.csv", cache_buster=None):
    """
    Load the fractionation dataset from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        cache_buster: Optional value used to invalidate Streamlit cache when source file changes
        
    Returns:
        pd.DataFrame or None: Loaded dataframe or None if file not found
    """
    try:
        df = pd.read_csv(file_path)

        # Force known categorical columns to object dtype even if encoded numerically
        forced_categorical = (
            get_equipment_columns()
            + get_lot_number_columns()
            + get_manual_categorical_columns()
        )
        for col in forced_categorical:
            if col in df.columns:
                df[col] = df[col].astype('object')

        return df
    except FileNotFoundError:
        st.error(f"File '{file_path}' not found. Please ensure it is in the correct directory.")
        return None


@st.cache_data
def load_variable_descriptions(file_path="data/variable_descriptions.json"):
    """
    Load variable descriptions from JSON file.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        dict: Dictionary of variable descriptions or empty dict if file not found
    """
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning("Variable descriptions file not found.")
        return {}


def classify_batch_quality(df, target_col='D49'):
    """
    Create Batch_Quality feature based on target variable (D49) values.
    
    Classification thresholds:
    - Good: > median + 0.5×std
    - Average: within ±0.5×std of median
    - Bad: < median - 0.5×std
    - Unknown: missing target value
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of the target column (default: 'D49')
        
    Returns:
        tuple: (df with Batch_Quality column, thresholds dict, quality counts)
    """
    if target_col not in df.columns:
        st.warning(f"⚠️ {target_col} column not found. Cannot create Batch_Quality feature.")
        return df, None, None
    
    # Calculate thresholds
    median_yield = df[target_col].median()
    std_yield = df[target_col].std()
    
    threshold_good = median_yield + 0.5 * std_yield
    threshold_bad = median_yield - 0.5 * std_yield
    
    thresholds = {
        'median': median_yield,
        'std': std_yield,
        'good': threshold_good,
        'bad': threshold_bad
    }
    
    # Classification function
    def classify(yield_value):
        if pd.isna(yield_value):
            return 'Unknown'
        elif yield_value > threshold_good:
            return 'Good'
        elif yield_value < threshold_bad:
            return 'Bad'
        else:
            return 'Average'
    
    # Create new column
    df_copy = df.copy()
    df_copy['Batch_Quality'] = df_copy[target_col].apply(classify)
    
    # Get quality counts
    quality_counts = df_copy['Batch_Quality'].value_counts()
    
    return df_copy, thresholds, quality_counts


def get_basic_stats(df):
    """
    Calculate basic dataset statistics.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Dictionary containing basic statistics
    """
    import numpy as np
    
    stats = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(exclude=[np.number]).columns),
        'total_cells': df.size,
        'total_missing': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / df.size) * 100
    }
    
    return stats


def get_column_type_analysis(df, var_descriptions):
    """
    Analyze and categorize columns by type (numeric vs categorical).
    
    Args:
        df (pd.DataFrame): Input dataframe
        var_descriptions (dict): Dictionary of variable descriptions
        
    Returns:
        tuple: (numeric_summary_df, categorical_summary_df)
    """
    import numpy as np
    
    # Get columns that should be treated as categorical
    forced_categorical = get_equipment_columns() + get_lot_number_columns() + get_manual_categorical_columns()
    forced_categorical = [col for col in forced_categorical if col in df.columns]
    
    # Get column lists - but exclude forced categorical from numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in forced_categorical]
    
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    # Add forced categorical columns
    categorical_cols = list(set(categorical_cols + forced_categorical))
    
    # Numeric columns summary
    if numeric_cols:
        numeric_summary = pd.DataFrame({
            'Column': numeric_cols,
            'Description': [var_descriptions.get(col, 'N/A') for col in numeric_cols],
            'Missing': [df[col].isnull().sum() for col in numeric_cols],
            'Mean': [df[col].mean() for col in numeric_cols],
            'Std': [df[col].std() for col in numeric_cols]
        })
    else:
        numeric_summary = pd.DataFrame()
    
    # Categorical columns summary
    if categorical_cols:
        categorical_summary = pd.DataFrame({
            'Column': categorical_cols,
            'Description': [var_descriptions.get(col, 'N/A') for col in categorical_cols],
            'Missing': [df[col].isnull().sum() for col in categorical_cols],
            'Unique Values': [df[col].nunique() for col in categorical_cols]
        })
    else:
        categorical_summary = pd.DataFrame()
    
    return numeric_summary, categorical_summary


def get_manual_categorical_columns():
    """
    Return manually forced categorical columns that may be encoded numerically in CSV.

    Returns:
        list: List of manually forced categorical column IDs
    """
    return ['CRATES_STAGING', 'CREATES_STAGING']


def get_missing_summary(df, var_descriptions):
    """
    Create a detailed summary of missing values for all columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        var_descriptions (dict): Dictionary of variable descriptions
        
    Returns:
        pd.DataFrame: Summary of missing values sorted by missing count
    """
    missing_df = df.isnull().sum().reset_index()
    missing_df.columns = ['Column Name', 'Missing Count']
    missing_df['% Missing'] = (missing_df['Missing Count'] / len(df)) * 100
    
    # Add variable descriptions
    missing_df['Description'] = missing_df['Column Name'].map(var_descriptions)
    
    # Reorder columns
    missing_df = missing_df[['Column Name', 'Description', 'Missing Count', '% Missing']]
    
    # Sort by most missing
    missing_df = missing_df.sort_values(by='Missing Count', ascending=False)
    
    return missing_df


def filter_columns_by_missing_threshold(df, threshold=30):
    """
    Filter out columns with missing values above a certain threshold.
    
    Args:
        df (pd.DataFrame): Input dataframe
        threshold (float): Maximum percentage of missing values allowed (default: 30)
        
    Returns:
        tuple: (filtered_df, columns_kept, columns_removed, missing_pct_per_col)
    """
    # Calculate missing percentage for each column
    missing_pct_per_col = (df.isnull().sum() / len(df)) * 100
    
    # Find columns to keep (<=threshold% missing)
    cols_to_keep = missing_pct_per_col[missing_pct_per_col <= threshold].index.tolist()
    
    # Find columns that were removed (>threshold% missing)
    cols_removed = missing_pct_per_col[missing_pct_per_col > threshold].index.tolist()
    
    # Create filtered dataframe
    df_filtered = df[cols_to_keep].copy()
    
    return df_filtered, cols_to_keep, cols_removed, missing_pct_per_col


def get_equipment_columns():
    """
    Return list of equipment column IDs.
    
    These columns represent equipment choices that should be treated as categorical:
    - B10-B13: BKA used (1st through 4th bowl)
    - C1: Tank used for FI (receiving tank)
    - C31: BKA35 used
    - D1: Tank used for FII+III
    - D24: Filter press used
    - D25: Sep line used for filtration
    - D46: Tank used for FIV1+IV4
    
    Returns:
        list: List of equipment column IDs
    """
    return ['B10', 'B11', 'B12', 'B13', 'C1', 'C31', 'D1', 'D24', 'D25', 'D46']


def get_lot_number_columns():
    """
    Return list of lot number column IDs.
    
    These columns represent lot numbers that should be treated as categorical:
    - C6: 4.0 M NaCl buffer lot number
    - C10: pH 4.0 SAAA buffer lot number
    - C17, C19: 95% EtOH (1st and 2nd) lot numbers
    - D3: pH 4.0 SAAA buffer lot number
    - D11, D13: 95% EtOH (1st and 2nd) lot numbers
    - D27, D29: Thin paper (1st and 2nd) lot numbers
    - D32, D34: Thick paper (1st and 2nd) lot numbers
    - D52: 20% EtOH used for postwash lot number
    - D54: FII+III PPT lot number
    - D20c: pH 4.0 SAAA Lot Number (Alcohol adjustment)
    - D36b: Retitration pH 4.0 SAAA Lot Number
    
    Returns:
        list: List of lot number column IDs
    """
    return ['C6', 'C10', 'C17', 'C19', 'D3', 'D11', 'D13', 'D27', 'D29', 
            'D32', 'D34', 'D52', 'D54', 'D20c', 'D36b']


def analyze_categorical_columns(df, column_list, var_descriptions):
    """
    Analyze categorical columns and return summary information.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column_list (list): List of column IDs to analyze
        var_descriptions (dict): Dictionary of variable descriptions
        
    Returns:
        pd.DataFrame: Summary dataframe with column info, unique values, etc.
    """
    # Filter to existing columns
    existing_cols = [col for col in column_list if col in df.columns]
    
    if not existing_cols:
        return pd.DataFrame()
    
    # Create summary data
    summary_data = []
    for col in existing_cols:
        unique_vals = df[col].dropna().unique()
        n_unique = len(unique_vals)
        non_null = df[col].notna().sum()
        desc = var_descriptions.get(col, 'N/A')
        
        # Sort values if possible
        try:
            unique_vals_sorted = sorted(unique_vals)
        except:
            unique_vals_sorted = list(unique_vals)
        
        summary_data.append({
            'Column': col,
            'Description': desc,
            'Non-Null': non_null,
            'Unique Values': n_unique,
            'Values': str(unique_vals_sorted)[:100] + '...' if len(str(unique_vals_sorted)) > 100 else str(unique_vals_sorted)
        })
    
    return pd.DataFrame(summary_data)


def get_categorical_unique_values(df, column_list):
    """
    Get detailed unique values for each categorical column.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column_list (list): List of column IDs
        
    Returns:
        dict: Dictionary mapping column IDs to their unique values
    """
    existing_cols = [col for col in column_list if col in df.columns]
    
    unique_values_dict = {}
    for col in existing_cols:
        unique_vals = df[col].dropna().unique()
        try:
            unique_values_dict[col] = sorted(unique_vals)
        except:
            unique_values_dict[col] = list(unique_vals)
    
    return unique_values_dict


def get_time_columns():
    """
    Return list of time-related column IDs (start/end times for all phases).
    
    These columns represent timestamps that may be removed for modeling:
    - Phase B: Thawing & Cryo Centrifuging
      B1, B2: Start/End plasma thawing at -10C
      B5, B6: Start/End pooling
      B9, B22: Start/End cryo centing
    - Phase C: Fraction I (FI)
      C2, C3: Start/End plasma receiving
      C15: Start buffer addition
      C22, C23: Start/End EtOH addition
      C32, C34: Start/End FI centing
    - Phase D: Fraction II+III (FII+III)
      D8: Start buffer addition
      D17, D18: Start/End EtOH addition
      D37, D39: Start/End filtration
      D41, D44: Start/End blowdry
    
    Returns:
        list: List of time column IDs
    """
    return [
        'B1', 'B2', 'B5', 'B6', 'B9', 'B22',  # Phase B
        'C2', 'C3', 'C15', 'C22', 'C23', 'C32', 'C34',  # Phase C
        'D8', 'D17', 'D18', 'D37', 'D39', 'D41', 'D44'  # Phase D
    ]


def create_column_selection_dataframe(df, var_descriptions, exclude_cols=None):
    """
    Create a comprehensive dataframe for column selection interface.
    
    Args:
        df (pd.DataFrame): Input dataframe
        var_descriptions (dict): Dictionary of variable descriptions
        exclude_cols (list): List of columns to exclude from table (default: ['D49', 'Batch_Quality'])
        
    Returns:
        pd.DataFrame: DataFrame with column information for selection
    """
    if exclude_cols is None:
        exclude_cols = ['D49', 'Batch_Quality']
    
    # Get time columns list
    time_columns = get_time_columns()
    
    # Get all columns except excluded ones
    all_cols_for_selection = [col for col in df.columns if col not in exclude_cols]
    
    # Create selection dataframe
    selection_df = pd.DataFrame({
        'Column': all_cols_for_selection,
        'Description': [var_descriptions.get(col, 'N/A')[:100] for col in all_cols_for_selection],
        '% Missing': [(df[col].isnull().sum() / len(df)) * 100 for col in all_cols_for_selection],
        'Missing Count': [df[col].isnull().sum() for col in all_cols_for_selection],
        'Non-Null Count': [df[col].notna().sum() for col in all_cols_for_selection],
        'Is Time Column': ['✓' if col in time_columns else '' for col in all_cols_for_selection]
    })
    
    # Sort by % missing (descending)
    selection_df = selection_df.sort_values(by='% Missing', ascending=False)
    
    return selection_df


def get_columns_above_missing_threshold(df, threshold=30):
    """
    Get list of columns with missing values above a certain threshold.
    
    Args:
        df (pd.DataFrame): Input dataframe
        threshold (float): Missing value percentage threshold (default: 30)
        
    Returns:
        list: List of column names above threshold
    """
    missing_pct_per_col = (df.isnull().sum() / len(df)) * 100
    return missing_pct_per_col[missing_pct_per_col > threshold].index.tolist()


def apply_column_removal(df, columns_to_remove):
    """
    Remove specified columns from dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns_to_remove (list): List of column names to remove
        
    Returns:
        pd.DataFrame: Filtered dataframe with columns removed
    """
    # Filter to only existing columns
    existing_cols_to_remove = [col for col in columns_to_remove if col in df.columns]
    
    if not existing_cols_to_remove:
        return df.copy()
    
    return df.drop(columns=existing_cols_to_remove)


def get_time_columns_in_dataframe(df):
    """
    Get list of time columns that actually exist in the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        list: List of time column names present in dataframe
    """
    time_columns = get_time_columns()
    return [col for col in time_columns if col in df.columns]


def merge_column_selections(current_selection, new_selection):
    """
    Merge two column selections, removing duplicates.
    
    Args:
        current_selection (list): Current list of selected columns
        new_selection (list): New list of columns to add
        
    Returns:
        list: Merged list without duplicates
    """
    if current_selection is None:
        current_selection = []
    if new_selection is None:
        new_selection = []
    
    return list(set(current_selection + new_selection))
