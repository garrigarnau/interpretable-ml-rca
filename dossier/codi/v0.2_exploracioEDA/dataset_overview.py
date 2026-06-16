"""
Dataset Overview Module
Contains functions for displaying dataset overview metrics and target variable visualizations.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def display_overview_metrics(basic_stats):
    """
    Display basic dataset overview metrics.
    
    Parameters:
    -----------
    basic_stats : dict
        Dictionary containing basic statistics (from get_basic_stats)
    """
    st.header("📈 Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Batches", basic_stats['total_rows'])
    col2.metric("Total Columns", basic_stats['total_columns'])
    col3.metric("Overall Missing Data", f"{basic_stats['missing_percentage']:.2f}%")
    
    st.divider()


def display_target_variable_visualization(df, target_col='D49', var_descriptions=None):
    """
    Display comprehensive visualization and statistics for the target variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset
    target_col : str, optional
        Name of the target column (default: 'D49')
    var_descriptions : dict, optional
        Dictionary mapping column names to descriptions
    """
    st.subheader(f"🎯 Target Variable: {target_col} (FII+III PPT Yield) Across Batches")
    
    if target_col not in df.columns:
        st.warning(f"⚠️ Target variable '{target_col}' not found in the dataset.")
        return
    
    # Get target description
    if var_descriptions:
        target_desc = var_descriptions.get(target_col, 'FII+III PPT yield')
        st.write(f"**{target_desc}**")
    
    # Create batch index for x-axis
    df_with_index = df.copy()
    df_with_index['Batch_Index'] = range(1, len(df_with_index) + 1)
    
    # Calculate and display statistics
    _display_target_statistics(df, target_col)
    
    # Create and display main plot
    _create_target_line_plot(df_with_index, target_col)
    
    # Create and display distribution plots
    _create_distribution_plots(df_with_index, target_col)
    
    st.divider()


def _display_target_statistics(df, target_col):
    """
    Display statistical metrics for the target variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset
    target_col : str
        Name of the target column
    """
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    target_data = df[target_col].dropna()
    
    if len(target_data) > 0:
        col_t1.metric("Mean Yield", f"{target_data.mean():.2f}")
        col_t2.metric("Median Yield", f"{target_data.median():.2f}")
        col_t3.metric("Std Dev", f"{target_data.std():.2f}")
        col_t4.metric(
            "Missing Values",
            f"{df[target_col].isnull().sum()} ({(df[target_col].isnull().sum()/len(df)*100):.1f}%)"
        )


def _create_target_line_plot(df_with_index, target_col):
    """
    Create an interactive line plot showing target variable across batches.
    
    Parameters:
    -----------
    df_with_index : pd.DataFrame
        Dataset with Batch_Index column
    target_col : str
        Name of the target column
    """
    target_data = df_with_index[target_col].dropna()
    
    # Create interactive plot with Plotly
    fig = go.Figure()
    
    # Add scatter plot with lines
    fig.add_trace(go.Scatter(
        x=df_with_index['Batch_Index'],
        y=df_with_index[target_col],
        mode='lines+markers',
        name=f'{target_col} Yield',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6, color='#1f77b4'),
        hovertemplate='<b>Batch %{x}</b><br>Yield: %{y:.2f}<extra></extra>'
    ))
    
    # Add mean line
    if len(target_data) > 0:
        fig.add_hline(
            y=target_data.mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {target_data.mean():.2f}",
            annotation_position="right"
        )
    
    fig.update_layout(
        title=f"{target_col} (FII+III PPT Yield) Distribution Across Batches",
        xaxis_title="Batch Number",
        yaxis_title="Yield Value",
        height=500,
        hovermode='x unified',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _create_distribution_plots(df_with_index, target_col):
    """
    Create histogram and box plot for target variable distribution.
    
    Parameters:
    -----------
    df_with_index : pd.DataFrame
        Dataset with Batch_Index column
    target_col : str
        Name of the target column
    """
    col_hist1, col_hist2 = st.columns(2)
    
    with col_hist1:
        fig_hist = px.histogram(
            df_with_index,
            x=target_col,
            nbins=30,
            title=f"{target_col} Value Distribution",
            labels={target_col: 'Yield Value', 'count': 'Frequency'}
        )
        fig_hist.update_layout(height=350)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col_hist2:
        fig_box = px.box(
            df_with_index,
            y=target_col,
            title=f"{target_col} Box Plot (Outlier Detection)",
            labels={target_col: 'Yield Value'}
        )
        fig_box.update_layout(height=350)
        st.plotly_chart(fig_box, use_container_width=True)


def display_target_by_quality(df, target_col='D49', quality_col='Batch_Quality', time_col='B1'):
    """
    Display target variable across batches colored by quality classification.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset (must include quality_col)
    target_col : str, optional
        Name of the target column (default: 'D49')
    quality_col : str, optional
        Name of the quality classification column (default: 'Batch_Quality')
    time_col : str, optional
        Name of the timestamp column (default: 'B1')
    """
    if quality_col not in df.columns:
        st.warning(f"⚠️ Quality column '{quality_col}' not found in the dataset.")
        return
    
    if target_col not in df.columns:
        st.warning(f"⚠️ Target variable '{target_col}' not found in the dataset.")
        return
    
    # Create batch index for x-axis
    df_with_index = df.copy()
    df_with_index['Batch_Index'] = range(1, len(df_with_index) + 1)
    
    # Check if time column exists and has valid data
    has_time_data = False
    if time_col in df.columns:
        # Try to convert to datetime if not already
        try:
            df_with_index[time_col] = pd.to_datetime(df_with_index[time_col])
            has_time_data = df_with_index[time_col].notna().sum() > 0
        except:
            has_time_data = False
    
    # Determine which x-axis to use
    use_time_axis = has_time_data
    x_col = time_col if use_time_axis else 'Batch_Index'
    
    # Define color mapping
    color_map = {
        'Good': '#28a745',
        'Average': '#ffc107',
        'Bad': '#dc3545',
        'Unknown': '#6c757d'
    }
    
    # Create interactive plot with Plotly
    fig = go.Figure()
    
    # Add traces for each quality category
    for quality in ['Good', 'Average', 'Bad', 'Unknown']:
        df_quality = df_with_index[df_with_index[quality_col] == quality]
        
        if len(df_quality) > 0:
            # Build hover template based on whether we have time data
            if use_time_axis:
                hover_template = '<b>Batch %{customdata}</b><br>Start Time: %{x}<br>Yield: %{y:.2f}<br>Quality: ' + quality + '<extra></extra>'
                customdata = df_quality['Batch_Index']
            else:
                hover_template = f'<b>Batch %{{x}}</b><br>Yield: %{{y:.2f}}<br>Quality: {quality}<extra></extra>'
                customdata = None
            
            fig.add_trace(go.Scatter(
                x=df_quality[x_col],
                y=df_quality[target_col],
                mode='markers',
                name=quality,
                marker=dict(
                    size=8,
                    color=color_map.get(quality, '#6c757d'),
                    line=dict(width=1, color='white')
                ),
                customdata=customdata,
                hovertemplate=hover_template
            ))
    
    # Add connecting line (all points, without quality distinction)
    fig.add_trace(go.Scatter(
        x=df_with_index[x_col],
        y=df_with_index[target_col],
        mode='lines',
        name='Trend',
        line=dict(color='lightgray', width=1),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Calculate mean for reference line
    target_data = df_with_index[target_col].dropna()
    if len(target_data) > 0:
        fig.add_hline(
            y=target_data.mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {target_data.mean():.2f}",
            annotation_position="right"
        )
    
    # Set x-axis title based on data type
    if use_time_axis:
        x_axis_title = "Start Time (B1)"
    else:
        x_axis_title = "Batch Number"
    
    fig.update_layout(
        title=f"{target_col} Distribution Over Time (Colored by Batch Quality)" if use_time_axis else f"{target_col} Distribution Across Batches (Colored by Batch Quality)",
        xaxis_title=x_axis_title,
        yaxis_title="Yield Value",
        height=500,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add note if time data is available
    if use_time_axis:
        st.caption(f"ℹ️ Timeline based on {time_col} (Start Time). Hover over points to see batch number.")
    
    st.plotly_chart(fig, use_container_width=True)


def display_missing_data_timeline(df, columns_to_analyze, time_col='B1', var_descriptions=None):
    """
    Display missing data patterns over time for specified columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset
    columns_to_analyze : list
        List of column names to analyze for missing patterns
    time_col : str, optional
        Name of the timestamp column (default: 'B1')
    var_descriptions : dict, optional
        Dictionary mapping column names to descriptions
    """
    if not columns_to_analyze:
        st.info("No columns to analyze for missing data timeline.")
        return
    
    if time_col not in df.columns:
        st.warning(f"⚠️ Time column '{time_col}' not found. Cannot create timeline.")
        return
    
    # Create a copy with batch index and time
    df_timeline = df.copy()
    df_timeline['Batch_Index'] = range(1, len(df_timeline) + 1)
    
    # Try to convert time column to datetime
    has_time_data = False
    try:
        df_timeline[time_col] = pd.to_datetime(df_timeline[time_col])
        has_time_data = df_timeline[time_col].notna().sum() > 0
    except:
        has_time_data = False
    
    # Determine x-axis
    use_time_axis = has_time_data
    x_col = time_col if use_time_axis else 'Batch_Index'
    
    # Filter to only existing columns
    columns_to_show = [col for col in columns_to_analyze if col in df.columns]
    
    if not columns_to_show:
        st.info("No valid columns found to display missing data timeline.")
        return
    
    st.write(f"**Missing Data Timeline for {len(columns_to_show)} Columns**")
    
    # Create missing data matrix (1 = missing, 0 = present)
    missing_matrix = df_timeline[columns_to_show].isnull().astype(int)
    
    # Add descriptive names if available
    if var_descriptions:
        column_labels = [f"{col} - {var_descriptions.get(col, 'N/A')[:50]}" for col in columns_to_show]
    else:
        column_labels = columns_to_show
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=missing_matrix.T.values,
        x=df_timeline[x_col],
        y=column_labels,
        colorscale=[[0, '#2ecc71'], [1, '#e74c3c']],  # Green for present, red for missing
        hovertemplate='<b>%{y}</b><br>' + 
                      ('Time: %{x}<br>' if use_time_axis else 'Batch: %{x}<br>') +
                      'Status: %{customdata}<extra></extra>',
        customdata=[['Missing' if val == 1 else 'Present' for val in row] for row in missing_matrix.T.values],
        showscale=True,
        colorbar=dict(
            title="Status",
            tickvals=[0.25, 0.75],
            ticktext=["Present", "Missing"],
            len=0.3,
            y=0.85
        )
    ))
    
    # Update layout
    x_axis_title = "Start Time (B1)" if use_time_axis else "Batch Number"
    title_suffix = "Over Time" if use_time_axis else "Across Batches"
    
    fig.update_layout(
        title=f"Missing Data Pattern {title_suffix} (Columns with >30% Missing)",
        xaxis_title=x_axis_title,
        yaxis_title="Column Name",
        height=max(400, len(columns_to_show) * 20),  # Dynamic height based on number of columns
        yaxis=dict(tickfont=dict(size=10)),
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add summary statistics
    col_ms1, col_ms2, col_ms3 = st.columns(3)
    
    total_cells = len(df_timeline) * len(columns_to_show)
    missing_cells = missing_matrix.sum().sum()
    
    col_ms1.metric("Total Columns Shown", len(columns_to_show))
    col_ms2.metric("Total Missing Cells", f"{missing_cells:,}")
    col_ms3.metric("Overall Missing %", f"{(missing_cells / total_cells * 100):.1f}%")
    
    if use_time_axis:
        st.caption(f"ℹ️ Timeline based on {time_col} (Start Time). Red indicates missing data, green indicates present data.")
