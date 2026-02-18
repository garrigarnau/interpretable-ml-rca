import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- CONFIGURATION ---
st.set_page_config(page_title="RCA Analysis Platform", layout="wide")

# --- CUSTOM CSS ---
st.markdown(
    """
<style>
    .block-container {padding-top: 1rem; padding-bottom: 2rem;}
    h1 {font-family: 'Helvetica', sans-serif; font-weight: 700; color: #2C3E50;}
    h3 {font-family: 'Helvetica', sans-serif; font-weight: 600; color: #34495E;}
    .stMetric {background-color: #F8F9F9; padding: 10px; border-radius: 5px; border-left: 5px solid #2E86C1;}
    div[data-testid="stSidebar"] {background-color: #F7F9F9;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Fractionation Root Cause Analysis")


# --- UTILS ---
def load_variable_titles():
    """Loads variable descriptions from JSON or returns empty dict."""
    path = os.path.join("data", "variable_descriptions.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


VARIABLE_TITLES = load_variable_titles()


def get_desc(col_name):
    """Returns 'ID: Description' if available, else just 'ID'."""
    return (
        f"{col_name}: {VARIABLE_TITLES[col_name]}"
        if col_name in VARIABLE_TITLES
        else col_name
    )


# --- DATA LOADING ---
@st.cache_data
def load_data():
    path = os.path.join("data", "fractionation_data.csv")
    if not os.path.exists(path):
        st.error(f"Critical Error: Data file not found at {path}")
        return None
    df = pd.read_csv(path)
    if "A1" in df.columns:
        df = df.set_index("A1")
    df["B1"] = pd.to_datetime(df["B1"], errors="coerce")
    return df


df_raw = load_data()

if df_raw is not None:
    # -------------------------------------------------------------------------
    # 1. SCOPE DEFINITION (SIDEBAR)
    # -------------------------------------------------------------------------
    st.sidebar.header("1. Experiment Scope")

    # Defaults
    min_date_avail = df_raw["B1"].min().date()
    max_date_avail = df_raw["B1"].max().date()

    # REMOVED strict min_value/max_value constraints to allow flexible selection
    experiment_dates = st.sidebar.date_input(
        "Experiment Time Range", value=[min_date_avail, max_date_avail]
    )

    # Filter Data (Intersection logic: handles ranges exceeding data automatically)
    if len(experiment_dates) == 2:
        df = df_raw[
            (df_raw["B1"].dt.date >= experiment_dates[0])
            & (df_raw["B1"].dt.date <= experiment_dates[1])
        ].copy()
    else:
        df = df_raw.copy()

    # Feature Selection
    st.sidebar.header("2. Feature Selection")
    all_cols = [c for c in df.columns if c != "B1"]

    target_col = st.sidebar.selectbox(
        "Target Variable",
        [c for c in all_cols if df[c].dtype in ["float64", "int64"]],
        format_func=get_desc,
    )

    potential_nums = [
        c for c in all_cols if df[c].dtype in ["float64", "int64"] and c != target_col
    ]
    selected_num = st.sidebar.multiselect(
        "Numerical Features", potential_nums, max_selections=20, format_func=get_desc
    )

    potential_cats = [
        c
        for c in all_cols
        if df[c].nunique() <= 20 and c != target_col and c not in selected_num
    ]
    selected_cat = st.sidebar.multiselect(
        "Categorical Features", potential_cats, format_func=get_desc
    )

    # -------------------------------------------------------------------------
    # 2. DATA PREVIEW (MAIN SCREEN)
    # -------------------------------------------------------------------------
    st.subheader("Selected Experiment Data")

    display_df = df[["B1", target_col] + selected_num + selected_cat].copy()
    display_df.columns = [get_desc(c) for c in display_df.columns]

    st.dataframe(
        display_df.sort_values(get_desc("B1")), use_container_width=True, height=250
    )
    st.caption(f"Showing {len(df)} available batches.")

    # -------------------------------------------------------------------------
    # 3. PREPROCESSING (INTERNAL)
    # -------------------------------------------------------------------------
    X_scope = df[selected_num + selected_cat].copy()
    if selected_cat:
        X_scope = pd.get_dummies(X_scope, columns=selected_cat, drop_first=True)

    y_scope = df[target_col]

    # Impute
    X_scope = X_scope.fillna(X_scope.median(numeric_only=True))
    y_scope = y_scope.fillna(y_scope.median())

    # Preserve date for splitting
    scope_data = X_scope.copy()
    scope_data["target_y"] = y_scope
    scope_data["date_col"] = df["B1"]

    # -------------------------------------------------------------------------
    # 4. VALIDATION STRATEGY
    # -------------------------------------------------------------------------
    st.divider()
    st.subheader("Validation Strategy")
    val_strategy = st.radio(
        "Choose Method:",
        ["Time-Based Split (Train/Test)", "5-Fold Cross Validation"],
        horizontal=True,
    )

    X_train, X_test, y_train, y_test = None, None, None, None
    is_cv = False

    if val_strategy == "Time-Based Split (Train/Test)":
        c1, c2 = st.columns(2)

        # Determine defaults (use available data as hint, but don't restrict)
        default_max = (
            df["B1"].max().date() if not df.empty else pd.Timestamp.today().date()
        )
        default_min = (
            df["B1"].min().date() if not df.empty else pd.Timestamp.today().date()
        )

        with c1:
            st.markdown("#### 📘 Training Set")
            # REMOVED min_value/max_value constraints
            train_dates = st.date_input(
                "Select Training Period",
                value=[default_min, default_max - pd.Timedelta(days=30)],
                key="train_range",
            )

        with c2:
            st.markdown("#### 📙 Test Set")
            # REMOVED min_value/max_value constraints
            test_dates = st.date_input(
                "Select Testing Period",
                value=[default_max - pd.Timedelta(days=29), default_max],
                key="test_range",
            )

        # Apply Intersection Logic
        if len(train_dates) == 2 and len(test_dates) == 2:
            # Masking automatically handles dates that don't exist in the data
            train_mask = (scope_data["date_col"].dt.date >= train_dates[0]) & (
                scope_data["date_col"].dt.date <= train_dates[1]
            )
            test_mask = (scope_data["date_col"].dt.date >= test_dates[0]) & (
                scope_data["date_col"].dt.date <= test_dates[1]
            )

            train_data = scope_data[train_mask]
            test_data = scope_data[test_mask]

            # Check Overlap
            overlap = train_data.index.intersection(test_data.index)
            if not overlap.empty:
                st.warning(
                    f"⚠️ **Overlap Detected:** {len(overlap)} batches are in BOTH sets. Please adjust dates."
                )

            # Display counts (helpful to see if a range yielded 0 results)
            if len(train_data) == 0:
                st.warning("⚠️ **Note:** Selected Training period contains 0 batches.")
            if len(test_data) == 0:
                st.warning("⚠️ **Note:** Selected Testing period contains 0 batches.")

            if len(train_data) > 0 and len(test_data) > 0:
                st.info(
                    f"**Split Status:** Training on **{len(train_data)}** batches | Testing on **{len(test_data)}** batches"
                )

            X_train = train_data.drop(["target_y", "date_col"], axis=1)
            y_train = train_data["target_y"]
            X_test = test_data.drop(["target_y", "date_col"], axis=1)
            y_test = test_data["target_y"]

    else:
        is_cv = True
        st.info(
            f"Using **5-Fold Cross Validation** on all {len(scope_data)} available batches."
        )
        X_train = scope_data.drop(["target_y", "date_col"], axis=1)
        y_train = scope_data["target_y"]

    # -------------------------------------------------------------------------
    # 5. EXECUTION & RESULTS
    # -------------------------------------------------------------------------
    if st.button("🚀 Run Root Cause Analysis", type="primary"):
        st.divider()

        # Safety Check: Cannot train on empty data
        if not is_cv and (len(X_train) == 0 or len(X_test) == 0):
            st.error(
                "Cannot proceed: One of the datasets (Train or Test) is empty based on the selected dates."
            )
        else:
            model = ExplainableBoostingRegressor()
            metrics_data = []
            plot_data = {}

            # --- FIX: DE-DUPLICATE COLUMNS ---
            X_train = X_train.loc[:, ~X_train.columns.duplicated()]

            if not is_cv:
                # Align columns
                X_test = X_test.loc[:, ~X_test.columns.duplicated()]
                X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

                with st.spinner("Training Model..."):
                    model.fit(X_train, y_train)
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)

                # Metrics
                metrics_data.append(
                    {
                        "Dataset": "Training Set",
                        "R²": r2_score(y_train, y_pred_train),
                        "MAE": mean_absolute_error(y_train, y_pred_train),
                        "RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train)),
                    }
                )
                metrics_data.append(
                    {
                        "Dataset": "Test Set",
                        "R²": r2_score(y_test, y_pred_test),
                        "MAE": mean_absolute_error(y_test, y_pred_test),
                        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test)),
                    }
                )

                plot_data["Training Set"] = (y_train, y_pred_train)
                plot_data["Test Set"] = (y_test, y_pred_test)

            else:
                # CV Mode
                with st.spinner("Running Cross Validation..."):
                    y_pred_cv = cross_val_predict(
                        model,
                        X_train,
                        y_train,
                        cv=KFold(5, shuffle=True, random_state=42),
                    )
                    model.fit(X_train, y_train)
                    y_pred_full = model.predict(X_train)

                metrics_data.append(
                    {
                        "Dataset": "5-Fold CV",
                        "R²": r2_score(y_train, y_pred_cv),
                        "MAE": mean_absolute_error(y_train, y_pred_cv),
                        "RMSE": np.sqrt(mean_squared_error(y_train, y_pred_cv)),
                    }
                )
                metrics_data.append(
                    {
                        "Dataset": "Full Data Fit",
                        "R²": r2_score(y_train, y_pred_full),
                        "MAE": mean_absolute_error(y_train, y_pred_full),
                        "RMSE": np.sqrt(mean_squared_error(y_train, y_pred_full)),
                    }
                )

                plot_data["5-Fold CV"] = (y_train, y_pred_cv)
                plot_data["Full Data Fit"] = (y_train, y_pred_full)

            # --- RESULTS DISPLAY ---
            st.subheader("Model Evaluation Metrics")

            metrics_df = pd.DataFrame(metrics_data).set_index("Dataset")
            st.table(
                metrics_df.style.format("{:.3f}").background_gradient(
                    cmap="Blues", subset=["R²"]
                )
            )

            # Interpretation
            r2_val = metrics_df.iloc[-1 if not is_cv else 0]["R²"]
            if r2_val > 0.6:
                st.success(
                    f"✅ **Strong Model:** Explains {r2_val:.1%} of variance. Root causes are reliable."
                )
            elif r2_val > 0.3:
                st.warning(
                    f"⚠️ **Moderate Model:** Explains {r2_val:.1%} of variance. Use insights as directional hints."
                )
            else:
                st.error(
                    f"❌ **Weak Model:** R² is {r2_val:.1%}. Consider selecting different features."
                )

            # Scatter Plots
            cols = st.columns(2)
            for idx, (name, (y_act, y_pred)) in enumerate(plot_data.items()):
                with cols[idx]:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=y_act,
                            y=y_pred,
                            mode="markers",
                            name="Batch",
                            text=y_act.index,
                            hovertemplate="<b>ID:</b> %{text}<br>Act: %{x:.2f}<br>Pred: %{y:.2f}",
                            marker=dict(color="#2E86C1", size=7, opacity=0.6),
                        )
                    )
                    mn, mx = (
                        min(y_act.min(), y_pred.min()),
                        max(y_act.max(), y_pred.max()),
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[mn, mx],
                            y=[mn, mx],
                            mode="lines",
                            name="Perfect Fit",
                            line=dict(color="black", dash="dash", width=2),
                        )
                    )

                    fig.update_layout(
                        title=f"{name}: Actual vs Predicted",
                        template="plotly_white",
                        xaxis_title="Actual Value",
                        yaxis_title="Predicted Value",
                        height=350,
                        margin=dict(l=20, r=20, t=40, b=20),
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # --- RCA EXPLANATIONS ---
            st.divider()
            st.subheader("Root Cause Drivers (Variable Impact)")

            ebm_global = model.explain_global()
            feature_names = ebm_global.data()["names"]

            cols = st.columns(2)
            for i, name in enumerate(feature_names):
                if " & " in name:
                    continue

                with cols[i % 2]:
                    data = ebm_global.data(i)
                    full_title = get_desc(name)

                    if data.get("names") is not None:
                        fig = px.bar(
                            x=data["names"][
                                : min(len(data["names"]), len(data["scores"]))
                            ],
                            y=data["scores"][
                                : min(len(data["names"]), len(data["scores"]))
                            ],
                            title=full_title,
                        )
                        fig.update_traces(marker_color="#2E86C1")
                    else:
                        x_v, y_v = data["lower_bounds"], data["scores"]
                        m = min(len(x_v), len(y_v))
                        fig = px.line(x=x_v[:m], y=y_v[:m], title=full_title)
                        fig.update_traces(line=dict(color="#2E86C1", width=3))

                    fig.update_layout(
                        template="plotly_white",
                        height=350,
                        xaxis_title="Variable Value",
                        yaxis_title="Impact on Target",
                        margin=dict(l=20, r=20, t=40, b=20),
                    )
                    st.plotly_chart(fig, use_container_width=True)
