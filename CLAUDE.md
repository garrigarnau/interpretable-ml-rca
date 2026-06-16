# CLAUDE.md - Project Context

## Project Overview

**TFG (Treball de Fi de Grau)** - Data Engineering degree final thesis.
**Title**: ML-based Root Cause Analysis (RCA) monitoring system for pharmaceutical production.
**Domain**: Grifols plasma fractionation (Fraction II+III immunoglobulin yield optimization).
**Approach**: Explainable Boosting Machine (EBM) glassbox model instead of black-box models, for GxP regulatory compliance.

## Key Business Context

- Target variable: **D49** (FII+III PPT yield %)
- Revenue variable: **D48** (FII+III PPT weight in kg)
- 151 process variables across 3 production phases (B=Cryo, C=Fraction I, D=Fraction II+III)
- Cold ethanol method (Cohn method) for plasma fractionation
- GxP/GMP regulated environment requiring full model auditability

## Tech Stack

- **Python 3.14**, Streamlit, Plotly
- **ML**: `interpret` (ExplainableBoostingRegressor), scikit-learn, xgboost (comparison only)
- **Data**: pandas, numpy, missingno
- **Dataset**: ~5 years historical production data, 151 variables

## Repository Structure

```
├── app.py                    # Main Streamlit MVP (production monitoring system)
├── app_filter.py             # Filtering utility module
├── app.ipynb                 # Jupyter notebook with all studies/analyses
├── requirements.txt          # Python dependencies
├── scripts/
│   ├── data_cleaning.py      # Data loading & preprocessing (25 functions)
│   ├── dataset_overview.py   # Visualization & statistics
│   └── extract_financial_results.py  # Revenue analysis
├── data/
│   ├── fractionation_data.csv         # Main dataset
│   ├── variable_descriptions.json     # 151 variable descriptions
│   └── variable_analysis_config.json  # Variable metadata (phase, actionable, correlations)
├── informes/                 # Project reports (initial, progress 1 & 2)
├── dossier/                  # Thesis documentation & code versions
│   ├── codi/                 # Version history (v0.2 EDA → v1.0 final)
│   ├── markdowns/            # 10-chapter thesis documentation
│   └── PDFs/                 # Formal thesis documents
└── old/                      # Previous app.py versions
```

## Core Methodology: Filter-Then-Interact

**3-Phase EBM Pipeline:**
1. **Phase 1 (Feature Ranking)**: EBM with interactions=0, rank features by main effect importance
2. **Phase 2 (Pruning)**: Select top-N features (default 15) via elbow detection or manual threshold
3. **Phase 3 (Interaction Hunting)**: Re-train EBM with interactions=10 on pruned features

**Performance**: R² improved from 0.33 (LASSO) → 0.52 (EBM), MAE from 1.4 → 0.98

## app.py - Streamlit MVP

5-phase data cleaning pipeline + 3-phase EBM + 3 main modules:
1. **Batch Monitoring Panel**: Real-time actual vs predicted yield
2. **RCA Analysis Module**: Waterfall decomposition per batch, confidence alarms
3. **Portfolio View**: Aggregate optimization potential, financial projections

Financial analysis: price sensitivity ($50-60/g), conservative 70% recovery factor, uncertainty bounds.

## Key Model Parameters

| Parameter | Value |
|-----------|-------|
| Missing threshold | 35% |
| Variance threshold | 0.10 |
| Correlation threshold | 0.90 |
| EBM learning rate | 0.05 |
| EBM max rounds | 5,000 |
| EBM max bins | 256 |
| Min samples/leaf | 10 |
| Train/Test split | 75/25 |
| CV folds | 5 |
| LASSO alpha | 0.03 |

## Key Findings

- Non-linear relationships discovered (e.g., filtration time C33 has bell-curve effect)
- Equipment IDs (non-actionable) have higher predictive weight than expected
- Top actionable variables: D36 (filtration, 62.1% lots), D24 (mixing, 50.4%), D40 (filtration, 43.0%)
- Focused Pareto strategy on 3 variables captures ~$5M/year improvement potential

## Development Commands

```bash
# Run Streamlit app
streamlit run app.py

# Run Jupyter notebook
jupyter notebook app.ipynb

# Install dependencies
pip install -r requirements.txt
```

## Language & Style

- Code: English (variables, functions, comments)
- Reports/documentation: Catalan/Spanish
- Academic context: UPC (Universitat Politecnica de Catalunya) or similar Catalan university
