# Estructura del Codi Versionat

Aquest directori conté el codi font del projecte organitzat per versions, seguint l'evolució cronològica del treball.

---

## v0.2_exploracioEDA/

**Data:** 2026-02-18 | **Milestone:** Primers scripts + anàlisi exploratòria

| Fitxer | Descripció |
|--------|------------|
| `data_cleaning.py` | Funcions inicials de càrrega de dades i preprocessament bàsic. Extret del commit `34e07d8`. |
| `dataset_overview.py` | Mòdul de visualització del dataset (estadístiques bàsiques, distribucions). Extret del commit `34e07d8`. |
| `eda_exploracio.ipynb` | Notebook d'exploració amb outputs executats. Conté: càrrega de dades, `df.shape`/`df.describe()`, visualització de missing values amb missingno, distribució de la variable objectiu D49, matriu de correlació de les top 20 variables numèriques. |

**Estat del projecte en aquesta versió:**
- Dataset carregat: 956 lots × ~150 variables
- Identificació inicial de la problemàtica de missing values
- Primeres visualitzacions de la distribució del yield (D49)

---

## v0.3_lasso_baseline/

**Data:** 2026-02-23 | **Milestone:** Model Lasso com a baseline

| Fitxer | Descripció |
|--------|------------|
| `data_cleaning.py` | Versió millorada amb enforcement de tipus categòrics (`object dtype`). Extret del commit `ac53550`. |
| `lasso_baseline.ipynb` | Notebook complet amb outputs executats. Implementa: pipeline de neteja (missing >35%, variància <0.10, correlació >0.90), StandardScaler, One-hot encoding, entrenament Lasso (alpha=0.01), validació creuada 5-fold, visualització dels coeficients top-20. |

**Resultats clau d'aquesta versió:**
- R² ≈ 0.33 (baseline)
- MAE ≈ 1.12
- 34 features amb coeficient no-zero
- Model lineal: no captura no-linealitats del procés

---

## v0.4_ebm_filter_interact/

**Data:** 2026-03-15 | **Milestone:** Pivot metodològic a EBM Filter-then-Interact

| Fitxer | Descripció |
|--------|------------|
| `ebm_filter_interact.ipynb` | Notebook amb l'estratègia completa Filter-then-Interact executada. Conté: Phase 1 (EBM additiu, interactions=0, min_samples_leaf=2, ranking de features), Phase 2 (selecció top-15 variables per importancia), Phase 3 (EBM amb 10 interaccions, min_samples_leaf=10), validació creuada 5-fold, shape functions de les top-3 variables, comparativa amb Lasso baseline. |

**Resultats clau d'aquesta versió:**
- R² ≈ 0.42 (millora +27% vs Lasso)
- MAE ≈ 0.95 (reducció -15.6%)
- Detecció de no-linealitats (bell curves)
- Variables categòriques (tancs) amb alt impacte predictiu
- Justificació del descart d'XGBoost (opacitat incompatible amb GxP)

---

## v0.7_mvp_streamlit/

**Data:** 2026-04-15 | **Milestone:** Primer MVP funcional amb Streamlit

| Fitxer | Descripció |
|--------|------------|
| `app_mvp.py` | Aplicació Streamlit operativa (~449 línies). Inclou: càrrega i neteja automàtica de dades, configuració d'hiperparàmetres EBM, entrenament del pipeline Filter-then-Interact, visualització de feature importance i shape functions, anàlisi local per lot (waterfall charts), simulador What-If amb sliders interactius. **No inclou** la secció d'impacte financer (afegida a v1.0). |

**Funcionalitats d'aquesta versió:**
- Panell de configuració (top-N features, max interactions, learning rate, etc.)
- Botó "Train EBM Pipeline"
- Shape Functions interactives (Plotly)
- Mòdul RCA: selecció de lot → waterfall → negative drivers
- What-If: sliders per simular canvis de paràmetres

**Per executar:** `streamlit run app_mvp.py`

---

## v1.0_final/

**Data:** 2026-06-04 | **Milestone:** Versió final completa

| Fitxer | Descripció |
|--------|------------|
| `app.py` | Dashboard principal Streamlit (~77 KB). EDA complet + Lasso + EBM + explicabilitat local. |
| `app_filter.py` | Pipeline EBM Filter-then-Interact complet (~90 KB). Versió de producció amb totes les funcionalitats: RCA, What-If, shape functions, **impacte financer**, business case, sensibilitat de preu. |
| `data_cleaning.py` | Funcions de càrrega i preprocessament (~16 KB). Pipeline complet de neteja amb tots els criteris. |
| `dataset_overview.py` | Mòdul de visualització del dataset (~14 KB). Estadístiques, distribucions, missing data. |
| `extract_financial_results.py` | Script standalone per al business case (~26 KB). Càlculs d'escenaris A i B, sensibilitat, ROI. |
| `app.ipynb` | Notebook d'exploració i experimentació (~1.1 MB). Conté tot el procés iteratiu de desenvolupament. |
| `requirements.txt` | Dependències Python del projecte. |

**Funcionalitats afegides respecte v0.7:**
- Mòdul d'impacte financer (traducció a $/lot)
- Business case integrat (Escenari A: 26.8M$/any, Escenari B: 5M$/any)
- Sensibilitat de preu (50/55/60 $/g)
- Script standalone `extract_financial_results.py`
- Coeficient de conservadorisme (29.4%)

---

## Evolució de mètriques entre versions

| Versió | Model | R² | MAE | Interpretable |
|--------|-------|----|-----|---------------|
| v0.3 | Lasso (alpha=0.01) | 0.33 | 1.12 | Sí (lineal) |
| v0.4 | EBM Additiu (Phase 1) | 0.38 | 1.02 | Sí (glass-box) |
| v0.4 | EBM Robust Leaf=10 (Phase 2) | 0.42 | 0.95 | Sí (glass-box) |

## Com executar

```bash
# Entorn
python -m venv .venv
source .venv/bin/activate
pip install -r v1.0_final/requirements.txt

# Notebooks (des de la carpeta del projecte)
jupyter notebook

# App Streamlit (versió final)
streamlit run v1.0_final/app_filter.py

# App MVP (versió simplificada)
streamlit run v0.7_mvp_streamlit/app_mvp.py
```
