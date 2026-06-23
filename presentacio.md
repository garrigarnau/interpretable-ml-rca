# Presentació TFG — Guió (17 minuts)

---

## Diapositiva 1 — Portada (0:15)

**Contingut:**
- Títol: "Disseny i avaluació d'una metodologia d'industrialització de la Root Cause Analysis (RCA) basada en models de Machine Learning"
- Arnau Garriga Torné
- Tutor: Joan Piedrafita
- Enginyeria de Dades, UAB — Juny 2026

**Imatge:** Logo UAB / Logo Grifols (opcional)

**Guió:**
> Bon dia, sóc l'Arnau Garriga i presento el meu TFG sobre com industrialitzar la Root Cause Analysis en entorns farmacèutics regulats utilitzant models de Machine Learning interpretables.

---

## Diapositiva 2 — RCA tradicional: estat actual i limitacions (1:00)

**Contingut:**
- Diagrama RCA clàssic (5 Whys, Ishikawa, FMEA)
- Limitacions:
  - Dependència de coneixement tàcit
  - Baixa repetibilitat entre equips
  - No escala amb volum de dades (150 variables)
  - No captura interaccions multivariants

**Imatge:** Diagrama Ishikawa genèric o esquema propi

**Guió:**
> Quan es detecta una desviació de procés en producció farmacèutica, es dispara una investigació de causa arrel. Actualment es fa amb metodologies clàssiques com 5 Whys o Ishikawa. El problema: depenen de l'experiència de l'investigador, no són repetibles entre equips, i amb 150 variables és impossible analitzar-les totes manualment. No capturen relacions no lineals ni interaccions.

---

## Diapositiva 3 — ML per RCA: per què els models actuals tenen més cost (1:15)

**Contingut:**
- Models actuals: Random Forest, XGBoost + SHAP/LIME
- Problemes en entorn GxP:
  - Explicació aproximada / Risc inconsistència
  - SHAP i LIME poden divergir amb variables correlacionades
  - Major càrrega validació (2 capes: model + explicació)
  - Pèrdua de forma funcional: magnitud sí, rang òptim no
  - Cost computacional i auditabilitat de cada explicació

**Imatge:** Mermaid diagram — "Explicació aproximada / Risc inconsistència" → "Major càrrega validació"

**Guió:**
> S'han aplicat models com XGBoost amb explicabilitat post-hoc via SHAP o LIME. Els models black-box sí que s'usen en farma, però tenen una càrrega de validació molt més alta: has de validar el model I l'explicació — dues capes en comptes d'una. SHAP i LIME poden donar explicacions divergents per la mateixa predicció amb variables correlacionades. I el més crític per RCA: SHAP et diu la magnitud i direcció, però no la forma de la relació — no saps quin és el rang òptim ni el cost de cada desviació. En GxP això es tradueix en més temps de validació, més risc d'inconsistència en auditoria, i menys utilitat operativa.

---

## Diapositiva 4 — Objectius del treball (1:00)

**Contingut:**
- Objectiu principal: Utilitzar un model interpretable-by-design per prioritzar hipòtesis de causa arrel amb criteris tècnics, traçables i alineats amb GxP
- Sub-objectius:
  1. Marc RCA data-driven alineat amb GxP
  2. Pipeline de dades reproduïble
  3. Model ML interpretable per predir desviacions
  4. Explicabilitat nativa per prioritzar hipòtesis
  5. Arquitectura de validació i industrialització

**Imatge:** Cap (text clar i net)

**Guió:**
> L'objectiu: utilitzar un model interpretable per disseny per prioritzar hipòtesis de causa arrel amb criteris tècnics i traçables. No substituïm l'expert, li donem eines quantitatives per reduir l'espai de recerca i accelerar la investigació, complint amb requisits regulatoris.

---

## Diapositiva 5 — Què és l'EBM i per què glass-box (1:30)

**Contingut:**
- Fórmula: f(x) = β₀ + Σᵢ fᵢ(xᵢ) + Σᵢ,ⱼ fᵢⱼ(xᵢ, xⱼ)
- GAM modern entrenat amb boosting
- Què desbloqueja:
  - Shape functions → forma exacta de cada relació
  - Interaccions parelles → efectes combinats
  - Contribucions locals → descomposició exacta per lot
  - No aproximació (vs SHAP)

**Imatge:** Diagrama conceptual EBM (estructura additiva)

**Guió:**
> L'EBM és un Generalized Additive Model modern entrenat amb boosting. La predicció és una suma de funcions individuals per variable més interaccions parelles. Cada terme és visualitzable i interpretable. La descomposició és exacta per construcció, no una aproximació. Ens desbloqueja: veure la forma de cada relació, capturar interaccions no lineals, i descompondre cada lot individualment per identificar què ha anat malament.

---

## Diapositiva 6 — Context: Grifols i el procés (1:00)

**Contingut:**
- Planta Grifols Los Angeles
- Fraccionament de plasma humà (mètode Cohn)
- 3 fases: (B) Cryo → (C) Fracció I → (D) Fracció II+III
- Target: D49 (yield % FII+III PPT)
- Clau: primeres etapes determinen rendiment i seguretat
- Plasma = matèria biològica, proteïnes sensibles a T i pH

**Imatge:** Esquema simplificat del procés de fraccionament (3 fases)

**Guió:**
> El cas d'estudi és Grifols Los Angeles, fraccionament de plasma humà per obtenir immunoglobulines. Mètode Cohn amb tres fases. La variable objectiu és el yield de Fracció II+III. Punt clau: el rendiment depèn de les primeres etapes. El plasma és biològic i les proteïnes són molt sensibles a temperatura i condicions químiques, fent el procés altament variable.

---

## Diapositiva 7 — Dataset i decisions de dades (0:45)

**Contingut:**
- Disponible: ~4.600 lots, 9 anys, 150 variables
- Decisió: només dades des de 2024 (canvi de procés → no-estacionarietat)
- 104 numèriques + 46 categòriques
- "Nosaltres donem les eines; els process engineers decideixen"
- Dataset final: 956 mostres × 56 variables

**Imatge:** Cap o timeline simple mostrant el tall a 2024

**Guió:**
> Disposem de 4.600 lots amb 150 variables en 9 anys. S'observa un canvi de procés a partir de 2024, així que entrenem només amb dades recents: 956 mostres. Important: tot aquest estudi el fem per trobar la millor configuració, però la idea és donar eines als process engineers perquè ells apliquin el seu criteri de domini.

---

## Diapositiva 8 — Neteja i preparació (0:45)

**Contingut:**
- Criteri de domini + validació experta
- Mapa de variables: correlacions, fase, tipus, accionabilitat
- Eliminacions: missings >35%, variància baixa, correlació >0.90, temporals sense utilitat causal
- Feature engineering: timestamps → durades agregades
- Resultat: 956 mostres × 56 variables

**Imatge:** Fragment del mapa de variables (variable_analysis_config.json visualitzat)

**Guió:**
> La neteja s'ha fet amb criteri de domini. Hem construït un mapa de variables documentant correlacions, fase i accionabilitat per automatitzar el procés. S'han eliminat variables amb massa missings, variància baixa, alta correlació i temporals sense causalitat. Timestamps transformats en durades. Resultat: 956 × 56.

---

## Diapositiva 9 — Baseline i proves prèvies (0:45)

**Contingut:**
- Proves: LASSO, Gradient Boosting, LASSO→EBM
- Baseline LASSO: MAE=1.126, R²=0.332
- Conclusió: selecció lineal descarta variables no lineals importants
- LASSO+EBM: pitjor resultat (R²=0.202)

**Imatge:** Cap (taula simple amb resultats baseline)

**Guió:**
> Vam provar diverses aproximacions de la literatura: LASSO com a baseline obté R²=0.332. Vam intentar usar LASSO per seleccionar variables i després aplicar EBM, però el resultat és el pitjor de tots — LASSO descarta variables amb efectes no lineals que l'EBM necessita. Calia una estratègia pròpia.

---

## Diapositiva 10 — Estratègia filter-then-interact (1:00)

**Contingut:**
- Diagrama 3 fases:
  - Fase 1: EBM (interactions=0) → ranking per importància
  - Fase 2: Selecció top-15 (elbow detection)
  - Fase 3: EBM (interactions=10) sobre les 15 seleccionades
- Hiperparàmetres: lr=0.05, max_bins=256, max_rounds=5000
- Resultats Fase 3: MAE=1.018, R²=0.470

**Imatge:** `images/fig_008_match_lasso_behavior_show_human_readable_name_next_to_encoded_feature_key.png` (ranking Fase 1)

**Guió:**
> La nostra estratègia: primer entrenem EBM sense interaccions per obtenir un ranking d'importància de variables. Seleccionem les top-15 on la importància cau significativament. Després reentrenem amb interaccions=10 sobre aquest conjunt reduït. Amb 15 variables, l'espai d'interaccions passa de 1.540 parells a 105, garantint densitat de dades suficient. Resultat: R²=0.470, ja superant el baseline.

---

## Diapositiva 11 — Fase 3: efectes principals i interaccions (0:45)

**Contingut:**
- Importància dels 15 efectes principals
- Interaccions detectades automàticament
- Model final: 25 termes (15 + 10)

**Imatge:** `images/plot_importances.png` (efectes principals en blau + interaccions en taronja)

**Guió:**
> El model final té 25 termes: 15 efectes principals i 10 interaccions detectades automàticament. Aquí veiem la importància de cada variable i quines parelles tenen efectes conjunts significatius. Tot és transparent i revisable per un enginyer de procés.

---

## Diapositiva 12 — Validació: configuracions Fase 4 (0:45)

**Contingut:**
- 7 configuracions avaluades amb 5-fold CV
- Taula de resultats (de l'article, Taula 3)
- Guanyador: Robust (Leaf=10): MAE=0.950, RMSE=1.368, R²=0.422
- R² inferior a Fase 3 (0.470) però validat i estable

**Imatge:** Cap (taula de configuracions)

**Guió:**
> Per validar, hem provat 7 configuracions d'EBM amb validació creuada de 5 folds. El R² baixa de 0.470 a 0.422, però ara és robust i estable entre folds. La configuració Robust amb Leaf=10 ofereix el millor compromís precisió-robustesa. En producció, la fiabilitat és tan important com la precisió puntual.

---

## Diapositiva 13 — Argument central: complexitat vs rendiment (0:45)

**Contingut:**
- EBM filter+interact: 25 termes, R²=0.477
- XGBoost tuned: 2.500 paràmetres, R²=0.369
- La interpretabilitat NO penalitza el rendiment
- El filtratge és l'element diferencial (p=0.037)

**Imatge:** `images/s14_complexity_vs_r2.png`

**Guió:**
> Aquesta figura condensa l'argument central del treball. El nostre model, amb només 25 termes interpretables, supera XGBoost amb 2.500 paràmetres per un 29% relatiu en R². La interpretabilitat no penalitza el rendiment. El filtratge és estadísticament significatiu amb p=0.037. Dos ordres de magnitud menys de complexitat, millor rendiment, i totalment auditable.

---

## Diapositiva 14 — Comparació de models (0:30)

**Contingut:**
- Taula benchmark 5 models (Taula 4 de l'article)
- EBM filter+interact guanya en MAE i R²

**Imatge:** `images/s14_model_comparison.png`

**Guió:**
> Aquí la comparació directa dels 5 models sobre test set. L'EBM filter+interact guanya tant en MAE com en R², sent alhora el model més interpretable i un dels més ràpids.

---

## Diapositiva 15 — Viabilitat computacional (0:30)

**Contingut:**
- Latència: 0.0015 ms/mostra (688K batches/s)
- Entrenament: 4.8s (reentrenament per lot possible)
- Empremta: 2.7 MB (contenidors lleugers, sense GPU)
- Context: cicle de lot 24-48h → decisió quasi-immediata

**Imatge:** `images/s15_training_inference_cost.png`

**Guió:**
> En aquest entorn necessitem respostes ràpides. El model permet 688.000 prediccions per segon, reentrenament en menys de 5 segons, i ocupa 2.7 MB. Podem desplegar-lo en temps real sense infraestructura dedicada, i reentrenar cada lot si cal.

---

## Diapositiva 16 — Drift: no tot són flors (0:30)

**Contingut:**
- PSI sobre finestres lliscants (50 lots, pas 25)
- Drift generalitzat des de juny 2025
- D36 arriba a PSI=6.8 (crític)
- Però: MAE es manté sota UCL → robustesa de l'estructura additiva
- Recomanació: reentrenament quan PSI>0.25 durant 3 finestres

**Imatge:** `images/s16_drift_psi_only.png`

**Guió:**
> No tot són flors. El monitoratge retrospectiu mostra drift significatiu des de juny 2025, especialment en D36 amb PSI=6.8. Però — i això és important — el MAE del model es manté dins dels límits de control. L'estructura additiva de l'EBM proporciona degradació gradual. Això valida la capa de monitoratge i la necessitat de reentrenament periòdic.

---

## Diapositiva 17 — Explicabilitat: shape functions (1:00)

**Contingut:**
- Shape function de C33 (temps de mescla FI)
- U-shape: òptim a 3 min, rang segur 1.2-9.3
- Penalització progressiva >10 min (fins -0.6 pp a 35 min)
- Com ho mira un process engineer:
  - "Quin és l'òptim?" → 3 min
  - "Quin rang és segur?" → 1.2 a 9.3
  - "Quin és el cost de cada minut extra?" → ~0.03 pp ≈ milers de $/lot

**Imatge:** `images/shape_c33.png`

**Guió:**
> Aquí és on l'EBM brilla per RCA. Aquesta shape function mostra C33, temps de mescla de Fracció I. Un enginyer pot veure immediatament: l'òptim és 3 minuts, operar entre 1.2 i 9.3 és segur, i cada minut per sobre de 10 costa 0.03 punts de yield, equivalent a milers de dòlars per lot. Això no t'ho dóna SHAP — SHAP et diu la magnitud però no la forma ni el rang òptim.

---

## Diapositiva 18 — Impacte operatiu i econòmic (1:00)

**Contingut:**
- Fórmula: Y_opt = Y_pred + (-Score_neg · rec · R²)
- Factor conservadorisme: 29.4% (R²=0.422 × rec=0.70)
- Escenari A (aspiracional): +1.01 pp → ~26.8 M$/any (100 lots)
- Escenari B (focalitzat, el més real): 3 variables (D36, D24, D40) → +0.19 pp → ~5 M$/any
- Preu referència: 55 $/g (validat amb experts interns)
- Escenari ajustat a realitat física i regulatòria

**Imatge:** Cap (taula o diagrama dels dos escenaris)

**Guió:**
> Per quantificar l'impacte, descomponem contribucions negatives lot a lot. Amb un factor conservador del 29.4% que integra la variància explicada i les friccions reals de planta, l'Escenari B — el més realista — focalitza en 3 variables que afecten >40% dels lots i estima ~5 milions anuals d'ingressos addicionals. El preu de 55 $/g l'hem validat amb experts de l'empresa. És un escenari ajustat a la realitat física i regulatòria.

---

## Diapositiva 19 — Conclusions (0:45)

**Contingut:**
- Filter-then-interact millora el baseline Lasso: -15.6% MAE, +27.1% R²
- La interpretabilitat NO penalitza el rendiment (supera XGBoost 29%)
- Sistema human-in-the-loop: model prioritza, expert decideix
- Base sòlida per desplegament progressiu en GMP/QMS

**Imatge:** `images/mapa-conceptual.png` (diagrama RCA de l'apèndix)

**Guió:**
> En conclusió: l'estratègia filter-then-interact millora consistentment el baseline, la interpretabilitat no penalitza el rendiment, i el sistema és un decision support — no un optimitzador automàtic. L'expert manté el control. Constitueix una base sòlida per a desplegament progressiu en entorns GMP.

---

## Diapositiva 20 — Treball futur (0:15)

**Contingut:**
- Validació formal GxP: IQ/OQ/PQ
- Ampliació temporal i nous escenaris
- Monitoratge i governança: versionat, reentrenament amb aprovacions QA
- Integració operativa: CAPA, MES/LIMS

**Imatge:** Cap

**Guió:**
> Com a treball futur: validació formal GxP amb IQ/OQ/PQ, ampliació de cobertura temporal, monitoratge amb governança de canvis, i integració directa amb sistemes CAPA i MES. Gràcies.

---

## Resum d'imatges utilitzades

| Diapositiva | Imatge | Font |
|-------------|--------|------|
| 10 | `images/fig_008_match_lasso_behavior_show_human_readable_name_next_to_encoded_feature_key.png` | Article Fig. ranking |
| 11 | `images/plot_importances.png` | Article Fig. 2a |
| 11 | `images/fig_014__extract_learned_terms_.png` | Article Fig. 2b |
| 13 | `images/s14_complexity_vs_r2.png` | Article Fig. 4 |
| 14 | `images/s14_model_comparison.png` | Article Fig. 3 |
| 15 | `images/s15_training_inference_cost.png` | Article Fig. 5 |
| 16 | `images/s16_drift_psi_only.png` | Article Fig. 6 |
| 17 | `images/shape_c33.png` | Article Fig. 7 |
| 19 | `images/mapa-conceptual.png` | Apèndix A1 |
