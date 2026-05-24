# DentalRx AI

**Live dashboard:** https://dentalrx-ai-sinasystems.streamlit.app

**Machine learning-based detection of regional anomalies in NHS dental antibiotic prescribing: a proof-of-concept decision support tool benchmarked against 2024 ADA/AAOS guidelines**

> Manuscript submitted to *JMIR Medical Informatics* (Q1, IF 3.8) · Preprint on medRxiv  
> Article type: Original Paper (tool development and evaluation)

---

## Overview

Dental antibiotic overprescribing is a documented public health problem: 67–80% of dental antibiotic prescriptions are estimated to be inappropriate, and dentists are among the top antibiotic prescribers in primary care. This contributes directly to antimicrobial resistance (AMR), a WHO-designated global health emergency.

**DentalRx AI** applies unsupervised machine learning (Isolation Forest) to publicly available NHS prescribing data to detect regional anomalies in dental antibiotic prescribing patterns across England. SHAP (SHapley Additive exPlanations) values provide feature-level explanations for each flagged region. Findings are benchmarked against the November 2024 ADA/AAOS guideline revision and surfaced via an interactive Streamlit dashboard.

This is, to our knowledge, the first published application of ML anomaly detection and SHAP explainability to dental antibiotic prescribing surveillance.

---

## Research gaps addressed

| Gap | Description |
|-----|-------------|
| GAP 1 | No published study has applied ML anomaly detection to dental prescribing surveillance — all existing ML stewardship work uses supervised classification |
| GAP 2 | SHAP explainability has never been applied to dental prescribing pattern analysis at the regional level |
| GAP 3 | No published study covers the period after the November 2024 ADA/AAOS guideline revision |
| GAP 4 | No open-source, publicly deployable clinical decision support tool exists for dental antibiotic stewardship monitoring |

---

## Data sources

| Dataset | Period | Format | Licence |
|---------|--------|--------|---------|
| [NHSBSA Dental Prescribing Dashboard](https://www.nhsbsa.nhs.uk/prescription-data/dispensing-data/dental-prescribing-dashboard) | Jan 2022 – Nov 2025 | Excel (.xlsx) | Open Government Licence |
| [NHSBSA FOI-02152](https://opendata.nhsbsa.net/dataset/foi-02152) | Jan 2020 – Dec 2023 | CSV | Open Government Licence |

**Data are not committed to this repository.** Download them from the links above and place them in `data/raw/`. See [Data download instructions](#data-download-instructions) below.

This study uses publicly available, anonymised, aggregate-level data. No individual patient data were accessed and no ethics approval was required.

---

## Technical architecture

```
Language:    Python 3.10+
ML method:   Isolation Forest (scikit-learn) — unsupervised anomaly detection
Explainability: SHAP — feature-level contribution scores per flagged region
Dashboard:   Streamlit + Folium choropleth map (England ICBs)
Guideline encoding: Rule-based flags derived from 2024 ADA/AAOS criteria
```

**Engineered features:**
1. Co-amoxiclav prescription rate per 1,000 population
2. Amoxicillin : co-amoxiclav ratio
3. Items per 1,000 population (all dental antibiotics)
4. Month-over-month change rate
5. Deviation from national mean by drug class

---

## Repository structure

```
dentalrx_ai/
├── data/
│   ├── raw/           # Downloaded NHSBSA files (not committed — see below)
│   └── processed/     # Cleaned, merged CSVs
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   ├── 02_features.ipynb     # Feature engineering walkthrough
│   └── 03_model.ipynb        # IsolationForest + SHAP analysis
├── src/
│   ├── preprocess.py         # Data cleaning and merging pipeline
│   ├── features.py           # Feature engineering functions
│   ├── guidelines.py         # 2024 ADA/AAOS rule-based flag encoding
│   ├── model.py              # IsolationForest training and scoring
│   └── explainer.py          # SHAP value computation and export
├── app/
│   └── app.py                # Streamlit dashboard
├── paper/                    # Manuscript drafts (JMIR IMRD format)
├── tests/                    # Unit tests for src modules
├── requirements.txt
├── LICENSE                   # MIT
└── README.md
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/dentalrx_ai.git
cd dentalrx_ai
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Data download instructions

1. **NHSBSA Dental Prescribing Dashboard**  
   Visit https://www.nhsbsa.nhs.uk/prescription-data/dispensing-data/dental-prescribing-dashboard  
   Download all available monthly Excel files → place in `data/raw/nhsbsa_dashboard/`

2. **NHSBSA FOI-02152**  
   Visit https://opendata.nhsbsa.net/dataset/foi-02152  
   Download all CSV files → place in `data/raw/foi_02152/`

---

## Running the pipeline

```bash
# 1. Clean and merge raw data
python src/preprocess.py

# 2. Engineer features
python src/features.py

# 3. Train model and compute anomaly scores
python src/model.py

# 4. Compute SHAP explanations
python src/explainer.py

# 5. Launch the dashboard
streamlit run app/app.py
```

---

## Running the tests

```bash
pytest tests/
```

---

## Dashboard

The Streamlit dashboard provides:
- **Choropleth map** of England ICBs coloured by anomaly score
- **SHAP bar charts** showing feature contributions per flagged region
- **Before/after timeline plots** relative to the November 2024 ADA/AAOS guideline revision

---

## Citation

If this work is useful to you, please cite the preprint until the journal version is available:

> [Author]. (2026). *Machine learning-based detection of regional anomalies in NHS dental antibiotic prescribing: a proof-of-concept decision support tool benchmarked against 2024 ADA/AAOS guidelines*. medRxiv. https://doi.org/[TBC]

---

## Ethical considerations

This study used publicly available, anonymised, aggregate-level data from the NHSBSA Open Data Portal under the Open Government Licence. No individual patient data were accessed and no ethics approval was required.

---

## Licence

This repository is licensed under the [MIT Licence](LICENSE).
