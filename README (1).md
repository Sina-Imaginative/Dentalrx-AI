# DentalRx AI

**Live dashboard:** https://dentalrx-ai-sinasystems.streamlit.app

ML-powered anomaly detection for NHS dental antibiotic prescribing surveillance across England, with SHAP explainability and 2024 ADA/AAOS guideline benchmarking.

---

## Overview

DentalRx AI applies unsupervised machine learning to publicly available NHS prescribing data to identify regional anomalies in dental antibiotic prescribing patterns across England's 42 Integrated Care Boards. The tool provides interpretable, feature-level explanations for flagged regions via an interactive dashboard.

Dental antibiotic overprescribing is a documented public health problem contributing to antimicrobial resistance (AMR), a WHO-designated global health emergency. This tool is designed to support antimicrobial stewardship efforts by surfacing regional prescribing anomalies from aggregate NHS data.

---

## Live Dashboard

The dashboard is publicly accessible at:

**https://dentalrx-ai-sinasystems.streamlit.app**

It provides:
- Regional anomaly scores across all 42 NHS England ICBs
- SHAP-based feature explanations for flagged regions
- Timeline analysis of prescribing trends with the November 2024 ADA/AAOS guideline revision annotated

---

## Data Sources

Both datasets are publicly available under the Open Government Licence v3.0. Raw data files are not committed to this repository — download them directly from the sources below.

| Dataset | Period | Source |
|---------|--------|--------|
| NHSBSA Dental Prescribing Dashboard | Jan 2022 – Feb 2026 | [nhsbsa.nhs.uk](https://www.nhsbsa.nhs.uk/prescription-data/dispensing-data/dental-prescribing-dashboard) |
| NHSBSA FOI-02152 (Part 4 — Dental Prescribing) | Jan 2020 – Dec 2023 | [opendata.nhsbsa.net](https://opendata.nhsbsa.net/dataset/foi-02152) |

No individual patient data were accessed. No ethics approval was required.

---

## Repository Structure

```
Dentalrx-AI/
├── data/
│   ├── raw/              # NHSBSA source files (not committed — download separately)
│   └── processed/        # Cleaned and feature-engineered datasets
├── notebooks/
│   ├── 01_preprocess.ipynb    # Data cleaning and reshaping
│   ├── 02_features.ipynb      # Feature engineering
│   └── 03_model.ipynb         # Anomaly detection and SHAP analysis
├── src/                  # Pipeline modules
│   ├── preprocess.py
│   ├── features.py
│   ├── guidelines.py
│   ├── model.py
│   └── explainer.py
├── app/
│   └── app.py            # Streamlit dashboard
├── tests/                # Unit tests
├── .streamlit/
│   └── config.toml       # Streamlit Cloud configuration
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

```bash
git clone https://github.com/Sina-Imaginative/Dentalrx-AI.git
cd Dentalrx-AI
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Data Download

**Step 1 — NHSBSA Dental Prescribing Dashboard**
Visit https://www.nhsbsa.nhs.uk/prescription-data/dispensing-data/dental-prescribing-dashboard
Download the Excel file and place it in `data/raw/`

**Step 2 — NHSBSA FOI-02152**
Visit https://opendata.nhsbsa.net/dataset/foi-02152
Download Part 4 (Dental Prescribing CSV) and place it in `data/raw/`

---

## Running the Dashboard Locally

```bash
streamlit run app/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

---

## Technical Stack

```
Language:       Python 3.11+
ML method:      Isolation Forest (scikit-learn)
Explainability: SHAP (TreeExplainer)
Dashboard:      Streamlit + Folium + Plotly
Licence:        MIT
```

---

## Ethical Considerations

This project uses publicly available, anonymised, aggregate-level data from the NHSBSA Open Data Portal under the Open Government Licence v3.0. No individual patient data were accessed and no ethics approval was required. Data are reported at the ICB level only.

---

## Citation

If you use this work in your research or build upon it, please cite:

> Memarzadeh S. (2026). *DentalRx AI: machine learning-based detection of regional anomalies in NHS dental antibiotic prescribing*. GitHub. https://github.com/Sina-Imaginative/Dentalrx-AI

---

## Licence

Copyright (c) 2026 Sina Memarzadeh

Licensed under the [MIT Licence](LICENSE).
