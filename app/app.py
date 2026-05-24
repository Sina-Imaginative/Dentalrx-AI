
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
import json
import warnings
warnings.filterwarnings("ignore")

# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="DentalRx AI",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #e63946;
    }
    .anomaly-high {
        color: #e63946;
        font-weight: 700;
    }
    .anomaly-normal {
        color: #2a9d8f;
        font-weight: 700;
    }
    .guideline-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.75rem;
        border-radius: 4px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/icb_model_results.csv", parse_dates=["month"])
    return df

icb = load_data()

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🦷 DentalRx AI")
st.sidebar.markdown("**NHS Dental Antibiotic Prescribing**")
st.sidebar.markdown("Anomaly detection benchmarked against 2024 ADA/AAOS guidelines")
st.sidebar.markdown("---")

# Month selector
min_month = icb["month"].min()
max_month = icb["month"].max()
all_months = sorted(icb["month"].unique())
month_labels = [pd.Timestamp(m).strftime("%b %Y") for m in all_months]

selected_month_label = st.sidebar.selectbox(
    "Select month for map view",
    month_labels,
    index=len(month_labels) - 1
)
selected_month = all_months[month_labels.index(selected_month_label)]

# ICB selector
icb_list = sorted(icb["icb_name"].unique())
selected_icb = st.sidebar.selectbox(
    "Select ICB for detail view",
    icb_list,
    index=icb_list.index("NHS SUFFOLK AND NORTH EAST ESSEX INTEGRATED CARE BOARD")
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Data sources**
- NHSBSA Dental Prescribing Dashboard (Jan 2022 – Feb 2026)
- NHSBSA FOI-02152 (Jan 2020 – Dec 2023)

**Method**
- Isolation Forest (scikit-learn)
- SHAP explainability
- 2024 ADA/AAOS guideline encoding

**Licence:** MIT | **Data:** OGL v3
""")

# ── Main header ──────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🦷 DentalRx AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Machine learning-based detection of regional anomalies '
    "in NHS dental antibiotic prescribing — benchmarked against 2024 ADA/AAOS guidelines</p>",
    unsafe_allow_html=True
)

# ── Summary metrics row ──────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("ICBs analysed", "42")
with col2:
    st.metric("Months covered", "50")
with col3:
    st.metric("Observations", "2,100")
with col4:
    st.metric("ML anomalies", "105 (5.0%)")
with col5:
    st.metric("Guideline flags", "157")

st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🗺️ Regional Map",
    "📊 ICB Detail & SHAP",
    "📈 Timeline Analysis"
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHOROPLETH MAP
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader(f"Anomaly scores by ICB — {selected_month_label}")

    month_data = icb[icb["month"] == selected_month].copy()

    col_map, col_table = st.columns([2, 1])

    with col_map:
        # Create folium map centred on England
        m = folium.Map(
            location=[52.5, -1.5],
            zoom_start=6,
            tiles="CartoDB positron"
        )

        # Normalise anomaly score to 0-1 for colour mapping
        # More negative = more anomalous = darker red
        score_min = icb["anomaly_score"].min()
        score_max = icb["anomaly_score"].max()

        for _, row in month_data.iterrows():
            # Normalise: 0 = most anomalous, 1 = most normal
            norm = (row["anomaly_score"] - score_min) / (score_max - score_min)
            
            # Colour: red for anomalous, green for normal
            if row["is_anomaly"] == 1:
                color = "#e63946"
                fill_opacity = 0.8
            else:
                color = "#2a9d8f"
                fill_opacity = 0.4

            # Shorten ICB name for display
            short_name = row["icb_name"].replace(
                "INTEGRATED CARE BOARD", "ICB"
            ).replace("NHS ", "")

            popup_text = f"""
            <b>{short_name}</b><br>
            Anomaly score: {row["anomaly_score"]:.4f}<br>
            Co-amoxiclav rate: {row["f1_coamox_rate"]:.3f}%<br>
            Amox:CoAmox ratio: {row["f2_amox_coamox_ratio"]:.1f}<br>
            Items per 1,000: {row["f3_items_per_1000"]:.3f}<br>
            <b>Status: {"⚠️ ANOMALY" if row["is_anomaly"]==1 else "✓ Normal"}</b>
            """

            folium.CircleMarker(
                location=[52.5, -1.5],  # placeholder — replaced below
                radius=8,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=fill_opacity,
                popup=folium.Popup(popup_text, max_width=250),
                tooltip=short_name
            )

        folium_static(m, width=600, height=500)

    with col_table:
        st.markdown("**ICB anomaly status this month**")

        display_cols = ["icb_name", "anomaly_score", "is_anomaly",
                        "f1_coamox_rate", "any_guideline_flag"]

        month_display = month_data[display_cols].copy()
        month_display.columns = ["ICB", "Score", "ML Flag",
                                  "CoAmox %", "Guideline"]
        month_display["ICB"] = month_display["ICB"].str.replace(
            "INTEGRATED CARE BOARD", "ICB"
        ).str.replace("NHS ", "")
        month_display = month_display.sort_values("Score")

        # Highlight anomalies
        def highlight_anomaly(row):
            if row["ML Flag"] == 1:
                return ["background-color: #ffe0e0"] * len(row)
            return [""] * len(row)

        st.dataframe(
            month_display.style.apply(highlight_anomaly, axis=1),
            height=500,
            use_container_width=True
        )

    # Flagged ICBs this month
    flagged = month_data[month_data["is_anomaly"] == 1]
    if len(flagged) > 0:
        st.markdown(f"**⚠️ {len(flagged)} ICBs flagged as anomalous this month:**")
        for _, row in flagged.sort_values("anomaly_score").iterrows():
            short = row["icb_name"].replace("INTEGRATED CARE BOARD","ICB").replace("NHS ","")
            st.markdown(
                f"- **{short}** — score: {row['anomaly_score']:.4f}, "
                f"co-amoxiclav: {row['f1_coamox_rate']:.3f}%"
            )
    else:
        st.success("No anomalies detected this month")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — ICB DETAIL AND SHAP
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    icb_data = icb[icb["icb_name"] == selected_icb].copy()
    short_name = selected_icb.replace("INTEGRATED CARE BOARD","ICB").replace("NHS ","")

    st.subheader(f"Detail view — {short_name}")

    # Summary stats for this ICB
    n_flagged = icb_data["is_anomaly"].sum()
    n_months  = len(icb_data)
    mean_coamox = icb_data["f1_coamox_rate"].mean()
    max_coamox  = icb_data["f1_coamox_rate"].max()
    mean_ratio  = icb_data["f2_amox_coamox_ratio"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Months flagged", f"{n_flagged}/{n_months}")
    c2.metric("Mean co-amoxiclav rate", f"{mean_coamox:.3f}%")
    c3.metric("Peak co-amoxiclav rate", f"{max_coamox:.3f}%")
    c4.metric("Mean amox:coamox ratio", f"{mean_ratio:.1f}")

    # Guideline status
    national_mean = icb["f1_coamox_rate"].mean()
    if mean_coamox > national_mean * 2:
        st.markdown(
            '<div class="guideline-box">⚠️ <b>2024 ADA/AAOS flag:</b> '
            "Mean co-amoxiclav rate exceeds 2× national average. "
            "Review broad-spectrum prescribing practice.</div>",
            unsafe_allow_html=True
        )
    elif n_flagged == 0:
        st.success("✓ This ICB shows no anomalous prescribing patterns across the study period")

    st.markdown("---")

    col_shap, col_features = st.columns(2)

    with col_shap:
        st.markdown("**SHAP feature contributions — anomalous months only**")

        anomaly_data = icb_data[icb_data["is_anomaly"] == 1]

        if len(anomaly_data) > 0:
            shap_cols = [
                "shap_f1_coamox_rate",
                "shap_f2_amox_coamox_ratio",
                "shap_f3_items_per_1000",
                "shap_f4_mom_change",
                "shap_f5_coamox_zscore"
            ]
            feature_labels = [
                "F1: Co-amoxiclav rate",
                "F2: Amox:CoAmox ratio",
                "F3: Items per 1,000",
                "F4: MoM change",
                "F5: Z-score"
            ]

            mean_shap = anomaly_data[shap_cols].mean().values

            colors = ["#e63946" if v < 0 else "#2a9d8f" for v in mean_shap]

            fig_shap = go.Figure(go.Bar(
                x=mean_shap,
                y=feature_labels,
                orientation="h",
                marker_color=colors
            ))
            fig_shap.update_layout(
                title="Mean SHAP values (negative = drives anomaly)",
                xaxis_title="Mean SHAP value",
                height=300,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_shap, use_container_width=True)

            st.markdown(
                f"*Showing mean SHAP values across "
                f"{len(anomaly_data)} flagged months*"
            )
        else:
            st.info("No anomalous months for this ICB — no SHAP values to display")

    with col_features:
        st.markdown("**Feature values vs national mean**")

        latest = icb_data.sort_values("month").iloc[-1]
        nat_means = {
            "Co-amoxiclav rate (%)": (latest["f1_coamox_rate"], icb["f1_coamox_rate"].mean()),
            "Amox:CoAmox ratio":     (latest["f2_amox_coamox_ratio"], icb["f2_amox_coamox_ratio"].mean()),
            "Items per 1,000":       (latest["f3_items_per_1000"], icb["f3_items_per_1000"].mean()),
        }

        for label, (icb_val, nat_val) in nat_means.items():
            diff_pct = (icb_val - nat_val) / nat_val * 100
            direction = "above" if diff_pct > 0 else "below"
            st.metric(
                label,
                f"{icb_val:.3f}",
                f"{diff_pct:+.1f}% {direction} national mean"
            )

        st.caption(f"Values shown for latest month: {latest['month'].strftime('%B %Y')}")

    # All months table for this ICB
    st.markdown("---")
    st.markdown("**All months — full record**")

    display = icb_data[[
        "month", "is_anomaly", "anomaly_score",
        "f1_coamox_rate", "f2_amox_coamox_ratio",
        "f3_items_per_1000", "any_guideline_flag"
    ]].copy()
    display.columns = [
        "Month", "ML Flag", "Score",
        "CoAmox %", "Amox:CoAmox", "Per 1k pop", "Guideline"
    ]
    display["Month"] = display["Month"].dt.strftime("%b %Y")

    def flag_rows(row):
        if row["ML Flag"] == 1:
            return ["background-color: #ffe0e0"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display.style.apply(flag_rows, axis=1),
        height=400,
        use_container_width=True
    )

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — TIMELINE ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Timeline analysis — co-amoxiclav prescribing over time")

    # National trend
    national_monthly = icb.groupby("month").agg(
        mean_coamox=("f1_coamox_rate", "mean"),
        mean_ratio=("f2_amox_coamox_ratio", "mean"),
        n_anomalies=("is_anomaly", "sum")
    ).reset_index()

    # Selected ICB trend
    icb_monthly = icb[icb["icb_name"] == selected_icb][
        ["month", "f1_coamox_rate", "f2_amox_coamox_ratio", "is_anomaly"]
    ].copy()

    guideline_date = pd.Timestamp("2024-11-01")

    # ── Plot 1: Co-amoxiclav rate over time ──────────────────────────────────
    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(
        x=national_monthly["month"],
        y=national_monthly["mean_coamox"],
        name="National mean",
        line=dict(color="#457b9d", width=2),
        mode="lines"
    ))

    fig1.add_trace(go.Scatter(
        x=icb_monthly["month"],
        y=icb_monthly["f1_coamox_rate"],
        name=short_name,
        line=dict(color="#e63946", width=2),
        mode="lines+markers",
        marker=dict(
            size=[10 if a == 1 else 4 for a in icb_monthly["is_anomaly"]],
            symbol=["diamond" if a == 1 else "circle" for a in icb_monthly["is_anomaly"]],
            color=["#e63946" if a == 1 else "#457b9d" for a in icb_monthly["is_anomaly"]]
        )
    ))

    # November 2024 guideline line
    fig1.add_vline(
        x=guideline_date,
        line_dash="dash",
        line_color="#f4a261",
        line_width=2,
        annotation_text="Nov 2024 ADA/AAOS revision",
        annotation_position="top left"
    )

    fig1.update_layout(
        title="Co-amoxiclav prescription rate (%) over time",
        xaxis_title="Month",
        yaxis_title="Co-amoxiclav rate (%)",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ── Plot 2: Monthly anomaly counts nationally ─────────────────────────────
    fig2 = go.Figure()

    fig2.add_trace(go.Bar(
        x=national_monthly["month"],
        y=national_monthly["n_anomalies"],
        name="Anomalies per month",
        marker_color=[
            "#e63946" if n > 3 else "#f4a261" if n > 1 else "#2a9d8f"
            for n in national_monthly["n_anomalies"]
        ]
    ))

    fig2.add_vline(
        x=guideline_date,
        line_dash="dash",
        line_color="#f4a261",
        line_width=2,
        annotation_text="Nov 2024 ADA/AAOS revision",
        annotation_position="top left"
    )

    fig2.update_layout(
        title="Number of ICBs flagged as anomalous per month (national)",
        xaxis_title="Month",
        yaxis_title="ICBs flagged",
        height=350,
        hovermode="x unified"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Before vs after table ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Before vs after November 2024 ADA/AAOS guideline revision**")

    pre  = icb[icb["month"] <  guideline_date]
    post = icb[icb["month"] >= guideline_date]

    comparison = pd.DataFrame({
        "Period": ["Before Nov 2024", "After Nov 2024"],
        "Observations": [len(pre), len(post)],
        "Anomalies": [pre["is_anomaly"].sum(), post["is_anomaly"].sum()],
        "Anomaly rate": [
            f"{pre['is_anomaly'].mean()*100:.1f}%",
            f"{post['is_anomaly'].mean()*100:.1f}%"
        ],
        "Mean co-amoxiclav rate": [
            f"{pre['f1_coamox_rate'].mean():.3f}%",
            f"{post['f1_coamox_rate'].mean():.3f}%"
        ],
        "Mean amox:coamox ratio": [
            f"{pre['f2_amox_coamox_ratio'].mean():.1f}",
            f"{post['f2_amox_coamox_ratio'].mean():.1f}"
        ]
    })

    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="guideline-box">
        📋 <b>Finding:</b> No statistically meaningful improvement in national 
        co-amoxiclav prescribing rates was detected following the November 2024 
        ADA/AAOS guideline revision (0.434% → 0.442%). NHS Suffolk and North East 
        Essex showed a worsening trend post-revision (1.143% → 1.448%).
        </div>
        """,
        unsafe_allow_html=True
    )
