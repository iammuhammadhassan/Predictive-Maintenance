import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="MainTenAI",
    page_icon="🛠️",
    layout="wide"
)

# ==========================================
# VISUAL THEME
# ==========================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Outfit:wght@400;500;700&display=swap');

    :root {
        --bg-1: #f5f7f4;
        --bg-2: #eaf4f1;
        --ink: #13212a;
        --muted: #4a5d67;
        --accent: #0f766e;
        --accent-2: #f59e0b;
        --critical: #c2410c;
        --card: rgba(255, 255, 255, 0.72);
        --stroke: rgba(19, 33, 42, 0.10);
    }

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .stApp {
        color: var(--ink);
        background:
            radial-gradient(circle at 12% 18%, rgba(15, 118, 110, 0.16), transparent 36%),
            radial-gradient(circle at 85% 12%, rgba(245, 158, 11, 0.18), transparent 30%),
            linear-gradient(135deg, var(--bg-1), var(--bg-2));
        background-attachment: fixed;
    }

    .stApp p,
    .stApp li,
    .stApp label,
    .stApp .stMarkdown,
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stMarkdownContainer"] span,
    .stApp [data-testid="stCaptionContainer"] {
        color: #1b3340;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 118, 110, 0.16), rgba(255, 255, 255, 0.85));
        border-right: 1px solid var(--stroke);
    }

    section[data-testid="stSidebar"] * {
        color: #102a36;
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    section[data-testid="stSidebar"] label {
        color: #0b2f3c !important;
        font-weight: 600;
    }

    section[data-testid="stSidebar"] [data-baseweb="select"] > div,
    section[data-testid="stSidebar"] [data-baseweb="input"] > div,
    section[data-testid="stSidebar"] [data-baseweb="slider"] {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
    }

    section[data-testid="stSidebar"] .stRadio > div,
    section[data-testid="stSidebar"] .stSelectbox > div,
    section[data-testid="stSidebar"] .stSlider > div {
        color: #102a36;
    }

    .hero {
        border: 1px solid var(--stroke);
        background: linear-gradient(130deg, rgba(15, 118, 110, 0.15), rgba(255, 255, 255, 0.86) 58%);
        border-radius: 20px;
        padding: 26px 28px;
        margin-bottom: 12px;
        box-shadow: 0 12px 40px rgba(19, 33, 42, 0.08);
        animation: rise 650ms ease-out;
    }

    .hero h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(1.6rem, 2.8vw, 2.4rem);
        font-weight: 700;
        letter-spacing: 0.4px;
        margin: 0;
        color: #0b2f3c;
    }

    .hero p {
        margin: 8px 0 0;
        color: var(--muted);
        font-size: 1.02rem;
    }

    .metric-card {
        border: 1px solid var(--stroke);
        background: var(--card);
        border-radius: 16px;
        padding: 14px 16px;
        box-shadow: 0 8px 22px rgba(19, 33, 42, 0.05);
        backdrop-filter: blur(5px);
        animation: rise 700ms ease-out;
    }

    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: var(--muted);
        margin-bottom: 4px;
    }

    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.45rem;
        font-weight: 700;
        color: #0d3a4a;
    }

    .analysis-section {
        color: #102a36;
    }

    .analysis-section h2,
    .analysis-section h3,
    .analysis-section h4,
    .analysis-section p,
    .analysis-section li,
    .analysis-section span,
    .analysis-section label {
        color: #102a36;
    }

    .analysis-section .stDownloadButton button,
    .analysis-section .stButton button {
        background: linear-gradient(135deg, #0f766e, #115e59);
        color: #ffffff !important;
        border: 1px solid rgba(15, 118, 110, 0.35);
        border-radius: 12px;
        font-weight: 700;
    }

    .analysis-section .stDownloadButton button:hover,
    .analysis-section .stButton button:hover {
        filter: brightness(1.04);
    }

    .analysis-section [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.80);
        border: 1px solid var(--stroke);
        border-radius: 14px;
    }

    .analysis-section [data-testid="stExpander"] summary,
    .analysis-section [data-testid="stExpander"] summary p {
        color: #0b2f3c !important;
        font-weight: 700;
    }

    .analysis-section .stDataFrame,
    .analysis-section [data-testid="stDataFrame"] {
        background: rgba(255, 255, 255, 0.88);
    }

    .status-card {
        border: 1px solid var(--stroke);
        background: rgba(255, 255, 255, 0.82);
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 10px 24px rgba(19, 33, 42, 0.06);
    }

    .status-badge {
        display: inline-block;
        padding: 8px 12px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.4px;
        animation: pulse 1.8s ease-in-out infinite;
    }

    .chip {
        display: inline-block;
        margin: 5px 7px 0 0;
        padding: 7px 10px;
        border-radius: 10px;
        border: 1px solid rgba(194, 65, 12, 0.3);
        background: rgba(194, 65, 12, 0.08);
        color: #9a3412;
        font-size: 0.88rem;
        font-weight: 500;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--stroke);
        border-radius: 14px;
        overflow: hidden;
    }

    @keyframes rise {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(15, 118, 110, 0.20); }
        50% { box-shadow: 0 0 0 9px rgba(15, 118, 110, 0.0); }
    }

    @media (max-width: 900px) {
        .hero {
            padding: 18px;
        }
        .metric-value {
            font-size: 1.2rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# LOAD MODELS
# ==========================================
@st.cache_resource
def load_models():
    return joblib.load("models/random_forest_model.pkl"), joblib.load("models/xgboost_model.pkl")


@st.cache_data
def load_dataset():
    base_dir = Path(__file__).resolve().parent
    candidates = [base_dir / "dataset" / "predictive_maintenance.csv", base_dir / "predictive_maintenance.csv"]
    for candidate in candidates:
        if candidate.exists():
            return pd.read_csv(candidate)
    raise FileNotFoundError("Could not find predictive_maintenance.csv")


@st.cache_data
def build_engineered_dataset(dataset: pd.DataFrame):
    engineered = dataset.copy()
    engineered["Temp_Diff"] = engineered["Process temperature [K]"] - engineered["Air temperature [K]"]
    engineered["Torque_Wear"] = engineered["Torque [Nm]"] * engineered["Tool wear [min]"]
    engineered["Torque_RPM"] = engineered["Torque [Nm]"] * engineered["Rotational speed [rpm]"]
    engineered = pd.get_dummies(engineered, columns=["Type"], drop_first=True)
    return engineered


def create_train_test_exports(engineered_dataset: pd.DataFrame):
    feature_columns = [column for column in engineered_dataset.columns if column not in ["Target", "Failure Type", "Product ID", "UDI"]]
    X = engineered_dataset[feature_columns].copy()
    y = engineered_dataset["Target"].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    train_df = X_train.copy()
    train_df["Target"] = y_train.values
    test_df = X_test.copy()
    test_df["Target"] = y_test.values
    return train_df, test_df


def save_engineered_exports(train_frame: pd.DataFrame, test_frame: pd.DataFrame):
    base_dir = Path(__file__).resolve().parent
    train_path = base_dir / "predictive_maintenance_train_engineered.csv"
    test_path = base_dir / "predictive_maintenance_test_engineered.csv"
    train_frame.to_csv(train_path, index=False)
    test_frame.to_csv(test_path, index=False)
    return train_path, test_path


def style_chart(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0.92)",
        plot_bgcolor="rgba(255,255,255,0.72)",
        font=dict(family="Outfit, sans-serif", color="#102a36", size=13),
        title_font=dict(color="#102a36"),
        legend=dict(font=dict(color="#102a36"), bgcolor="rgba(255,255,255,0.72)"),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(16, 42, 54, 0.10)",
        zeroline=False,
        linecolor="rgba(16, 42, 54, 0.18)",
        tickfont=dict(color="#102a36"),
        title_font=dict(color="#102a36"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(16, 42, 54, 0.10)",
        zeroline=False,
        linecolor="rgba(16, 42, 54, 0.18)",
        tickfont=dict(color="#102a36"),
        title_font=dict(color="#102a36"),
    )
    return fig


def make_mpl_figure(figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize, dpi=160)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.grid(True, axis="y", color="#d6e2e8", linewidth=0.8, alpha=0.9)
    ax.grid(False, axis="x")
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
    ax.tick_params(colors="#102a36", labelsize=10)
    ax.title.set_color("#102a36")
    ax.xaxis.label.set_color("#102a36")
    ax.yaxis.label.set_color("#102a36")
    return fig, ax


def render_mpl(fig):
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


rf, xgb = load_models()
dataset_df = load_dataset()
engineered_df = build_engineered_dataset(dataset_df)
train_df, test_df = create_train_test_exports(engineered_df)
train_csv_path, test_csv_path = save_engineered_exports(train_df, test_df)

# ==========================================
# TITLE
# ==========================================
st.markdown(
    """
    <div class="hero">
        <h1>MainTenAI</h1>
        <p>Elegant, real-time machine risk intelligence powered by Random Forest and XGBoost.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================
# SIDEBAR INPUTS
# ==========================================
st.sidebar.markdown("## Machine Parameters")
st.sidebar.caption("Tune the operating profile and compare model behavior in real time.")

air_temp = st.sidebar.slider("Air Temperature (K)", 295.0, 305.0, 300.0)
process_temp = st.sidebar.slider("Process Temperature (K)", 305.0, 315.0, 310.0)
rpm = st.sidebar.slider("Rotational Speed (RPM)", 1100, 2900, 1500)
torque = st.sidebar.slider("Torque (Nm)", 0.0, 80.0, 40.0)
tool_wear = st.sidebar.slider("Tool Wear (min)", 0, 260, 100)

machine_type = st.sidebar.selectbox("Machine Type", ["H", "M", "L"])
model_choice = st.sidebar.radio("Model", ["Random Forest", "XGBoost"])

# ==========================================
# FEATURE ENGINEERING
# ==========================================
temp_diff = process_temp - air_temp
torque_wear = torque * tool_wear
torque_rpm = torque * rpm

type_L = 1 if machine_type == "L" else 0
type_M = 1 if machine_type == "M" else 0

# ==========================================
# STANDARD INPUT (FOR XGBOOST)
# ==========================================
input_df = pd.DataFrame({
    'Air_temperature_K': [air_temp],
    'Process_temperature_K': [process_temp],
    'Rotational_speed_rpm': [rpm],
    'Torque_Nm': [torque],
    'Tool_wear_min': [tool_wear],
    'Temp_Diff': [temp_diff],
    'Torque_Wear': [torque_wear],
    'Torque_RPM': [torque_rpm],
    'Type_L': [type_L],
    'Type_M': [type_M]
})

# ==========================================
# RF FORMAT CONVERTER
# ==========================================
def convert_for_rf(df):
    return pd.DataFrame({
        'Air temperature [K]': df['Air_temperature_K'],
        'Process temperature [K]': df['Process_temperature_K'],
        'Rotational speed [rpm]': df['Rotational_speed_rpm'],
        'Torque [Nm]': df['Torque_Nm'],
        'Tool wear [min]': df['Tool_wear_min'],
        'Temp_Diff': df['Temp_Diff'],
        'Torque_Wear': df['Torque_Wear'],
        'Torque_RPM': df['Torque_RPM'],
        'Type_L': df['Type_L'],
        'Type_M': df['Type_M']
    })

# ==========================================
# MODEL SELECTION
# ==========================================
if model_choice == "Random Forest":
    model_input = convert_for_rf(input_df)
    model = rf
else:
    model_input = input_df
    model = xgb

# ==========================================
# PREDICTION
# ==========================================
prediction = model.predict(model_input)[0]
probability = model.predict_proba(model_input)[0][1]
health = round((1 - probability) * 100, 2)

# ==========================================
# STATUS HELPERS
# ==========================================
if probability < 0.2:
    risk_label = "LOW RISK"
    risk_color = "#166534"
    risk_bg = "rgba(22, 101, 52, 0.12)"
elif probability < 0.5:
    risk_label = "MEDIUM RISK"
    risk_color = "#a16207"
    risk_bg = "rgba(245, 158, 11, 0.17)"
elif probability < 0.8:
    risk_label = "HIGH RISK"
    risk_color = "#b45309"
    risk_bg = "rgba(217, 119, 6, 0.18)"
else:
    risk_label = "CRITICAL RISK"
    risk_color = "#b91c1c"
    risk_bg = "rgba(220, 38, 38, 0.16)"

# ==========================================
# TABS
# ==========================================
prediction_tab, dataset_tab, analysis_tab = st.tabs(["Prediction", "Dataset Viewer", "Analysis"])

with prediction_tab:
    # ==========================================
    # METRICS
    # ==========================================
    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Failure Probability</div>
            <div class="metric-value">{probability:.2%}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    col2.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Health Score</div>
            <div class="metric-value">{health:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    col3.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Torque</div>
            <div class="metric-value">{torque:.1f} Nm</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    col4.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Tool Wear</div>
            <div class="metric-value">{tool_wear} min</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # ==========================================
    # RISK VISUALS
    # ==========================================
    left, right = st.columns([2, 1])

    with left:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "Failure Risk (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#0f766e'},
                'steps': [
                    {'range': [0, 30], 'color': "#86efac"},
                    {'range': [30, 60], 'color': "#fde047"},
                    {'range': [60, 80], 'color': "#fdba74"},
                    {'range': [80, 100], 'color': "#fca5a5"}
                ],
                'threshold': {'line': {'color': '#7c2d12', 'width': 4}, 'value': probability * 100}
            }
        ))
        fig.update_layout(
            margin=dict(l=20, r=20, t=45, b=20),
            height=350,
            paper_bgcolor='rgba(255,255,255,0.55)',
            font=dict(family='Outfit, sans-serif', color='#163441')
        )
        st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # STATUS PANEL
    # ==========================================
    with right:
        st.markdown("### Machine Status")
        st.markdown(
            f"""
            <div class="status-card">
                <div class="status-badge" style="color:{risk_color}; background:{risk_bg}; border:1px solid {risk_color}33;">
                    {risk_label}
                </div>
                <p style="margin:12px 0 4px;color:#284451;"><b>Model:</b> {model_choice}</p>
                <p style="margin:0;color:#44606b;">The system continuously estimates failure likelihood from current operating conditions.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ==========================================
    # RESULT
    # ==========================================
    st.divider()

    if prediction == 1:
        st.error(f"⚠️ FAILURE PREDICTED ({probability:.2%})")
    else:
        st.success(f"✅ MACHINE HEALTHY ({(1 - probability):.2%})")

    # ==========================================
    # SNAPSHOT
    # ==========================================
    st.subheader("Machine Snapshot")
    with st.expander("View model-ready feature vector", expanded=True):
        st.dataframe(model_input, use_container_width=True)

    # ==========================================
    # ENGINEERING FEATURES
    # ==========================================
    st.subheader("Engineering Indicators")

    eng_df = pd.DataFrame({
        "Feature": ["Temp Diff", "Torque Wear", "Torque RPM"],
        "Value": [temp_diff, torque_wear, torque_rpm]
    })

    bar_fig = px.bar(
        eng_df,
        x="Feature",
        y="Value",
        color="Feature",
        color_discrete_sequence=["#0f766e", "#0891b2", "#f59e0b"]
    )
    bar_fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=30, b=20), yaxis_title="Magnitude", xaxis_title="")
    style_chart(bar_fig)
    st.plotly_chart(bar_fig, use_container_width=True)

    # ==========================================
    # AI INSIGHTS
    # ==========================================
    st.subheader("AI Insights")

    insights = []

    if torque > 50:
        insights.append("High torque detected")
    if tool_wear > 150:
        insights.append("High tool wear detected")
    if temp_diff > 12:
        insights.append("Possible overheating risk")
    if rpm < 1300:
        insights.append("Low RPM under load")

    if len(insights) == 0:
        st.success("No major risk detected")
    else:
        chips_html = "".join([f"<span class='chip'>{item}</span>" for item in insights])
        st.markdown(chips_html, unsafe_allow_html=True)

with dataset_tab:
    st.subheader("Dataset Viewer")
    st.caption("Browse the raw maintenance records, filter by machine type or failure state, and export a cleaned slice.")

    dataset_col1, dataset_col2, dataset_col3 = st.columns(3)
    with dataset_col1:
        type_options = sorted(dataset_df["Type"].dropna().astype(str).unique().tolist())
        type_filter = st.multiselect("Filter by Type", type_options, default=type_options)
    with dataset_col2:
        failure_options = sorted(dataset_df["Failure Type"].dropna().astype(str).unique().tolist())
        failure_filter = st.multiselect("Filter by Failure Type", failure_options, default=failure_options)
    with dataset_col3:
        target_filter = st.selectbox("Target", ["All", "Failure Only", "No Failure Only"])

    filtered_dataset = dataset_df.copy()
    if type_filter:
        filtered_dataset = filtered_dataset[filtered_dataset["Type"].astype(str).isin(type_filter)]
    if failure_filter:
        filtered_dataset = filtered_dataset[filtered_dataset["Failure Type"].astype(str).isin(failure_filter)]
    if target_filter == "Failure Only":
        filtered_dataset = filtered_dataset[filtered_dataset["Target"] == 1]
    elif target_filter == "No Failure Only":
        filtered_dataset = filtered_dataset[filtered_dataset["Target"] == 0]

    view_col1, view_col2, view_col3, view_col4 = st.columns(4)
    view_col1.markdown(f"<div class='metric-card'><div class='metric-label'>Rows</div><div class='metric-value'>{len(filtered_dataset):,}</div></div>", unsafe_allow_html=True)
    view_col2.markdown(f"<div class='metric-card'><div class='metric-label'>Columns</div><div class='metric-value'>{filtered_dataset.shape[1]}</div></div>", unsafe_allow_html=True)
    view_col3.markdown(f"<div class='metric-card'><div class='metric-label'>Failure Rate</div><div class='metric-value'>{filtered_dataset['Target'].mean():.2%}</div></div>", unsafe_allow_html=True)
    view_col4.markdown(f"<div class='metric-card'><div class='metric-label'>Machine Types</div><div class='metric-value'>{filtered_dataset['Type'].nunique()}</div></div>", unsafe_allow_html=True)

    st.dataframe(filtered_dataset, use_container_width=True, height=420)
    st.download_button(
        "Download filtered data as CSV",
        data=filtered_dataset.to_csv(index=False).encode("utf-8"),
        file_name="predictive_maintenance_filtered.csv",
        mime="text/csv"
    )

with analysis_tab:
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.subheader("Analysis")
    st.caption("A deeper visual scan of class balance, machine types, feature relationships, and engineered feature structure.")

    analysis_col1, analysis_col2, analysis_col3, analysis_col4 = st.columns(4)
    analysis_col1.markdown(f"<div class='metric-card'><div class='metric-label'>Records</div><div class='metric-value'>{len(dataset_df):,}</div></div>", unsafe_allow_html=True)
    analysis_col2.markdown(f"<div class='metric-card'><div class='metric-label'>Failures</div><div class='metric-value'>{dataset_df['Target'].sum():,}</div></div>", unsafe_allow_html=True)
    analysis_col3.markdown(f"<div class='metric-card'><div class='metric-label'>Failure Rate</div><div class='metric-value'>{dataset_df['Target'].mean():.2%}</div></div>", unsafe_allow_html=True)
    analysis_col4.markdown(f"<div class='metric-card'><div class='metric-label'>Avg Tool Wear</div><div class='metric-value'>{dataset_df['Tool wear [min]'].mean():.1f}</div></div>", unsafe_allow_html=True)

    split_col1, split_col2, split_col3 = st.columns(3)
    split_col1.markdown(f"<div class='metric-card'><div class='metric-label'>Train Rows</div><div class='metric-value'>{len(train_df):,}</div></div>", unsafe_allow_html=True)
    split_col2.markdown(f"<div class='metric-card'><div class='metric-label'>Test Rows</div><div class='metric-value'>{len(test_df):,}</div></div>", unsafe_allow_html=True)
    split_col3.markdown(f"<div class='metric-card'><div class='metric-label'>Train/Test Ratio</div><div class='metric-value'>{len(train_df)/len(test_df):.1f}:1</div></div>", unsafe_allow_html=True)

    st.markdown("### Class Balance")
    balance = dataset_df["Target"].map({0: "No Failure", 1: "Failure"}).value_counts().reindex(["No Failure", "Failure"])
    fig, ax = make_mpl_figure((8, 4.5))
    sns.barplot(x=balance.index, y=balance.values, ax=ax, palette=["#0f766e", "#f59e0b"])
    ax.set_xlabel("Target")
    ax.set_ylabel("Count")
    ax.set_title("Failure vs No Failure")
    for index, value in enumerate(balance.values):
        ax.text(index, value, f"{int(value):,}", ha="center", va="bottom", color="#102a36", fontweight="bold")
    render_mpl(fig)

    st.markdown("### Machine Type Distribution")
    type_counts = dataset_df["Type"].value_counts().reindex(["H", "M", "L"])
    fig, ax = make_mpl_figure((8, 4.5))
    sns.barplot(x=type_counts.index, y=type_counts.values, ax=ax, palette=["#0f766e", "#0891b2", "#f59e0b"])
    ax.set_xlabel("Machine Type")
    ax.set_ylabel("Count")
    ax.set_title("Machine Type Frequency")
    for index, value in enumerate(type_counts.values):
        ax.text(index, value, f"{int(value):,}", ha="center", va="bottom", color="#102a36", fontweight="bold")
    render_mpl(fig)

    st.markdown("### Failure Type Breakdown")
    failure_type_counts = dataset_df["Failure Type"].value_counts()
    fig, ax = make_mpl_figure((10, 5))
    sns.barplot(y=failure_type_counts.index, x=failure_type_counts.values, ax=ax, palette="viridis")
    ax.set_xlabel("Count")
    ax.set_ylabel("Failure Type")
    ax.set_title("Failure Type Frequency")
    for index, value in enumerate(failure_type_counts.values):
        ax.text(value, index, f" {int(value):,}", va="center", color="#102a36", fontweight="bold")
    render_mpl(fig)

    st.markdown("### Torque and Wear Relationship")
    fig, ax = make_mpl_figure((9, 5.5))
    sns.scatterplot(
        data=dataset_df,
        x="Tool wear [min]",
        y="Torque [Nm]",
        hue=dataset_df["Target"].map({0: "No Failure", 1: "Failure"}),
        palette={"No Failure": "#0f766e", "Failure": "#f59e0b"},
        alpha=0.65,
        ax=ax,
        s=36,
    )
    ax.set_xlabel("Tool Wear (min)")
    ax.set_ylabel("Torque (Nm)")
    ax.set_title("Torque vs Tool Wear")
    ax.legend(title="Outcome", frameon=True, facecolor="white", edgecolor="#cbd5e1")
    render_mpl(fig)

    feature_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Temp_Diff",
        "Torque_Wear",
        "Torque_RPM",
    ]
    engineered_stats = engineered_df[feature_cols].describe().transpose()[["mean", "std", "min", "max"]]
    st.markdown("### Engineered Feature Summary")
    st.dataframe(engineered_stats, use_container_width=True)

    st.markdown("### Correlation Matrix")
    corr_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Target"
    ]
    fig, ax = make_mpl_figure((9, 7))
    sns.heatmap(
        dataset_df[corr_cols].corr(numeric_only=True),
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Correlation"},
    )
    ax.set_title("Correlation Between Numeric Features")
    render_mpl(fig)

    st.markdown("### Engineered Feature Correlations")
    engineered_corr_cols = ["Temp_Diff", "Torque_Wear", "Torque_RPM", "Target"]
    fig, ax = make_mpl_figure((7, 5.5))
    sns.heatmap(
        engineered_df[engineered_corr_cols].corr(numeric_only=True),
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Correlation"},
    )
    ax.set_title("Engineered Features vs Target")
    render_mpl(fig)

    st.markdown("### Failure Rate by Machine Type")
    failure_rate = dataset_df.groupby("Type")["Target"].mean().reindex(["H", "M", "L"])
    fig, ax = make_mpl_figure((8, 4.5))
    sns.barplot(x=failure_rate.index, y=failure_rate.values, ax=ax, palette=["#0f766e", "#0891b2", "#f59e0b"])
    ax.set_xlabel("Machine Type")
    ax.set_ylabel("Failure Rate")
    ax.set_title("Failure Rate by Type")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    for index, value in enumerate(failure_rate.values):
        ax.text(index, value, f"{value:.1%}", ha="center", va="bottom", color="#102a36", fontweight="bold")
    render_mpl(fig)

    st.markdown("### Feature Impact by Failure")
    box_cols = st.columns(3)
    box_features = ["Torque [Nm]", "Tool wear [min]", "Rotational speed [rpm]"]
    for column, feature in zip(box_cols, box_features):
        with column:
            fig, ax = make_mpl_figure((7, 4.8))
            sns.boxplot(
                data=dataset_df,
                x="Target",
                y=feature,
                ax=ax,
                palette=["#0f766e", "#f59e0b"],
            )
            ax.set_xlabel("Target")
            ax.set_ylabel(feature)
            ax.set_title(feature)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["No Failure", "Failure"])
            render_mpl(fig)

    st.markdown("### Engineered Train/Test Preview")
    preview_left, preview_right = st.columns(2)
    with preview_left:
        st.markdown("#### Training Sample")
        st.dataframe(train_df.head(10), use_container_width=True)
    with preview_right:
        st.markdown("#### Testing Sample")
        st.dataframe(test_df.head(10), use_container_width=True)

    with st.expander("Dataset summary statistics"):
        st.dataframe(dataset_df.describe(include='all').transpose(), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# FOOTER
# ==========================================
st.divider()
st.caption("MainTenAI | RF + XGBoost | Crafted for clarity and fast decisions")