

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

st.set_page_config(page_title="CarApp | UK Road Collisions", page_icon="🚗", layout="wide")

# Presentation formatting. No external assets or services are required.
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; max-width: 1240px;}
      [data-testid="stMetric"] {border: 1px solid #DDE4EB; border-radius: 12px; padding: 12px;}
      .hero {background: linear-gradient(115deg, #152941, #166277); color: white;
             padding: 25px 30px; border-radius: 14px; margin-bottom: 18px;}
      .hero h1 {color: white; margin: 0 0 7px 0;}
      .hero p {color: #E1EDF2; margin: 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🚗 CarApp</h1>
      <p>Great Britain road collision analysis (2025) · UWS Advanced Data Science</p>
    </div>
    """,
    unsafe_allow_html=True,
)

SEVERITY_LABELS = {1: "Fatal", 2: "Serious", 3: "Slight"}
SEVERITY_COLORS = {"Fatal": "#BC4651", "Serious": "#E09B38", "Slight": "#21867C"}

# Code descriptions derived from the DfT STATS19 field definitions.
FACTORS = {
    "Weather conditions": (
        "weather_conditions",
        {1: "Fine", 2: "Rain", 3: "Snow", 4: "Fine / high winds",
         5: "Rain / high winds", 6: "Snow / high winds", 7: "Fog or mist",
         8: "Other", 9: "Unknown"},
    ),
    "Lighting conditions": (
        "light_conditions",
        {-1: "Not reported", 1: "Daylight", 4: "Dark / lights lit",
         5: "Dark / lights unlit", 6: "Dark / no lighting",
         7: "Dark / lighting unknown"},
    ),
    "Road surface": (
        "road_surface_conditions",
        {-1: "Not reported", 1: "Dry", 2: "Wet or damp", 3: "Snow",
         4: "Frost or ice", 5: "Flood", 9: "Unknown"},
    ),
    "Speed limit": ("speed_limit", {}),
    "Road type": (
        "road_type",
        {1: "Roundabout", 2: "One-way street", 3: "Dual carriageway",
         6: "Single carriageway", 7: "Slip road", 9: "Unknown"},
    ),
    "Urban or rural": (
        "urban_or_rural_area",
        {1: "Urban", 2: "Rural", 3: "Unallocated"},
    ),
}

MODEL_FEATURES = [
    "speed_limit", "road_type", "light_conditions", "weather_conditions",
    "road_surface_conditions", "urban_or_rural_area",
]
CATEGORICAL_FEATURES = [
    "road_type", "light_conditions", "weather_conditions",
    "road_surface_conditions", "urban_or_rural_area",
]


@st.cache_data(show_spinner=False)
def load_csv(file_contents: bytes) -> pd.DataFrame:
    """Read the uploaded data without modifying its original columns."""
    from io import BytesIO
    return pd.read_csv(BytesIO(file_contents))


@st.cache_data(show_spinner="Training and evaluating the Decision Tree...")
def evaluate_decision_tree(data: pd.DataFrame) -> dict:
    """Reproduce the notebook's original stratified 80/20 Decision Tree evaluation.

    Importantly, one-hot encoding is fitted only on training data inside a
    pipeline. The held-out test set is never used to fit the model.
    """
    valid = data["collision_severity"].isin([1, 2, 3])
    model_data = data.loc[valid, MODEL_FEATURES + ["collision_severity"]].copy()
    model_data["speed_limit"] = pd.to_numeric(model_data["speed_limit"], errors="coerce")
    model_data = model_data.dropna(subset=["speed_limit"])

    if len(model_data) < 20:
        raise ValueError("The uploaded file has too few valid collision records for modelling.")

    X = model_data[MODEL_FEATURES]
    y = model_data["collision_severity"].isin([1, 2]).astype(int)
    if y.nunique() < 2:
        raise ValueError("The dataset must contain both Slight and Serious/Fatal collisions.")

    # Match the exact method and hyperparameters in the coursework notebook.
    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("numerical", "passthrough", ["speed_limit"]),
    ])
    classifier = DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=100, class_weight="balanced", random_state=42
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    return {
        "model": pipeline,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": accuracy_score(y_test, predictions),
        "report": classification_report(
            y_test, predictions, labels=[0, 1],
            target_names=["Slight", "Serious/Fatal"],
            output_dict=True, zero_division=0,
        ),
        "matrix": confusion_matrix(y_test, predictions, labels=[0, 1]),
    }


def format_factor(df: pd.DataFrame, column: str, lookup: dict) -> pd.Series:
    if lookup:
        return df[column].map(lookup).fillna("Other / unmapped")
    return pd.to_numeric(df[column], errors="coerce").astype("Int64").astype(str) + " mph"


st.sidebar.header("Dataset")
uploaded = st.sidebar.file_uploader(
    "Upload DfT 2025 collision CSV", type=["csv"],
    help="Use dft-road-casualty-statistics-collision-2025.csv",
)
st.sidebar.caption("The CSV is processed within the running app; it is not uploaded to GitHub.")
st.sidebar.divider()
st.sidebar.markdown("**Data source:** UK Department for Transport, Road Safety Open Data (2025).")

if uploaded is None:
    st.info("To begin, upload `dft-road-casualty-statistics-collision-2025.csv` using the sidebar.")
    st.write("The dashboard includes data quality checks, interactive analysis, and Decision Tree results.")
    st.stop()

try:
    df = load_csv(uploaded.getvalue())
except Exception as exc:
    st.error(f"Unable to read the uploaded CSV: {exc}")
    st.stop()

needed = set(MODEL_FEATURES + ["collision_severity"])
missing = sorted(needed - set(df.columns))
if missing:
    st.error("The CSV is missing required columns: " + ", ".join(missing))
    st.stop()

# Keep severity codes numeric; retain original source rows unchanged.
df = df.copy()
df["collision_severity"] = pd.to_numeric(df["collision_severity"], errors="coerce")

n_fatal = int(df["collision_severity"].eq(1).sum())
n_serious = int(df["collision_severity"].eq(2).sum())
n_slight = int(df["collision_severity"].eq(3).sum())
metric_cols = st.columns(4)
for col, caption, count in zip(
    metric_cols,
    ["Recorded collisions", "Fatal", "Serious", "Slight"],
    [len(df), n_fatal, n_serious, n_slight],
):
    col.metric(caption, f"{count:,}")
st.caption("These are counts of police-recorded injury collisions, not casualties or traffic risk.")

tab_overview, tab_explore, tab_quality, tab_ml = st.tabs([
    "📊 Dataset overview", "🔎 Collision analysis", "🧹 Data preprocessing", "🌳 Decision Tree",
])

with tab_overview:
    left, right = st.columns(2)
    with left:
        st.subheader("Severity breakdown")
        counts = pd.DataFrame({
            "Severity": ["Slight", "Serious", "Fatal"],
            "Collisions": [n_slight, n_serious, n_fatal],
        })
        chart = px.bar(
            counts, x="Severity", y="Collisions", color="Severity",
            color_discrete_map=SEVERITY_COLORS, text="Collisions",
        )
        chart.update_traces(texttemplate="%{text:,}", textposition="outside")
        chart.update_layout(showlegend=False)
        st.plotly_chart(chart, use_container_width=True)
    with right:
        st.subheader("By speed limit")
        speeds = (
            df["speed_limit"].value_counts().sort_index()
            .rename_axis("Speed limit (mph)").reset_index(name="Collisions")
        )
        st.plotly_chart(
            px.bar(speeds, x="Speed limit (mph)", y="Collisions"),
            use_container_width=True,
        )
    st.subheader("First 20 records")
    st.dataframe(df.head(20), hide_index=True, use_container_width=True)

with tab_explore:
    st.subheader("Compare conditions and severity")
    st.write("Choose an environmental or road factor, a severity group, and how to display the result.")
    col1, col2, col3 = st.columns(3)
    with col1:
        factor = st.selectbox("Road or environmental factor", list(FACTORS))
    with col2:
        selection = st.selectbox(
            "Collision severity", ["Fatal", "Serious", "Slight", "Serious/Fatal", "All"]
        )
    with col3:
        view_mode = st.selectbox(
            "View results as", ["Number of collisions", "Percentage within condition"],
            disabled=(selection == "All"),
        )

    col_name, labels = FACTORS[factor]
    categories = format_factor(df, col_name, labels)
    if selection == "All":
        matched = pd.Series(True, index=df.index)
        view_mode = "Number of collisions"
    elif selection == "Serious/Fatal":
        matched = df["collision_severity"].isin([1, 2])
    else:
        code = {"Fatal": 1, "Serious": 2, "Slight": 3}[selection]
        matched = df["collision_severity"].eq(code)

    work = pd.DataFrame({"Condition": categories, "Match": matched})
    totals = work.groupby("Condition").size()
    subcounts = work.loc[work["Match"]].groupby("Condition").size()
    result = pd.DataFrame({
        "Selected collisions": subcounts,
        "Total collisions in condition": totals,
    }).fillna(0).astype(int)
    result["Percentage within condition"] = (
        100 * result["Selected collisions"] / result["Total collisions in condition"]
    ).round(2)
    result = result.reset_index().sort_values("Selected collisions", ascending=False)

    measure = "Selected collisions" if view_mode == "Number of collisions" else "Percentage within condition"
    chart_col, data_col = st.columns([1.55, 1])
    with chart_col:
        plot = result.sort_values(measure)
        figure = px.bar(plot, x=measure, y="Condition", orientation="h")
        figure.update_layout(height=455, yaxis_title=None)
        st.plotly_chart(figure, use_container_width=True)
    with data_col:
        st.dataframe(result, hide_index=True, use_container_width=True)
        st.download_button(
            "Download results as CSV", result.to_csv(index=False).encode("utf-8"),
            file_name="carapp_condition_summary.csv", mime="text/csv",
        )
    st.info(
        "The percentage is the proportion of recorded collisions in a condition that had the "
        "selected severity. It is not the chance of having an accident, and it does not prove causation."
    )

with tab_quality:
    st.subheader("Dataset checks and preprocessing")
    st.write("These checks do not delete or change the original uploaded records.")
    left, middle, right = st.columns(3)
    left.metric("Rows", f"{len(df):,}")
    middle.metric("Exact duplicate rows", f"{int(df.duplicated().sum()):,}")
    right.metric("Missing cells (NaN)", f"{int(df.isna().sum().sum()):,}")
    quality = pd.DataFrame({
        "Column": df.columns,
        "Data type": df.dtypes.astype(str).values,
        "Missing values": df.isna().sum().values,
        "Missing (%)": (100 * df.isna().mean()).round(4).values,
    })
    st.dataframe(quality.sort_values("Missing values", ascending=False),
                 hide_index=True, use_container_width=True)
    st.caption(
        "DfT special codes (for example, -1 or 9) can mean unknown or unreported values "
        "even when Pandas does not identify them as missing."
    )
    with st.expander("See examples of special codes"):
        tests = {
            "light_conditions": [-1],
            "weather_conditions": [9],
            "road_surface_conditions": [-1, 9],
            "urban_or_rural_area": [3],
        }
        rows = [{
            "Column": col, "Special codes": ", ".join(map(str, codes)),
            "Records": int(df[col].isin(codes).sum()),
        } for col, codes in tests.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.write(
        "For modelling, category codes are one-hot encoded using the training data only. "
        "The numeric speed limit is used directly."
    )

with tab_ml:
    st.subheader("Decision Tree: slight vs serious/fatal")
    st.write(
        "This tab reproduces the Decision Tree trained in the notebook. "
        "It fits the model on 80% of the uploaded dataset and evaluates it on the remaining 20%."
    )
    st.caption(
        "Inputs: speed limit, road type, lighting, weather, road surface, and urban/rural area. "
        "Model: maximum depth 5, minimum 100 samples per leaf, balanced class weights, random seed 42."
    )
    try:
        results = evaluate_decision_tree(df[MODEL_FEATURES + ["collision_severity"]])
    except Exception as exc:
        st.error(f"Model training or evaluation failed: {exc}")
        st.stop()

    report = results["report"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{results['accuracy'] * 100:.2f}%")
    c2.metric("Serious/Fatal recall", f"{report['Serious/Fatal']['recall'] * 100:.1f}%")
    c3.metric("Serious/Fatal F1-score", f"{report['Serious/Fatal']['f1-score']:.2f}")
    st.caption(
        f"Training collisions: {results['train_rows']:,} · "
        f"Test collisions: {results['test_rows']:,}"
    )

    scores = pd.DataFrame({
        label: {
            "Precision": report[label]["precision"],
            "Recall": report[label]["recall"],
            "F1-score": report[label]["f1-score"],
            "Test records": int(report[label]["support"]),
        }
        for label in ["Slight", "Serious/Fatal"]
    }).T
    st.subheader("Classification report")
    st.dataframe(scores.style.format({
        "Precision": "{:.2f}", "Recall": "{:.2f}", "F1-score": "{:.2f}",
        "Test records": "{:,.0f}",
    }), use_container_width=True)

    st.subheader("Confusion matrix")
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ConfusionMatrixDisplay(
        confusion_matrix=results["matrix"],
        display_labels=["Slight", "Serious/Fatal"],
    ).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Actual vs predicted severity")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.warning(
        "The model misses many serious/fatal collisions. Its predictions are exploratory "
        "and should not be used to make decisions about individual incidents."
    )

    with st.expander("View the first three levels of the Decision Tree"):
        pipeline = results["model"]
        tree = pipeline.named_steps["classifier"]
        features = pipeline.named_steps["preprocessor"].get_feature_names_out()
        fig, ax = plt.subplots(figsize=(21, 9))
        plot_tree(
            tree, feature_names=features,
            class_names=["Slight", "Serious/Fatal"],
            max_depth=3, filled=True, rounded=True, fontsize=7, ax=ax,
        )
        ax.set_title("Simplified view of the trained Decision Tree (first three levels)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

st.divider()
st.caption(
    "CarApp · Department for Transport 2025 collision records · "
    "Coursework demonstration of exploratory analysis and classification"
)
