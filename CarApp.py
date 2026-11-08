
import streamlit as st
import pandas as pd

# Configure the application
st.set_page_config(
    page_title="CarApp",
    layout="wide"
)

st.title("CarApp - UK Road Collision Analysis 2025")

st.write("Interactive analysis of road traffic collisions in Great Britain.")

# Upload dataset
uploaded_file = st.file_uploader(
    "Upload your road collision CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # Create application tabs
    tab1, tab2, tab3 = st.tabs([
        "Dataset Overview",
        "Data Preprocessing",
        "Collision Analysis"
    ])

    # TAB 1 - DATASET OVERVIEW
    with tab1:

        st.header("Dataset Overview")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Collisions", len(df))

        col2.metric(
            "Fatal Collisions",
            (df["collision_severity"] == 1).sum()
        )

        col3.metric(
            "Serious Collisions",
            (df["collision_severity"] == 2).sum()
        )

        st.subheader("Dataset Preview")
        st.dataframe(df.head(20))

    # TAB 2 - DATA PREPROCESSING
    with tab2:

        st.header("Data Quality and Preprocessing")

        st.subheader("Missing Values")

        missing = pd.DataFrame({
            "Missing Values": df.isnull().sum(),
            "Missing Percentage":
                (df.isnull().mean() * 100).round(4)
        })

        st.dataframe(missing)

        st.subheader("Duplicate Records")

        duplicate_count = df.duplicated().sum()

        st.metric("Duplicate Rows", duplicate_count)

        st.subheader("Column Data Types")

        data_types = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values
        })

        st.dataframe(data_types)

        st.info(
            "Data quality checks are displayed above. "
            "The original dataset has not been modified."
        )

    # TAB 3 - COLLISION ANALYSIS
    with tab3:

        st.header("Collision Analysis")

        severity_codes = {
            "Fatal": 1,
            "Serious": 2,
            "Slight": 3
        }

        severity = st.selectbox(
            "Select Collision Severity",
            list(severity_codes.keys())
        )

        filtered_df = df[
            df["collision_severity"] == severity_codes[severity]
        ]

        category = st.selectbox(
            "Analyse Collisions By",
            [
                "weather_conditions",
                "light_conditions",
                "road_surface_conditions",
                "speed_limit"
            ]
        )

        counts = filtered_df[category].value_counts()

        st.subheader("Collision Counts")
        st.dataframe(counts.to_frame(name="Collisions"))

        st.subheader("Collision Chart")
        st.bar_chart(counts)

else:
    st.info("Upload your 2025 road collision CSV to begin.")
