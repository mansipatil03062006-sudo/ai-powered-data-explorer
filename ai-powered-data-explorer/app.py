import io 
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="AI-Powered Data Explorer", page_icon="📊", layout="wide")

st.title("📊 AI-Powered Data Explorer")
st.caption("Upload a CSV, explore data quality, build interactive charts, and discover patterns.")

@st.cache_data
def load_csv(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))

uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is None:
    st.info("Upload a CSV file in the sidebar, or try the included sample dataset.")
    sample_path = Path(__file__).parent / "data" / "sample_sales.csv"
    if st.button("Load sample sales dataset"):
        st.session_state["sample_data"] = pd.read_csv(sample_path)
    if "sample_data" not in st.session_state:
        st.stop()
    df = st.session_state["sample_data"].copy()
else:
    try:
        df = load_csv(uploaded_file.getvalue())
    except Exception as exc:
        st.error(f"Could not read this CSV: {exc}")
        st.stop()

if df.empty:
    st.warning("This CSV has no rows.")
    st.stop()

# Overview metrics
st.subheader("Dataset Overview")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows", f"{len(df):,}")
m2.metric("Columns", f"{df.shape[1]:,}")
m3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
m4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

tabs = st.tabs(["Data Preview", "Data Quality", "Visualizations", "AI Insights"])

with tabs[0]:
    st.subheader("Explore your data")
    search = st.text_input("Search all columns")
    preview = df.copy()
    if search:
        mask = preview.astype(str).apply(
            lambda col: col.str.contains(search, case=False, na=False, regex=False)
        ).any(axis=1)
        preview = preview[mask]
    st.caption(f"Showing {len(preview):,} of {len(df):,} rows")
    st.dataframe(preview, use_container_width=True, height=360)
    st.subheader("Statistical summary")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

with tabs[1]:
    st.subheader("Missing values")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        st.success("No missing values found.")
    else:
        missing_df = missing.rename_axis("Column").reset_index(name="Missing values")
        st.plotly_chart(px.bar(missing_df, x="Column", y="Missing values"), use_container_width=True)
        st.dataframe(missing_df, use_container_width=True)
    st.subheader("Duplicates")
    st.write(f"**{int(df.duplicated().sum()):,}** duplicate rows detected.")
    st.subheader("Column profile")
    profile = pd.DataFrame({
        "Column": df.columns.astype(str),
        "Data type": df.dtypes.astype(str).values,
        "Unique values": df.nunique(dropna=True).values,
        "Missing values": df.isna().sum().values,
    })
    st.dataframe(profile, use_container_width=True)

with tabs[2]:
    st.subheader("Interactive charts")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    chart_type = st.selectbox("Chart type", ["Histogram", "Bar chart", "Scatter plot", "Box plot"])
    if chart_type == "Histogram":
        if numeric_cols:
            col = st.selectbox("Numeric column", numeric_cols)
            fig = px.histogram(df, x=col, title=f"Distribution of {col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric columns found.")
    elif chart_type == "Bar chart":
        if categorical_cols:
            col = st.selectbox("Category column", categorical_cols)
            counts = df[col].astype(str).value_counts().head(25).rename_axis(col).reset_index(name="Count")
            st.plotly_chart(px.bar(counts, x=col, y="Count", title=f"Top values in {col}"), use_container_width=True)
        else:
            st.warning("No categorical columns found.")
    elif chart_type == "Scatter plot":
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("X-axis", numeric_cols)
            y_options = [c for c in numeric_cols if c != x_col]
            y_col = st.selectbox("Y-axis", y_options)
            color_col = st.selectbox("Color by (optional)", ["None"] + categorical_cols)
            fig = px.scatter(df, x=x_col, y=y_col, color=None if color_col == "None" else color_col)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("A scatter plot needs at least two numeric columns.")
    else:
        if numeric_cols:
            col = st.selectbox("Numeric column", numeric_cols)
            st.plotly_chart(px.box(df, y=col, title=f"Box plot of {col}"), use_container_width=True)
        else:
            st.warning("No numeric columns found.")

    if len(numeric_cols) >= 2:
        st.subheader("Correlation heatmap")
        corr = df[numeric_cols].corr()
        st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Numeric feature correlations"), use_container_width=True)

with tabs[3]:
    st.subheader("Automated insights")
    st.write("Rule-based statistical insights plus optional machine-learning anomaly detection.")
    missing_cols = df.columns[df.isna().any()].tolist()
    if missing_cols:
        for col in missing_cols:
            st.warning(f"**{col}** contains {int(df[col].isna().sum())} missing values.")
    else:
        st.success("No missing values detected.")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) >= 3 and series.nunique() > 1:
            skew = float(series.skew())
            if abs(skew) >= 1:
                direction = "right" if skew > 0 else "left"
                st.info(f"**{col}** has a {direction}-skewed distribution (skewness {skew:.2f}).")

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
        strong = upper[upper.abs() >= 0.7].sort_values(key=np.abs, ascending=False)
        if not strong.empty:
            st.markdown("**Strong correlations (|r| ≥ 0.70):**")
            for (a, b), value in strong.items():
                st.write(f"- {a} and {b}: **{value:.2f}**")

    if len(numeric_cols) and len(df) >= 10:
        st.divider()
        st.subheader("Anomaly detection (Isolation Forest)")
        selected = st.multiselect("Features to analyze", numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))])
        contamination = st.slider("Expected anomaly fraction", 0.01, 0.20, 0.05, 0.01)
        if st.button("Detect anomalies"):
            if not selected:
                st.warning("Select at least one feature.")
            else:
                clean = df[selected].replace([np.inf, -np.inf], np.nan).dropna()
                if len(clean) < 10:
                    st.warning("At least 10 complete rows are required.")
                else:
                    model = IsolationForest(contamination=contamination, random_state=42)
                    labels = model.fit_predict(clean)
                    result = df.loc[clean.index].copy()
                    result["Anomaly"] = np.where(labels == -1, "Potential anomaly", "Typical")
                    st.metric("Potential anomalies", int((labels == -1).sum()))
                    st.dataframe(result[result["Anomaly"] == "Potential anomaly"], use_container_width=True)

# Download cleaned data: drop duplicates, preserve missing values for user review.
cleaned = df.drop_duplicates()
st.sidebar.download_button(
    "Download deduplicated CSV",
    data=cleaned.to_csv(index=False).encode("utf-8"),
    file_name="deduplicated_data.csv",
    mime="text/csv",
)
st.sidebar.caption("Uploaded data is processed by this app; no external AI API is used.")
