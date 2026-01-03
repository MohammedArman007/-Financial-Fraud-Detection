import streamlit as st
import pandas as pd
import plotly.express as px
from preprocessing import load_data, preprocess_data
from model import train_models

# Streamlit Config
st.set_page_config(page_title="Financial Fraud Detection", layout="wide")

st.title("💰 Financial Fraud Detection Dashboard")
st.markdown("Detect suspicious transactions using Machine Learning & Analytics.")

# File uploader
uploaded_file = st.file_uploader("📂 Upload a dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")
else:
    st.info("No file uploaded. Loading default dataset (`transactions_10000.csv`).")
    df = load_data("transactions_10000.csv")

if df is not None:

    # ----------------------------
    # KPI CARDS (Top Metrics)
    # ----------------------------
    total_txn = len(df)
    fraud_txn = df["is_fraud"].sum() if "is_fraud" in df.columns else 0
    fraud_percent = (fraud_txn / total_txn * 100) if total_txn > 0 else 0
    avg_amount = df["amount"].mean() if "amount" in df.columns else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", f"{total_txn:,}")
    col2.metric("Fraudulent %", f"{fraud_percent:.2f}%")
    col3.metric("Avg. Transaction Amount", f"${avg_amount:,.2f}")

    # ----------------------------
    # Tabs for navigation
    # ----------------------------
    tab1, tab2, tab3, tab4 = st.tabs([" Raw Data", " Preprocessing", " Models", " Analytics"])

    # --- RAW DATA TAB ---
    with tab1:
        st.subheader(" Raw Data Preview")

        # Fraud filter
        fraud_only = st.checkbox("Show only fraudulent transactions", value=False)
        if fraud_only and "is_fraud" in df.columns:
            df = df[df["is_fraud"] == 1]

        # Search box
        search_value = st.text_input("🔍 Search transactions")
        if search_value:
            df = df[df.apply(lambda row: row.astype(str).str.contains(search_value, case=False).any(), axis=1)]

        # Row slider
        row_count = st.slider("Number of rows to preview", 5, 100, 10)
        st.dataframe(df.head(row_count))

    # --- PREPROCESSING TAB ---
    with tab2:
        st.subheader(" Preprocessed Data")
        df_processed, numeric_cols = preprocess_data(df)
        st.dataframe(df_processed.head(10))

    # --- MODELS TAB ---
    with tab3:
        st.subheader(" Model Performance Reports")
        models, reports, test_data = train_models(df_processed)

        for model_name, report in reports.items():
            st.write(f"### {model_name}")
            st.json(report)

    # --- ANALYTICS TAB ---
    with tab4:
        st.header(" Advanced Fraud Analytics")

        # Fraud Distribution by Amount
        if "is_fraud" in df.columns and "amount" in df.columns:
            st.subheader(" Fraud Distribution by Amount")
            fig = px.histogram(df, x="amount", color="is_fraud", 
                               title="Transaction Amount vs Fraud", 
                               marginal="box", 
                               nbins=50)
            st.plotly_chart(fig, use_container_width=True)

        # Fraud by Merchant
        if "merchant" in df.columns and "is_fraud" in df.columns:
            st.subheader(" Fraud by Merchant")
            fraud_by_merchant = df[df["is_fraud"] == 1]["merchant"].value_counts().head(10)
            fig = px.bar(fraud_by_merchant, 
                         x=fraud_by_merchant.index, 
                         y=fraud_by_merchant.values, 
                         title="Top 10 Merchants with Fraud Cases", 
                         labels={"x": "Merchant", "y": "Fraud Cases"})
            st.plotly_chart(fig, use_container_width=True)

        # Fraud by Location
        if "location" in df.columns and "is_fraud" in df.columns:
            st.subheader(" Fraud by Location")
            fraud_by_location = df[df["is_fraud"] == 1]["location"].value_counts().head(10)
            fig = px.bar(fraud_by_location, 
                         x=fraud_by_location.index, 
                         y=fraud_by_location.values, 
                         title="Top 10 Locations with Fraud Cases", 
                         labels={"x": "Location", "y": "Fraud Cases"})
            st.plotly_chart(fig, use_container_width=True)

        # Fraud Trend Over Time
        if "date" in df.columns and "is_fraud" in df.columns:
            st.subheader(" Fraud Trend Over Time")
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            trend = df.groupby("date")["is_fraud"].sum().reset_index()
            fig = px.line(trend, x="date", y="is_fraud", title="Fraud Cases Over Time")
            st.plotly_chart(fig, use_container_width=True)

        # Top High-Value Fraudulent Transactions
        if "is_fraud" in df.columns and "amount" in df.columns:
            st.subheader(" Top 10 High-Value Fraudulent Transactions")
            top_fraud = df[df["is_fraud"] == 1].sort_values(by="amount", ascending=False).head(10)
            st.dataframe(top_fraud)
