import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='dark')

def create_monthly_orders(df):
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])

    filtered_data = df[df['Tanggal'].dt.year.isin([2024, 2025])]

    filtered_data['month'] = filtered_data['Tanggal'].dt.month
    filtered_data['year'] = filtered_data['Tanggal'].dt.year

    monthly_data = (
        filtered_data.groupby(['year', 'month'])
        .agg(order_count=('Tanggal', 'count'), revenue=('Total Biaya Produksi/Hari', 'sum'))
        .reset_index()
    )
    return monthly_data


all_data_df = pd.read_csv("produksi_dataset_clean.csv")


# Dataset
datetime_cols = ["Tanggal"]
all_data_df.sort_values(by="Tanggal", inplace=True)
all_data_df.reset_index(inplace=True)

for col in datetime_cols:
    all_data_df[col] = pd.to_datetime(all_data_df[col])

min_date = all_data_df["Tanggal"].min()
max_date = all_data_df["Tanggal"].max()

# Sidebar
with st.sidebar:
    # Title
    st.markdown("### Capstone Project DB4-PS007")

    st.markdown("<hr style='margin: 15px 0; border-color: #47663B;'>", unsafe_allow_html=True)

    st.markdown("### Contact Me")

    st.markdown(
        """
        - Email: rizkaindahpuspita@gmail.com
        - Github: [rizkaindahp](https://github.com/rizkaindahp)
        - Linkedin: [Rizka Indah Puspita](https://linkedin.com/in/rizka-indah-puspita)
        """,
        unsafe_allow_html=True
    )

    st.markdown("<hr style='margin: 15px 0; border-color: #47663B;'>", unsafe_allow_html=True)

    # Date Range
    start_date, end_date = st.date_input(
        label="Select Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

# Main
main_df = all_data_df[(all_data_df["Tanggal"] >= str(start_date)) & (all_data_df["Tanggal"] <= str(end_date))]


monthly_orders_df = create_monthly_orders(main_df)

# Title
st.header("Dashboard Produksi Wajit")
# Header
st.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='color: #47663B;'>Dashboard Produksi Wajit</h1>
    </div>
    """,
    unsafe_allow_html=True
)