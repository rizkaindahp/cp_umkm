
# Mengimpor library yang diperlukan
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
sns.set(style='dark')
from sklearn.metrics import accuracy_score

# Membaca data bersih dari file CSV
df_eda = pd.read_csv("dataset/produksi_dataset_clean.csv")
X = df_eda.drop(columns=['Tanggal','Total Biaya Produksi/Hari','Nama Toko']) 
y = df_eda["Total Biaya Produksi/Hari"]


# Memuat model dari file joblib
model_path = "model/random_forest_model.pkl"
model = pickle.load(open(model_path, "rb"))

# Memprediksi data yang sama dengan yang digunakan untuk training
RF = RandomForestRegressor(n_estimators=30, max_depth=16)
RF.fit(X.values, y) 
y_pred = RF.predict(X.values)
mse_value_matrix = mean_squared_error(y, y_pred)
# accuracy = round((accuracy * 100), 2)

# Menyiapkan DataFrame untuk user input
df_final = X
df_final["Total Biaya Produksi/Hari"] = y

# Mengatur konfigurasi halaman Streamlit
st.set_page_config(page_title="Production Cost Predict (Wajit)", page_icon=":rice_cracker:")

# Menampilkan judul halaman
st.title("Production Cost Predict (Wajit)")
# Menampilkan mse model
st.write(
    f"**_Model's Mean Squared Error (RandomForest)_** :  :green[**{mse_value_matrix:.2f}**]"
)
st.write("")

# Membuat tab untuk single prediction dan multi-prediction
tab1, tab2 = st.tabs(["Prediction", "Dashboard"])

# Bagian Predict
with tab1:
    st.sidebar.header("**User Input** Sidebar")

    # Menambahkan input untuk Gula Pasir dengan batasan nilai minimum dan maksimum
    gula_pasir = st.sidebar.number_input(
        label="**Gula Pasir (Rp)**",
        min_value=float(df_final["Gula Pasir (Rp)"].min()),  # Ubah ke float
        max_value=float(df_final["Gula Pasir (Rp)"].max()),  # Ubah ke float
        step=0.01  # Ubah langkah menjadi 0.01 agar bisa menerima angka desimal
    )

    # Menampilkan informasi tentang batas nilai Gula Pasir
    st.sidebar.write(
        f":orange[Min] value: :orange[**{df_final['Gula Pasir (Rp)'].min():.2f}**], "
        f":red[Max] value: :red[**{df_final['Gula Pasir (Rp)'].max():.2f}**]"
    )
    st.sidebar.write("")


    # Menambahkan input untuk Gula Merah dengan batasan nilai minimum dan maksimum
    gula_merah = st.sidebar.number_input(
        label="**Gula Merah (Rp)**",
        min_value=float(df_final["Gula Merah (Rp)"].min()),
        max_value=float(df_final["Gula Merah (Rp)"].max()),
        step=0.01 # Mengatur langkah menjadi 1 agar hanya menerima nilai integer
    )
    # Menampilkan informasi tentang batas nilai Gula Merah
    st.sidebar.write(
        f":orange[Min] value: :orange[**{df_final['Gula Merah (Rp)'].min():.2f}**], :red[Max] value: :red[**{df_final['Gula Merah (Rp)'].max():.2f}**]"
    )
    st.sidebar.write("")

    # Menambahkan input untuk Kelapa dengan batasan nilai minimum dan maksimum
    kelapa = st.sidebar.number_input(
        label="**Kelapa (Rp)**",
        min_value=int(df_final["Kelapa (Rp)"].min()),
        max_value=int(df_final["Kelapa (Rp)"].max()),
        step=1 # Mengatur langkah menjadi 1 agar hanya menerima nilai integer
    )
    # Menampilkan informasi tentang batas nilai Kelapa
    st.sidebar.write(
        f":orange[Min] value: :orange[**{df_final['Kelapa (Rp)'].min()}**], :red[Max] value: :red[**{df_final['Kelapa (Rp)'].max()}**]"
    )
    st.sidebar.write("")

    # Menambahkan input untuk Beras Ketan dengan batasan nilai minimum dan maksimum
    beras_ketan = st.sidebar.number_input(
        label="**Beras Ketan (Rp)**",
        min_value=int(df_final["Beras Ketan (Rp)"].min()),
        max_value=int(df_final["Beras Ketan (Rp)"].max()),
        step=1 # Mengatur langkah menjadi 1 agar hanya menerima nilai integer
    )
    # Menampilkan informasi tentang batas nilai Beras Ketan
    st.sidebar.write(
        f":orange[Min] value: :orange[**{df_final['Beras Ketan (Rp)'].min()}**], :red[Max] value: :red[**{df_final['Beras Ketan (Rp)'].max()}**]"
    )
    st.sidebar.write("")

    # Menambahkan input untuk Vanili Bubuk dengan batasan nilai minimum dan maksimum
    vanili_bubuk = st.sidebar.number_input(
        label="**Vanili Bubuk (Rp)**",
        min_value=int(df_final["Vanili Bubuk (Rp)"].min()),
        max_value=int(df_final["Vanili Bubuk (Rp)"].max()),
        step=1 # Mengatur langkah menjadi 1 agar hanya menerima nilai integer
    )
    # Menampilkan informasi tentang batas nilai Vanili Bubuk
    st.sidebar.write(
        f":orange[Min] value: :orange[**{df_final['Vanili Bubuk (Rp)'].min()}**], :red[Max] value: :red[**{df_final['Vanili Bubuk (Rp)'].max()}**]"
    )
    st.sidebar.write("")

    # Menambahkan input untuk Daun Jagung dengan batasan nilai minimum dan maksimum
    daun_jagung = st.sidebar.number_input(
        label="**Daun Jagung (Rp)**",
        min_value=int(df_final["Daun Jagung (Rp)"].min()),
        max_value=int(df_final["Daun Jagung (Rp)"].max()),
        step=1 # Mengatur langkah menjadi 1 agar hanya menerima nilai integer
    )
    # Menampilkan informasi tentang batas nilai Daun Jagung
    st.sidebar.write(
        f":orange[Min] value: :orange[**{df_final['Daun Jagung (Rp)'].min()}**], :red[Max] value: :red[**{df_final['Daun Jagung (Rp)'].max()}**]"
    )
    st.sidebar.write("")

     # Menambahkan input untuk Jumlah Produksi dengan batasan nilai minimum dan maksimum
    jumlah_produksi = st.sidebar.number_input(
        label="**Jumlah Produksi (Rp)**",
        min_value=int(df_final["Jumlah Produksi"].min()),
        max_value=int(df_final["Jumlah Produksi"].max()),
        step=1 # Mengatur langkah menjadi 1 agar hanya menerima nilai integer
    )
    # Menampilkan informasi tentang batas nilai Jumlah Produksi
    st.sidebar.write(
        f":orange[Min] value: :orange[**{df_final['Jumlah Produksi'].min()}**], :red[Max] value: :red[**{df_final['Jumlah Produksi'].max()}**]"
    )
    st.sidebar.write("")

     # Menambahkan input untuk Gaji Karyawan dengan batasan nilai minimum dan maksimum
    gaji_karyawan = st.sidebar.number_input(
        label="**Gaji Karyawan (Rp)**",
        min_value=int(df_final["Gaji Karyawan"].min()),
        max_value=int(df_final["Gaji Karyawan"].max()),
        step=1 # Mengatur langkah menjadi 1 agar hanya menerima nilai integer
    )
    # Menampilkan informasi tentang batas nilai Gaji Karyawan
    st.sidebar.write(
        f":orange[Min] value: :orange[**{df_final['Gaji Karyawan'].min()}**], :red[Max] value: :red[**{df_final['Gaji Karyawan'].max()}**]"
    )
    st.sidebar.write("")

     # Menambahkan input untuk Biaya Overhead dengan batasan nilai minimum dan maksimum
    biaya_overhead = st.sidebar.number_input(
        label="**Biaya Overhead (Listrik, Gas, Plastik dll)**",
        min_value=int(df_final["Biaya Overhead (Listrik, Gas, Plastik dll)"].min()),
        max_value=int(df_final["Biaya Overhead (Listrik, Gas, Plastik dll)"].max()),
        step=1 # Mengatur langkah menjadi 1 agar hanya menerima nilai integer
    )
    # Menampilkan informasi tentang batas nilai Biaya Overhead
    st.sidebar.write(
        f":orange[Min] value: :orange[**{df_final['Biaya Overhead (Listrik, Gas, Plastik dll)'].min()}**], :red[Max] value: :red[**{df_final['Biaya Overhead (Listrik, Gas, Plastik dll)'].max()}**]"
    )
    st.sidebar.write("")

    # Membuat DataFrame untuk input pengguna
    data = {
        "Gula Pasir": gula_pasir,
        "Gula Merah": gula_merah,
        "Kelapa": kelapa,
        "Beras Ketan": beras_ketan,
        "Daun Jagung": daun_jagung,
        "Vanili Bubuk": vanili_bubuk,
        "Jumlah Produksi": jumlah_produksi,
        "Gaji Karyawan": gaji_karyawan,
        "Biaya Overhead": biaya_overhead,
    }

    preview_df = pd.DataFrame(data, index=["input"])

    # Menampilkan DataFrame hasil input pengguna
    st.header("User Input as DataFrame")
    st.write("")
    # st.dataframe(preview_df.iloc[:, :6])
    # st.dataframe(preview_df.iloc[:, 6:])
    st.dataframe(preview_df)

    # Menambahkan tombol prediksi
    predict_btn = st.button("**Predict**", type="primary")

    # Melakukan prediksi saat tombol ditekan
    st.write("")
    if predict_btn:
        # Membuat input fitur sesuai dengan yang digunakan dalam pelatihan model
        inputs = [[gula_pasir, gula_merah, kelapa, beras_ketan, daun_jagung, vanili_bubuk, jumlah_produksi, gaji_karyawan, biaya_overhead]]

        # Pastikan semua input dalam format float agar sesuai dengan model
        inputs = [[float(i) for i in inputs[0]]]

        # Melakukan prediksi dengan model
        prediction = model.predict(inputs)[0]  # Ambil hasil prediksi pertama

        # Menampilkan bar progress selama prediksi berlangsung
        bar = st.progress(0)
        status_text = st.empty()

        for i in range(1, 101):
            status_text.text(f"{i}% complete")
            bar.progress(i)
            time.sleep(0.01)
            if i == 100:
                time.sleep(1)
                status_text.empty()
                bar.empty()

        
        # Menampilkan hasil prediksi dan deskripsi hasilnya
        if prediction >= 1000 and prediction <= 3000 :
            result = ":green[**Rendah**]"
            desc = 'Ini menunjukkan biaya produksi yang dikeluarkan sangat rendah'
        elif prediction >= 3001 and prediction <= 6000 :
            result = ":orange[**Sedang**]"
            desc = 'Ini menunjukkan biaya produksi yang dikeluarkan sedang'
        elif prediction > 6001  :
            result = ":red[**Tinggi**]"
            desc = 'Ini menunjukkan biaya produksi yang dikeluarkan sangat tinggi'
     
        
        # Mengalikan hasil prediksi dengan 1000
        prediction_final = prediction * 1000

        # Menampilkan hasil prediksi dengan format yang lebih jelas
        st.write("")
        st.subheader("Hasil Prediksi:")
        st.success(f"💰 **Prediksi Biaya Produksi: Rp {prediction_final:,.2f}**")
        st.subheader(result)
        st.write(desc)


with tab2:
    st.header("Visualisasi Dashboard")

    # Function
    def create_tot_monthly_production(df):
        # Convert 'Tanggal' to datetime format
        df['Tanggal'] = pd.to_datetime(df['Tanggal'])

        # Filter data for the years 2024 and 2025
        filtered_data = df[df['Tanggal'].dt.year.isin([2024, 2025])]

        # Extract month and year from 'Tanggal'
        filtered_data['month'] = filtered_data['Tanggal'].dt.month
        filtered_data['year'] = filtered_data['Tanggal'].dt.year

        # Group by 'year' and 'month' and calculate the sum of 'Total Biaya Produksi/Hari'
        monthly_data = (
            filtered_data.groupby(['year', 'month'])
            .agg(total_cost=('Total Biaya Produksi/Hari', 'sum'))  # Sum of 'Total Biaya Produksi/Hari'
            .reset_index()
        )
        
        return monthly_data

    def count_jumlah_produksi(df):
        # Grouping by 'Nama Toko' and calculating the sum of 'Jumlah Produksi'
        crop_summary = df.groupby('Nama Toko')[['Jumlah Produksi']].sum()

        # Sorting the results from highest to lowest based on 'Jumlah Produksi'
        crop_summary = crop_summary.sort_values(by='Jumlah Produksi', ascending=True)

        return crop_summary

    # Dataset
    datetime_cols = ["Tanggal"]
    df_eda.sort_values(by="Tanggal", inplace=True)
    df_eda.reset_index(inplace=True)

    for col in datetime_cols:
        df_eda[col] = pd.to_datetime(df_eda[col])

    # Filter by 'Nama Toko' (Store Name)
    store_names = ['All'] + list(df_eda['Nama Toko'].unique())  # Add 'All' option to store names
    selected_store = st.selectbox("Select Store", store_names)  # Add selectbox for store name

    # Filter dataset based on selected store
    if selected_store == "All":
        df_eda_filtered = df_eda  # If "All" is selected, no filter is applied
    else:
        df_eda_filtered = df_eda[df_eda['Nama Toko'] == selected_store]  # Filter by selected store

    min_date = df_eda_filtered["Tanggal"].min()
    max_date = df_eda_filtered["Tanggal"].max()

    # Date Range filter
    start_date, end_date = st.date_input(
        label="Select Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

    main_df = df_eda_filtered[(df_eda_filtered["Tanggal"] >= str(start_date)) & (df_eda_filtered["Tanggal"] <= str(end_date))]
    monthly_production_df = create_tot_monthly_production(main_df)
    count_jml_production_df = count_jumlah_produksi(main_df)

    ##  =============== Monthly Total Cost Production Plot =================
    st.subheader("Monthly Total Cost Production (2024-2025)")

    fig, ax = plt.subplots(figsize=(12, 8))

    for year in [2024, 2025]:
        # Filter data for each year
        yearly_data = monthly_production_df[monthly_production_df['year'] == year]

        # Plot data
        ax.plot(
            yearly_data['month'],
            yearly_data['total_cost'],  # Use 'total_cost' instead of 'order_count'
            marker='o',
            label=f'Total Cost {year}'
        )

    # Add chart details
    ax.set_title("Monthly Total Cost Production (2024-2025)", fontsize=14)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Total Cost", fontsize=12)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.legend(title="Year")
    ax.grid(alpha=0.5)
    plt.tight_layout()

    st.pyplot(fig)

    st.markdown("<hr style='margin: 15px 0; border-color: #47663B;'>", unsafe_allow_html=True)

    ##  =============== Plot Jumlah Produksi =================
    st.subheader("Production Quantity per Store Toko")

    # Bar plot of total "Jumlah Produksi" per "Nama Toko"
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use Seaborn's dark blue palette and apply it to Matplotlib
    colors = sns.color_palette("Blues_d", len(count_jml_production_df))

    # Adding the bar chart with gradient colors
    bars = ax.bar(count_jml_production_df.index, count_jml_production_df['Jumlah Produksi'], color=colors)

    # Adding labels and title
    ax.set_xlabel('Nama Toko')
    ax.set_ylabel('Total Jumlah Produksi')
    ax.set_title('Total Production Quantity per Store Toko')

    # Rotate x labels for better readability
    ax.set_xticks(range(len(count_jml_production_df.index)))
    ax.set_xticklabels(count_jml_production_df.index, rotation=45)

    st.pyplot(fig)

    st.markdown("<hr style='margin: 15px 0; border-color: #47663B;'>", unsafe_allow_html=True)


    