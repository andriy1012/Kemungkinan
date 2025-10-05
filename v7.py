import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import datetime

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Harga By Andriy",
    page_icon="📈",
    layout="wide"
)

st.title("Terkadang ya bener kadang juga bener juga inshaallah")
st.markdown("Upload data CSV dan dapatkan kemungkinanya jangan lupa Doa")

# Upload file
uploaded_file = st.file_uploader("Upload file CSV data saham", type=['csv'])

def get_models():
    """Return dictionary of models"""
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5),
        'SVR': SVR(kernel='rbf', C=1.0)
    }
    return models

def evaluate_model(model, X_test, y_test):
    """Evaluate model dan return metrics"""
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'Predictions': y_pred
    }

if uploaded_file is not None:
    try:
        # Baca file CSV
        df = pd.read_csv(uploaded_file)

        st.success(f"✅ File berhasil diupload! {len(df)} baris data")

        # Tampilkan data dengan lebih baik
        st.subheader("📋 Data yang Diupload")

        # Tampilkan informasi data
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Baris", len(df))
        with col2:
            st.metric("Total Kolom", len(df.columns))
        with col3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            st.metric("Kolom Numerik", len(numeric_cols))
        with col4:
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'tanggal' in col.lower()]
            st.metric("Kolom Tanggal", len(date_cols))

        # Tampilkan preview data (10 baris pertama dan terakhir)
        st.write("**Preview Data (10 Baris Pertama):**")
        st.dataframe(df.head(10), use_container_width=True)

        # Tampilkan info kolom
        st.write("**Informasi Kolom:**")
        col_info = pd.DataFrame({
            'Kolom': df.columns,
            'Tipe Data': df.dtypes.values,
            'Non-Null Count': df.count().values,
            'Null Count': df.isnull().sum().values
        })
        st.dataframe(col_info, use_container_width=True)

        # Pilih kolom secara manual
        st.subheader("🔧 Konfigurasi Data")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Pilih kolom tanggal:**")
            date_col = st.selectbox("Kolom tanggal:", df.columns, key="date_col")

        with col2:
            st.write("**Pilih kolom harga:**")
            # Cari kolom numerik untuk harga
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                price_col = st.selectbox("Kolom harga:", numeric_cols, key="price_col")
            else:
                st.error("❌ Tidak ada kolom numerik dalam data!")
                st.stop()

        # Tampilkan statistik kolom harga yang dipilih
        st.write(f"**Statistik {price_col}:**")
        price_stats = df[price_col].describe()
        stats_df = pd.DataFrame({
            'Statistik': price_stats.index,
            'Nilai': price_stats.values
        })
        st.dataframe(stats_df, use_container_width=True)

        # Pilih algoritma
        st.write("**Pilih algoritma:**")
        algorithms = st.multiselect(
            "Pilih satu atau lebih algoritma:",
            ['Linear Regression', 'Ridge Regression', 'Random Forest', 'Gradient Boosting', 'SVR'],
            default=['Linear Regression', 'Random Forest']
        )

        if not algorithms:
            st.error("❌ Pilih minimal satu algoritma!")
            st.stop()

        # Konversi tanggal
        try:
            df['date_processed'] = pd.to_datetime(df[date_col])
            df = df.sort_values('date_processed').reset_index(drop=True)
            use_date = True
            st.success("✅ Kolom tanggal berhasil dikonversi")
        except Exception as e:
            st.warning(f"⚠️ Tidak bisa mengkonversi tanggal: {str(e)}, menggunakan nomor urut")
            df['date_processed'] = range(len(df))
            use_date = False

        # Siapkan data untuk training - SANGAT SEDERHANA
        df_clean = df[['date_processed', price_col]].copy()
        df_clean = df_clean.dropna()

        st.info(f"📊 Data setelah cleaning: {len(df_clean)} baris (dari {len(df)} awal)")

        if len(df_clean) < 10:
            st.error(f"❌ Data terlalu sedikit setelah cleaning! Minimal 10 data, hanya ada {len(df_clean)}")
            st.stop()

        # Tampilkan data yang sudah dibersihkan
        with st.expander("🔍 Lihat Data yang Sudah Dibersihkan"):
            st.dataframe(df_clean.head(10), use_container_width=True)

            # Tampilkan grafik data historis
            st.write("**Grafik Data Historis:**")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(
                x=df_clean['date_processed'],
                y=df_clean[price_col],
                mode='lines+markers',
                name=f'Harga {price_col}',
                line=dict(color='white', width=2)
            ))
            fig_hist.update_layout(
                title='Grafik Harga Historis',
                xaxis_title='Tanggal',
                yaxis_title=f'Harga {price_col}',
                height=400
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # Buat features SANGAT SEDERHANA
        df_clean['day_num'] = range(len(df_clean))
        df_clean['price_lag_1'] = df_clean[price_col].shift(1)
        df_clean['price_lag_2'] = df_clean[price_col].shift(2)
        df_clean['price_rolling_3'] = df_clean[price_col].rolling(3).mean()
        df_clean = df_clean.dropna()

        st.info(f"📊 Data final untuk training: {len(df_clean)} baris")

        if st.button("🚀 MULAI TRAINING DAN PREDIKSI", type="primary"):

            # SPLIT DATA MANUAL - TIDAK PAKAI train_test_split
            split_point = int(len(df_clean) * 0.8)  # 80% training, 20% testing

            # Training data (first 80%)
            X_train = df_clean[['day_num', 'price_lag_1', 'price_lag_2', 'price_rolling_3']].iloc[:split_point]
            y_train = df_clean[price_col].iloc[:split_point]

            # Testing data (last 20%)
            X_test = df_clean[['day_num', 'price_lag_1', 'price_lag_2', 'price_rolling_3']].iloc[split_point:]
            y_test = df_clean[price_col].iloc[split_point:]

            st.write(f"📚 Training data: {len(X_train)} baris")
            st.write(f"🧪 Testing data: {len(X_test)} baris")

            # Train models
            models = get_models()
            trained_models = {}
            results = {}

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, model_name in enumerate(algorithms):
                if model_name in models:
                    status_text.text(f"Training {model_name}...")

                    try:
                        model = models[model_name]
                        model.fit(X_train, y_train)
                        trained_models[model_name] = model

                        # Evaluate model
                        metrics = evaluate_model(model, X_test, y_test)
                        results[model_name] = metrics

                        st.success(f"✅ {model_name} berhasil ditraining! (MAE: Rp {metrics['MAE']:,.0f})")

                    except Exception as e:
                        st.error(f"❌ Error training {model_name}: {str(e)}")

                    progress_bar.progress((i + 1) / len(algorithms))

            status_text.text("Training selesai!")

            # Tampilkan hasil perbandingan model
            st.subheader("📊 Perbandingan Performa Model")

            if results:
                results_df = pd.DataFrame(results).T
                results_df = results_df[['MAE', 'RMSE', 'R2', 'MAPE']]
                results_df.columns = ['MAE (Error)', 'RMSE', 'R² Score', 'MAPE (%)']

                # Sort by MAE (best first)
                results_df = results_df.sort_values('MAE (Error)')

                col1, col2 = st.columns([2, 1])

                with col1:
                    # Style the dataframe
                    styled_df = results_df.style.format({
                        'MAE (Error)': '{:,.0f}',
                        'RMSE': '{:,.0f}',
                        'R² Score': '{:.3f}',
                        'MAPE (%)': '{:.2f}%'
                    }).highlight_min(subset=['MAE (Error)', 'RMSE', 'MAPE (%)'], color='lightgreen') \
                      .highlight_max(subset=['R² Score'], color='lightgreen')

                    st.dataframe(styled_df, use_container_width=True)

                with col2:
                    best_model = results_df.index[0]
                    best_mae = results_df.iloc[0]['MAE (Error)']
                    best_mape = results_df.iloc[0]['MAPE (%)']

                    st.metric("🎯 Model Terbaik", best_model)
                    st.metric("📉 Error Terkecil", f"Rp {best_mae:,.0f}")
                    st.metric("🎯 Akurasi Terbaik", f"{best_mape:.2f}%")
                    st.metric("📈 Skor R² Terbaik", f"{results_df.iloc[0]['R² Score']:.3f}")

            # PREDIKSI 7 HARI KE DEPAN UNTUK SEMUA MODEL
            st.subheader("🔮 Prediksi 7 Hari Ke Depan - Semua Model")

            # Ambil data terakhir
            last_data = df_clean.iloc[-1:].copy()
            current_price = last_data[price_col].iloc[0]

            # Buat prediksi untuk semua model
            all_predictions = {}

            for model_name, model in trained_models.items():
                predictions = []
                current_features = last_data[['day_num', 'price_lag_1', 'price_lag_2', 'price_rolling_3']].copy()

                for day in range(7):
                    try:
                        # Predict harga
                        next_price = model.predict(current_features)[0]
                        predictions.append(max(next_price, 0.1))  # Pastikan harga tidak negatif

                        # Update features untuk hari berikutnya
                        current_features = current_features.copy()
                        current_features['day_num'] += 1
                        current_features['price_lag_2'] = current_features['price_lag_1']
                        current_features['price_lag_1'] = next_price
                        current_features['price_rolling_3'] = np.mean([
                            current_features['price_lag_1'],
                            current_features['price_lag_2'],
                            current_features['price_lag_2']  # Approximate
                        ])

                    except Exception as e:
                        st.error(f"Error prediksi {model_name} hari {day+1}: {e}")
                        predictions.append(np.nan)

                all_predictions[model_name] = predictions

            # Buat tanggal prediksi
            if use_date:
                last_date = df_clean['date_processed'].max()
                future_dates = [last_date + datetime.timedelta(days=i+1) for i in range(7)]
            else:
                last_day = df_clean['day_num'].max()
                future_dates = [f"Hari {last_day + i + 1}" for i in range(7)]

            # Tampilkan prediksi model terbaik
            best_model_name = results_df.index[0]
            best_predictions = all_predictions[best_model_name]

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Prediksi Detail - {best_model_name}:**")
                pred_df = pd.DataFrame({
                    'Tanggal': [d.strftime('%d/%m/%Y') if hasattr(d, 'strftime') else d for d in future_dates],
                    'Prediksi': [f"Rp {p:,.0f}" for p in best_predictions],
                    'Perubahan': [f"{((p - current_price)/current_price*100):+.1f}%" for p in best_predictions]
                })
                st.dataframe(pred_df, use_container_width=True)

            with col2:
                st.write("**Statistik Prediksi:**")

                avg_prediction = np.mean(best_predictions)
                total_change = ((best_predictions[-1] - current_price) / current_price) * 100
                trend = "📈 NAIK" if total_change > 0 else "📉 TURUN"
                best_mape = results_df.loc[best_model_name, 'MAPE (%)']

                st.metric("Harga Terakhir", f"Rp {current_price:,.0f}")
                st.metric("Rata-rata Prediksi", f"Rp {avg_prediction:,.0f}")
                st.metric("Trend 7 Hari", trend, f"{total_change:+.1f}%")
                st.metric("Tingkat Akurasi", f"{100 - best_mape:.1f}%")

            # GRAFIK PREDIKSI SEMUA MODEL
            st.subheader("📊 Grafik Prediksi - Semua Model")

            fig = go.Figure()

            # Data historis (30 hari terakhir atau semua jika kurang)
            show_history = min(30, len(df_clean))
            hist_dates = df_clean['date_processed'].iloc[-show_history:]
            hist_prices = df_clean[price_col].iloc[-show_history:]

            # Plot data historis
            fig.add_trace(go.Scatter(
                x=hist_dates,
                y=hist_prices,
                mode='lines+markers',
                name='Data Historis',
                line=dict(color='black', width=3),
                marker=dict(size=6)
            ))

            # Warna untuk setiap model
            colors = ['red', 'blue', 'green', 'orange', 'purple']

            # Plot prediksi setiap model
            for i, (model_name, predictions) in enumerate(all_predictions.items()):
                color = colors[i % len(colors)]

                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=predictions,
                    mode='lines+markers',
                    name=f'{model_name}',
                    line=dict(color=color, width=2, dash='dash'),
                    marker=dict(size=6)
                ))

            fig.update_layout(
                title='Perbandingan Prediksi 7 Hari Ke Depan - Semua Model',
                xaxis_title='Tanggal',
                yaxis_title=f'Harga ({price_col})',
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # DOWNLOAD HASIL
            st.subheader("💾 Download Hasil Prediksi")

            # Gabungkan semua prediksi
            result_data = {'Tanggal': future_dates}

            for model_name, predictions in all_predictions.items():
                result_data[model_name] = predictions

            result_df = pd.DataFrame(result_data)
            csv = result_df.to_csv(index=False)

            st.download_button(
                label="📥 Download Semua Prediksi sebagai CSV",
                data=csv,
                file_name="hasil_prediksi_semua_model.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"❌ Terjadi error: {str(e)}")
        st.info("""
        **Tips mengatasi error:**
        1. Pastikan file CSV memiliki data numerik
        2. Pastikan kolom tanggal bisa dikonversi ke format tanggal
        3. Pastikan ada minimal 10 baris data
        4. Coba pilih algoritma yang lebih sederhana terlebih dahulu
        """)

else:
    # Petunjuk penggunaan
    st.info("""
    ## 📋 Cara Menggunakan:

    1. **Upload file CSV** data saham Anda
    2. **Pilih kolom tanggal** dan **kolom harga**
    3. **Pilih algoritma** yang ingin digunakan (bisa multiple)
    4. **Klik tombol** "MULAI TRAINING DAN PREDIKSI"
    5. **Lihat hasil** prediksi 7 hari ke depan dari semua model

    ### Algoritma yang Tersedia:
    - 📊 **Linear Regression** - Cepat dan sederhana
    - 🛡️ **Ridge Regression** - Linear dengan regularization
    - 🌳 **Random Forest** - Robust terhadap noise
    - 🚀 **Gradient Boosting** - Akurasi tinggi
    - 🔄 **SVR** - Baik untuk pattern kompleks
    """)

# Footer
st.markdown("---")
st.markdown("Dibuat dengan Sepenuh-penuhnya 😂🥴😵‍💫 Cuman ngira^ tapi pakek logika")
