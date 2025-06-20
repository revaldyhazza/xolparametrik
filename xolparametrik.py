import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from fitter import Fitter
import scipy.stats as stats
import warnings
import io

st.set_page_config(page_title="Pricing XoL Parametrik", page_icon="💵")
warnings.filterwarnings('ignore')

# Kamus untuk nama distribusi yang ramah pengguna
distribution_names = {
    'weibull_min': 'Distribusi Weibull',
    'lognorm': 'Distribusi Lognormal',
    'gamma': 'Distribusi Gamma',
    'pareto': 'Distribusi Pareto',
    'expon': 'Distribusi Eksponensial'
}

# Daftar distribusi yang akan digunakan
distributions = ['weibull_min', 'lognorm', 'gamma', 'pareto', 'expon']

# Fungsi untuk menghitung metrik
def calculate_metrics(data, dist_name, params, seed=42):
    dist = getattr(stats, dist_name)
    
    # Untuk Lognormal, paksa loc=0 untuk memastikan nilai positif
    if dist_name == 'lognorm':
        params = (params[0], 0, params[2])  # s, loc=0, scale
    
    # Set seed untuk data acak
    np.random.seed(seed)
    fitted_data = dist.rvs(*params, size=len(data))
    
    # RMSE
    rmse = np.sqrt(np.mean((data - fitted_data) ** 2))
    
    # Log-Likelihood
    try:
        log_likelihood = np.sum(dist.logpdf(data, *params))
    except:
        log_likelihood = np.nan
    
    # AIC dan BIC
    k = len(params)  # Jumlah parameter
    n = len(data)    # Jumlah data
    aic = 2 * k - 2 * log_likelihood if not np.isnan(log_likelihood) else np.nan
    bic = k * np.log(n) - 2 * log_likelihood if not np.isnan(log_likelihood) else np.nan
    
    # Kolmogorov-Smirnov (KS) Statistic
    ks_stat, _ = stats.ks_2samp(data, fitted_data)
    
    # Mean, Variansi, Standar Deviasi, Skewness, dan Kurtosis
    mean = np.mean(fitted_data)
    variance = np.var(fitted_data)
    std_dev = np.std(fitted_data)
    skewness = stats.skew(fitted_data)
    kurtosis = stats.kurtosis(fitted_data)
    
    return {
        'RMSE': rmse,
        'Log-Likelihood': log_likelihood,
        'AIC': aic,
        'BIC': bic,
        'KS': ks_stat,
        'Mean': mean,
        'Variance': variance,
        'Std Dev': std_dev,
        'Skewness': skewness,
        'Kurtosis': kurtosis
    }

# Fungsi untuk membaca file dengan caching
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    return df

# Fungsi untuk fitting distribusi dengan caching
@st.cache_data
def fit_distributions(data, distributions, _timeout=60):
    # Validasi data untuk Lognormal
    valid_distributions = distributions.copy()
    if 'lognorm' in valid_distributions and np.any(data <= 0):
        st.warning("Data mengandung nilai nol atau negatif. Distribusi Lognormal hanya mendukung data positif. Menghapus Lognormal dari fitting.")
        valid_distributions.remove('lognorm')
    
    if not valid_distributions:
        st.error("Tidak ada distribusi yang valid untuk di-fit ke data ini.")
        st.stop()
    
    f = Fitter(data, distributions=valid_distributions, timeout=_timeout)
    f.fit()
    return f

# Fungsi untuk simulasi Monte Carlo dengan seed
def monte_carlo_simulation(dist_name, params, n_iterations=1000, seed=42):
    np.random.seed(seed)  # Mengatur seed untuk reproduksibilitas
    dist = getattr(stats, dist_name)
    
    # Generate data acak menggunakan inverse CDF untuk lognorm dan gamma
    if dist_name == 'lognorm':
        # Parameter untuk lognorm: s (shape), loc, scale
        s, loc, scale = params
        # Pastikan loc=0 untuk data positif
        loc = 0
        # Generate probabilitas uniform [0,1]
        uniform_data = np.random.uniform(0, 1, n_iterations)
        # Gunakan ppf (inverse CDF) untuk menghasilkan data acak
        simulated_data = dist.ppf(uniform_data, s, loc=loc, scale=scale)
    elif dist_name == 'gamma':
        # Parameter untuk gamma: a (shape), loc, scale
        a, loc, scale = params
        # Generate probabilitas uniform [0,1]
        uniform_data = np.random.uniform(0, 1, n_iterations)
        # Gunakan ppf (inverse CDF) untuk menghasilkan data acak
        simulated_data = dist.ppf(uniform_data, a, loc=loc, scale=scale)
    else:
        # Untuk distribusi lain (weibull_min, pareto, expon), gunakan rvs
        simulated_data = dist.rvs(*params, size=n_iterations)
    
    # Validasi hasil simulasi
    if np.any(simulated_data <= 0):
        st.error(f"Simulasi Monte Carlo untuk {dist_name} menghasilkan nilai nol atau negatif. Ini tidak valid untuk distribusi ini.")
        return None
    
    return simulated_data

# Fungsi untuk alokasi klaim ke UR dan layer
def allocate_claims(simulated_data, ur, layers):
    results = []
    for claim in simulated_data:
        claim_allocation = {'UR': 0, 'Layer 1': 0, 'Layer 2': 0, 'Layer 3': 0, 'Layer 4': 0, 'Layer 5': 0, 'Layer 6': 0}
        remaining_claim = max(0, claim)  # Pastikan klaim tidak negatif
        
        # Alokasi ke UR
        claim_allocation['UR'] = min(remaining_claim, ur)
        remaining_claim -= claim_allocation['UR']
        
        # Alokasi ke layer 1-6
        for i, layer_limit in enumerate(layers, 1):
            if remaining_claim <= 0:
                break
            claim_allocation[f'Layer {i}'] = min(remaining_claim, layer_limit)
            remaining_claim -= claim_allocation[f'Layer {i}']
        
        results.append(claim_allocation)
    
    return pd.DataFrame(results)

# Judul aplikasi
st.title("Pricing Reasuransi Excess of Loss (XoL) Metode Parametrik")
st.write("Silakan unggah file dengan format .csv/.xls/.xlsx, pilih kolom untuk dilakukan fitting distribusi, dan lakukan simulasi Monte Carlo menggunakan distribusi yang dapat dipilih dengan seed yang dapat diatur.")

# Upload file
uploaded_file = st.file_uploader("Unggah file data", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # Membaca file dengan caching
    try:
        df = load_data(uploaded_file)
        
        st.write("Preview Data:")
        st.dataframe(df, hide_index=True)

        # Pilih kolom
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_columns:
            selected_column = st.selectbox("Pilih kolom untuk fitting distribusi", numeric_columns)
            
            # Ambil data dari kolom yang dipilih
            data = df[selected_column].dropna().values

            if len(data) > 0:
                # Menampilkan mean dan standard deviation
                st.subheader("Statistik Data")
                st.write(f"**Rata-rata (Mean):** {np.mean(data):.4f}")
                st.write(f"**Standar Deviasi (Std Dev):** {np.std(data):.4f}")

                # Visualisasi histogram data
                st.subheader("Histogram Data")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data, kde=True, stat="density", bins=150, ax=ax)
                ax.set_title(f'Histogram {selected_column}')
                ax.set_xlabel('Nilai')
                ax.set_ylabel('Densitas')
                st.pyplot(fig)

                # Fitting distribusi dengan caching
                st.subheader("Proses Fitting Distribusi")
                with st.spinner("Sedang melakukan fitting distribusi..."):
                    f = fit_distributions(data, distributions, _timeout=60)

                # Cek apakah ada distribusi yang berhasil di-fit
                if not f.fitted_param:
                    st.error("Tidak ada distribusi yang berhasil di-fit ke data ini. Silakan cek data atau coba distribusi lain.")
                    st.stop()

                # Ringkasan semua distribusi diurutkan berdasarkan RMSE
                st.subheader("Ringkasan Semua Distribusi (Diurutkan Berdasarkan RMSE)")
                metrics_scores = {}
                for dist_name in distributions:
                    if dist_name in f.fitted_param:
                        params = f.fitted_param[dist_name]
                        metrics = calculate_metrics(data, dist_name, params, seed=42)
                        metrics_scores[dist_name] = metrics
                
                # Urutkan berdasarkan RMSE
                sorted_distributions = sorted(metrics_scores.items(), key=lambda x: x[1]['RMSE'])
                summary_df = pd.DataFrame({
                    'Distribusi': [distribution_names.get(dist, dist) for dist, _ in sorted_distributions],
                    'RMSE': [metrics['RMSE'] for _, metrics in sorted_distributions],
                    'Log-Likelihood': [metrics['Log-Likelihood'] for _, metrics in sorted_distributions],
                    'AIC': [metrics['AIC'] for _, metrics in sorted_distributions],
                    'BIC': [metrics['BIC'] for _, metrics in sorted_distributions],
                    'KS': [metrics['KS'] for _, metrics in sorted_distributions],
                    'Mean': [metrics['Mean'] for _, metrics in sorted_distributions],
                    'Variance': [metrics['Variance'] for _, metrics in sorted_distributions],
                    'Std Dev': [metrics['Std Dev'] for _, metrics in sorted_distributions],
                    'Skewness': [metrics['Skewness'] for _, metrics in sorted_distributions],
                    'Kurtosis': [metrics['Kurtosis'] for _, metrics in sorted_distributions],
                    'Parameter': [f.fitted_param.get(dist, {}) for dist, _ in sorted_distributions]
                })
                st.dataframe(summary_df, hide_index=True)

                # Distribusi terbaik berdasarkan RMSE dalam bentuk tabel
                st.subheader("Informasi terkait Distribusi Terbaik")
                best_dist_name, best_metrics = sorted_distributions[0]
                best_params = f.fitted_param.get(best_dist_name, {})
                friendly_name = distribution_names.get(best_dist_name, best_dist_name)
                
                # Membuat tabel untuk distribusi terbaik
                best_dist_df = pd.DataFrame({
                    'Metrik': ['Distribusi', 'RMSE', 'Log-Likelihood', 'AIC', 'BIC', 'KS Statistic', 'Mean', 'Variance', 'Std Dev', 'Skewness', 'Kurtosis', 'Parameter'],
                    'Nilai': [
                        friendly_name,
                        f"{best_metrics['RMSE']:.4f}",
                        'N/A' if np.isnan(best_metrics['Log-Likelihood']) else f"{best_metrics['Log-Likelihood']:.4f}",
                        'N/A' if np.isnan(best_metrics['AIC']) else f"{best_metrics['AIC']:.4f}",
                        'N/A' if np.isnan(best_metrics['BIC']) else f"{best_metrics['BIC']:.4f}",
                        f"{best_metrics['KS']:.4f}",
                        f"{best_metrics['Mean']:.4f}",
                        f"{best_metrics['Variance']:.4f}",
                        f"{best_metrics['Std Dev']:.4f}",
                        f"{best_metrics['Skewness']:.4f}",
                        f"{best_metrics['Kurtosis']:.4f}",
                        str(best_params)
                    ]
                })
                st.table(best_dist_df)

                # Visualisasi 3 distribusi terbaik
                st.subheader("Plot Distribusi Terbaik")
                plt.figure(figsize=(10, 6))
                f.plot_pdf(Nbest=3)
                plt.title('Fitting Distribusi Terbaik')
                plt.xlabel('Nilai')
                plt.ylabel('Densitas')
                st.pyplot(plt.gcf())

                # Pilih distribusi untuk simulasi Monte Carlo
                st.subheader("Simulasi Monte Carlo")
                dist_options = [distribution_names.get(dist, dist) for dist in distributions if dist in f.fitted_param]
                selected_dist = st.selectbox("Pilih distribusi untuk dilakukan simulasi Monte Carlo", dist_options)
                
                # Slider untuk mengatur seed
                seed_value = st.slider("Atur seed untuk Simulasi Monte Carlo", min_value=0, max_value=1000, value=42, step=1)
                
                # Input untuk UR dan Layer 1-6
                st.subheader("Masukkan Nilai UR dan Layer Excess of Loss")
                # Tambah info rentang nilai data
                min_value = np.min(data)
                max_value = np.max(data)
                st.info(f"Data yang diunggah memiliki rentang nilai dari {min_value:,.0f} sampai {max_value:,.0f}.")
                
                ur = st.number_input("Ultimate Retention (UR)", min_value=0, step=200, format="%d")
                layer_1 = st.number_input("Layer 1", min_value=0, step=200, format="%d")
                layer_2 = st.number_input("Layer 2", min_value=0, step=200, format="%d")
                layer_3 = st.number_input("Layer 3", min_value=0, step=200, format="%d")
                layer_4 = st.number_input("Layer 4", min_value=0, step=200, format="%d")
                layer_5 = st.number_input("Layer 5", min_value=0, step=200, format="%d")
                layer_6 = st.number_input("Layer 6", min_value=0, step=200, format="%d")
                
                layers = [layer_1, layer_2, layer_3, layer_4, layer_5, layer_6]
                
                # Tampilkan nilai UR
