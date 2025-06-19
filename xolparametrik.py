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

# Fungsi untuk menghitung parameter secara manual dengan error handling
def calculate_manual_parameters(data, dist_name):
    try:
        if dist_name == 'lognorm':
            if np.any(data <= 0):
                raise ValueError("Data mengandung nilai nol atau negatif, tidak valid untuk Lognormal.")
            log_data = np.log(data)
            mu = np.mean(log_data)
            sigma = np.std(log_data)
            if np.isnan(mu) or np.isnan(sigma):
                raise ValueError("Gagal menghitung mu atau sigma untuk Lognormal.")
            return (sigma, mu), {'sigma': sigma, 'mu': mu}
        
        elif dist_name == 'gamma':
            mean_data = np.mean(data)
            var_data = np.var(data)
            if var_data == 0:
                raise ValueError("Variansi data nol, tidak dapat menghitung parameter Gamma.")
            alpha = (mean_data ** 2) / var_data
            beta = var_data / mean_data
            if np.isnan(alpha) or np.isnan(beta):
                raise ValueError("Gagal menghitung alpha atau beta untuk Gamma.")
            return (alpha, beta), {'alpha': alpha, 'beta': beta}
        
        elif dist_name == 'pareto':
            if np.any(data <= 0):
                raise ValueError("Data mengandung nilai nol atau negatif, tidak valid untuk Pareto.")
            beta = np.min(data)
            log_term = np.log(data / beta)
            if np.any(np.isnan(log_term)) or np.any(np.isinf(log_term)):
                raise ValueError("Gagal menghitung log(data/beta) untuk Pareto.")
            alpha = max(1 / np.mean(log_term), 1.5)  # Constrain alpha to avoid heavy tails
            if np.isnan(alpha):
                raise ValueError("Gagal menghitung alpha untuk Pareto.")
            return (alpha, beta), {'alpha': alpha, 'beta': beta}
        
        elif dist_name == 'expon':
            theta = np.mean(data)
            if np.isnan(theta):
                raise ValueError("Gagal menghitung theta untuk Eksponensial.")
            return (theta,), {'theta': theta}
        
        elif dist_name == 'weibull_min':
            shape, loc, scale = stats.weibull_min.fit(data)
            if np.isnan(shape) or np.isnan(loc) or np.isnan(scale):
                raise ValueError("Gagal menghitung parameter untuk Weibull.")
            return (shape, loc, scale), {'shape': shape, 'loc': loc, 'scale': scale}
        
    except Exception as e:
        st.error(f"Kesalahan saat menghitung parameter untuk {distribution_names.get(dist_name, dist_name)}: {str(e)}")
        return None, None

# Fungsi untuk menghitung metrik
def calculate_metrics(data, dist_name, params, seed=42):
    np.random.seed(seed)
    if dist_name == 'lognorm':
        sigma, mu = params
        fitted_data = np.exp(mu + sigma * np.random.normal(size=len(data)))
    elif dist_name == 'gamma':
        alpha, beta = params
        fitted_data = stats.gamma.ppf(np.random.uniform(size=len(data)), alpha, scale=beta)
    elif dist_name == 'pareto':
        alpha, beta = params
        fitted_data = beta / (np.random.uniform(size=len(data)) ** (1 / alpha))
    elif dist_name == 'expon':
        (theta,) = params
        fitted_data = -theta * np.log(np.random.uniform(size=len(data)))
    elif dist_name == 'weibull_min':
        shape, loc, scale = params
        fitted_data = stats.weibull_min.ppf(np.random.uniform(size=len(data)), shape, loc, scale)
    
    # RMSE
    rmse = np.sqrt(np.mean((data - fitted_data) ** 2))
    
    # Log-Likelihood
    try:
        if dist_name == 'lognorm':
            log_likelihood = np.sum(stats.lognorm.logpdf(data, sigma, loc=0, scale=np.exp(mu)))
        elif dist_name == 'gamma':
            log_likelihood = np.sum(stats.gamma.logpdf(data, alpha, scale=beta))
        elif dist_name == 'pareto':
            log_likelihood = np.sum(stats.pareto.logpdf(data, alpha, scale=beta))
        elif dist_name == 'expon':
            log_likelihood = np.sum(stats.expon.logpdf(data, scale=theta))
        elif dist_name == 'weibull_min':
            log_likelihood = np.sum(stats.weibull_min.logpdf(data, shape, loc, scale))
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
    valid_distributions = distributions.copy()
    if 'lognorm' in valid_distributions and np.any(data <= 0):
        st.warning("Data mengandung nilai nol atau negatif. Distribusi Lognormal hanya mendukung data positif. Menghapus Lognormal dari fitting.")
        valid_distributions.remove('lognorm')
    if 'pareto' in valid_distributions and np.any(data <= 0):
        st.warning("Data mengandung nilai nol atau negatif. Distribusi Pareto hanya mendukung data positif. Menghapus Pareto dari fitting.")
        valid_distributions.remove('pareto')
    
    if not valid_distributions:
        st.error("Tidak ada distribusi yang valid untuk di-fit ke data ini.")
        st.stop()
    
    f = Fitter(data, distributions=valid_distributions, timeout=_timeout)
    f.fit()
    return f

# Fungsi untuk simulasi Monte Carlo dengan seed
def monte_carlo_simulation(dist_name, params, n_iterations=1000, seed=42):
    np.random.seed(seed)
    U = np.random.uniform(size=n_iterations)
    
    if dist_name == 'lognorm':
        sigma, mu = params
        simulated_data = np.exp(mu + sigma * stats.norm.ppf(U))
    elif dist_name == 'gamma':
        alpha, beta = params
        simulated_data = stats.gamma.ppf(U, alpha, scale=beta)
    elif dist_name == 'pareto':
        alpha, beta = params
        simulated_data = beta / (U ** (1 / alpha))
    elif dist_name == 'expon':
        (theta,) = params
        simulated_data = -theta * np.log(1 - U)
    elif dist_name == 'weibull_min':
        shape, loc, scale = params
        simulated_data = stats.weibull_min.ppf(U, shape, loc, scale)
    
    if np.any(simulated_data <= 0):
        st.error(f"Simulasi Monte Carlo untuk {dist_name} menghasilkan nilai nol atau negatif. Ini tidak valid untuk distribusi ini.")
        return None
    
    return simulated_data

# Fungsi untuk alokasi klaim ke UR dan layer
def allocate_claims(simulated_data, ur, layers):
    results = []
    for claim in simulated_data:
        claim_allocation = {'UR': 0, 'Layer 1': 0, 'Layer 2': 0, 'Layer 3': 0, 'Layer 4': 0, 'Layer 5': 0, 'Layer 6': 0}
        remaining_claim = max(0, claim)
        
        claim_allocation['UR'] = min(remaining_claim, ur)
        remaining_claim -= claim_allocation['UR']
        
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
    try:
        df = load_data(uploaded_file)
        
        st.write("Preview Data:")
        st.dataframe(df, hide_index=True)

        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_columns:
            selected_column = st.selectbox("Pilih kolom untuk fitting distribusi", numeric_columns)
            
            data = df[selected_column].dropna().values

            if len(data) > 0:
                st.subheader("Statistik Data")
                st.write(f"**Rata-rata (Mean):** {np.mean(data):.4f}")
                st.write(f"**Standar Deviasi (Std Dev):** {np.std(data):.4f}")

                st.subheader("Histogram Data")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data, kde=True, stat="density", bins=150, ax=ax)
                ax.set_title(f'Histogram {selected_column}')
                ax.set_xlabel('Nilai')
                ax.set_ylabel('Densitas')
                st.pyplot(fig)

                st.subheader("Proses Fitting Distribusi")
                with st.spinner("Sedang melakukan fitting distribusi..."):
                    f = fit_distributions(data, distributions, _timeout=60)

                if not f.fitted_param:
                    st.error("Tidak ada distribusi yang berhasil di-fit ke data ini. Silakan cek data atau coba distribusi lain.")
                    st.stop()

                st.subheader("Ringkasan Semua Distribusi (Diurutkan Berdasarkan RMSE)")
                metrics_scores = {}
                for dist_name in distributions:
                    if dist_name in f.fitted_param:
                        params, _ = calculate_manual_parameters(data, dist_name)
                        if params is None:
                            continue
                        metrics = calculate_metrics(data, dist_name, params, seed=42)
                        metrics_scores[dist_name] = metrics
                
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
                    'Parameter': [calculate_manual_parameters(data, dist)[1] for dist, _ in sorted_distributions]
                })
                st.dataframe(summary_df, hide_index=True)

                st.subheader("Informasi terkait Distribusi Terbaik")
                best_dist_name, best_metrics = sorted_distributions[0]
                best_params, best_params_display = calculate_manual_parameters(data, best_dist_name)
                if best_params is None:
                    st.error("Gagal menghitung parameter untuk distribusi terbaik.")
                    st.stop()
                friendly_name = distribution_names.get(best_dist_name, best_dist_name)
                
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
                        str(best_params_display)
                    ]
                })
                st.table(best_dist_df)

                st.subheader("Plot Distribusi Terbaik")
                plt.figure(figsize=(10, 6))
                f.plot_pdf(Nbest=3)
                plt.title('Fitting Distribusi Terbaik')
                plt.xlabel('Nilai')
                plt.ylabel('Densitas')
                st.pyplot(plt.gcf())

                st.subheader("Simulasi Monte Carlo")
                dist_options = [distribution_names[dist] for dist in distributions if dist in f.fitted_param]
                selected_dist = st.selectbox("Pilih distribusi untuk dilakukan simulasi Monte Carlo", dist_options)
                
                dist_name = [k for k, v in distribution_names.items() if v == selected_dist][0]
                params_simulasi, params_display = calculate_manual_parameters(data, dist_name)
                
                if params_simulasi is None or params_display is None:
                    st.error(f"Tidak dapat melanjutkan simulasi karena parameter untuk {selected_dist} tidak valid.")
                    st.stop()
                
                st.markdown(f"### Parameter untuk Distribusi {selected_dist}")
                params_df = pd.DataFrame({
                    'Parameter': list(params_display.keys()),
                    'Nilai': [f"{value:.4f}" for value in params_display.values()]
                })
                st.dataframe(params_df, use_container_width=True, hide_index=True)
                
                seed_value = st.slider("Atur seed untuk Simulasi Monte Carlo", min_value=0, max_value=1000, value=42, step=1)
                
                st.subheader("Masukkan Nilai UR dan Layer Excess of Loss")
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
                
                st.subheader("Nilai UR dan Layer yang Dimasukkan")
                layers_df = pd.DataFrame({
                    'UR': [ur],
                    'Layer 1': [layer_1],
                    'Layer 2': [layer_2],
                    'Layer 3': [layer_3],
                    'Layer 4': [layer_4],
                    'Layer 5': [layer_5],
                    'Layer 6': [layer_6]
                })
                st.dataframe(layers_df, use_container_width=True, hide_index=True)
                
                st.subheader("Faktor Loading (%)")
                risk_adjustment = st.number_input("Risk Adjustment (%)", min_value=0, value=10, step=1, format="%d") / 100
                profit = st.number_input("Profit (%)", min_value=0, value=5, step=1, format="%d") / 100
                operating_expenses = st.number_input("Operating Expenses (%)", min_value=0, value=5, step=1, format="%d") / 100
                komisi = st.number_input("Komisi (%)", min_value=0, value=2, step=1, format="%d") / 100
                
                st.subheader("Parameter Loading yang telah diinput")
                percentages_df = pd.DataFrame({
                    'Risk Adjustment (%)': [int(risk_adjustment * 100)],
                    'Profit (%)': [int(profit * 100)],
                    'Operating Expenses (%)': [int(operating_expenses * 100)],
                    'Komisi (%)': [int(komisi * 100)]
                })
                st.dataframe(percentages_df, use_container_width=True, hide_index=True)
                
                if st.button("Run Simulasi Monte Carlo"):
                    with st.spinner("Menjalankan simulasi Monte Carlo..."):
                        n_iterations = 1000
                        simulated_data = monte_carlo_simulation(dist_name, params_simulasi, n_iterations=n_iterations, seed=seed_value)
                        
                        if simulated_data is None:
                            st.stop()
                        
                        st.subheader(f"Hasil Simulasi Monte Carlo (Distribusi {selected_dist} Seed: {seed_value})")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.histplot(simulated_data, kde=True, stat="density", bins=150, ax=ax)
                        ax.set_title(f'Histogram Hasil Simulasi Monte Carlo ({selected_dist}, Seed: {seed_value})')
                        ax.set_xlabel('Nilai')
                        ax.set_ylabel('Densitas')
                        st.pyplot(fig)
                        
                        st.subheader("Perbandingan Statistik Data Asli dan Simulasi")
                        stats_df = pd.DataFrame({
                            'Statistik': ['Rata-rata', 'Standar Deviasi'],
                            'Data Asli': [np.mean(data), np.std(data)],
                            'Simulasi': [np.mean(simulated_data), np.std(simulated_data)]
                        })
                        stats_df['Data Asli'] = stats_df['Data Asli'].map('{:.4f}'.format)
                        stats_df['Simulasi'] = stats_df['Simulasi'].map('{:.4f}'.format)
                        st.dataframe(stats_df, hide_index=True)
                        
                        st.subheader("Alokasi Klaim Simulasi ke UR dan Layer")
                        claims_df = allocate_claims(simulated_data, ur, layers)
                        
                        st.write("Preview Alokasi Klaim:")
                        claims_df_display = claims_df.apply(lambda x: x.map(lambda y: int(y) if y.is_integer() else y))
                        st.dataframe(claims_df_display, hide_index=True)
                        st.write("Rata-rata Alokasi Klaim:")
                        avg_claims = claims_df.mean()
                        avg_claims_df = pd.DataFrame({
                            'UR': [int(avg_claims['UR']) if avg_claims['UR'].is_integer() else avg_claims['UR']],
                            'Layer 1': [int(avg_claims['Layer 1']) if avg_claims['Layer 1'].is_integer() else avg_claims['Layer 1']],
                            'Layer 2': [int(avg_claims['Layer 2']) if avg_claims['Layer 2'].is_integer() else avg_claims['Layer 2']],
                            'Layer 3': [int(avg_claims['Layer 3']) if avg_claims['Layer 3'].is_integer() else avg_claims['Layer 3']],
                            'Layer 4': [int(avg_claims['Layer 4']) if avg_claims['Layer 4'].is_integer() else avg_claims['Layer 4']],
                            'Layer 5': [int(avg_claims['Layer 5']) if avg_claims['Layer 5'].is_integer() else avg_claims['Layer 5']],
                            'Layer 6': [int(avg_claims['Layer 6']) if avg_claims['Layer 6'].is_integer() else avg_claims['Layer 6']]
                        })
                        st.dataframe(avg_claims_df, use_container_width=True, hide_index=True)
                        
                        risk_premium = avg_claims * n_iterations
                        risk_premium_df = pd.DataFrame({
                            'UR': [int(risk_premium['UR']) if risk_premium['UR'].is_integer() else risk_premium['UR']],
                            'Layer 1': [int(risk_premium['Layer 1']) if risk_premium['Layer 1'].is_integer() else risk_premium['Layer 1']],
                            'Layer 2': [int(risk_premium['Layer 2']) if risk_premium['Layer 2'].is_integer() else risk_premium['Layer 2']],
                            'Layer 3': [int(risk_premium['Layer 3']) if risk_premium['Layer 3'].is_integer() else risk_premium['Layer 3']],
                            'Layer 4': [int(risk_premium['Layer 4']) if risk_premium['Layer 4'].is_integer() else risk_premium['Layer 4']],
                            'Layer 5': [int(risk_premium['Layer 5']) if risk_premium['Layer 5'].is_integer() else risk_premium['Layer 5']],
                            'Layer 6': [int(risk_premium['Layer 6']) if risk_premium['Layer 6'].is_integer() else risk_premium['Layer 6']]
                        })
                        st.subheader("Risk Premium")
                        st.dataframe(risk_premium_df, use_container_width=True, hide_index=True)
                        
                        denominator_ur = 1 - profit - operating_expenses
                        denominator_layers = 1 - profit - operating_expenses - komisi
                        if denominator_ur <= 0:
                            st.error("Profit + Operating Expenses tidak boleh >= 100% untuk UR!")
                        elif denominator_layers <= 0:
                            st.error("Profit + Operating Expenses + Komisi tidak boleh >= 100% untuk Layer!")
                        else:
                            xol_premium = pd.Series(index=risk_premium.index, dtype=float)
                            xol_premium['UR'] = (risk_premium['UR'] * (1 + risk_adjustment)) / denominator_ur
                            for i in range(1, 7):
                                xol_premium[f'Layer {i}'] = (risk_premium[f'Layer {i}'] * (1 + risk_adjustment)) / denominator_layers
                            xol_premium_df = pd.DataFrame({
                                'UR': [int(xol_premium['UR']) if xol_premium['UR'].is_integer() else xol_premium['UR']],
                                'Layer 1': [int(xol_premium['Layer 1']) if xol_premium['Layer 1'].is_integer() else xol_premium['Layer 1']],
                                'Layer 2': [int(xol_premium['Layer 2']) if xol_premium['Layer 2'].is_integer() else xol_premium['Layer 2']],
                                'Layer 3': [int(xol_premium['Layer 3']) if xol_premium['Layer 3'].is_integer() else xol_premium['Layer 3']],
                                'Layer 4': [int(xol_premium['Layer 4']) if xol_premium['Layer 4'].is_integer() else xol_premium['Layer 4']],
                                'Layer 5': [int(xol_premium['Layer 5']) if xol_premium['Layer 5'].is_integer() else xol_premium['Layer 5']],
                                'Layer 6': [int(xol_premium['Layer 6']) if xol_premium['Layer 6'].is_integer() else xol_premium['Layer 6']]
                            })
                            st.subheader("Premi Excess of Loss (XoL)")
                            st.dataframe(xol_premium_df, use_container_width=True, hide_index=True)
                            
                            rol = {}
                            for i, layer_value in enumerate(layers, 1):
                                if layer_value > 0:
                                    rol[f'Layer {i}'] = 100 * (xol_premium[f'Layer {i}'] / (layer_value * n_iterations))
                                else:
                                    rol[f'Layer {i}'] = 0
                            rol_df = pd.DataFrame({
                                'Layer 1 (%)': [f"{rol['Layer 1']:,.2f}"],
                                'Layer 2 (%)': [f"{rol['Layer 2']:,.2f}"],
                                'Layer 3 (%)': [f"{rol['Layer 3']:,.2f}"],
                                'Layer 4 (%)': [f"{rol['Layer 4']:,.2f}"],
                                'Layer 5 (%)': [f"{rol['Layer 5']:,.2f}"],
                                'Layer 6 (%)': [f"{rol['Layer 6']:,.2f}"]
                            })
                            st.subheader("Rate On Line (RoL)")
                            st.dataframe(rol_df, use_container_width=True, hide_index=True)
                            
                            min_deposit_premium = xol_premium[['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5', 'Layer 6']].sum()
                            st.info(f"Premi Minimum Deposit yang harus dibayarkan adalah {int(min_deposit_premium) if min_deposit_premium.is_integer() else min_deposit_premium:,.2f}")
                        
                        sim_df = pd.DataFrame(simulated_data, columns=['Klaim Acak'])
                        sim_df = pd.concat([sim_df, claims_df], axis=1)
                        sim_df = sim_df.apply(lambda x: x.map(lambda y: int(y) if y.is_integer() else y))
                        
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            sim_df.to_excel(writer, index=False, sheet_name='Simulation_Results')
                        excel_data = output.getvalue()
                        
                        st.download_button(
                            label="Unduh Hasil Simulasi dan Alokasi Klaim (Excel)",
                            data=excel_data,
                            file_name=f"Simulasi Monte Carlo Distribusi {selected_dist} Seed: {seed_value}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

            else:
                st.error("Kolom yang dipilih tidak memiliki data yang valid.")
        else:
            st.error("Tidak ada kolom numerik dalam file yang diunggah.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {str(e)}")
else:
    st.info("Silakan unggah file CSV atau Excel untuk memulai.")
