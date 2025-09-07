import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import urllib.parse

# --- Page Config ---
st.set_page_config(
    page_title="CLT Simulator Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- App Title ---
st.title("🏛️ Central Limit Theorem (CLT) Workbench")
st.markdown("An interactive app for Executives and Data Scientists to explore the CLT.")

# --- State Management and URL Params ---
def get_from_params(param, default, type_func):
    """Helper to get and type-cast a URL parameter."""
    try:
        return type_func(st.query_params.get(param, default))
    except (ValueError, TypeError):
        return default

# --- Sidebar Controls ---
with st.sidebar:
    st.header("👤 Persona")
    persona = st.radio(
        "Choose your view",
        ("Executive", "Data Scientist"),
        horizontal=True,
        key='persona',
        index=["Executive", "Data Scientist"].index(get_from_params('persona', "Executive", str))
    )

    ab_mode = st.toggle("A/B Comparison Mode", key='ab_mode', value=get_from_params('ab_mode', False, bool))

    # --- Simulation Controls ---
    if ab_mode:
        st.header("Shared Controls")
        M = st.slider("Number of Samples (M)", 100, 20000, get_from_params('M', 1000, int), key='M')
        seed = st.number_input("Random Seed", 0, 10000, get_from_params('seed', 42, int), key='seed')

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Scenario A")
            dist_name_a = st.selectbox("Distribution", ("Normal", "Uniform", "Exponential", "Pareto"), key="dist_name_a", index=["Normal", "Uniform", "Exponential", "Pareto"].index(get_from_params('dist_name_a', "Normal", str)))
            n_a = st.slider("Sample Size (n)", 1, 5000, get_from_params('n_a', 30, int), key="n_a")
        with col2:
            st.subheader("Scenario B")
            dist_name_b = st.selectbox("Distribution", ("Normal", "Uniform", "Exponential", "Pareto"), key="dist_name_b", index=["Normal", "Uniform", "Exponential", "Pareto"].index(get_from_params('dist_name_b', "Uniform", str)))
            n_b = st.slider("Sample Size (n)", 1, 5000, get_from_params('n_b', 100, int), key="n_b")
    else:
        st.header("🔬 Simulation Controls")
        dist_name = st.selectbox("Base Distribution", ("Normal", "Uniform", "Exponential", "Pareto"), key="dist_name", index=["Normal", "Uniform", "Exponential", "Pareto"].index(get_from_params('dist_name', "Normal", str)))
        n = st.slider("Sample Size (n)", 1, 5000, get_from_params('n', 30, int), key="n")
        M = st.slider("Number of Samples (M)", 100, 20000, get_from_params('M', 1000, int), key="M_single")
        seed = st.number_input("Random Seed", 0, 10000, get_from_params('seed', 42, int), key="seed_single")

    # --- Distribution Parameters ---
    st.divider()
    st.header("Distribution Parameters")
    if ab_mode:
        with st.expander("Scenario A Parameters", expanded=True):
            if dist_name_a == "Normal": dist_params_a = {'loc': st.slider("Mean (μ)", -10.0, 10.0, get_from_params('loc_a', 0.0, float), 0.1, key="loc_a"), 'scale': st.slider("Std Dev (σ)", 0.1, 10.0, get_from_params('scale_a', 1.0, float), 0.1, key="scale_a")}
            elif dist_name_a == "Uniform": dist_params_a = {'loc': st.slider("Start", -10.0, 10.0, get_from_params('loc_a', 0.0, float), 0.1, key="loc_a"), 'scale': st.slider("Width", 0.1, 20.0, get_from_params('scale_a', 5.0, float), 0.1, key="scale_a")}
            elif dist_name_a == "Exponential": dist_params_a = {'scale': st.slider("Scale (λ⁻¹)", 0.1, 10.0, get_from_params('scale_a', 1.0, float), 0.1, key="scale_a")}
            elif dist_name_a == "Pareto": dist_params_a = {'b': st.slider("Shape (b)", 0.1, 5.0, get_from_params('b_a', 2.0, float), 0.1, key="b_a")}
        with st.expander("Scenario B Parameters"):
            if dist_name_b == "Normal": dist_params_b = {'loc': st.slider("Mean (μ)", -10.0, 10.0, get_from_params('loc_b', 0.0, float), 0.1, key="loc_b"), 'scale': st.slider("Std Dev (σ)", 0.1, 10.0, get_from_params('scale_b', 2.0, float), 0.1, key="scale_b")}
            elif dist_name_b == "Uniform": dist_params_b = {'loc': st.slider("Start", -10.0, 10.0, get_from_params('loc_b', 2.0, float), 0.1, key="loc_b"), 'scale': st.slider("Width", 0.1, 20.0, get_from_params('scale_b', 5.0, float), 0.1, key="scale_b")}
            elif dist_name_b == "Exponential": dist_params_b = {'scale': st.slider("Scale (λ⁻¹)", 0.1, 10.0, get_from_params('scale_b', 2.0, float), 0.1, key="scale_b")}
            elif dist_name_b == "Pareto": dist_params_b = {'b': st.slider("Shape (b)", 0.1, 5.0, get_from_params('b_b', 1.0, float), 0.1, key="b_b")}
    else:
        if dist_name == "Normal": dist_params = {'loc': st.slider("Mean (μ)", -10.0, 10.0, get_from_params('loc', 0.0, float), 0.1, key="loc"), 'scale': st.slider("Std Dev (σ)", 0.1, 10.0, get_from_params('scale', 1.0, float), 0.1, key="scale")}
        elif dist_name == "Uniform": dist_params = {'loc': st.slider("Start", -10.0, 10.0, get_from_params('loc', 0.0, float), 0.1, key="loc"), 'scale': st.slider("Width", 0.1, 20.0, get_from_params('scale', 5.0, float), 0.1, key="scale")}
        elif dist_name == "Exponential": dist_params = {'scale': st.slider("Scale (λ⁻¹)", 0.1, 10.0, get_from_params('scale', 1.0, float), 0.1, key="scale")}
        elif dist_name == "Pareto": dist_params = {'b': st.slider("Shape (b)", 0.1, 5.0, get_from_params('b', 2.0, float), 0.1, key="b")}

    # --- UX Controls ---
    st.divider()
    quick_preview = st.toggle("🚀 Quick Preview", value=True, help="Cap simulations at M=200 for faster interaction.")
    if quick_preview: M = min(M, 200) if ab_mode else min(st.session_state.M_single, 200)

    # --- Glossary and Export ---
    with st.expander("📚 Glossary & Sharing"):
        st.markdown("Key terms and sharing options.")
        if st.button("Copy Link to Clipboard", use_container_width=True):
            st.query_params.from_dict(st.session_state)
            st.success("Link updated in browser URL. Copy it to share.")

# --- Core Simulation & Helper Functions ---
def get_dist_from_params(dist_name, dist_params):
    """Helper to reconstruct a scipy.stats distribution from name and params."""
    if dist_name == "Normal":
        return stats.norm(**dist_params)
    elif dist_name == "Uniform":
        return stats.uniform(**dist_params)
    elif dist_name == "Exponential":
        return stats.expon(**dist_params)
    elif dist_name == "Pareto":
        return stats.pareto(**dist_params)
    # This path should ideally not be reached if inputs are validated
    raise ValueError(f"Unknown distribution: {dist_name}")

@st.cache_data(ttl=600)
def run_simulation(dist_name, dist_params, n, M, seed):
    """Runs the CLT simulation using hashable parameters."""
    dist = get_dist_from_params(dist_name, dist_params)
    np.random.seed(seed)
    samples = dist.rvs(size=(M, n))
    return pd.DataFrame({'Sample Mean': np.mean(samples, axis=1)})

def get_stats(data):
    mean, var, skew = np.mean(data), np.var(data), stats.skew(data)
    shapiro_p = stats.shapiro(data)[1]
    return {"Mean": mean, "Variance": var, "Skewness": skew, "p-value": shapiro_p}

def plot_hist(ax, data, dist_name, dist_params, n, title):
    """Plots the histogram and theoretical PDF."""
    dist = get_dist_from_params(dist_name, dist_params)
    ax.hist(data, bins=50, density=True, alpha=0.7, color="#0072B2", label="Sample Means")
    clt_mu, clt_sigma = dist.mean(), dist.std() / np.sqrt(n)
    x = np.linspace(data.min(), data.max(), 200)
    ax.plot(x, stats.norm.pdf(x, clt_mu, clt_sigma), 'r--', lw=2.5, label="Theoretical PDF")
    ax.set_title(title, fontsize=14); ax.set_xlabel("Sample Mean"); ax.legend()

def plot_qq(ax, data, title):
    stats.probplot(data, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor('#0072B2'); ax.get_lines()[1].set_color('r')
    ax.set_title(title, fontsize=14); ax.set_xlabel("Theoretical Quantiles")

# --- Main App Logic ---
if ab_mode:
    # A/B Mode
    st.header("A/B Comparison")
    with st.spinner(f"Running Simulations A ({dist_name_a}, n={n_a}) and B ({dist_name_b}, n={n_b})..."):
        means_a = run_simulation(dist_name_a, dist_params_a, n_a, M, seed)['Sample Mean']
        means_b = run_simulation(dist_name_b, dist_params_b, n_b, M, seed)['Sample Mean']
    stats_a, stats_b = get_stats(means_a), get_stats(means_b)

    # KPIs
    st.markdown("##### Comparative Metrics (B vs. A)")
    kpi_cols = st.columns(3)
    kpi_cols[0].metric("Δ Mean", f"{stats_b['Mean']:.3f}", f"{stats_b['Mean'] - stats_a['Mean']:.3f}")
    kpi_cols[1].metric("Δ Variance", f"{stats_b['Variance']:.3f}", f"{stats_b['Variance'] - stats_a['Variance']:.3f}", delta_color="inverse")
    kpi_cols[2].metric("Δ Skewness", f"{stats_b['Skewness']:.3f}", f"{stats_b['Skewness'] - stats_a['Skewness']:.3f}")

    # Charts
    hist_cols = st.columns(2)
    fig_a, ax_a = plt.subplots(); plot_hist(ax_a, means_a, dist_name_a, dist_params_a, n_a, f"Scenario A: {dist_name_a} (n={n_a})"); hist_cols[0].pyplot(fig_a)
    fig_b, ax_b = plt.subplots(); plot_hist(ax_b, means_b, dist_name_b, dist_params_b, n_b, f"Scenario B: {dist_name_b} (n={n_b})"); hist_cols[1].pyplot(fig_b)

    if st.checkbox("Show side-by-side Q-Q Plots", key='show_qq_ab'):
        qq_cols = st.columns(2)
        fig_qa, ax_qa = plt.subplots(); plot_qq(ax_qa, means_a, "Q-Q Plot (A)"); qq_cols[0].pyplot(fig_qa)
        fig_qb, ax_qb = plt.subplots(); plot_qq(ax_qb, means_b, "Q-Q Plot (B)"); qq_cols[1].pyplot(fig_qb)
else:
    # Single Mode
    with st.spinner(f"Simulating {M} samples of size {n} from a {dist_name} distribution..."):
        sample_means = run_simulation(dist_name, dist_params, n, M, seed)['Sample Mean']

    # Preset Narratives
    if n < 30: st.info("💡 **Small n:** With a small sample size (n < 30), expect noisier sample means and slower convergence to normality.")
    if dist_name == "Pareto": st.info("💡 **Heavy-Tailed Distribution (Pareto):** This distribution has outliers that can cause the sample mean to stabilize much slower.")

    # --- View Routing ---
    if persona == "Executive":
        st.header("📈 Executive Dashboard")
        stats_data = get_stats(sample_means)
        p_value = stats_data["p-value"]
        st.markdown("##### Key Performance Indicators")
        kpi_cols = st.columns(4)
        kpi_cols[0].metric("Sample Mean", f"{stats_data['Mean']:.3f}", help="The average of all sample means.")
        kpi_cols[1].metric("Variance", f"{stats_data['Variance']:.3f}", help="Spread of sample means. Decreases as 'n' increases.")
        kpi_cols[2].metric("Skewness", f"{stats_data['Skewness']:.3f}", help="Asymmetry. Near 0 is symmetric.")
        kpi_cols[3].metric("Normality (p-value)", f"{p_value:.3f}", help="If p ≥ 0.05, we consider it normal.", delta_color="normal" if p_value >= 0.05 else "inverse")
        
        st.markdown("> **Takeaway:** The distribution of sample means becomes taller, narrower, and more bell-shaped as sample size `n` increases, visually confirming the Central Limit Theorem.")
        fig, ax = plt.subplots(figsize=(10, 5)); plot_hist(ax, sample_means, dist_name, dist_params, n, f"Distribution of Sample Means (n={n}, M={M})"); st.pyplot(fig)

    elif persona == "Data Scientist":
        st.header("🔬 Data Scientist Deep Dive")
        tab1, tab2 = st.tabs(["CLT Diagnostics", "Base Distribution Inspector"])
        with tab1:
            st.markdown("> **Takeaway:** The Q-Q plot's linearity and the histogram's bell shape reinforce the normality conclusion from the Shapiro-Wilk test.")
            col1, col2 = st.columns(2)
            with col1: fig_hist, ax_hist = plt.subplots(); plot_hist(ax_hist, sample_means, dist_name, dist_params, n, "Distribution of Sample Means"); st.pyplot(fig_hist)
            with col2: fig_qq, ax_qq = plt.subplots(); plot_qq(ax_qq, sample_means, "Q-Q Plot vs. Normal"); st.pyplot(fig_qq)
        with tab2:
            st.markdown("> **Takeaway:** This chart shows the underlying distribution we are sampling from. The CLT works even if this distribution is not normal itself.")
            dist = get_dist_from_params(dist_name, dist_params)
            single_sample = dist.rvs(size=n, random_state=seed)
            fig, ax = plt.subplots()
            ax.hist(single_sample, bins=30, density=True, alpha=0.7, color="#009E73", label=f"One Sample (n={n})")
            if dist_name != "Pareto":
                x = np.linspace(dist.ppf(0.001), dist.ppf(0.999), 200)
                ax.plot(x, dist.pdf(x), 'k--', lw=2, label="Theoretical PDF")
            ax.set_title(f"'{dist_name}' Base Distribution vs. One Sample"); ax.legend(); st.pyplot(fig)

# --- Downloads in Sidebar ---
with st.sidebar:
    st.divider()
    st.markdown("### 📥 Export")
    if ab_mode:
        col1, col2 = st.columns(2)
        col1.download_button("CSV (A)", means_a.to_csv(), f"clt_A.csv", "text/csv")
        col2.download_button("CSV (B)", means_b.to_csv(), f"clt_B.csv", "text/csv")
    else:
        st.download_button("Download Sample Means (CSV)", sample_means.to_csv(), f"clt_data.csv", "text/csv")
        summary_md = f"# CLT Summary\n- Dist: {dist_name}\n- n: {n}\n- M: {M}\n" + "\n".join([f"- {k}: {v:.3f}" for k,v in get_stats(sample_means).items()])
        st.download_button("Download Summary (MD)", summary_md, "clt_summary.md", "text/markdown")
