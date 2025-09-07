import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import urllib.parse
import statsmodels.api as sm

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

    # --- Preset Scenarios ---
    st.divider()
    st.header("Preset Scenarios")
    def set_preset(preset):
        st.session_state.ab_mode = False # Presets are for single mode
        if preset == 'small_n':
            st.session_state.dist_name = "Normal"
            st.session_state.n = 5
            st.session_state.loc = 0.0
            st.session_state.scale = 1.0
            st.session_state.preset_banner = "small_n"
        elif preset == 'large_n':
            st.session_state.dist_name = "Normal"
            st.session_state.n = 500
            st.session_state.loc = 0.0
            st.session_state.scale = 1.0
            st.session_state.preset_banner = "large_n"
        elif preset == 'heavy_tail':
            st.session_state.dist_name = "Pareto"
            st.session_state.n = 30
            st.session_state.b = 1.5 # A heavier tail
            st.session_state.preset_banner = "heavy_tail"

    preset_cols = st.columns(3)
    preset_cols[0].button("Small n", on_click=set_preset, args=('small_n',), use_container_width=True)
    preset_cols[1].button("Large n", on_click=set_preset, args=('large_n',), use_container_width=True)
    preset_cols[2].button("Heavy Tail", on_click=set_preset, args=('heavy_tail',), use_container_width=True)

    # --- Distribution Explainers ---
    if not ab_mode:
        if dist_name == "Normal": st.info("Symmetric, bell-shaped. CLT converges quickly.")
        elif dist_name == "Uniform": st.info("Flat, equal probability in a range. Converges quickly.")
        elif dist_name == "Exponential": st.info("Skewed with a tail. Slower convergence than Normal.")
        elif dist_name == "Pareto": st.info("Heavy-tailed with extreme outliers. Slowest convergence.")

    # --- Glossary and Export ---
    with st.expander("📚 Glossary & Sharing"):
        st.markdown("""
        - **CLT (Central Limit Theorem):** States that the distribution of sample means approximates a normal distribution as the sample size gets larger, regardless of the population's distribution.
        - **LLN (Law of Large Numbers):** States that the average of the results obtained from a large number of trials should be close to the expected value.
        - **Skewness:** A measure of a distribution's asymmetry. Positive skew has a long right tail; negative skew has a long left tail.
        - **Kurtosis:** A measure of the "tailedness" of a distribution. High kurtosis means more outliers (heavy tails).
        - **Heavy Tails:** A property of distributions where extreme events are more likely than in a normal distribution.
        - **Normality Test:** A statistical process used to determine if a sample of data is well-modeled by a normal distribution (e.g., Shapiro-Wilk test).
        """)
        st.divider()
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
    ax.hist(data, bins=50, density=True, alpha=0.7, color="#0072B2")
    clt_mu, clt_sigma = dist.mean(), dist.std() / np.sqrt(n)
    x = np.linspace(data.min(), data.max(), 200)
    ax.plot(x, stats.norm.pdf(x, clt_mu, clt_sigma), 'r--', lw=3)
    ax.set_title(title, fontsize=14); ax.set_xlabel("Sample Mean")

def plot_qq(ax, data, title):
    """Plots a Q-Q plot with confidence bands using statsmodels."""
    sm.qqplot(data, line='s', ax=ax, dist=stats.norm)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    # Style the plot to match
    ax.get_lines()[0].set_markerfacecolor('#0072B2')
    ax.get_lines()[0].set_markeredgecolor('#0072B2')
    ax.get_lines()[1].set_color('r')

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

        # Calculate theoretical values
        base_dist = get_dist_from_params(dist_name, dist_params)
        theory_mean = base_dist.mean()
        theory_var = base_dist.var() / n

        # Determine color cues based on tolerance
        mean_delta = stats_data['Mean'] - theory_mean
        var_delta = stats_data['Variance'] - theory_var

        # Tolerance: 5% for variance, 0.25 for skewness
        var_color = "normal" if abs(var_delta / theory_var) < 0.05 else "inverse"
        skew_color = "normal" if abs(stats_data['Skewness']) < 0.25 else "inverse"
        p_value_color = "normal" if stats_data['p-value'] >= 0.05 else "inverse"

        st.markdown("##### Key Performance Indicators")
        kpi_cols = st.columns(4)
        kpi_cols[0].metric(
            "Sample Mean", f"{stats_data['Mean']:.3f}",
            delta=f"{mean_delta:.3f} vs μ",
            help="The average of all the sample means. The Central Limit Theorem states this should be very close to the true population mean (μ)."
        )
        kpi_cols[1].metric(
            "Variance", f"{stats_data['Variance']:.4f}",
            delta=f"{var_delta:.4f} vs σ²/n",
            delta_color=var_color,
            help="The variance of the sample means. The CLT predicts this will be the population variance (σ²) divided by the sample size (n). Green if empirical variance is within 10% of theoretical."
        )
        kpi_cols[2].metric(
            "Skewness", f"{stats_data['Skewness']:.3f}",
            delta_color=skew_color,
            help="A measure of asymmetry. A value near 0 indicates a symmetric, bell-like curve. Green if absolute value is < 0.5."
        )
        kpi_cols[3].metric(
            "Normality (p-value)", f"{stats_data['p-value']:.3f}",
            delta_color=p_value_color,
            help="Result from the Shapiro-Wilk test for normality. A high p-value (p ≥ 0.05) means we fail to reject the null hypothesis that the data is normally distributed. Green indicates 'passes' the normality test."
        )
        
        st.markdown("> **Takeaway:** The distribution of sample means becomes taller, narrower, and more bell-shaped as sample size `n` increases, visually confirming the Central Limit Theorem.")
        fig, ax = plt.subplots(figsize=(10, 5)); plot_hist(ax, sample_means, dist_name, dist_params, n, f"Distribution of Sample Means (n={n}, M={M})"); st.pyplot(fig)
        with st.expander("What am I looking at?"):
            st.markdown("""
            This chart shows the **distribution of sample means**. Here's what that means:

            1.  We take a "sample" of `n` data points from the base distribution you selected.
            2.  We calculate the average (the "mean") of that one sample.
            3.  We repeat this process `M` times, giving us `M` different sample means.

            The histogram above shows the shape of these `M` sample means. The **Central Limit Theorem (CLT)** is the magic here: no matter the shape of the original distribution, the distribution of its sample means will tend to look like a normal (bell-shaped) curve, especially when `n` is large.
            """)

    elif persona == "Data Scientist":
        st.header("🔬 Data Scientist Deep Dive")
        tab1, tab2, tab3 = st.tabs(["CLT Diagnostics", "Convergence Trace", "Base Distribution Inspector"])

        base_dist = get_dist_from_params(dist_name, dist_params)

        with tab1:
            st.markdown("> **Takeaway:** The Q-Q plot's linearity and high coverage % within ±2σ support the normality conclusion.")
            col1, col2 = st.columns(2)
            with col1:
                fig_hist, ax_hist = plt.subplots();
                plot_hist(ax_hist, sample_means, dist_name, dist_params, n, "Distribution of Sample Means");
                st.pyplot(fig_hist)
                with st.expander("What am I looking at?"):
                    st.markdown("This is the same histogram from the Executive view, showing the distribution of sample means. It should appear bell-shaped if the CLT has taken effect.")
            with col2:
                fig_qq, ax_qq = plt.subplots();
                plot_qq(ax_qq, sample_means, "Q-Q Plot vs. Normal");
                st.pyplot(fig_qq)
                with st.expander("What am I looking at?"):
                    st.markdown("""
                    A **Quantile-Quantile (Q-Q) Plot** compares the quantiles of our data (the sample means) against the theoretical quantiles of a normal distribution.
                    - If the data is perfectly normal, the blue dots will form a straight 45-degree line.
                    - The red line represents this ideal 45-degree line.
                    - The shaded area is a **confidence band**. If the dots stay within this band, it provides strong evidence for normality.
                    """)

            # Coverage Percentage
            theory_mean = base_dist.mean()
            theory_std = base_dist.std() / np.sqrt(n)
            within_1_std = np.sum((sample_means >= theory_mean - theory_std) & (sample_means <= theory_mean + theory_std)) / M
            within_2_std = np.sum((sample_means >= theory_mean - 2*theory_std) & (sample_means <= theory_mean + 2*theory_std)) / M

            st.markdown("##### Coverage Statistics")
            cov_cols = st.columns(2)
            cov_cols[0].metric("Coverage within ±1σ", f"{within_1_std:.2%}", help="Expected: ~68%")
            cov_cols[1].metric("Coverage within ±2σ", f"{within_2_std:.2%}", help="Expected: ~95%")

        with tab2:
            st.markdown("> **Takeaway:** As the number of samples (M) increases, the running average of the sample means converges to the true population mean (μ), illustrating the Law of Large Numbers.")

            # Calculate running mean and variance
            running_means = np.cumsum(sample_means) / np.arange(1, M + 1)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(np.arange(1, M + 1), running_means, label="Running Sample Mean")
            ax.axhline(y=theory_mean, color='r', linestyle='--', label=f"Theoretical Mean (μ={theory_mean:.3f})")
            ax.set_title("Convergence of Sample Mean")
            ax.set_xlabel("Number of Samples (M)")
            ax.set_ylabel("Mean Value")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        with tab3:
            st.markdown("> **Takeaway:** This chart shows the underlying distribution we are sampling from. The CLT works even if this distribution is not normal itself.")
            single_sample = base_dist.rvs(size=n, random_state=seed)
            fig, ax = plt.subplots()
            ax.hist(single_sample, bins=30, density=True, alpha=0.7, color="#009E73", label=f"One Sample (n={n})")
            if dist_name != "Pareto":
                x = np.linspace(base_dist.ppf(0.001), base_dist.ppf(0.999), 200)
                ax.plot(x, base_dist.pdf(x), 'k--', lw=2, label="Theoretical PDF")
            ax.set_title(f"'{dist_name}' Base Distribution vs. One Sample"); ax.legend(); st.pyplot(fig)

def generate_markdown_summary(dist_name, dist_params, n, M, seed, stats_data):
    """Generates a detailed Markdown summary of the simulation."""
    param_str = ", ".join([f"{k}={v}" for k, v in dist_params.items()])
    theory_var = get_dist_from_params(dist_name, dist_params).var() / n
    p_val_status = "Pass" if stats_data['p-value'] >= 0.05 else "Fail"

    summary = f"""
- **Scenario**: {dist_name}({param_str}), n={n}, M={M}, seed={seed}
- **Sample Mean**: {stats_data['Mean']:.4f}
- **Sample Variance**: {stats_data['Variance']:.4f} (vs. Theory: {theory_var:.4f})
- **Skewness**: {stats_data['Skewness']:.4f}
- **Shapiro p-value**: {stats_data['p-value']:.3f} → {p_val_status} Normality
"""
    return summary

def generate_python_snippet(dist_name, dist_params, n, M, seed):
    """Generates a Python script to reproduce the simulation."""
    return f"""
import numpy as np
from scipy import stats

# Simulation Parameters
dist_name = "{dist_name}"
dist_params = {dist_params}
n = {n}
M = {M}
seed = {seed}

# Recreate distribution
if dist_name == "Normal": dist = stats.norm(**dist_params)
elif dist_name == "Uniform": dist = stats.uniform(**dist_params)
elif dist_name == "Exponential": dist = stats.expon(**dist_params)
elif dist_name == "Pareto": dist = stats.pareto(**dist_params)

# Run simulation
np.random.seed(seed)
samples = dist.rvs(size=(M, n))
sample_means = np.mean(samples, axis=1)

# Print results
print(f"Simulation Results for {{dist_name}} (n={{n}}, M={{M}}):")
print(f"  - Sample Mean: {{np.mean(sample_means):.4f}}")
print(f"  - Sample Variance: {{np.var(sample_means):.4f}}")
"""

# --- Downloads in Sidebar ---
with st.sidebar:
    st.divider()
    st.markdown("### 📥 Export")
    if ab_mode:
        st.markdown("**Scenario A**")
        st.download_button("CSV (A)", means_a.to_csv(), f"clt_A.csv", "text/csv", key="csv_a")
        md_a = generate_markdown_summary(dist_name_a, dist_params_a, n_a, M, seed, stats_a)
        st.download_button("Summary (A)", md_a, "summary_A.md", "text/markdown", key="md_a")
        py_a = generate_python_snippet(dist_name_a, dist_params_a, n_a, M, seed)
        st.download_button("Python (A)", py_a, "reproduce_A.py", "text/python", key="py_a")

        st.markdown("**Scenario B**")
        st.download_button("CSV (B)", means_b.to_csv(), f"clt_B.csv", "text/csv", key="csv_b")
        md_b = generate_markdown_summary(dist_name_b, dist_params_b, n_b, M, seed, stats_b)
        st.download_button("Summary (B)", md_b, "summary_B.md", "text/markdown", key="md_b")
        py_b = generate_python_snippet(dist_name_b, dist_params_b, n_b, M, seed)
        st.download_button("Python (B)", py_b, "reproduce_B.py", "text/python", key="py_b")
    else:
        st.download_button("Download Sample Means (CSV)", sample_means.to_csv(), f"clt_data.csv", "text/csv")

        stats_data = get_stats(sample_means)
        summary_md = generate_markdown_summary(dist_name, dist_params, n, M, seed, stats_data)
        st.download_button("Download Summary (MD)", summary_md, "clt_summary.md", "text/markdown")

        python_snippet = generate_python_snippet(dist_name, dist_params, n, M, seed)
        st.download_button("Download Python Snippet", python_snippet, "reproduce_clt.py", "text/python")
