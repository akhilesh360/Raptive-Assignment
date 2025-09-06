import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
import json
import textwrap
from urllib.parse import urlencode

# --- Page Configuration ---
st.set_page_config(page_title="CLT Explorer", page_icon="📊", layout="wide")

# --- Core Simulation Logic ---
@st.cache_data
def run_simulation(dist_name, dist_params, n, M, seed):
    np.random.seed(seed)
    if dist_name == "Normal": base_dist = stats.norm(loc=dist_params.get('μ', 0), scale=dist_params.get('σ', 1))
    elif dist_name == "Exponential": base_dist = stats.expon(scale=1/dist_params.get('λ', 1))
    elif dist_name == "Poisson": base_dist = stats.poisson(mu=dist_params.get('λ', 3))
    elif dist_name == "Bernoulli": base_dist = stats.bernoulli(p=dist_params.get('p', 0.5))
    elif dist_name == "Pareto": base_dist = stats.pareto(b=dist_params.get('α', 2.5))
    else: raise ValueError(f"Unknown distribution: {dist_name}")

    sample_means = np.array([base_dist.rvs(size=n).mean() for _ in range(M)])
    base_sample = base_dist.rvs(size=1000)
    theoretical_mean = base_dist.mean()
    theoretical_var = base_dist.var() / n if n > 0 else 0
    stats_dict = {
        "theoretical_mean": theoretical_mean, "theoretical_var": theoretical_var,
        "theoretical_std": np.sqrt(theoretical_var), "empirical_mean": sample_means.mean(),
        "empirical_var": sample_means.var(ddof=1), "empirical_skew": stats.skew(sample_means),
        "empirical_kurt": stats.kurtosis(sample_means)
    }
    return sample_means, base_dist, base_sample, stats_dict

# --- Plotting Functions ---
def create_hero_histogram(sample_means, theoretical_mean, theoretical_std, title):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=sample_means, name='Sample Means', histnorm='probability density', marker_color='#1f77b4', opacity=0.7))
    x_norm = np.linspace(min(sample_means), max(sample_means), 500)
    y_norm = stats.norm.pdf(x_norm, loc=theoretical_mean, scale=theoretical_std)
    fig.add_trace(go.Scatter(x=x_norm, y=y_norm, mode='lines', name='Theoretical Normal', line=dict(color='red', width=2)))
    fig.update_layout(title_text=title, xaxis_title="Sample Mean", yaxis_title="Density", template="plotly_white", showlegend=False)
    return fig

def create_qq_plot(sample_means):
    (osm, osr), (slope, intercept, r) = stats.probplot(sample_means, dist="norm", fit=True)
    fig = go.Figure(data=[go.Scatter(x=osm, y=osr, mode='markers', name='Data'), go.Scatter(x=osm, y=slope*osm + intercept, mode='lines', name='Fit', line=dict(color='red'))])
    fig.update_layout(title=f'Q-Q Plot (R²={r**2:.4f})', xaxis_title='Theoretical Quantiles', yaxis_title='Sample Quantiles', template="plotly_white")
    return fig

def create_base_dist_plot(base_dist, base_sample, dist_name):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=base_sample, name='Base Sample', histnorm='probability density', marker_color='#2ca02c'))
    if dist_name in ["Normal", "Exponential", "Pareto"]:
        x_vals = np.linspace(base_sample.min(), base_sample.max(), 200)
        y_vals = base_dist.pdf(x_vals)
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='Theoretical PDF', line=dict(color='purple')))
    else:
        x_vals, counts = np.unique(base_sample, return_counts=True)
        y_vals_pmf = base_dist.pmf(x_vals)
        fig.add_trace(go.Bar(x=x_vals, y=counts/len(base_sample), name='Empirical PMF'))
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals_pmf, mode='markers', name='Theoretical PMF', marker=dict(color='purple', size=8)))
    fig.update_layout(title=f'Histogram of Base Distribution ({dist_name})', xaxis_title='Value', yaxis_title='Density / Probability', template="plotly_white")
    return fig

def create_convergence_plot(base_dist, n, M):
    means = []
    m_steps = np.linspace(100, M, 20, dtype=int)
    for m_step in m_steps:
        sample_means = np.array([base_dist.rvs(size=n).mean() for _ in range(m_step)])
        means.append(sample_means.mean())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m_steps, y=means, mode='lines+markers', name='Empirical Mean'))
    fig.add_hline(y=base_dist.mean(), line_dash="dash", line_color="red", name="Theoretical Mean")
    fig.update_layout(title='Convergence of Sample Mean as M Grows', xaxis_title='Number of Repetitions (M)', yaxis_title='Mean of Sample Means', template="plotly_white")
    return fig

# --- UI Component Functions ---
def get_scenario_params(suffix, defaults):
    st.markdown(f"#### Scenario {suffix}")
    dist_options = ("Normal", "Exponential", "Poisson", "Bernoulli", "Pareto")
    dist_key = f'dist_{suffix}'
    dist_name = st.selectbox("Distribution", dist_options, key=dist_key, index=dist_options.index(defaults.get(dist_key, "Normal")))
    params = {}
    if dist_name == "Normal":
        params['μ'] = st.slider("Mean (μ)", -10.0, 10.0, float(defaults.get(f'μ_{suffix}', 0.0)), 0.1, key=f'μ_{suffix}')
        params['σ'] = st.slider("Std Dev (σ)", 0.1, 10.0, float(defaults.get(f'σ_{suffix}', 1.0)), 0.1, key=f'σ_{suffix}')
    elif dist_name == "Exponential":
        params['λ'] = st.slider("Rate (λ)", 0.1, 10.0, float(defaults.get(f'λ_{suffix}', 1.0)), 0.1, key=f'λ_{suffix}')
    elif dist_name == "Poisson":
         params['λ'] = st.slider("Rate (λ)", 1.0, 20.0, float(defaults.get(f'λ_{suffix}', 3.0)), 0.5, key=f'λ_{suffix}')
    elif dist_name == "Bernoulli":
        params['p'] = st.slider("Probability (p)", 0.0, 1.0, float(defaults.get(f'p_{suffix}', 0.5)), 0.01, key=f'p_{suffix}')
    elif dist_name == "Pareto":
        params['α'] = st.slider("Shape (α)", 0.1, 5.0, float(defaults.get(f'α_{suffix}', 2.5)), 0.1, key=f'α_{suffix}')
        if params['α'] <= 2: st.warning("For Pareto with α ≤ 2, variance is infinite and the CLT does not apply in its standard form.")
    
    n = st.slider("Sample Size (n)", 1, 500, int(defaults.get(f'n_{suffix}', 30)), 1, key=f'n_{suffix}')
    M = st.slider("Repetitions (M)", 100, 10000, int(defaults.get(f'M_{suffix}', 1000)), 100, key=f'M_{suffix}')
    return dist_name, params, n, M

def render_executive_view(sample_means, stats_dict, n, title="Executive Summary"):
    st.header(title)
    fig = create_hero_histogram(sample_means, stats_dict['theoretical_mean'], stats_dict['theoretical_std'], "Distribution of Sample Means")
    st.plotly_chart(fig, use_container_width=True)
    return fig

def render_data_scientist_view(sample_means, base_dist, base_sample, stats_dict, dist_name, n, M):
    st.header("Detailed Analysis")
    tab1, tab2, tab3 = st.tabs(["Convergence", "Diagnostics", "Base Distribution"])
    with tab1:
        st.subheader("Convergence of Sample Means to Normality")
        st.plotly_chart(create_convergence_plot(base_dist, n, M), use_container_width=True)
    with tab2:
        st.subheader("Goodness-of-Fit and Diagnostics")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("###### Normality Tests")
            ks_stat, ks_p = stats.kstest(sample_means, 'norm', args=(stats_dict['empirical_mean'], np.sqrt(stats_dict['empirical_var'])))
            sw_stat, sw_p = stats.shapiro(sample_means)
            st.metric("Kolmogorov-Smirnov p-value", f"{ks_p:.4f}")
            st.metric("Shapiro-Wilk p-value", f"{sw_p:.4f}")
        with c2: st.plotly_chart(create_qq_plot(sample_means), use_container_width=True)
    with tab3:
        st.subheader("Underlying Base Distribution")
        st.plotly_chart(create_base_dist_plot(base_dist, base_sample, dist_name), use_container_width=True)
        m, v, s, k = base_dist.stats(moments='mvsk')
        st.table(pd.DataFrame({"Moment": ["Mean", "Variance", "Skewness", "Kurtosis"], "Value": [f"{m:.3f}", f"{v:.3f}", f"{s:.3f}", f"{k:.3f}"]}))

def render_reproducibility_section(params):
    with st.expander("Export & Reproduce"):
        st.code(f"http://localhost:8501?{urlencode(params)}", language=None)
        code_snippet = f"run_simulation(dist_name='{params.get('dist_A')}', dist_params={params.get('params_A')}, n={params.get('n_A')}, M={params.get('M_A')}, seed={params.get('seed')})"
        st.code(textwrap.dedent(code_snippet), language="python")
        settings_json = json.dumps(params, indent=2, default=str)
        st.download_button("Download Settings (JSON)", settings_json, "clt_settings.json")

def render_glossary():
    with st.sidebar.expander("Glossary", expanded=False):
        st.markdown("""
        - **CLT (Central Limit Theorem):** States that the distribution of sample means approximates a normal distribution as the sample size grows.
        - **LLN (Law of Large Numbers):** States that the average of results from many trials should be close to the expected value.
        - **Skewness:** Measure of a distribution's asymmetry.
        - **Kurtosis:** Measure of a distribution's "tailedness".
        - **Heavy-Tailed Distribution:** A distribution with tails heavier than an exponential one, meaning more outliers.
        """)

# --- Main Application ---
def main():
    st.title("📊 Central Limit Theorem Dashboard")
    with st.sidebar:
        st.header("Controls")
        defaults = {k: v[0] for k, v in st.query_params.items()}
        ab_mode = st.toggle("A/B Comparison Mode", key='ab_mode', value=bool(defaults.get('ab_mode', False)))
        st.subheader("Presets")
        if st.button("Small n (n=5)", use_container_width=True, disabled=ab_mode): st.query_params.n_A = 5; st.rerun()
        if st.button("Large n (n=200)", use_container_width=True, disabled=ab_mode): st.query_params.n_A = 200; st.rerun()
        if st.button("Heavy Tail (Pareto, α=1.5)", use_container_width=True, disabled=ab_mode):
            st.query_params.dist_A = "Pareto"; st.query_params.α_A = 1.5; st.rerun()
        
        params = {}
        if ab_mode:
            c1, c2 = st.columns(2)
            with c1: params['A'] = get_scenario_params("A", defaults)
            with c2: params['B'] = get_scenario_params("B", defaults)
        else:
            params['A'] = get_scenario_params("A", defaults)
        
        view_mode = st.radio("View Mode", ("Executive View", "Data Scientist View"), key='view_mode', horizontal=True, disabled=ab_mode)
        seed = st.number_input("Random Seed", 0, 1000, int(defaults.get('seed', 42)), 1, key='seed')
        render_glossary()

    try:
        if ab_mode:
            dist_A, params_A, n_A, M_A = params['A']
            dist_B, params_B, n_B, M_B = params['B']
            means_A, _, _, stats_A = run_simulation(dist_A, params_A, n_A, M_A, seed)
            means_B, _, _, stats_B = run_simulation(dist_B, params_B, n_B, M_B, seed)
            st.header("A/B Comparison Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean Δ (A-B)", f"{stats_A['empirical_mean'] - stats_B['empirical_mean']:+.3f}")
            c2.metric("Variance Δ (A-B)", f"{stats_A['empirical_var'] - stats_B['empirical_var']:+.3f}")
            c3.metric("Skewness Δ (A-B)", f"{stats_A['empirical_skew'] - stats_B['empirical_skew']:+.3f}")
            c1, c2 = st.columns(2)
            with c1: fig_A = render_executive_view(means_A, stats_A, n_A, f"Scenario A: {dist_A}")
            with c2: fig_B = render_executive_view(means_B, stats_B, n_B, f"Scenario B: {dist_B}")
        else:
            dist, dist_params, n, M = params['A']
            means, base_dist, base_sample, stats_dict = run_simulation(dist, dist_params, n, M, seed)
            if view_mode == "Executive View":
                render_executive_view(means, stats_dict, n)
                all_params = {'dist_A': dist, 'params_A': dist_params, 'n_A': n, 'M_A': M, 'seed': seed}
                render_reproducibility_section(all_params)
            else:
                render_data_scientist_view(means, base_dist, base_sample, stats_dict, dist, n, M)
                all_params = {'dist_A': dist, 'params_A': dist_params, 'n_A': n, 'M_A': M, 'seed': seed}
                render_reproducibility_section(all_params)
    except (ValueError, TypeError, KeyError) as e:
        st.error(f"An error occurred: {e}. Please check your parameters.")

if __name__ == "__main__":
    main()
