import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# --- Core Functions ---
@st.cache_data(ttl=600)
def generate_sample_means(dist, n, k, seed):
    """Generate k sample means of size n from a given distribution."""
    np.random.seed(seed)
    samples = dist.rvs(size=(k, n))
    return np.mean(samples, axis=1)

def get_metrics(sample_means, mu, sigma, n):
    """Compute metrics for a sample means distribution."""
    mean_val = np.mean(sample_means)
    var_val = np.var(sample_means, ddof=1)
    skew_val = stats.skew(sample_means)
    shapiro_test = stats.shapiro(sample_means)
    theory_var = (sigma ** 2) / n
    return {
        "mean": mean_val,
        "var": var_val,
        "skew": skew_val,
        "shapiro_p": shapiro_test.pvalue,
        "theory_var": theory_var,
        "delta_mean": mean_val - mu,
        "delta_var": var_val - theory_var,
    }

def plot_histogram(sample_means, mu, sigma, n, label="Scenario"):
    """Plot histogram of sample means with theoretical overlay."""
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(sample_means, bins=40, density=True, alpha=0.7, color="skyblue", label="Empirical")
    x_vals = np.linspace(min(sample_means), max(sample_means), 400)
    y_vals = stats.norm.pdf(x_vals, mu, sigma/np.sqrt(n))
    ax.plot(x_vals, y_vals, "r--", linewidth=2, label="Theoretical Normal")
    ax.set_title(f"Distribution of Sample Means ({label})")
    ax.set_xlabel("Sample Mean")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def plot_qq(sample_means, label="Scenario"):
    """Plot Q-Q plot against Normal distribution."""
    fig, ax = plt.subplots(figsize=(4,4))
    stats.probplot(sample_means, dist="norm", plot=ax)
    ax.set_title(f"Q-Q Plot ({label})")
    return fig

# --- App Main ---
def main():
    st.set_page_config(page_title="Central Limit Theorem Lab", page_icon="📊", layout="wide")
    st.title("📊 Central Limit Theorem Lab")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Controls")

        # Distribution parameters
        mu = st.slider("Mean (μ)", -5.0, 5.0, 0.0)
        sigma = st.slider("Std Dev (σ)", 0.1, 5.0, 1.0)

        # CLT controls
        quick_preview = st.toggle("Quick Preview", value=True)
        n = st.slider("Sample Size (n)", 1, 1000, 30)
        k_slider = st.slider("Number of Samples (k)", 100, 10000, 1000)
        k = 200 if quick_preview else k_slider

        # Presets
        st.subheader("Presets")
        if st.button("Small n (n=5)"):
            n, sigma = 5, 1.0
            st.session_state.preset_message = "Expect noisier sample means; slower normality."
        if st.button("Large n (n=200)"):
            n, sigma = 200, 1.0
            st.session_state.preset_message = "Sample means converge quickly to Normal."
        if st.button("Heavy Tail (Pareto)"):
            sigma = 2.0
            st.session_state.preset_message = "Heavy-tailed distribution → outliers, slower convergence."

        # Seed
        seed = st.number_input("Random Seed", 0, 10000, 42)

        # A/B comparison toggle
        ab_mode = st.toggle("A/B Comparison Mode")

        # Share link
        st.subheader("Share Scenario")
        current_params = {"mu": mu, "sigma": sigma, "n": n, "k": k, "seed": seed}
        st.experimental_set_query_params(**current_params)
        st.caption("Copy page URL to share this setup.")

    # Preset message banner
    if st.session_state.get("preset_message"):
        st.info(st.session_state.preset_message)

    # --- Tabs ---
    if ab_mode:
        tab1, tab2, tab3 = st.tabs(["A/B Comparison", "Glossary", "Animation"])
    else:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Distribution", "CLT Demo", "Data", "Methods", "Animation"])

    # --- A/B Comparison Mode ---
    if ab_mode:
        with tab1:
            st.header("A/B Comparison of Scenarios")

            c1, c2 = st.columns(2)
            with c1:
                distA = stats.norm(loc=mu, scale=sigma)
                sample_means_A = generate_sample_means(distA, n, k, seed)
                metricsA = get_metrics(sample_means_A, mu, sigma, n)
                st.pyplot(plot_histogram(sample_means_A, mu, sigma, n, "Scenario A"))

            with c2:
                distB = stats.poisson(mu if mu>0 else 3)  # Example: Poisson λ
                sample_means_B = generate_sample_means(distB, n, k, seed)
                metricsB = get_metrics(sample_means_B, mu, sigma, n)
                st.pyplot(plot_histogram(sample_means_B, mu, sigma, n, "Scenario B (Poisson)"))

            # Delta Metrics
            st.subheader("Δ Metrics (A - B)")
            d1, d2, d3 = st.columns(3)
            d1.metric("Δ Mean", f"{metricsA['mean']-metricsB['mean']:.3f}")
            d2.metric("Δ Variance", f"{metricsA['var']-metricsB['var']:.3f}")
            d3.metric("Δ Skewness", f"{metricsA['skew']-metricsB['skew']:.3f}")

            # Optional Q-Q plots
            if st.checkbox("Show Q-Q Plots"):
                q1, q2 = st.columns(2)
                q1.pyplot(plot_qq(sample_means_A, "Scenario A"))
                q2.pyplot(plot_qq(sample_means_B, "Scenario B"))

        with tab2:
            st.header("Glossary")
            st.markdown("""
            - **Mean (μ):** Average of values.
            - **Variance (σ²):** Spread of values.
            - **Skewness:** Asymmetry of distribution.
            - **Q-Q Plot:** Compares quantiles to check Normality.
            """)

        with tab3:
            # Animation in A/B mode
            show_animation_tab(mu, sigma, n, k, seed)

    # --- Standard Mode ---
    else:
        with tab1:
            st.header("Normal Distribution")
            st.markdown("> **Takeaway:** PDF shows density; CDF shows cumulative probability.")
            dist = stats.norm(loc=mu, scale=sigma)
            x = np.linspace(mu-4*sigma, mu+4*sigma, 500)
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
            ax1.plot(x, dist.pdf(x), "b-", linewidth=2)
            ax1.set_title("PDF"); ax1.set_ylabel("Density")
            ax2.plot(x, dist.cdf(x), "r-", linewidth=2)
            ax2.set_title("CDF"); ax2.set_ylabel("Cumulative Probability")
            st.pyplot(fig)

        with tab2:
            st.header("CLT Demonstration")
            st.markdown("> **Takeaway:** As k grows, sample means cluster near μ and approximate Normal.")

            with st.spinner("Simulating sample means…"):
                sample_means = generate_sample_means(dist, n, k, seed)
            st.pyplot(plot_histogram(sample_means, mu, sigma, n, "Executive View"))

            # KPI Strip
            st.subheader("Key Metrics")
            metrics = get_metrics(sample_means, mu, sigma, n)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sample Mean", f"{metrics['mean']:.3f}", f"{metrics['delta_mean']:.3f} vs μ")
            c2.metric("Variance", f"{metrics['var']:.3f}", f"{metrics['delta_var']:.3f} vs Theory")
            c3.metric("Skewness", f"{metrics['skew']:.3f}")
            c4.metric("Shapiro p", f"{metrics['shapiro_p']:.3f}", "Pass" if metrics['shapiro_p']>=0.05 else "Fail")

        with tab3:
            st.header("Simulated Data")
            df = pd.DataFrame(sample_means, columns=["Sample Mean"])
            st.dataframe(df.head(10))
            st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode(), "sample_means.csv", "text/csv")
            insight = f"""
            # Insight Summary
            - n = {n}, k = {k}, μ = {mu}, σ = {sigma}
            - Sample mean = {metrics['mean']:.3f}, variance = {metrics['var']:.3f}
            - Shapiro p = {metrics['shapiro_p']:.3f} → {"Normal" if metrics['shapiro_p']>=0.05 else "Not Normal"}
            """
            st.download_button("⬇️ Download Insight (Markdown)", insight.encode(), "insight.md", "text/markdown")

        with tab4:
            st.header("Methods")
            st.markdown("""
            **Central Limit Theorem (CLT):**
            - For large n, distribution of sample means ~ Normal(μ, σ/√n).
            - Variance of sample means = σ²/n.

            **Reproducibility Code:**
            ```python
            run_simulation(mu, sigma, n, k, seed)
            ```
            """)

        with tab5:
            # Animation in normal mode
            show_animation_tab(mu, sigma, n, k, seed)

# --- Helper: Animation Tab ---
def show_animation_tab(mu, sigma, n, k, seed):
    st.header("Convergence of Sample Means (Animated)")
    st.markdown("> **Takeaway:** As we add more samples, the histogram of sample means tightens and approaches Normal.")

    frames = st.slider("Frames (increments of 100 samples)", 5, 50, 20)
    anim_speed = st.slider("Animation speed (seconds per frame)", 0.1, 1.0, 0.3)

    if st.button("▶️ Start Animation"):
        with st.spinner("Generating animation frames..."):
            all_samples = generate_sample_means(stats.norm(mu, sigma), n, frames*100, seed)

        placeholder = st.empty()
        for f in range(1, frames+1):
            subset = all_samples[:f*100]
            fig = plot_histogram(subset, mu, sigma, n, f"{f*100} samples")
            placeholder.pyplot(fig)
            plt.close(fig)
            st.sleep(anim_speed)

        st.success("✅ Animation finished! Notice how the distribution stabilizes.")

if __name__ == "__main__":
    main()
