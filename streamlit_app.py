import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# --- Core Functions ---
@st.cache_data
def generate_sample_means(dist, n, k, seed):
    """Generate k sample means of size n from a given distribution."""
    np.random.seed(seed)
    # Each row is a sample, so we generate a k x n matrix
    samples = dist.rvs(size=(k, n))
    # Calculate the mean of each row (each sample)
    return np.mean(samples, axis=1)

def main():
    st.set_page_config(
        page_title="Normal Distribution & CLT Lab",
        page_icon="📊",
        layout="wide"
    )
    st.title("📊 Normal Distribution & CLT Lab")

    with st.expander("What is this app?", expanded=True):
        st.markdown("""
        This application demonstrates two fundamental concepts in statistics:
        1.  **The Normal Distribution:** A bell-shaped curve described by its mean (μ) and standard deviation (σ). You can adjust these parameters in the **Distribution** tab.
        2.  **The Central Limit Theorem (CLT):** This theorem states that if you take many samples from *any* distribution and calculate their means, the distribution of those sample means will look like a Normal distribution. Explore this in the **Central Limit Theorem** tab.
        """)

    # --- Sidebar for controls ---
    with st.sidebar:
        st.header("Distribution Parameters")
        mu = st.slider("Mean (μ)", -5.0, 5.0, 0.0, 0.1, help="The center of the Normal distribution.")
        sigma = st.slider("Standard Deviation (σ)", 0.1, 5.0, 1.0, 0.1, help="The spread or width of the Normal distribution.")

        st.header("CLT Simulation Controls")
        n = st.slider("Sample Size (n)", 1, 1000, 30, help="The number of data points in each sample.")
        k = st.slider("Number of Samples (k)", 100, 10000, 1000, help="The number of times we draw a sample of size n.")

        st.header("Probability Shading")
        # Ensure the range slider's bounds are reasonable relative to the distribution
        plot_range_min = mu - 4 * sigma
        plot_range_max = mu + 4 * sigma
        prob_range = st.slider("Range [a, b]", plot_range_min, plot_range_max, (mu - sigma, mu + sigma), 0.1, help="Select a range [a, b] to calculate P(a ≤ X ≤ b).")

        st.header("Reproducibility")
        if 'seed' not in st.session_state:
            st.session_state.seed = np.random.randint(0, 10000)

        def resample():
            st.session_state.seed = np.random.randint(0, 10000)

        st.number_input("Random Seed", 0, 10000, key='seed')
        st.button("Resample", on_click=resample, use_container_width=True)

    # --- Main area for plots and data ---
    # Create the normal distribution object from scipy.stats
    dist = stats.norm(loc=mu, scale=sigma)

    tab1, tab2, tab3 = st.tabs(["Distribution", "Central Limit Theorem", "Data"])

    with tab1:
        st.header("Normal Distribution PDF & CDF")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # PDF Plot
        x = np.linspace(plot_range_min, plot_range_max, 1000)
        y_pdf = dist.pdf(x)
        ax1.plot(x, y_pdf, 'b-')
        ax1.set_title("Probability Density Function (PDF)")
        ax1.set_ylabel("Density")
        ax1.grid(True)

        # Shading
        a, b = prob_range
        x_fill = np.linspace(a, b, 100)
        y_fill = dist.pdf(x_fill)
        ax1.fill_between(x_fill, y_fill, color='blue', alpha=0.3)

        # CDF Plot
        y_cdf = dist.cdf(x)
        ax2.plot(x, y_cdf, 'r-')
        ax2.set_title("Cumulative Distribution Function (CDF)")
        ax2.set_ylabel("Cumulative Probability")
        ax2.grid(True)

        st.pyplot(fig)

        # Calculated Probability
        prob = dist.cdf(b) - dist.cdf(a)
        st.metric(f"Probability P({a:.2f} ≤ X ≤ {b:.2f})", f"{prob:.4f}")

    with tab2:
        st.header("Central Limit Theorem Demonstration")

        sample_means = generate_sample_means(dist, n, k, st.session_state.seed)

        fig_clt, ax_clt = plt.subplots(figsize=(8, 6))

        # Histogram of sample means
        ax_clt.hist(sample_means, bins=50, density=True, alpha=0.7, label="Empirical Sample Means")

        # Theoretical normal distribution overlay
        clt_mu = mu
        clt_sigma = sigma / np.sqrt(n)
        x_clt = np.linspace(min(sample_means), max(sample_means), 1000)
        y_clt = stats.norm.pdf(x_clt, loc=clt_mu, scale=clt_sigma)
        ax_clt.plot(x_clt, y_clt, 'r--', linewidth=2, label=f"Theoretical N(μ, σ/√n)")

        # 95% CI
        ci_lower = clt_mu - 1.96 * clt_sigma
        ci_upper = clt_mu + 1.96 * clt_sigma
        ax_clt.axvspan(ci_lower, ci_upper, color='red', alpha=0.1, label='95% Confidence Interval')

        ax_clt.set_title("Distribution of Sample Means")
        ax_clt.set_xlabel("Sample Mean Value")
        ax_clt.set_ylabel("Density")
        ax_clt.legend()
        ax_clt.grid(True)

        st.pyplot(fig_clt)

        # Stats Panel
        st.subheader("Statistical Properties")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Theoretical Mean", f"{clt_mu:.4f}")
            st.metric("Theoretical Variance", f"{clt_sigma**2:.4f}")
        with col2:
            st.metric("Empirical Mean", f"{np.mean(sample_means):.4f}")
            st.metric("Empirical Variance", f"{np.var(sample_means, ddof=1):.4f}")

    with tab3:
        st.header("Simulated Sample Means Data")

        # Regenerate for this tab to ensure data is always available
        sample_means_df = generate_sample_means(dist, n, k, st.session_state.seed)
        df = pd.DataFrame(sample_means_df, columns=["Sample Mean"])

        st.dataframe(df.head(10))

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download All Sample Means (CSV)",
            data=csv,
            file_name=f"clt_sample_means_n{n}_k{k}.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()
