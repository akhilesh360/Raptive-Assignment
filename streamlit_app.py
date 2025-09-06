import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def get_distribution(dist_name, **kwargs):
    """Returns a scipy.stats distribution object based on the name and parameters."""
    if dist_name == "Normal":
        return stats.norm(loc=kwargs["mean"], scale=kwargs["std_dev"])
    elif dist_name == "Poisson":
        return stats.poisson(mu=kwargs["lam"])
    elif dist_name == "Exponential":
        return stats.expon(scale=1/kwargs["lam_exp"])
    return None

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="Statistical Distribution Explorer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📊 Statistical Distribution Explorer")
    st.write(
        "An interactive web app to explore the properties of common statistical distributions."
    )

    # --- Sidebar Controls ---
    st.sidebar.header("Distribution Parameters")
    dist_name = st.sidebar.selectbox(
        "Choose a distribution",
        ("Normal", "Poisson", "Exponential"),
        help="Select the statistical distribution you want to explore.",
    )

    seed = st.sidebar.number_input(
        "Random Seed",
        value=42,
        min_value=0,
        max_value=1000,
        step=1,
        help="Set a seed for the random number generator to ensure reproducibility.",
    )
    np.random.seed(seed)

    sample_size = st.sidebar.slider(
        "Sample Size", 50, 5000, 1000, 10, help="Number of random samples to generate."
    )

    params = {}
    if dist_name == "Normal":
        st.sidebar.subheader("Normal Distribution")
        params["mean"] = st.sidebar.slider("Mean (μ)", -10.0, 10.0, 0.0, 0.1)
        params["std_dev"] = st.sidebar.slider("Standard Deviation (σ)", 0.1, 10.0, 1.0, 0.1)
    elif dist_name == "Poisson":
        st.sidebar.subheader("Poisson Distribution")
        params["lam"] = st.sidebar.slider("Lambda (λ)", 1.0, 20.0, 5.0, 0.5)
    elif dist_name == "Exponential":
        st.sidebar.subheader("Exponential Distribution")
        params["lam_exp"] = st.sidebar.slider("Lambda (λ)", 0.1, 10.0, 1.0, 0.1)

    dist = get_distribution(dist_name, **params)
    if not dist:
        st.error("Invalid distribution selected.")
        return

    # --- Main Panel ---
    st.header(f"{dist_name} Distribution")

    # Generate sample data
    sample = dist.rvs(size=sample_size)

    # --- Statistics Panel ---
    st.sidebar.subheader("Descriptive Statistics")
    mean, var, skew, kurt = dist.stats(moments='mvsk')
    st.sidebar.metric("Mean", f"{mean:.4f}")
    st.sidebar.metric("Variance", f"{var:.4f}")
    st.sidebar.metric("Skewness", f"{skew:.4f}")
    st.sidebar.metric("Kurtosis", f"{kurt:.4f}")

    # --- Data Download ---
    df = pd.DataFrame(sample, columns=["value"])
    csv = df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Download Sample Data (CSV)",
        data=csv,
        file_name=f"{dist_name.lower()}_sample.csv",
        mime='text/csv',
    )

    # --- Visualizations ---
    col1, col2 = st.columns(2)

    # PDF/PMF Plot
    with col1:
        fig, ax = plt.subplots()
        if dist_name in ["Normal", "Exponential"]:
            x = np.linspace(dist.ppf(0.001), dist.ppf(0.999), 200)
            ax.plot(x, dist.pdf(x), 'r-', lw=2, label='PDF')
            ax.hist(sample, bins=50, density=True, alpha=0.6, label='Sample Histogram')
            ax.set_title(f"{dist_name} PDF vs. Sample Histogram")
            ax.set_ylabel("Density")
        else: # Poisson
            x = np.arange(dist.ppf(0.001), dist.ppf(0.999))
            ax.plot(x, dist.pmf(x), 'ro', ms=8, label='PMF')
            ax.vlines(x, 0, dist.pmf(x), colors='r', lw=2)
            ax.hist(sample, bins=np.arange(x[0], x[-1]+2)-0.5, density=True, alpha=0.6, label='Sample Histogram')
            ax.set_title(f"{dist_name} PMF vs. Sample Histogram")
            ax.set_ylabel("Probability")
        ax.legend()
        st.pyplot(fig)

    # CDF Plot
    with col2:
        fig, ax = plt.subplots()
        if dist_name in ["Normal", "Exponential"]:
            x = np.linspace(dist.ppf(0.001), dist.ppf(0.999), 200)
            ax.plot(x, dist.cdf(x), 'b-', lw=2, label='CDF')
        else: # Poisson
            x = np.arange(dist.ppf(0.001), dist.ppf(0.999))
            ax.step(x, dist.cdf(x), 'b-', where='post', lw=2, label='CDF')

        # Add empirical CDF
        ecdf_x = np.sort(sample)
        ecdf_y = np.arange(1, len(sample) + 1) / len(sample)
        ax.plot(ecdf_x, ecdf_y, 'g--', lw=2, label='Empirical CDF')

        ax.set_title(f"{dist_name} CDF vs. Empirical CDF")
        ax.set_ylabel("Cumulative Probability")
        ax.legend()
        st.pyplot(fig)

    # --- Interactive Demo ---
    if dist_name == "Normal":
        st.header("Central Limit Theorem (CLT) Demo")
        st.write(
            "The Central Limit Theorem states that the distribution of sample means "
            "approximates a normal distribution, regardless of the population's distribution, "
            "as the sample size gets larger."
        )

        clt_sample_size = st.slider("CLT Sample Size", 10, 500, 30, 5)
        num_samples = st.slider("Number of Samples", 100, 5000, 1000, 100)

        sample_means = [np.mean(dist.rvs(size=clt_sample_size)) for _ in range(num_samples)]

        fig, ax = plt.subplots()
        ax.hist(sample_means, bins=50, density=True, alpha=0.7, label='Histogram of Sample Means')

        # Overlay normal distribution
        clt_mean = np.mean(sample_means)
        clt_std = np.std(sample_means)
        x = np.linspace(clt_mean - 4*clt_std, clt_mean + 4*clt_std, 100)
        ax.plot(x, stats.norm.pdf(x, clt_mean, clt_std), 'r-', lw=2, label='Normal Approximation')

        ax.set_title("Distribution of Sample Means")
        ax.set_xlabel("Sample Mean")
        ax.set_ylabel("Density")
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
