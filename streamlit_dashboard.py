import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson, expon

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Interactive Distribution Explorer",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Interactive Distribution Explorer")
st.markdown("""
This app is purely **educational**. Explore how different statistical distributions behave, 
and understand key properties such as shape, spread, and the **Central Limit Theorem (CLT)**.

- Select a distribution (Normal, Poisson, Exponential)
- Adjust parameters using sliders
- Visualize how the distribution changes
- Play with the CLT demo by sampling repeatedly from a distribution
""")

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.header("Choose Distribution & Parameters")

# Select distribution
dist_name = st.sidebar.selectbox("Distribution", ["Normal", "Poisson", "Exponential"])

# Parameter sliders
if dist_name == "Normal":
    mu = st.sidebar.slider("Mean (μ)", -5.0, 5.0, 0.0, 0.1)
    sigma = st.sidebar.slider("Standard Deviation (σ)", 0.1, 5.0, 1.0, 0.1)
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
    y = norm.pdf(x, mu, sigma)
elif dist_name == "Poisson":
    lam = st.sidebar.slider("Lambda (λ)", 1, 20, 5)
    x = np.arange(0, 30)
    y = poisson.pmf(x, lam)
elif dist_name == "Exponential":
    scale = st.sidebar.slider("Scale (1/λ)", 0.1, 5.0, 1.0, 0.1)
    x = np.linspace(0, 20, 500)
    y = expon.pdf(x, scale=scale)

# -------------------------------
# Plot distribution
# -------------------------------
fig, ax = plt.subplots()
ax.plot(x, y, lw=2)
ax.fill_between(x, y, alpha=0.3)
ax.set_title(f"{dist_name} Distribution")
ax.set_xlabel("x")
ax.set_ylabel("Density / Probability")
st.pyplot(fig)

# -------------------------------
# Central Limit Theorem Demo
# -------------------------------
st.header("🎲 Central Limit Theorem Demo")
st.markdown("Sample repeatedly from the chosen distribution and see how the **sample mean** converges to a Normal distribution.")

n_samples = st.slider("Sample size (per trial)", 10, 500, 50, 10)
n_trials = st.slider("Number of trials", 100, 2000, 500, 100)

# Generate sample means
if dist_name == "Normal":
    samples = np.random.normal(mu, sigma, (n_trials, n_samples))
elif dist_name == "Poisson":
    samples = np.random.poisson(lam, (n_trials, n_samples))
elif dist_name == "Exponential":
    samples = np.random.exponential(scale, (n_trials, n_samples))

sample_means = samples.mean(axis=1)

# Plot histogram of sample means
fig2, ax2 = plt.subplots()
ax2.hist(sample_means, bins=30, density=True, alpha=0.6, color="blue")
ax2.set_title("Distribution of Sample Means (CLT)")
ax2.set_xlabel("Sample Mean")
ax2.set_ylabel("Density")
st.pyplot(fig2)

st.success("✅ Explore how the CLT works: regardless of the original distribution, the means approach Normal!")
