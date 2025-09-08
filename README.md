# Distribution Lab (Streamlit) — Normal + CLT

An interactive Streamlit app that demonstrates how parameters change a Normal
distribution and visualizes the Central Limit Theorem (CLT).

## Live Demo
<https://raptive-assignment-ds7u4sk28vqkqdvrekpzwa.streamlit.app/>

## Features
- Adjustable μ and σ with instant PDF/CDF updates
- Probability shading P(a ≤ X ≤ b)
- CLT simulation: histogram of sample means vs. N(μ, σ/√n)
- Stats: mean, variance, skewness, kurtosis
- Download simulated data as CSV

## Run Locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy (Streamlit Community Cloud)
- New app → connect this repo → path: streamlit_app.py → Python `runtime.txt` set to 3.11.
- Deploy!
