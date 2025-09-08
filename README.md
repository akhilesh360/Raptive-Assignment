# Distribution Lab — Normal Distribution & Central Limit Theorem (CLT)

An interactive **Streamlit app** to explore the **Normal distribution** and visualize the **Central Limit Theorem (CLT)** in action.

---

## Demo  
👉 **[Live App](https://raptive-assignment-ds7u4sk28vqkqdvrekpzwa.streamlit.app/)**  

---

## Features
- Adjustable **μ (mean)** and **σ (standard deviation)** with instant PDF/CDF updates  
- Probability shading: `P(a ≤ X ≤ b)`  
- CLT simulation: histogram of sample means vs. theoretical `N(μ, σ/√n)`  
- Summary statistics: mean, variance, skewness, kurtosis  
- Download simulated data as CSV  

---

## Dashboard Preview

### Normal Distribution Playground  
*(Adjust mean & standard deviation interactively)*  
![Normal Dist Screenshot](screenshots/normal-dist-demo.png)

### Central Limit Theorem Simulation  
*(Histogram of sample means compared to theoretical normal)*  
![CLT Screenshot](screenshots/clt-sim-demo.png)

---

## Quick Start

```bash
# Clone repo
git clone https://github.com/akhilesh360/distribution-lab.git
cd distribution-lab

# Install dependencies
pip install -r requirements.txt

# Run app locally
streamlit run streamlit_app.py
