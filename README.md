# Statistical Distribution Explorer

An interactive Streamlit web app to visualize and understand common statistical distributions: Normal, Poisson, and Exponential.

## ✨ Features

*   **Interactive Controls**: Adjust distribution parameters (e.g., mean, standard deviation, lambda) using sliders.
*   **Multiple Distributions**: Choose between Normal, Poisson, and Exponential distributions.
*   **Visualizations**:
    *   Probability Density/Mass Function (PDF/PMF) plots.
    *   Cumulative Distribution Function (CDF) plots.
*   **Interactive Demo**: A simulation of the Central Limit Theorem for the Normal distribution.
*   **Key Statistics**: View calculated mean, variance, skewness, and kurtosis.
*   **Data Export**: Download a sample of generated data as a CSV file.
*   **Reproducibility**: Set a random seed to get the same results every time.

## 🚀 Live Demo

[**Launch the Streamlit App**](https://your-streamlit-app-url.streamlit.app)

*(Note: You will need to deploy this app to Streamlit Cloud to get a shareable link.)*

## 🛠️ Setup and Installation

To run this app locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run streamlit_app.py
    ```

The app should now be open in your web browser.
