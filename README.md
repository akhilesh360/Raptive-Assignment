# Raptive Assignment - Revenue Analysis: Time-on-Page Relationship Study

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B.svg)](https://raptive-assignment-ds7u4sk28vqkqdvrekpzwa.streamlit.app/)

## DEMO - **[https://raptive-assignment-ds7u4sk28vqkqdvrekpzwa.streamlit.app/](https://raptive-assignment-ds7u4sk28vqkqdvrekpzwa.streamlit.app/)**

## System Workflow

Study the link between user engagement (time on page) and revenue, across platforms, browsers, and sites. Detect Simpson's Paradox and extract actionable insights for optimization.

> **Clean → Explore → Model → Visualize → Recommend**

## Dashboard Preview

### Overall Time vs Revenue  
![Overall Scatter](outputs/01_scatter_overall.png)

### Platform Analysis  
![Platform Scatter](outputs/02_scatter_by_platform.png)

### Browser Analysis  
![Browser Scatter](outputs/03_scatter_by_browser.png)

### Partial Effects (Controlled Model)  
![Partial Effects](outputs/04_partial_effects.png)

## Key Features

- Full data science workflow: cleaning, EDA, regression modeling, advanced visualization
- Simpson's Paradox detection—compare overall and segmented correlations
- Executive-ready charts and summary statistics
- Platform/browser/site-specific business recommendations

## Quick Start

```bash
git clone https://github.com/akhilesh360/Raptive-Assignment.git
cd Raptive-Assignment

# Install dependencies
pip install -r requirements.txt

# Run analysis
python revenue_analysis.py

# View charts
open outputs/
```

Or launch the dashboard:
```bash
streamlit run streamlit_app.py
```

## How It Works

- **Data Cleaning:** Handles missing values, outliers, deduplication
- **Exploration:** Descriptive statistics, grouped summaries
- **Statistical Modeling:** Linear and multiple regression with categorical controls
- **Visualization:** Publication-quality charts for executive reporting
- **Business Insights:** Segmented strategies and recommendations based on findings

## Tech Stack

- Python 3.8+
- Streamlit (interactive dashboard)
- Pandas, NumPy (data handling)
- Matplotlib, Seaborn (visualization)
- Statsmodels (regression modeling)

## Project Structure

```
Raptive-Assignment/
├── streamlit_app.py          # Interactive dashboard
├── revenue_analysis.py       # Main analysis script
├── outputs/                  # Generated charts and summaries
├── requirements.txt          # Dependencies
└── README.md                 # Project overview
```

## Contributing

Pull requests welcome! For major changes, open an issue first.

## License

MIT License – see [LICENSE](LICENSE) for details.

## Author

**Sai Akhilesh Veldi**  
[GitHub](https://github.com/akhilesh360) • [LinkedIn](https://www.linkedin.com/in/saiakhileshveldi/) • [Portfolio](https://akhilesh360.github.io/SAIPORTFOLIO/)

---
