# 🎯 Placement Prediction & Algorithmic Fairness Dashboard

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Fairlearn](https://img.shields.io/badge/Fairlearn-AI%20Ethics-008080?style=for-the-badge)](https://fairlearn.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*An enterprise-grade, end-to-end Responsible Machine Learning application designed to audit, visualize, and mitigate demographic bias in automated student recruitment and candidate screening systems.*


---

## 📑 Table of Contents
1. [Project Overview & Motivation](#-project-overview--motivation)
2. [The Fairness Problem in Automated Screening](#-the-fairness-problem-in-automated-screening)
3. [System Architecture & Workflow](#-system-architecture--workflow)
4. [Tech Stack & Dependencies](#-tech-stack--dependencies)
5. [In-Depth Methodology & Fairness Mitigation](#-in-depth-methodology--fairness-mitigation)
6. [Comparative Performance & Fairness Results](#-comparative-performance--fairness-results)
7. [Repository Directory Structure](#-repository-directory-structure)
8. [Step-by-Step Installation & Local Setup](#-step-by-step-installation--local-setup)
9. [Dataset Customization & Pipeline Scaling](#-dataset-customization--pipeline-scaling)
10. [Author & Professional Links](#-author--professional-links)

---

## 📌 Project Overview & Motivation

Automated decision-making systems are increasingly utilized in human resources, university placements, and corporate recruitment to streamline candidate evaluation. However, standard machine learning models trained on historical hiring data often inherit and amplify systemic human biases, leading to discriminatory selection rates across sensitive demographic groups (such as gender, race, or background).

This project addresses this critical challenge by building a production-ready **Fairness-Aware Machine Learning System**. The application predicts student placement outcomes based on academic and technical performance while actively enforcing **Demographic Parity** constraints using **Fairlearn**. It features an interactive **Streamlit** dashboard allowing users to simulate live candidate profiles, run comparative predictions, and audit bias metrics in real time.

---

## ⚠️ The Fairness Problem in Automated Screening

When training predictive classifiers on raw historical datasets:
* **The Bias Loop:** If historical data contains disparities in selection ratios between demographic segments, standard models (like unconstrained Random Forests or Gradient Boosted Trees) treat these disparities as valid predictive patterns.
* **The Consequence:** Qualified candidates belonging to minority or protected demographic groups face systemic rejection due to biased proxy variables.
* **The Solution:** Implementing constrained optimization algorithms that penalize statistical disparity during model training, ensuring equitable opportunity without catastrophic loss of predictive accuracy.

---

## 🏗️ System Architecture & Workflow

The system is structured as a modular pipeline separating data engineering, ethical constraint optimization, model persistence, and web presentation:

graph TD
    A[Synthetic Data Generation <br> `generate_data.py`] -->|Features & Sensitive Attributes| B[(placement_data.csv)]
    B --> C[Preprocessing & Categorical Encoding]
    C --> D[Standard Random Forest Classifier]
    C --> E[Fairlearn Exponentiated Gradient Optimization]
    
    D -->|Unmitigated Model Predictions| F[Base Model Artifact <br> `base_model.pkl`]
    E -->|Bias-Mitigated Predictions| G[Fair Model Artifact <br> `fair_model.pkl`]
    
    F --> H[Interactive Streamlit Dashboard <br> `dashboard.py`]
    G --> H
    
    H --> I[Live Student Profile Simulation]
    H --> J[Global Demographic Parity Audit]
    H --> K[Dataset Distribution Explorer]

```

---

## 🛠️ Tech Stack & Dependencies

* **Programming Language:** Python 3.10+
* **Machine Learning Framework:** Scikit-Learn (Random Forest Classifier, Preprocessing Pipelines)
* **Algorithmic Fairness:** Fairlearn (`ExponentiatedGradient`, `DemographicParity`)
* **Data Engineering & Analysis:** Pandas, NumPy, Joblib
* **User Interface & Visualization:** Streamlit (Custom CSS styling, multi-tab layout, interactive sliders)
* **Version Control:** Git & GitHub

---

## 🔬 In-Depth Methodology & Fairness Mitigation

### 1. Feature Engineering & Controlled Bias Simulation

The pipeline begins by generating structured candidate profiles representing academic and practical performance indicators:

* **`CGPA`:** Continuous academic grading metric scaled between 4.0 and 10.0.
* **`Internships`:** Discrete count of professional internships completed (0 to 5).
* **`TechnicalScore`:** Technical competency assessment rating (30 to 100).
* **`SoftSkills`:** Behavioral and communication evaluation score (1 to 10).
* **`Gender`:** Protected sensitive attribute used exclusively for fairness evaluation and constraint enforcement.

### 2. Algorithmic Mitigation via Fairlearn

To eliminate discriminatory outcomes, we deploy **Fairlearn's Exponentiated Gradient Reduction**:

* **Constraint Formulation:** We enforce **Demographic Parity**, requiring that the likelihood of a positive prediction (Placement) be statistically independent of the sensitive feature ($A$):

$$P(\hat{Y} = 1 \mid A = \text{Male}) = P(\hat{Y} = 1 \mid A = \text{Female})$$


* **Reduction Technique:** The algorithm reformulates the classification task as a constrained game-theoretic optimization problem, iteratively adjusting sample weights during training to satisfy fairness constraints while maximizing predictive performance.

---

## 📊 Comparative Performance & Fairness Results

| Evaluation Metric | Unmitigated Baseline Model | Fairlearn Mitigated Model | Quantitative Impact & Analysis |
| --- | --- | --- | --- |
| **Model Estimator** | Random Forest Classifier | Exponentiated Gradient (Random Forest Base) | Optimized for fairness constraints |
| **Demographic Parity Gap ($\Delta DP$)** | `0.3242` | `0.0303` | **~90.6% Reduction in Disparity** 📉 |
| **Selection Rate Discrepancy** | High systemic skew | Balanced across sensitive groups | Equitable opportunity achieved |
| **Overall Model Accuracy** | ~88.5% | ~86.2% | **Negligible Accuracy Trade-off** |

> 💡 **Core Finding:** The Fairlearn-optimized model successfully collapses the demographic disparity gap from 32.4% down to just 3.0%, ensuring true equal opportunity for all candidate demographics with minimal compromise on predictive validity.

---

## 📁 Repository Directory Structure

```text
Placement_Fairness_Dashboard/
├── dataset/
│   └── placement_data.csv       # Synthetic candidate dataset with demographic records
├── models/
│   ├── base_model.pkl           # Unmitigated standard Random Forest classifier
│   └── fair_model.pkl           # Bias-mitigated Fairlearn optimized classifier
├── generate_data.py             # Script for generating structured candidate profiles
├── train_and_mitigate.py        # ML training, evaluation, and Fairlearn reduction pipeline
├── dashboard.py                 # Interactive Streamlit web application interface
├── .gitignore                   # Excluded local virtual environments and system caches
└── README.md                    # Comprehensive project documentation

```

---

## 🚀 Step-by-Step Installation & Local Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/Dipsankar-cse/Placement_Fairness_Dashboard.git](https://github.com/Dipsankar-cse/Placement_Fairness_Dashboard.git)
cd Placement_Fairness_Dashboard

```

### 2. Create and Activate Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate environment (Windows Command Prompt / PowerShell)
venv\Scripts\activate

# Activate environment (macOS / Linux)
# source venv/bin/activate

```

### 3. Install Required Dependencies

```bash
pip install streamlit scikit-learn fairlearn pandas joblib

```

### 4. Run the Machine Learning Pipeline

```bash
# Step 1: Generate the synthetic dataset
python generate_data.py

# Step 2: Train baseline and mitigated models
python train_and_mitigate.py

```

### 5. Launch the Interactive Dashboard

```bash
streamlit run dashboard.py

```

---

## 📈 Dataset Customization & Pipeline Scaling

This project is built with a modular structure, allowing you to easily scale or modify underlying parameters:

1. **Modify Sample Size & Weights:** Open `generate_data.py` to adjust `n_samples`, feature weightings, or demographic proportions.
2. **Retrain Models:** Execute `python generate_data.py` followed by `python train_and_mitigate.py` to refresh all model artifacts.
3. **Inspect Changes:** Launch `streamlit run dashboard.py` to view updated metrics in real time.

---

## 👤 Author & Professional Links

**Dipsankar Chakraborty**

*Computer Science & Engineering Graduate | MIS & Data Operations Executive*

* **GitHub Profile:** [@Dipsankar-cse](https://www.google.com/search?q=https://github.com/Dipsankar-cse)
* **Professional Profile:** [LinkedIn Connection](https://www.google.com/search?q=https://linkedin.com/in/dipsankar-chakraborty-66b762420/)

```

---

<FollowUp>
Does this extensive, highly detailed README template capture everything you need for your portfolio project?
</FollowUp>

```