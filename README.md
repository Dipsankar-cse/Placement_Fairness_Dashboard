# 🎯 Placement Prediction & Algorithmic Fairness Dashboard

An interactive machine learning application built with **Python**, **Scikit-Learn**, **Fairlearn**, and **Streamlit** to predict student placement outcomes and mitigate demographic bias in hiring algorithms.

---

## 📌 Project Overview
Automated screening models in recruitment can unintentionally perpetuate demographic disparities. This project demonstrates how algorithmic fairness constraints can be integrated into predictive machine learning workflows to achieve high prediction accuracy while ensuring equitable decision-making across demographic groups.

- **Baseline Model:** Standard Random Forest Classifier exhibiting demographic parity disparities.
- **Fairness Mitigation:** Implemented **Fairlearn's `ExponentiatedGradient`** with `DemographicParity` constraints to reduce bias.
- **Interactive Dashboard:** A Streamlit web interface featuring a custom gradient header banner, live scenario simulation, and demographic parity evaluations.

---

## 📊 Key Results

| Metric | Unmitigated Base Model | Fairlearn Mitigated Model | Improvement |
| :--- | :---: | :---: | :---: |
| **Demographic Parity Gap** | `0.3242` | `0.0303` | **~90.6% Reduction in Bias** |

---

## 🛠️ Tech Stack & Dependencies
- **Language:** Python 3.10+
- **Machine Learning & Fairness:** Scikit-Learn, Fairlearn, Joblib
- **Data Manipulation:** Pandas, NumPy
- **Web Application:** Streamlit

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/Dipsankar-cse/Placement_Fairness_Dashboard.git](https://github.com/Dipsankar-cse/Placement_Fairness_Dashboard.git)
cd Placement_Fairness_Dashboard