# AI Ethics Audit & Fairness Assessment: Adult Census Income Prediction

An end-to-end algorithmic fairness and ethical audit framework evaluating demographic bias in machine learning classifiers. This project investigates structural disparities within the benchmark **Adult Census Dataset**, measures disparate impact against protected attributes, and implements in-processing mitigation techniques using **Fairlearn** and **Scikit-Learn**.

---

## 🚀 Project Overview
Automated decision systems often inherit and amplify historical labor market biases. This repository houses an in-depth ethical audit notebook that:
1. **Establishes a Baseline:** Trains an unconstrained `RandomForestClassifier` to predict annual income (>\\$50K) and exposes deep outcome disparities.
2. **Quantifies Fairness:** Leverages `Fairlearn`'s `MetricFrame` to compute group-level selection rates, accuracy, and precision across protected classes (`sex`).
3. **Applies Mitigation:** Implements an in-processing wrapper (`ExponentiatedGradient`) coupled with a strict `DemographicParity` constraint.
4. **Visualizes Trade-offs:** Generates comparative visualizations analyzing the balance between model accuracy and demographic equity.

---

## 📊 Key Results Summary

| Metric | Unconstrained Baseline | Post-Mitigation (ExponentiatedGradient) |
| :--- | :--- | :--- |
| **Overall Accuracy** | **85.82%** | **80.17%** |
| **Selection Rate (Female)** | 0.0890 | 0.2338 |
| **Selection Rate (Male)** | 0.2607 | 0.2877 |
| **Selection Rate Disparity ($\Delta$)** | **0.1717** | **0.0539** |
| **Disparate Impact Ratio** | ~0.341 (Violates 4/5ths Rule) | ~0.812 (Compliant > 0.80) |

---

## 🛠️ Tech Stack & Dependencies
* **Python** (3.10+)
* **Fairlearn** (Algorithmic fairness toolkit & reductions)
* **Scikit-Learn** (Baseline modeling & data preprocessing)
* **Pandas / NumPy** (Data manipulation & matrix operations)
* **Matplotlib / Seaborn** (Data visualization & disparity plotting)

---

## ⚙️ Quick Start & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/ai-ethics-audit-adult-census.git](https://github.com/your-username/ai-ethics-audit-adult-census.git)
   cd ai-ethics-audit-adult-census
