# CardioNexus-AI

**Adaptive Differentially Private Federated Learning with Federated SHAP Explainability for Multi-Hospital Heart Disease Prediction**

A federated learning framework that combines **FedProx** (non-IID aggregation), **Adaptive Differential Privacy** (DP-SGD via Opacus), and **Federated SHAP** explainability for privacy-preserving, interpretable cardiovascular disease prediction across multiple simulated hospital clients.

---

## 🔍 Overview

Existing federated learning approaches for heart disease prediction address privacy, data heterogeneity, and interpretability **in isolation**. This project proposes a unified framework that:

- Handles **non-IID hospital data** using FedProx-based aggregation
- Provides **formal privacy guarantees** via adaptive DP-SGD (noise decreases as training converges)
- Produces **genuinely federated explanations** by aggregating local SHAP values across hospitals — without sharing raw patient data
- Audits **demographic fairness** (gender, age group) using Fairlearn

---

## 🏗️ Architecture

```
Hospital 1 (Cleveland)          Hospital 2 (Hungarian)
   Local DNN                       Local DNN
   + FedProx loss                  + FedProx loss
   + Opacus DP-SGD                 + Opacus DP-SGD
   + Local SHAP                    + Local SHAP
        │                               │
        └──────────────┬────────────────┘
                        ▼
                 Central Server
        FedProx weighted aggregation
        Federated SHAP aggregation
        Fairness evaluation (Fairlearn)
```

**Model:** 10 → 64 → 32 → 16 → 1 (PyTorch DNN, BCE loss, Adam optimizer)

---

## 📊 Datasets

| Dataset | Source | Patients | Features |
|---|---|---|---|
| Cleveland Heart Disease | UCI ML Repository | 303 | 10 (common schema) |
| Hungarian Heart Disease | UCI ML Repository | 294 | 10 (common schema) |

Both datasets were cleaned, missing values imputed, multi-class targets binarised, and features Min-Max normalised independently per client.

---

## 🧪 Results

### 1. FedAvg vs. FedProx (10 rounds)

| Configuration | Final Accuracy | Peak Accuracy |
|---|---|---|
| FedAvg (baseline) | 72.5% | 75.0% (R9) |
| FedProx (μ=0.01) | 71.7% | 80.0% (R6) |
| **FedProx (μ=0.1)** | **82.5%** | 83.3% (R8) |
| FedProx (μ=1.0) | 55.8% | 55.8% (flat) |

### 2. Privacy-Utility Trade-off (FedProx + Adaptive DP-SGD, 20 rounds)

| Noise Schedule σ(t) | Final Accuracy | Final ε | Privacy Level |
|---|---|---|---|
| max(1.0 − 0.07(t−1), 0.3) | 76.7% | 74.9 | Weak |
| max(1.5 − 0.02(t−1), 1.0) | 55.8% | 4.7 | Strong (underfits) |
| **max(1.0 − 0.02(t−1), 0.6)** | **78.3%** | **15.2** | Balanced (selected) |

![Accuracy vs Communication Rounds](images/fig1_accuracy.png)
![Privacy Budget vs Rounds](images/fig2_privacy.png)

### 3. Federated SHAP Explainability

| Rank | Cleveland (local) | Hungarian (local) | Global (federated) |
|---|---|---|---|
| 1 | exang (0.2922) | exang (0.2906) | **exang (0.2914)** |
| 2 | cp (0.0448) | cp (0.0453) | cp (0.0450) |
| 3 | sex (0.0365) | thalach (0.0314) | sex (0.0337) |
| 4 | thalach (0.0355) | sex (0.0308) | thalach (0.0335) |
| 5 | oldpeak (0.0176) | oldpeak (0.0169) | oldpeak (0.0172) |

> Both hospitals independently rank **exercise-induced angina (exang)** as the dominant predictor, with near-identical SHAP magnitudes (0.2922 vs 0.2906) — validating that federated SHAP aggregation produces consistent, clinically meaningful explanations without raw data sharing.

![Federated SHAP Feature Importance](images/fig3_shap.png)

### 4. Demographic Fairness Audit

| Site | Gender Accuracy Gap | Age Group Accuracy Gap |
|---|---|---|
| Cleveland | 13.5% (F 81.4% / M 68.0%) | 16.5% (Young 84.1% / Old 67.6%) |
| Hungarian | 7.1% (F 86.4% / M 79.3%) | 4.5% (Young 84.7% / Old 80.7%) |

![Fairness Comparison](images/fig4_fairness.png)

> Subgroup-level disparities at Cleveland exceed the 5% threshold despite strong aggregate accuracy — highlighting the need for fairness auditing in FL-based clinical systems, an aspect absent from prior FL-CVD literature.

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.11 |
| FL Framework | [Flower (flwr)](https://flower.ai/) |
| Deep Learning | PyTorch |
| Differential Privacy | Opacus (DP-SGD) |
| Explainability | SHAP (KernelExplainer) |
| Fairness | Fairlearn |
| Class Balancing | imbalanced-learn |

---

## 📁 Project Structure

```
CardioNexus_AI/
├── data/                  # Raw and cleaned datasets
├── code/
│   ├── model.py           # Shared DNN architecture
│   ├── preprocess.py       # Data cleaning & normalization
│   ├── client.py          # Flower client (FedAvg/FedProx)
│   ├── client_dp.py       # Flower client with Adaptive DP-SGD
│   ├── run_fedavg.py       # FedAvg baseline experiment
│   ├── run_fedprox.py      # FedProx hyperparameter sweep
│   ├── run_dp.py           # FedProx + Adaptive DP experiment
│   ├── federated_shap.py   # Federated SHAP aggregation
│   ├── fairness_eval.py    # Demographic fairness audit
│   └── make_figures.py     # Generate result figures
└── results/                # Logs and generated figures
```

---

## 🚀 Running the Experiments

```bash
# 1. Preprocess data
cd code
python preprocess.py

# 2. FedAvg baseline
python run_fedavg.py

# 3. FedProx (set MU in run_fedprox.py)
python run_fedprox.py

# 4. FedProx + Adaptive DP-SGD
python run_dp.py

# 5. Federated SHAP explainability
python federated_shap.py

# 6. Fairness evaluation
python fairness_eval.py

# 7. Generate figures
python make_figures.py
```

---

## 📌 Key Contributions

1. **FedProx-based non-IID aggregation** tuned for heterogeneous multi-hospital cardiovascular data
2. **Adaptive DP-SGD** with a decaying noise schedule, achieving 78.3% accuracy at ε = 15.2
3. **Federated SHAP aggregation** — the first genuinely federated (not post-hoc centralised) explainability pipeline validated across two independent hospital sites
4. **Demographic fairness audit** revealing site-specific disparities invisible in aggregate metrics

---

## 👤 Author

**Swarnali Ghosh**
B.Tech CSE (AI & ML), IEM/UEM Kolkata
Research Intern, IEEE EMBS Pune Chapter
[github.com/swarnali2005](https://github.com/swarnali2005)

---

## 📄 Citation

If you use this work, please cite (paper details to be updated upon publication).
