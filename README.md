<div align="center">

# Hi, I'm Qamar Usman 👋

### Machine Learning Engineer · Kaggle Expert (Top 0.4% Globally)

I build end-to-end ML systems — from causal time-series forecasting and quantitative trading strategies, to healthcare risk models, to autonomous LangGraph agents — and ship them as real, working products, not just notebooks.

[![Kaggle](https://img.shields.io/badge/Kaggle-qamarmath-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/qamarmath)
[![Email](https://img.shields.io/badge/Email-usmanqamar874%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:usmanqamar874@gmail.com)
[![Location](https://img.shields.io/badge/Based%20in-Punjab%2C%20Pakistan-333333?style=for-the-badge)](#)

</div>

---

## 👤 About Me

I'm a Machine Learning Engineer with 2+ years of production experience and a Kaggle track record spanning **58+ competitions across finance, biology, healthcare, and NLP**. My work sits at the intersection of three things:

- **Rigor** — causal, leakage-free validation for time-series and survival data; energy-based physical scoring for structure prediction; fairness-stratified evaluation for clinical risk models.
- **Breadth** — I move comfortably between gradient-boosted trees (LightGBM/XGBoost/CatBoost), deep learning (PyTorch, transformer fine-tuning, protein/RNA language models), and LLM-powered agentic systems (LangGraph, RAG, tool orchestration).
- **Shipping** — every project below is either a scored Kaggle solution, a deployed app (Streamlit/FastAPI/Docker/Azure), or both. I don't stop at a leaderboard score.

I hold a **BS in Mathematics** (Government College University Faisalabad), which is the backbone of how I approach modeling — I default to understanding *why* a method works before reaching for a library.

---

## 💼 Currently

- 🔭 **Machine Learning Engineer** at **[VFixall](https://vfixall.com)** (Remote) — since Dec 2023
  - Built demand-forecasting pipelines (XGBoost + LightGBM) that cut stockouts by **40%**
  - Automated EDA workflows, reducing analysis time from 8 hours to 3 (**60% faster**)
  - Shipped Plotly/Streamlit dashboards used by **15+ stakeholders**
- 🏆 Actively competing in **AI Agent Security — Multi-Step Tool Attacks** and **RSNA Knee Abnormality Detection** on Kaggle
- 📈 Sharpening research toward publishable work (arXiv / Zenodo) as a path into research-engineer roles
- 💬 Ask me about: causal time-series validation, volatility-constrained trading strategies, survival analysis, or building agentic pipelines with LangGraph

---

## 🏅 Kaggle Achievements

<div align="center">

| 🎖️ Rank | 🥇 Medals | 🧪 Competitions | 📊 Percentile |
|:---:|:---:|:---:|:---:|
| **Expert** | 2 🥈 Silver · 1 🥉 Bronze | 58+ (60 solo entries) | **Top 0.4%** of 212,000+ Kagglers |

</div>

**Medal-winning competitions:**

| Competition | Result | Field |
|---|---|---|
| 🧬 [Stanford RNA 3D Structure Prediction](https://www.kaggle.com/code/qamarmath/beginner-to-advanced-rna-3d-structure-predicti) | 🥈 Silver — Rank **57 / 1,516** (Top 4%) | Structural Biology / Deep Learning |
| 🧠 [Child Mind Institute — Problematic Internet Use](https://www.kaggle.com/code/qamarmath/fork-of-handling-overfitting-val-qwk-0-457-4c34cf) | 🥈 Silver — Rank **76 / 3,559** (Top 3%) | Behavioral Health / Tabular ML |
| 📈 [Hull Tactical — Market Prediction](https://www.kaggle.com/code/qamarmath/hull-tactical-causal-lgbm-market-volatility) | 🥉 Bronze — Rank **364 / 3,677** (Top 10%) | Quantitative Finance |

📌 Full competition history: **[kaggle.com/qamarmath](https://www.kaggle.com/qamarmath)**

---

## 🚀 Featured Projects

### 💹 Quantitative Finance & Time-Series Forecasting

<table>
<tr>
<td width="50%" valign="top">

**[Hull Tactical — S&P 500 Excess Return Prediction](https://github.com/Qamar-usman-ai/hull-tactical-asset-allocation-lgbm)** 🥉
Causal LightGBM pipeline converting excess-return forecasts into a volatility-constrained (≤120%) allocation strategy.
**1.70 Sharpe Ratio**, 4.2% outperformance on a 6-month live test window.
`LightGBM` `Polars` `Time-Series` `[Kaggle Notebook](https://www.kaggle.com/code/qamarmath/hull-tactical-causal-lgbm-market-volatility)`

</td>
<td width="50%" valign="top">

**[Rohlik Orders Forecasting](https://github.com/Qamar-usman-ai/Rohlik-Orders-Forecasting-Challenge)**
XGBoost (DART) demand forecasting across 7 Central European warehouses, with cyclical time encodings and TF-IDF holiday features.
**3.37% MAPE, 0.9856 R²** — deployed as a [live Streamlit app](https://rohlik-orders-forecasting-challenge-htxwnjy5vm3sjudwkhsz5t.streamlit.app/).
`XGBoost` `Streamlit` `Feature Engineering`

</td>
</tr>
<tr>
<td width="50%" valign="top">

**[SME Financial Health Predictor](https://github.com/Qamar-usman-ai/SME-Financial-Health-Predictor)**
3-class classifier for Southern African SME financial health from 9,618 survey responses with up to 46.7% missingness per feature.
**0.8849 F1 — Rank 28 of 950+ participants.**
`Random Forest` `Word2Vec Embeddings`

</td>
<td width="50%" valign="top">

**[ADIA Lab Structural Break Challenge](https://github.com/Qamar-usman-ai/ADIA-Lab-Structural-Break-Challenge)**
Regime-shift detection across 10,001 synthetic time series (23.7M+ rows) using FFT frequency features, entropy, and statistical moments.
**0.9224 training ROC-AUC**, 94% break-detection precision.
`LightGBM` `Signal Processing`

</td>
</tr>
<tr>
<td colspan="2" valign="top">

**[Stock Market AI Analyst](https://github.com/Qamar-usman-ai/stock-market-analyst)**
Cloud-native LangGraph agent orchestrating technical-indicator extraction, Finnhub news-sentiment scraping, Plotly visualization, and SARIMA forecasting into autonomous BUY/HOLD/SELL investment reports via Groq LLaMA 3.3 70B. Deployed on Docker / Azure Container Apps.
`LangGraph` `FastAPI` `SARIMA` `LLM Integration`

</td>
</tr>
</table>

### 🏥 Healthcare & Behavioral Analytics

<table>
<tr>
<td width="50%" valign="top">

**[Stanford RNA 3D Structure Prediction](https://www.kaggle.com/code/qamarmath/beginner-to-advanced-rna-3d-structure-predicti)** 🥈
Predicted 3D RNA folds from sequence alone using an RNA-language-model + structure-prediction deep ensemble (up to 20 model variants), ranked by a multi-term physical energy function (bond, angle, stacking, H-bond, electrostatics).

</td>
<td width="50%" valign="top">

**[Child Mind Institute — Problematic Internet Use](https://www.kaggle.com/code/qamarmath/fork-of-handling-overfitting-val-qwk-0-457-4c34cf)** 🥈
Predicted internet-use severity in adolescents from wearable fitness data. 7-fold Stratified LightGBM with Nelder-Mead threshold tuning.
**QWK 0.463** on final submission.

</td>
</tr>
<tr>
<td width="50%" valign="top">

**[Early Pediatric Sepsis Detection](https://github.com/Qamar-usman-ai/Early-Sepsis-Detection-Model)**
XGBoost early-warning system predicting sepsis onset **6 hours before clinical recognition** from 331K+ measurements across 2,649 patients (2.07% positive class), using TF-IDF drug-exposure profiles.
**0.9675 PR-AUC**, patient-independent Group K-Fold validation.

</td>
<td width="50%" valign="top">

**[PII Detection in Educational Data](https://www.kaggle.com/code/qamarmath/fork-of-detecting-student-pii-with-fine-tuned-debe)**
Token-classification NER pipeline detecting 7 PII entity types across 6,807 student essays using a 4-model weighted DeBERTa ensemble.
**0.95323 F5** on the public leaderboard — Rank 209/2,048 (Top 10%).

</td>
</tr>
<tr>
<td width="50%" valign="top">

**[Survival Prediction After HCT](https://github.com/Qamar-usman-ai/Survival-Prediction-1)**
Fairness-aware survival model for post-transplant patients. Nelson-Aalen cumulative-hazard transformation + Word2Vec categorical embeddings + LightGBM/XGBoost/CatBoost ensemble, evaluated via **race-stratified Concordance Index**.

</td>
<td width="50%" valign="top">

**[ANNITIA — Liver Disease Risk Prediction](https://github.com/Qamar-usman-ai/ANNITIA-Survival-Analysis-Challenge)**
Time-to-event pipeline predicting hepatic events and mortality from longitudinal biomarker trajectories (up to 22 visits/patient). Ensembled CoxNet, Random Survival Forest, and a custom PyTorch DeepSurv model.
**Up to 0.968 C-index** on the death outcome.

</td>
</tr>
</table>

### 🤖 Deep Learning, Research & Agentic AI

<table>
<tr>
<td width="50%" valign="top">

**[LangGraph Autonomous Lead-Generation Agent](https://github.com/Qamar-usman-ai/langgraph)**
Autonomous agent that discovers real leads (name, email, organization) via web search and scraping, using a genuine **plan-act-reflect loop** — cyclic state graph, conditional routing, self-correction — not a hardcoded pipeline.

</td>
<td width="50%" valign="top">

**[Stock Market AI Analyst](https://github.com/Qamar-usman-ai/stock-market-analyst)**
4-tool LangGraph agent (data, sentiment, visualization, forecasting) generating full AI investment reports via Groq LLaMA 3.3, deployed as a containerized FastAPI service.

</td>
</tr>
</table>

📌 Two more agentic/finance and healthcare projects above overlap categories intentionally — the underlying skills transfer directly between them.

---

## 🛠️ Tech Stack

**Languages & Core**
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-4479A1?style=flat-square&logo=postgresql&logoColor=white)

**Modeling**
![LightGBM](https://img.shields.io/badge/LightGBM-blue?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-EB5E28?style=flat-square)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=flat-square&logoColor=black)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)

**LLMs & Agentic AI**
![LangGraph](https://img.shields.io/badge/LangGraph-1C1C1C?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square)
![HuggingFace](https://img.shields.io/badge/🤗%20Transformers-FFD21E?style=flat-square)
![RAG](https://img.shields.io/badge/RAG-4B0082?style=flat-square)

**Data & Deployment**
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Polars](https://img.shields.io/badge/Polars-CD792C?style=flat-square)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Azure](https://img.shields.io/badge/Azure-0078D4?style=flat-square&logo=microsoftazure&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?style=flat-square&logo=githubactions&logoColor=white)

---

## 📊 GitHub Stats

<div align="center">
<img height="165" src="https://github-readme-stats.vercel.app/api?username=Qamar-usman-ai&show_icons=true&theme=default&hide_border=true&count_private=true" />
<img height="165" src="https://github-readme-stats.vercel.app/api/top-langs/?username=Qamar-usman-ai&layout=compact&hide_border=true" />
</div>

<div align="center">
<img src="https://github-readme-streak-stats.herokuapp.com/?user=Qamar-usman-ai&hide_border=true" />
</div>

> Note: these widgets render live from the GitHub Readme Stats service and update automatically — no action needed once this file is in your profile repo.

---

## 🎓 Education & Certifications

**BS Mathematics** — Government College University Faisalabad (2014 – 2019)
*Strong foundation for ML algorithm development and statistical modeling.*

- ✅ TensorFlow Developer Specialization (2023)
- ✅ Machine Learning Specialization — Stanford University / Coursera (2023)
- ✅ IBM Data Science Professional Certificate (2023)

---

## 📫 Let's Connect

<div align="center">

[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/qamarmath)
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:usmanqamar874@gmail.com)

📱 +92 304-6515636   |   📍 Punjab, Pakistan   |   🗣️ English (Professional) · Urdu (Native)

⭐ If any of the projects above are useful to you, a star on the repo goes a long way.

</div>
