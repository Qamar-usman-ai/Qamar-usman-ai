<div align="center">

# 👋 Hi, I'm Qamar Usman

### Machine Learning Engineer | Agentic AI • LLMs • RAG • Medical AI • Computer Vision • Time Series

[![Kaggle](https://img.shields.io/badge/Kaggle-Top_0.4%25_Expert-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/qamarmath)
[![GitHub](https://img.shields.io/badge/GitHub-Qamar--usman--ai-181717?style=for-the-badge&logo=github)](https://github.com/Qamar-usman-ai)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/qamar-usman-ai)
[![Email](https://img.shields.io/badge/Email-usmanqamar874@gmail.com-EA4335?style=for-the-badge&logo=gmail)](mailto:usmanqamar874@gmail.com)

📍 Bahawalnagar, Punjab, Pakistan | Open to Remote & Global Opportunities

</div>

---

## 👨‍💻 About Me

I am a **research-driven Machine Learning Engineer** with a strong Mathematics foundation, currently building end-to-end production ML pipelines at **VFIXALL**. My work spans:

- 🤖 **Agentic AI & LLMs** — Multi-agent systems, ReAct agents, tool-use pipelines, AutoGen
- 🔍 **RAG Systems** — LangChain, FAISS, semantic search, PDF/web document Q&A
- 🏥 **Medical & Healthcare AI** — EEG analysis, RNA folding, cancer detection, sepsis prediction
- 👁️ **Computer Vision** — EfficientNet, ResNet, CNNs for medical imaging
- 📈 **Time Series Forecasting** — XGBoost, cyclical encoding, MAPE-optimized pipelines
- ☁️ **Cloud & MLOps** — Azure ML, Docker, GitHub Actions, MLflow, CI/CD

> **Kaggle Global Rank: Top 0.4% worldwide | 807 of 202,646 competitors | 2× Silver Medalist**

---

## 📊 GitHub Stats

<div align="center">

![GitHub Stats](https://github-readme-stats.vercel.app/api?username=Qamar-usman-ai&show_icons=true&theme=tokyonight&hide_border=true)
![GitHub Streak](https://github-readme-streak-stats.herokuapp.com/?user=Qamar-usman-ai&theme=tokyonight&hide_border=true)
![Top Languages](https://github-readme-stats.vercel.app/api/top-langs/?username=Qamar-usman-ai&layout=compact&theme=tokyonight&hide_border=true)

</div>

---

## 🚀 Projects

---

### 🤖 AGENTIC AI & LLM SYSTEMS

---

#### 📈 [Stock Market Analyst Agent](https://github.com/Qamar-usman-ai/stock-market-analyst)
> **Agentic AI | ReAct | Tool-Use | LLM Pipelines**

An autonomous **multi-step agentic AI** system that independently researches, analyzes, and reports on stock market data — no human intervention needed after launch.

**How it works:**
1. 🔎 Agent autonomously queries real-time financial data and web sources
2. 🧠 ReAct reasoning loop: Think → Act → Observe → Repeat
3. 📊 LLM-powered analysis synthesizes signals and trends
4. 📄 Auto-generates structured PDF reports with insights

**Tech Stack:** Python • LangChain Agents • OpenAI API • Web Search Tool • ReAct Framework • PDF Generation

```
Input: "Analyze NVIDIA stock this week"
  ↓ Agent searches web + financial APIs
  ↓ LLM reasons over retrieved data
  ↓ Agent generates structured analysis
Output: Automated PDF report with insights
```

---

#### 💬 [Chat with PDFs & Websites (RAG)](https://chat-with-pdfs.streamlit.app)
> **RAG | LangChain | FAISS | Gemini / OpenAI**

Production-ready **Retrieval-Augmented Generation** app. Upload any PDF or paste a website URL and have a full conversation with the content using state-of-the-art LLMs.

**Architecture:**
```
PDF / URL → Text Extraction → Chunking → FAISS Vector Store
                                                ↓
User Question → Semantic Search → Retrieved Context → LLM → Answer
```

**Tech Stack:** Streamlit • LangChain • FAISS • Gemini / Gemma / OpenAI • Python

🔗 [Live Demo](https://chat-with-pdfs.streamlit.app)

---

#### 💬 [Chat with Your Data (CSV/Excel/SQL)](https://csv-chat.streamlit.app)
> **Agentic Data Analysis | NL2SQL | Google Generative AI**

Interact with structured data — CSV, Excel, SQL — using **plain English**. No SQL knowledge needed.

**Tech Stack:** Streamlit • LangChain • Google Generative AI • Pandas • Python

🔗 [Live Demo](https://csv-chat.streamlit.app)

---

#### 🛡️ PII Detection in Student Writing (NER)
> **DeBERTa-v3 Ensemble | Token Classification | Educational AI**

Automated anonymization of student writing to enable privacy-safe educational research at scale.

- **Model:** "Piiranha" — ensemble of 3 DeBERTa-v3-large models
- **Inference:** Softmax Weighted Average across ensemble
- **Optimization:** Micro F5-Score (recall-weighted) for maximum anonymization coverage
- **Impact:** Enables large-scale NLP research on student data without privacy violations

---

#### 📐 Math Misconception Classification (MAP@3: 0.9428)
> **Specialized Encoder | Educational NLP**

High-precision classifier identifying student math misconceptions to enable targeted tutoring.

- **Model:** `jhu-clsp/ettin-encoder-400m` — mathematics-specialized encoder
- **Training:** 3 epochs, FP16 Mixed Precision, structured context engineering
- **Performance:** **0.9428 MAP@3**

---

### 🏥 MEDICAL & HEALTHCARE AI

---

#### 🧬 [Stanford RNA 3D Folding Challenge](https://www.kaggle.com/qamarmath) — 🥈 Silver Medal | Rank 57/1,516 (Top 4%)
> **Transformers | 3D Structure Prediction | Ensemble Learning**

Solved one of biology's grand challenges: predicting 3D atomic coordinates of RNA molecules from sequence alone — accelerating RNA-based medicines, cancer immunotherapies, and CRISPR gene editing.

**Pipeline:**
```
RNA Sequence
    ↓
RNA2nd: 18-layer Transformer (secondary structure)
    ↓
MSA2XYZ: 3D coordinate generator
    ↓
20-Model Ensemble → OpenMM Energy Minimization → Final 3D Structure
```

- **Evaluation Metric:** TM-score (global topology alignment)
- **Ensemble:** 20 models for structural diversity
- **Physics:** OpenMM thermodynamic stability refinement

---

#### 🧠 Child Mind Institute: Problematic Internet Use — 🥈 Silver Medal | Rank 76/3,559 (Top 3%)
> **LightGBM | Mental Health AI | Behavioral Prediction**

Early detection of problematic internet use in youth using physical activity and fitness data as mental health proxies — enabling early intervention for depression and anxiety.

- **Model:** LightGBM Regressor + 7-Fold Stratified K-Fold
- **Optimization:** Nelder-Mead threshold tuning for max QWK
- **Result:** Final QWK = **0.463**

---

#### 🧠 HMS: Harmful Brain Activity Classification — Top 11% | Rank 312/2,767
> **ResNet18d | EEG Signal Processing | Neurocritical Care**

Deep learning pipeline automating seizure and harmful brain activity detection from EEG signals.

- **Architecture:** ResNet18d on single-channel log-normalized spectrograms
- **Loss:** KL Divergence on expert neurologist soft labels
- **Training:** 5-Fold CV + Cosine Annealing scheduler

---

#### 🏥 [CIBMTR: Post-HCT Survival Predictions](https://github.com/Qamar-usman-ai/Survival-Prediction-1) — Top 10% | Rank 341/3,325
> **Ensemble Learning | Fairness ML | Healthcare**

Equitable survival prediction for stem cell transplant patients across demographic groups.

- **Ensemble:** XGBoost + CatBoost + LightGBM
- **Fairness Metric:** Stratified Concordance Index (Mean − SD across racial groups)
- **RMSE:** 0.2757

🔗 [GitHub](https://github.com/Qamar-usman-ai/Survival-Prediction-1)

---

#### 🛡️ [Skin Cancer Classification (96.59% AUROC)](https://github.com/Qamar-usman-ai/EfficientNet-B0-Achieve-0.965-AUROC-in-Skin-cancer)
> **EfficientNet-B0 | Transfer Learning | Medical Imaging**

- Transfer learning with EfficientNet-B0; weighted BCE loss for class imbalance
- **AUROC: 96.59% | Specificity: 91.76%**

🔗 [GitHub](https://github.com/Qamar-usman-ai/EfficientNet-B0-Achieve-0.965-AUROC-in-Skin-cancer)

---

#### 🫁 [Pneumonia Detection via CNN (94% Accuracy)](https://github.com/Qamar-usman-ai/Pneumonia-Detection-via-CNN-94-Test-Accuracy)
> **CNN | Chest X-Ray | Binary Classification**

CNN classifier for pneumonia vs. normal chest X-rays achieving **94% test accuracy**.

🔗 [GitHub](https://github.com/Qamar-usman-ai/Pneumonia-Detection-via-CNN-94-Test-Accuracy)

---

#### 🏥 Pediatric Sepsis Early Detection (PR-AUC: 0.9675)
> **XGBoost | Imbalanced Learning | Clinical AI**

Early sepsis detection in pediatric ICU patients — 2.07% positive rate handled with strategic undersampling + TF-IDF drug exposure features.

- **Model:** XGBoost + Stratified Group K-Fold
- **PR-AUC: 0.9675 | Accuracy: 91–96%**

---

### ⏰ TIME SERIES & DATA PLATFORMS

---

#### 🛒 [Rohlik Orders Forecasting (3.37% MAPE)](https://github.com/Qamar-usman-ai/Rohlik-Orders-Forecasting-Challenge)
> **XGBoost | Feature Engineering | Streamlit Dashboard**

Daily order forecasting for grocery warehouse operations with advanced time series feature engineering.

- **Features:** Cyclical encoding (sin/cos), TF-IDF holiday analysis, lag features
- **MAPE: 3.37%**
- **Deployed:** Interactive Streamlit dashboard for EDA and model insights

🔗 [GitHub](https://github.com/Qamar-usman-ai/Rohlik-Orders-Forecasting-Challenge)

---

#### 🤖 [AutoML Platform](https://automated-ml.streamlit.app)
> **Automated ML | Bayesian Optimization | Streamlit**

End-to-end automated ML pipeline for classification and regression — no code needed.

🔗 [GitHub](https://github.com/Qamar-usman-ai/Automated-ml) | [Live Demo](https://automated-ml.streamlit.app)

---

#### 📊 [EDA & Data Visualization Platform](https://data-visualization-eda.streamlit.app)
> **Plotly | ReportLab | Automated Analysis**

Comprehensive EDA platform with interactive visualizations and professional PDF report generation.

🔗 [Live Demo](https://data-visualization-eda.streamlit.app)

---

## 🛠️ Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Azure](https://img.shields.io/badge/Azure-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-006400?style=for-the-badge)

</div>

---

## 🏆 Kaggle Achievements

| Medal | Competition | Rank |
|---|---|---|
| 🥈 Silver | Stanford RNA 3D Folding | 57 / 1,516 (Top 4%) |
| 🥈 Silver | Child Mind Institute: PIU | 76 / 3,559 (Top 3%) |
| 🏅 Top 10% | CIBMTR Survival Prediction | 341 / 3,325 |
| 🏅 Top 11% | HMS Brain Activity | 312 / 2,767 |

**🌍 Global Rank: 807 / 202,646 — Top 0.4% Worldwide**

---

## 📜 Certifications

- 🏅 TensorFlow Developer Certificate — Google
- 🏅 IBM Data Science Professional Certificate — IBM / Coursera
- 🏅 Machine Learning Specialization — Stanford / Andrew Ng

---

<div align="center">

*"Advancing AI through competitive excellence and meaningful real-world applications"*

![Visitor Count](https://komarev.com/ghpvc/?username=Qamar-usman-ai&color=blue&style=for-the-badge)

</div>
