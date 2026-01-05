<h1 align="center">Qamar Usman</h1>
<h3 align="center">Machine Learning Engineer | LLMs • Transformers • Medical AI • Time Series</h3>

<p align="center">
  <a href="https://github.com/Qamar-usman-ai?tab=repositories">
    <img src="https://img.shields.io/badge/Kaggle-Expert-blue?style=for-the-badge&logo=kaggle" alt="Kaggle Expert">
  </a>
  <a href="https://www.kaggle.com/qamarmath">
    <img src="https://img.shields.io/badge/Top-0.4%25-purple?style=for-the-badge" alt="Top 0.4%">
  </a>
  <img src="https://img.shields.io/badge/Projects-17+-orange?style=for-the-badge" alt="Projects">
  <img src="https://img.shields.io/badge/Medals-2_Silver-yellow?style=for-the-badge" alt="Silver Medals">
</p>

---

## 📊 GitHub Stats

<p align="center">
  <img src="https://github-readme-stats.vercel.app/api?username=Qamar-usman-ai&show_icons=true&theme=radical&count_private=true&hide_border=true" alt="GitHub Stats" height="180"/>
  <img src="https://github-readme-streak-stats.herokuapp.com/?user=Qamar-usman-ai&theme=radical&hide_border=true" alt="GitHub Streak" height="180"/>
  <img src="https://github-readme-stats.vercel.app/api/top-langs/?username=Qamar-usman-ai&layout=compact&theme=radical&hide_border=true" alt="Top Languages" height="180"/>
</p>

---

## 👨‍💻 About Me

I am a research-driven Machine Learning Engineer with a strong foundation in Mathematics, specializing in:

- 🤖 **Large Language Models (LLMs)** & **Retrieval-Augmented Generation (RAG)**
- 📈 **Time Series Forecasting**
- 👁️ **Computer Vision**
- 🏥 **Medical & Healthcare AI**
- 🏗️ **Deep Learning and Transformer-based architectures**

My work combines rigorous ML engineering, research experimentation, and practical deployment of real-world AI systems. I currently work at **VFIXALL**, building end-to-end production ML pipelines.

---

## 🧪 Research & Project Highlights

### 🔬 Medical & Healthcare AI

#### **🧠 Child Mind Institute: Problematic Internet Use Prediction**
**🥈 Kaggle Silver Medal (Top 3%) | Rank: 76 / 3,559 Teams**

**Overview:** Developed a machine learning solution for the Child Mind Institute to detect early signs of problematic internet usage in youth. This research uses accessible physical activity and fitness data as proxies for mental health indicators, enabling early intervention for depression and anxiety.

**Technical Approach:**
- **Model:** LightGBM Regressor with 7-Fold Stratified K-Fold validation
- **Optimization:** Nelder-Mead method for threshold tuning to maximize Quadratic Weighted Kappa (QWK)
- **Performance:** Achieved Final QWK of 0.463, proving high generalization on unseen fitness data

**Resources:** [Kaggle Notebook](https://www.kaggle.com/code/qamarmath/ensemble-models)

---

#### **🧬 Stanford RNA 3D Folding Challenge**
**🥈 Kaggle Silver Medal (Top 4%) | Rank: 57 / 1,516 Teams**

**Project Overview:** Solved one of biology's "grand challenges" by predicting 3D atomic coordinates of RNA molecules from primary sequences. This accelerates RNA-based medicine including cancer immunotherapies and CRISPR gene editing.

**Technical Architecture:**
- **Dual-Stage Pipeline:** RNA2nd (18-layer Transformer) + MSA2XYZ (3D coordinate generator)
- **Ensemble Strategy:** 20-model ensemble for structural diversity and accuracy
- **Physics Integration:** OpenMM energy minimization for thermodynamic stability
- **Evaluation:** Optimized for TM-score focusing on global topology

**Resources:** [Kaggle Notebook](https://www.kaggle.com/code/qamarmath/maximizing-variance-ensemble-pipeline)

---

#### **🧠 HMS: Harmful Brain Activity Classification**
**🏅 Top 11% | Rank: 312 / 2,767 Teams**

**Project Overview:** Deep learning pipeline to automate detection of seizures and harmful brain activity patterns from EEG signals for neurocritical care applications.

**Technical Approach:**
- **Model Architecture:** ResNet18d backbone modified for single-channel spectrogram data
- **Signal Processing:** Normalized logarithmic representations with custom transformations
- **Loss Function:** Kullback-Leibler Divergence for expert neurologist "soft labels"
- **Training:** 5-Fold Cross-Validation with Cosine Annealing scheduler

**Impact:** Enables faster neurocritical care and accurate epilepsy drug development.

---

#### **🏥 CIBMTR: Equity in post-HCT Survival Predictions**
**🏅 Top 10% | Rank: 341 / 3,325 Teams**

**Project Overview:** Predictive models for Hematopoietic Cell Transplantation survival rates with focus on equitable outcomes across demographic groups.

**Technical Approach:**
- **Model Ensemble:** XGBoost + CatBoost + LightGBM combination
- **Fairness Metric:** Stratified Concordance Index (Mean - SD across racial groups)
- **Performance:** Ensemble achieved best stability with RMSE 0.2757

**Resources:** [GitHub Repository](https://github.com/Qamar-usman-ai/Survival-Prediction-1)

---

#### **🛡️ Skin Cancer Classification: EfficientNet-B0**
**State-of-the-Art Performance | 96.59% AUROC**

**Technical Architecture:**
- **Core Model:** EfficientNet-B0 with Transfer Learning
- **Imbalance Handling:** Weighted BCE loss and Stratified K-Fold validation
- **Augmentation:** Geometric and color-space regularization
- **Performance:** 96.59% AUROC, 91.76% specificity

**Resources:** [GitHub Repository](https://github.com/Qamar-usman-ai/EfficientNet-B0-Achieve-0.965-AUROC-in-Skin-cancer)

---

#### **🏥 Pediatric Sepsis Early Detection**
**PHEMS Hackathon | PR-AUC: 0.9675**

**Technical Approach:**
- **Handling Imbalance:** Strategic undersampling for 2.07% sepsis cases
- **Feature Engineering:** Drug exposure (TF-IDF), temporal dynamics
- **Model:** XGBoost with Stratified Group K-Fold Cross-Validation
- **Performance:** 0.9675 PR-AUC, 91-96% accuracy

**Resources:** [GitHub Repository](https://github.com/Qamar-usman-ai/Early-Sepsis-Detection-Model)

---

### 🤖 Large Language Models & RAG

#### **🛡️ PII Detection in Student Writing (NER)**
**Automated Data Anonymization for Educational Science**

**Technical Architecture:**
- **Model Ensemble:** "Piiranha" ensemble of three DeBERTa-v3 models
- **Weighted Inference:** Softmax Weighted Average approach
- **Token Classification:** Custom token alignment for complex formatting
- **Optimization:** Micro F5-Score focus (recall-weighted)

**Impact:** Enables educational research while protecting student privacy.

---

#### **📐 Math Misconception Classification**
**High-Precision Educational Tool | MAP@3: 0.9428**

**Technical Architecture:**
- **Foundation Model:** jhu-clsp/ettin-encoder-400m (mathematics-specialized)
- **Context Engineering:** Structured inputs with answer key logic
- **Optimization:** 3 epochs with FP16 Mixed Precision
- **Performance:** Achieved 0.9428 MAP@3

---

### 📚 AI-Powered Document Analysis Tools

#### **Chat with PDFs or Websites** 📚💬
**AI-Powered Document Analysis Tool**

**Features:** Upload PDFs or input website URLs to query content using Gemini, Gemma, or OpenAI models with semantic search and chat history.

**Tech Stack:** Streamlit, LangChain, FAISS  
**Live Demo:** [chat-with-pdfs.streamlit.app](https://chat-with-pdfs.streamlit.app/)

---

#### **Chat with Your Data** 📊💬
**AI-Powered Data Analysis Tool**

**Features:** Interact with CSV, Excel, and SQL data using natural language queries powered by Google Generative AI.

**Tech Stack:** Streamlit, LangChain  
**Live Demo:** [csv-chat.streamlit.app](https://csv-chat.streamlit.app/)

---

### ⏰ Time Series Forecasting & Data Analysis

#### **Rohlik Orders Forecasting Challenge** 🛒📈
**Time Series Forecasting | 3.37% MAPE**

**Technical Approach:**
- **Model:** XGBoost with advanced feature engineering
- **Feature Engineering:** Cyclical encoding, TF-IDF holiday analysis
- **Performance:** 3.37% MAPE accuracy
- **Deployment:** Interactive Streamlit dashboard

**Resources:** [GitHub Repository](https://github.com/Qamar-usman-ai/Rohlik-Orders-Forecasting-Challenge)

---

#### **Automated Machine Learning Platform** 🤖⚡
**Streamlit AutoML Platform**

**Features:** Automated end-to-end ML pipeline for classification and regression tasks.

**Tech Stack:** Streamlit, Bayesian Optimization  
**Live Demo:** [automated-ml.streamlit.app](https://automated-ml.streamlit.app/)

---

#### **Data Visualization EDA Platform** 📊🔍
**Automated Data Analysis Platform**

**Features:** Perform comprehensive EDA, generate interactive visualizations, and create professional PDF reports.

**Tech Stack:** Streamlit, Plotly, ReportLab  
**Live Demo:** [data-visualization-eda.streamlit.app](https://data-visualization-eda.streamlit.app/)

---

## 🛠️ Tech Stack

### **Languages & Frameworks:**
<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/Transformers-FFD700?style=for-the-badge&logo=huggingface&logoColor=black" alt="Transformers">
</p>

<p align="left">
  <img src="https://img.shields.io/badge/Scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/XGBoost-337D98?style=for-the-badge" alt="XGBoost">
  <img src="https://img.shields.io/badge/LightGBM-7BF2E9?style=for-the-badge" alt="LightGBM">
  <img src="https://img.shields.io/badge/CatBoost-FF6B35?style=for-the-badge" alt="CatBoost">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
</p>

### **ML Domains:**
<p align="left">
  <img src="https://img.shields.io/badge/LLMs-Transformers-yellow" alt="LLMs">
  <img src="https://img.shields.io/badge/RAG-Systems-blue" alt="RAG">
  <img src="https://img.shields.io/badge/Computer_Vision-OpenCV-green" alt="CV">
  <img src="https://img.shields.io/badge/Time_Series-Forecasting-orange" alt="Time Series">
  <img src="https://img.shields.io/badge/Medical_AI-Healthcare-red" alt="Medical AI">
</p>

### **Tools & Platforms:**
<p align="left">
  <img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white" alt="Git">
  <img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white" alt="GitHub Actions">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge" alt="MLflow">
</p>

---

## 📈 Project Statistics

| Category | Count | Key Projects |
|----------|-------|--------------|
| **Medical AI** | 6 | RNA Folding, Brain Activity, Sepsis Detection |
| **LLMs & NLP** | 4 | PII Detection, Math Misconception, RAG Systems |
| **Time Series** | 3 | Rohlik Forecasting, AutoML Platform |
| **Production Tools** | 4 | Streamlit Apps, Automated ML |
| **Total Projects** | **17+** | Actively maintained with CI/CD |

---

## 🌐 Connect With Me

<p align="center">
  <a href="https://www.kaggle.com/qamarmath" target="_blank">
    <img src="https://img.shields.io/badge/Kaggle-Profile-blue?style=for-the-badge&logo=kaggle" alt="Kaggle" height="35">
  </a>
  <a href="https://github.com/Qamar-usman-ai" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-000000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" height="35">
  </a>
  <a href="https://linkedin.com/in/qamar-usman-ai" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" height="35">
  </a>
</p>

---

## 🎯 Professional Philosophy

**"Research-Driven Engineering for Real-World Impact"**

- **🔬 Scientific Rigor:** Combining academic research with practical implementation
- **⚡ Production Excellence:** Building scalable, maintainable ML systems
- **🤝 Collaborative Innovation:** Contributing to open-source communities
- **🌍 Positive Impact:** Focusing on healthcare, education, and scientific advancement
- **📈 Continuous Growth:** Learning through competition and collaboration

---

<div align="center">
  
  ![Visitor Count](https://komarev.com/ghpvc/?username=Qamar-usman-ai&color=blue&style=flat-square&label=Profile+Views)
  
  ### *"Advancing AI through competitive excellence and meaningful applications"*
  
  **📍 Location:** Bahāwalnagar, Punjab, Pakistan | Remote & Global Opportunities
  
</div>
