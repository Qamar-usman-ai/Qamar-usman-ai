# 👋 Hi, I'm Qamar Usman

## 💫 About Me
I am a **research-driven Machine Learning Engineer** with a strong foundation in **Mathematics**, specializing in:

- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)
- Time Series Forecasting
- Computer Vision
- Medical & Healthcare AI
- Deep Learning and Transformer-based architectures

My work combines **rigorous ML engineering**, **research experimentation**, and **practical deployment of real-world AI systems**.

🏢 **Currently working at:** **VFIXALL**  
🚀 Building **end-to-end production-grade ML pipelines**

---

# 🧪 Research & Project Highlights

---

## 🔬 Medical & Healthcare AI

### 🧠 Child Mind Institute — Problematic Internet Use Prediction  
**🥈 Kaggle Silver Medal (Top 3%) | Rank: 76 / 3,559 Teams**

#### 📌 Overview
Developed a machine learning solution to detect **early signs of problematic internet usage in youth**. This research leverages **physical activity and fitness data** as proxies for mental health indicators, bypassing traditional clinical barriers to enable early intervention for **depression and anxiety**.

#### 🛠️ Technical Approach
- **Model:** LightGBM Regressor  
- **Validation:** 7-Fold Stratified K-Fold Cross-Validation  
- **Optimization:** Nelder–Mead method for threshold tuning  
- **Metric:** Quadratic Weighted Kappa (QWK)  
- **Final Score:** **0.463 QWK**, demonstrating strong generalization on unseen data

🔗 **Resources**
- Kaggle Notebook — Full Implementation & Research

---

## 🧬 Structural Bioinformatics & Deep Learning

### 🧪 Stanford RNA 3D Folding Challenge  
**🥈 Kaggle Silver Medal (Top 4%) | Rank: 57 / 1,516 Teams**

#### 📌 Project Overview
Successfully tackled one of biology’s **grand challenges**: predicting the **3D atomic structure of RNA molecules** directly from their primary sequences. This work contributes to accelerating **RNA-based medicine**, including **cancer immunotherapies** and **CRISPR gene editing**, by uncovering the folds and functions of natural RNA—often referred to as the *dark matter of biology*.

#### 🛠️ Technical Architecture

**Dual-Stage Neural Network Pipeline**

- **RNA Language Model (RNA2nd)**
  - 18-layer encoder transformer
  - Captures long-range dependencies across sequences up to 2,400 nucleotides

- **Structure Prediction Model (MSA2XYZ)**
  - Multi-cycle refinement architecture
  - Converts sequence embeddings into global 3D atomic coordinates
  - Predicts key atoms: **P, C4', N1/N9**

- **Ensemble Strategy**
  - Integrated **20+ model variants**
  - Ensures structural diversity, robustness, and accuracy

#### ⚙️ Optimization & Physics-Based Constraints
- **Segmented Prediction:** Overlapping chunking for long RNA sequences (>480 nt) with mathematical stitching
- **Energy-Based Refinement:** Used **OpenMM** to enforce bond, angle, and stacking energy constraints
- **Evaluation Metric:** Optimized for **TM-score**, prioritizing global structural topology

🔗 **Resources**
- Kaggle Notebook — Silver Medal Solution

---

## 🧠 HMS — Harmful Brain Activity Classification  
**Top 11% | Rank: 312 / 2,767 Teams**

#### 📌 Project Overview
Developed a deep learning pipeline to automatically detect **seizures and harmful brain activity patterns** from EEG signals. The system reduces the manual review burden for critically ill patients, enabling faster neurocritical care decisions and improved epilepsy drug development.

#### 🛠️ Technical Approach
- **Model Architecture:** ResNet18d backbone (timm), adapted for single-channel spectrograms
- **Signal Processing:**
  - Log-normalized EEG spectrograms
  - Standardized 512×512 image transformations
- **Loss Function:** KL Divergence to model soft neurologist labels
- **Training Strategy:**  
  - 5-Fold Cross-Validation  
  - Cosine Annealing Learning Rate Scheduler  
  - 9 training epochs for stable convergence

#### 📊 Key Results
- **Best Fold Test Loss:** 0.56 (KL Divergence)
- **Targets:** Seizure (SZ), LPD, GPD, LRDA, GRDA, Other
- **Impact:** Effectively differentiates complex EEG edge cases with expert disagreement

🔗 **Resources**
- Kaggle Competition — HMS Harmful Brain Activity Classification

---

## 🏥 CIBMTR — Equity in Post-HCT Survival Predictions  
**Top 10% | Rank: 341 / 3,325 Teams**

#### 📌 Project Overview
Developed equitable predictive models for **allogeneic Hematopoietic Cell Transplantation (HCT)** survival. The focus was minimizing predictive bias across **race, socioeconomic status, and geography** to promote fairness and rebuild trust in healthcare AI.

#### 🛠️ Technical Approach
- **Model Ensemble:** XGBoost, CatBoost, LightGBM
- **Fairness Metric:** Stratified Concordance Index (Mean − Std across racial groups)
- **Data Handling:** Synthetic datasets mirroring real-world clinical disparities

#### 📊 Performance Summary

| Model | Val RMSE | Val MSE | Status |
|-----|---------|---------|-------|
| LightGBM | 0.2790 | 0.0778 | Slight Overfitting |
| XGBoost | 0.2765 | 0.0764 | High Generalization |
| CatBoost | 0.2765 | 0.0764 | High Generalization |
| **Ensemble** | **0.2757** | **0.0760** | **Best Stability** |

🔗 **Resources**
- GitHub: https://github.com/Qamar-usman-ai/Survival-Prediction-1  
- Kaggle Competition — CIBMTR

---

## 🛡️ Skin Cancer Classification — EfficientNet-B0  
**State-of-the-Art Diagnostic Performance | AUROC: 96.59%**

#### 🧠 Project Overview
Developed a high-precision deep learning system for **Malignant vs Benign skin lesion classification**, combining two major medical imaging datasets to improve early cancer detection while reducing unnecessary biopsies.

#### 🛠️ Technical Architecture
- **Core Model:** EfficientNet-B0 (ImageNet pre-trained)
- **Fine-Tuning:** Last 3 blocks + custom classifier head
- **Imbalance Handling:** Weighted BCE loss (pos_weight = 1.14)
- **Augmentation:** Albumentations (Flip, Rotation, Brightness/Contrast)
- **Training:** FP16 mixed precision on Tesla T4 GPUs

#### 📊 Test Metrics (2,660 Images)

| Metric | Value |
|------|------|
| AUROC | **96.59%** |
| Accuracy | 90.19% |
| Recall | 88.54% |
| Specificity | 91.76% |

#### ⚕️ Clinical Implications
High specificity significantly reduces false positives and unnecessary biopsies while maintaining strong malignant detection.

🔗 **Resources**
- GitHub: https://github.com/Qamar-usman-ai  
- Streamlit Web App  
- Kaggle Research Documentation

---

## 🫁 Pneumonia Detection via Custom CNN  
**Test Accuracy: 94.01%**

#### 🧠 Overview
Built an automated diagnostic system for Pneumonia detection from chest X-rays using a **custom CNN** trained on merged multi-source datasets.

#### 🛠️ Technical Details
- 5 Convolutional Blocks with BatchNorm & Dropout
- Image size: 150×150 (grayscale)
- Optimizer: RMSprop
- Scheduler: ReduceLROnPlateau
- Augmentation: ImageDataGenerator

#### 📊 Results (2,420 Test Images)

| Metric | Value |
|------|------|
| Accuracy | 94.01% |
| Precision | 0.96 |
| Recall | 0.95 |
| F1-Score | 0.94 |

#### 🚀 Deployment
- **Backend:** FastAPI
- **Frontend:** Web-based image upload
- **Inference:** uvicorn app:app

🔗 **Resources**
- GitHub: https://github.com/Qamar-usman-ai/Pneumonia-Detection-via-CNN-94-Test-Accuracy  
- Kaggle Notebook

---

## 🏥 Pediatric Sepsis Early Detection  
**PR-AUC: 0.9675 | PHEMS Hackathon**

- Dataset: 331,639 time points from 2,649 patients
- Extreme imbalance handling (2.07% positives)
- TF-IDF drug exposure features
- XGBoost with Stratified Group K-Fold CV

🔗 **Resources**
- GitHub: https://github.com/Qamar-usman-ai

---

## 🤖 Large Language Models & RAG

### 🛡️ PII Detection in Student Writing (NER)
- DeBERTa-v3 ensemble (Piiranha)
- Softmax-weighted inference
- Optimized for **Micro F5-Score**
- High recall for student safety

### 📐 Math Misconception Classification
- Ettin-Encoder-400M
- MAP@3: **0.9428**
- 65 pedagogical misconception classes

---

## 📊 Time Series Forecasting & Analytics

### 🛒 Rohlik Orders Forecasting Challenge
- XGBoost forecasting
- MAPE: **3.37%**
- Advanced feature engineering
- Streamlit dashboard

🔗 GitHub: https://github.com/Qamar-usman-ai/Rohlik-Orders-Forecasting-Challenge

---

## 🤖 Automated ML Platforms
- AutoML for Classification & Regression
- Bayesian Hyperparameter Optimization
- Streamlit-based deployment

🔗 GitHub: https://github.com/Qamar-usman-ai

---

## 🛠️ Tech Stack

### Languages & Frameworks
Python · TensorFlow · PyTorch · Keras · Transformers · FastAPI  
Scikit-learn · XGBoost · LightGBM · CatBoost  

### ML Domains
LLMs · RAG · NLP · Computer Vision · Medical AI  
Time Series Forecasting · AutoML · Deep Learning  

### Tools
Git · GitHub Actions · Docker · Streamlit · MLflow  
NumPy · Pandas · Plotly · SciPy  

---

## 📫 Connect With Me
🌐 GitHub: https://github.com/Qamar-usman-ai

⭐ If you find my work useful, feel free to star the repositories!
