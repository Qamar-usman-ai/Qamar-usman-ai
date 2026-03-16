# 👋 Hi, I'm Qamar Usman

### Machine Learning Engineer | Kaggle Expert | Medical AI · Agentic AI · LLMs · Business Forecasting

[![Kaggle](https://img.shields.io/badge/Kaggle-Expert%20Rank%20735-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://kaggle.com/qamarmath)
[![GitHub followers](https://img.shields.io/github/followers/Qamar-usman-ai?style=social)](https://github.com/Qamar-usman-ai)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/qamar-usman)
[![Profile Views](https://komarev.com/ghpvc/?username=Qamar-usman-ai&color=blue&style=flat)](https://github.com/Qamar-usman-ai)

📍 Pakistan &nbsp;|&nbsp; 🌍 Open to Remote Roles Worldwide &nbsp;|&nbsp; ⏰ Available EST / GMT / CET

---

## 🏆 Competition Achievements

> Ranked in the **Top 0.4% of 203,000+ ML engineers globally** on Kaggle — sustained performance across 36 competitions in healthcare, education, finance, and biology.

| Platform | Competition | Rank | Result |
|----------|------------|------|--------|
| 🥈 **Kaggle** | Stanford RNA 3D Folding | **57 / 1,516** | Top 4% — Silver Medal |
| 🥈 **Kaggle** | Child Mind Institute: PIU | **76 / 3,559** | Top 3% — Silver Medal |
| 🏅 **Kaggle** | CIBMTR Survival Prediction | **341 / 3,325** | Top 10% |
| 🏅 **Kaggle** | HMS Brain Activity (EEG) | **312 / 2,767** | Top 11% |
| 🏅 **Kaggle** | PII Detection in Student Writing | **209 / 2,048** | Top 10% |
| 🏅 **Kaggle** | ECG Digitization (PhysioNet) | **203 / 1,424** | Top 14% |
| 🏅 **Zindi** | Financial Health Prediction | **28 / 900** | Top 3% |
| 🏅 **Zindi** | Barbados Traffic Analysis | **40 / 222** | Top 18% |
| 🏅 **Zindi** | DigiCow Farmer Adoption | **88 / 387** | Top 23% |

**Kaggle Global Rank: 735 / 203,000+ — Top 0.4% Worldwide | 2× Silver Medals | 36 Competitions**

---

## 📂 Complete Project Index

| # | Project | Domain | Key Result |
|---|---------|--------|-----------|
| | **🏥 SECTION 1 — MEDICAL AI** | | |
| 1 | [HMS Brain Activity Classification](#-1-hms-harmful-brain-activity-classification) | Neurology / ICU | Rank 312/2,767 |
| 2 | [Skin Cancer Detection](#-2-skin-cancer-detection) | Oncology | AUROC 96.59% |
| 3 | [Pediatric Sepsis Detection](#-3-pediatric-sepsis-early-detection) | ICU / Emergency | PR-AUC 0.9675 |
| 4 | [ECG Digitization](#-4-ecg-digitization--physionet) | Cardiology | Rank 203/1,424 |
| 5 | [Post-HCT Survival Prediction](#-5-post-hct-survival-prediction) | Oncology / Transplant | Rank 341/3,325 |
| | **🤖 SECTION 2 — AGENTIC AI & LLMs** | | |
| 6 | [Stock Market Analyst Agent](#-6-stock-market-analyst-agent) | Finance / Agentic AI | Autonomous ReAct agent |
| 7 | [Product Research Agent](#-7-product-research-agent) | E-commerce / Agentic AI | Multi-agent research pipeline |
| 8 | [Smart AI Tour Planner](#-8-smart-ai-tour-planner) | Travel / Agentic AI | Personalized itinerary agent |
| 9 | [LinkedIn Viral Post Generator](#-9-linkedin-viral-post-generator-agent) | Content AI / Agents | Multi-agent content pipeline |
| 10 | [Chat with PDFs & Websites (RAG)](#-10-chat-with-pdfs--websites-rag-system) | LLMs / RAG | Production-ready Q&A |
| 11 | [Chat with CSV / SQL (NL2SQL)](#-11-chat-with-your-data--nl2sql-agent) | Data Analysis / LLMs | Natural language → SQL |
| 12 | [AutoML Platform](#-12-automl-platform) | AutoML / MLOps | End-to-end automated ML |
| | **🥇 SECTION 3 — COMPETITION ML** | | |
| 13 | [RNA 3D Structure Prediction](#-13-rna-3d-structure-prediction--silver-medal) | Computational Biology | Rank 57/1,516 — Silver Medal |
| 14 | [Problematic Internet Use](#-14-problematic-internet-use-in-children--silver-medal) | Mental Health AI | Rank 76/3,559 — Silver Medal |
| 15 | [PII Detection in Student Writing](#-15-pii-detection-in-student-writing) | NLP / Privacy | F5 Score 0.953 |
| | **📊 SECTION 4 — BUSINESS FORECASTING** | | |
| 16 | [Rohlik Orders Forecasting](#-16-rohlik-grocery-orders-forecasting) | Supply Chain | 3.37% MAPE |
| 17 | [Financial Health Prediction](#-17-dataorg-financial-health-prediction--zindi) | Fintech / SME | Rank 28/900 — Top 3% |
| 18 | [Barbados Traffic Analysis](#-18-barbados-traffic-analysis--zindi) | Smart Cities | Rank 40/222 — Top 18% |
| 19 | [DigiCow Farmer Adoption](#-19-digicow-farmer-training-adoption--zindi) | AgriTech | Rank 88/387 |
| 20 | [agriBORA Maize Price Forecasting](#-20-agribora-maize-price-forecasting--zindi) | Agriculture / Finance | Weekly price forecasting |

---
---

# 🏥 SECTION 1 — MEDICAL AI

> I build AI systems that solve life-or-death clinical problems. Every project in this section addresses a real gap in healthcare where delayed or incorrect decisions cause serious harm. The goal in each case is the same: give clinicians better information, faster.

---

## 🧠 1. HMS Harmful Brain Activity Classification
**Kaggle Research Competition | Rank 312 / 2,767 Teams (Top 11%)**

### The Problem — A Silent Crisis in Intensive Care

Every year, critically ill patients in ICUs suffer **seizures and dangerous brain activity patterns that go undetected for hours**. Continuous EEG monitoring generates enormous amounts of data — but there are not enough neurologists in the world to review all of it in real time.

The result is devastating:
- A seizure goes unnoticed for 3–4 hours
- Brain damage accumulates with every passing minute
- Treatment begins too late to prevent permanent harm

**This project automates EEG pattern detection so neurologists can focus their attention exactly where patients need it most.**

### What the Model Must Detect

| Pattern | What It Means Clinically |
|---------|--------------------------|
| **Seizure** | Uncontrolled electrical brain activity — requires immediate treatment |
| **LPD** | Lateralized Periodic Discharges — associated with seizure risk |
| **GPD** | Generalized Periodic Discharges — linked to poor patient outcomes |
| **LRDA** | Lateralized Rhythmic Delta Activity — may indicate focal brain injury |
| **GRDA** | Generalized Rhythmic Delta Activity — associated with encephalopathy |
| **Other** | Normal or non-harmful patterns |

### Why This Is Hard
Raw EEG is a sequence of voltage measurements from up to 19 electrodes sampled at 256 Hz. A single 10-minute segment contains millions of data points. The signal is extremely noisy — motion artifacts, electrode contact issues, and electrical interference all contaminate the recording. And critically, **expert neurologists do not always agree** on the correct label — so the "ground truth" is a soft probability distribution, not a hard label.

### My Solution — ResNet18d on Log-Normalized Spectrograms

```
Raw EEG Signal
(19 electrodes × time, sampled at 256 Hz)
              │
              ▼
Convert to Spectrogram:
  → Short-Time Fourier Transform (STFT)
  → Log-normalize the frequency amplitudes
  → Output: 2D image (frequency × time)
  → Why? Seizure patterns have characteristic
    VISUAL signatures in frequency space that
    a CNN can detect far better than a 1D signal model
              │
              ▼
ResNet18d (pre-trained, fine-tuned):
  Block 1: Conv2D(64) → BatchNorm → ReLU → MaxPool
  Block 2: Residual blocks (64 → 128 filters)
  Block 3: Residual blocks (128 → 256 filters)
  Block 4: Residual blocks (256 → 512 filters)
  → Global Average Pooling
  → Fully Connected → 6 output classes
              │
              ▼
KL Divergence Loss
  (NOT cross-entropy — because labels are soft
   probability distributions from multiple
   neurologist opinions, not hard 0/1 labels.
   KL divergence is mathematically correct here.)
              │
              ▼
5-Fold Cross-Validation + Cosine Annealing LR
              │
              ▼
Output: Probability distribution over 6 brain activity classes
```

### Key Technical Decisions Explained

**Why spectrograms instead of raw signals?**
Raw EEG sequences are long and noisy. CNNs trained on spectrograms benefit from decades of image classification research. More importantly, seizure patterns manifest as specific visual signatures in the frequency domain — rhythmic bursts at characteristic frequencies — that a ResNet can detect reliably.

**Why KL Divergence loss?**
The training labels were collected by having multiple expert neurologists independently label each EEG segment. Their labels were averaged into a probability vector (e.g., [0.7 seizure, 0.2 LPD, 0.1 other]). Cross-entropy loss assumes hard labels (one class = 1.0, rest = 0.0). KL Divergence correctly measures the difference between two probability distributions — exactly what this problem requires.

**Why Cosine Annealing?**
Cosine annealing gradually reduces the learning rate following a cosine curve, which allows the model to make large updates early in training and fine-grained adjustments later — consistently outperforming fixed learning rates on this type of problem.

### Results
| Metric | Value |
|--------|-------|
| Competition Rank | **312 / 2,767 (Top 11%)** |
| Loss Function | KL Divergence |
| Architecture | ResNet18d |
| Training | 5-Fold CV + Cosine Annealing |
| Competition Host | Harvard Medical School + Kaggle |

### Real-World Impact
Deployed in an ICU, a system like this could screen EEG streams in real time and alert neurologists to suspicious segments within seconds. **Seizures detected in minutes instead of hours** means less brain damage, shorter ICU stays, and better patient outcomes.

---

## 🔬 2. Skin Cancer Detection
**Personal Project | AUROC: 96.59% | Specificity: 91.76%**

### The Problem — The Deadliest Cancer You Can See

Skin cancer is the most common cancer in the world — **over 5 million new cases per year** in the US alone. Melanoma, the deadliest form, has a 98%+ survival rate when caught at Stage 1, but below 20% when caught at Stage 4.

The gap between those two outcomes is simply: **time to detection**.

Most people never visit a dermatologist for routine skin checks. Even when they do, trained dermatologists misdiagnose suspicious lesions 20–30% of the time on visual inspection alone. And in low-income countries, dermatologists are simply not available.

**The opportunity:** Smartphones now have cameras capable of capturing medically usable dermoscopic images. An AI screening tool deployed as an app could flag high-risk lesions for urgent specialist referral — bringing early detection to anyone with a phone.

### The Core Technical Challenge — Extreme Class Imbalance

In the real world, only about 5% of examined skin lesions are malignant. This creates a dangerous trap: a model that predicts "benign" for every single case achieves 95% accuracy — but has zero clinical value and would let every cancer through undetected.

Standard accuracy is the wrong metric. The right question is: **across all cases where the model assigns a higher risk score to a malignant case vs a benign case, how often does it rank them correctly?** That is AUROC — and it is immune to class imbalance.

### My Solution — Transfer Learning with EfficientNet-B0

```
Input: Dermoscopic skin lesion image (RGB, any resolution)
              │
              ▼
Preprocessing Pipeline:
  → Resize to 224×224 pixels
  → Normalize (ImageNet mean=[0.485,0.456,0.406],
               std=[0.229,0.224,0.225])
  → Training augmentation:
     Random horizontal + vertical flip
     Random rotation (±30°)
     Color jitter (brightness, contrast, saturation)
     Random zoom (0.8–1.2×)
  → Augmentation forces the model to be invariant
    to orientation, lighting, and scale — which
    dermatoscopes vary in real clinical use
              │
              ▼
EfficientNet-B0 (pre-trained on ImageNet):
  → ImageNet pre-training means the model already
    knows edges, textures, colors, and shapes
  → Fine-tune all layers on ISIC dermoscopy dataset
  → Replace final classification head:
    1280 features → Dropout(0.3) → Linear(1)
              │
              ▼
Weighted Binary Cross-Entropy Loss:
  → Malignant class weighted 10× more than benign
  → Because missing one cancer is far worse than
    one unnecessary biopsy referral
              │
              ▼
Output: Malignancy probability score (0.0 → 1.0)
Threshold optimized for sensitivity ≥ 0.95
(catch at least 95% of all cancers)
```

### Why EfficientNet over ResNet or VGG?

EfficientNet simultaneously scales network depth, width, and input resolution using a compound scaling method. For the same accuracy, it requires 8× fewer parameters and is 6× faster than comparable ResNet architectures. In a mobile screening app where inference must run on a phone, this matters enormously.

### Results
| Metric | Value |
|--------|-------|
| **AUROC** | **96.59%** |
| **Specificity** | 91.76% |
| Architecture | EfficientNet-B0 |
| Dataset | ISIC 2020 (33,000+ images) |
| Framework | PyTorch |

---

## 🚨 3. Pediatric Sepsis Early Detection
**Personal Project | PR-AUC: 0.9675 | Pediatric ICU Clinical Data**

### The Problem — The Golden Hour That Kills

Sepsis is the body's catastrophic response to infection — when the immune system turns against its own organs. It is **the leading cause of preventable death in hospitals worldwide**, killing over 11 million people every year. In children, it progresses faster and is more deadly than in adults.

The critical fact: **for every hour that antibiotic treatment is delayed after sepsis onset, mortality increases by approximately 7%.** The first hour of treatment — the "golden hour" — is the difference between survival and death.

The problem: early sepsis looks exactly like a dozen other conditions. Fever, elevated heart rate, and falling blood pressure are common to many illnesses. A busy ICU nurse managing 6 patients simultaneously cannot continuously monitor every vital sign pattern for every patient and catch the subtle early signals that indicate sepsis is beginning.

**This project builds an automated early warning system that detects sepsis risk from routine clinical measurements — alerting staff hours before the condition becomes critical.**

### The Data Reality — Why This Problem Is Hard

Before building any model, I analyzed the data carefully:

```python
# Understanding the class imbalance problem:
# Only 2.07% of patient-hours are positive (sepsis onset)
# This is the fundamental challenge

positive_rate = train_df['sepsis_label'].mean()
print(f"Positive rate: {positive_rate:.2%}")
# Output: Positive rate: 2.07%

# Missing value analysis — ICU data is always incomplete:
# Lab tests are ordered when clinically needed, not every hour
# The pattern of missing tests is itself a clinical signal

for col in clinical_cols:
    missing_pct = train_df[col].isnull().mean() * 100
    print(f"{col}: {missing_pct:.1f}% missing")
```

**Key data challenges I had to solve:**
- **2.07% positive rate** — extreme imbalance. A naive model predicts "no sepsis" for everything and gets 98% accuracy while being clinically useless.
- **High missingness** — lab values like creatinine and lactate are only measured when ordered. Missing values follow clinical patterns that encode information.
- **Patient grouping** — standard K-Fold leaks data. If a patient's hour 3 is in training and hour 4 is in validation, the model memorizes patient patterns, not generalizable clinical signals.
- **Free-text drug data** — medication exposure is recorded as text. Needs intelligent encoding.

### My Complete Solution

```
Raw ICU Data:
(Vitals: HR, MAP, SpO2, Temp, RR per hour)
(Labs: WBC, Creatinine, Lactate, Bilirubin)
(Drugs: Antibiotics, Vasopressors — free text)
              │
              ▼
Feature Engineering:

  Temporal Rolling Features:
  → Mean, std, min, max, trend slope over
    3-hour, 6-hour, 12-hour windows
  → "Is the patient getting worse?" signals

  Missingness Features:
  → % of labs missing in last 6 hours
  → Was lactate ordered? (Doctors order lactate
    when they suspect sepsis — ordering pattern
    = clinical suspicion signal)

  Drug Encoding:
  → TF-IDF on medication names
  → Antibiotic started = strong sepsis suspicion signal
  → Vasopressor started = hemodynamic failure signal

  Clinical Score Features:
  → SOFA score components
  → Quick SOFA (qSOFA) indicator
  → Shock index (HR / Systolic BP)
              │
              ▼
Class Imbalance Strategy:
  → Random undersampling majority class to 10:1 ratio
  → Preserves enough negatives for boundary learning
  → Avoids synthetic data artifacts from oversampling
              │
              ▼
XGBoost Classifier:
  → Stratified Group K-Fold CV
    (groups = patient IDs — complete patient
     isolation between train and validation)
  → scale_pos_weight tuned for residual imbalance
  → Early stopping on PR-AUC
              │
              ▼
Threshold optimization:
  → Maximize F2 score (recall-weighted)
  → Clinical requirement: catch ≥95% of sepsis cases
  → Accept higher false alarm rate as the price
    of missing no true cases
              │
              ▼
Output: Sepsis risk score per patient per hour
        Alert generated when score > optimized threshold
```

### Why PR-AUC Instead of ROC-AUC?

ROC-AUC looks impressive on imbalanced datasets even for bad models, because the huge number of true negatives inflates the true negative rate. PR-AUC (Precision-Recall AUC) shows how well the model performs specifically on the minority class — the sepsis cases that actually matter. At **PR-AUC = 0.9675**, the model maintains extremely high precision even as recall approaches 1.0.

### Results
| Metric | Value |
|--------|-------|
| **PR-AUC** | **0.9675** |
| Accuracy | 91–96% |
| Positive Rate | 2.07% |
| Model | XGBoost + Stratified Group K-Fold |
| Framework | Scikit-learn + XGBoost |

---

## ❤️ 4. ECG Digitization — PhysioNet
**Kaggle Research Competition | Rank 203 / 1,424 Teams (Top 14%)**

### The Problem — A Billion Heartbeats Locked in Paper

The entire cardiology history of the 20th century exists only on paper. Billions of ECG recordings — lifetime cardiac histories, rare arrhythmia patterns, longitudinal disease studies — are stored as yellowed printouts in filing cabinets across the world's hospitals. They cannot be analyzed by AI. They cannot be shared digitally. Every day, some are lost to water damage, fire, or simple decay.

Getting this data into digital form would:
- Enable AI training on 60+ years of real cardiac history
- Allow patients to share their complete cardiac record with any doctor worldwide
- Unlock research into long-term heart disease progression that is currently impossible

**This competition asked: can a computer vision system automatically extract the precise ECG waveform from a photograph of a paper printout?**

### Why This Is Genuinely Hard

This is not a simple scan-to-text problem:
- Paper records are **physically degraded** — yellowed, stained, wrinkled, photographed at angles
- **Grid lines overlap the signal** and must be removed without damaging the waveform
- **Multiple leads** (up to 12 independent channels) must be correctly identified and separated
- **Handwritten annotations** partially obscure signal segments
- Signal **amplitude must be calibrated to real units** (millivolts) using a reference pulse
- **Broken signal segments** from fold creases must be intelligently reconnected

### My Solution — Computer Vision + Signal Processing Pipeline

```
Input: Photograph or scan of paper ECG printout
              │
              ▼
Stage 1 — Image Preprocessing:
  → Perspective correction (deskew tilted photos)
  → Adaptive thresholding (handle yellowing/staining)
  → Grid line removal (morphological operations:
    detect horizontal + vertical periodic lines,
    subtract them from the image)
  → Lead boundary detection (find where each of
    the 12 leads begins and ends)
              │
              ▼
Stage 2 — Waveform Extraction per Lead:
  → Binarize: signal pixels vs background
  → Skeletonize: reduce signal to 1-pixel-wide centerline
  → Gap filling: reconnect segments broken by fold lines
  → Extract: y-coordinate (voltage proxy) at each x (time)
              │
              ▼
Stage 3 — Physical Calibration:
  → Detect calibration pulse (standard 1mV, 0.2s reference
    square wave present on every ECG recording)
  → Compute pixel-to-voltage scale factor
  → Detect time grid lines → compute pixel-to-time scale
  → Apply scales: pixel coordinates → mV vs seconds
              │
              ▼
Output: Structured digital ECG time-series
  → Same format as modern digital ECG machines
  → One array per lead: voltage (mV) vs time (s)
  → Ready for downstream AI analysis or storage
```

### Results
| Metric | Value |
|--------|-------|
| Competition Rank | **203 / 1,424 (Top 14%)** |
| Competition Host | PhysioNet / Kaggle |
| Task | Waveform extraction from images |
| Framework | OpenCV + NumPy + SciPy |

---

## 🔬 5. Post-HCT Survival Prediction
**Kaggle Research Competition | Rank 341 / 3,325 Teams (Top 10%)**

### The Problem — The Last Hope With No Roadmap

Hematopoietic Cell Transplantation (HCT) — stem cell transplant — is the last treatment option for thousands of patients with leukemia, lymphoma, and other blood cancers when all other treatments have failed. It is also one of the most dangerous medical procedures, with mortality rates of 20–40% in the first year from complications including graft failure, infection, and organ damage.

**The clinical problem:** Doctors currently have limited tools to predict which patients will survive and which will not. Decisions about who receives a transplant, which donor to choose, and how aggressively to condition the patient are made largely on experience and broad statistics — not personalized prediction.

**The equity problem:** Research showed that existing survival prediction models performed well for white patients but significantly worse for Black, Hispanic, and other minority patients — a life-and-death disparity caused by historical underrepresentation in training data.

**My goal:** Build a survival prediction model that is both accurate AND equitable — performing consistently well across all racial groups.

### My Solution — Ensemble with Fairness-Aware Evaluation

```
Clinical Input Features:
  → Diagnosis (leukemia type, disease stage)
  → Donor information (related/unrelated, HLA match)
  → Conditioning regimen (intensity, drugs used)
  → Patient demographics (age, sex, race, weight)
  → Prior treatments (number, types, responses)
  → Transplant center characteristics
              │
              ▼
Feature Engineering:
  → Word2Vec embeddings for medical codes
    (diagnosis codes and drug names encode
     clinical relationships — "AML" and "ALL"
     are both leukemias and should be
     near each other in feature space)
  → Age × conditioning intensity interaction
    (elderly patients tolerate less aggressive
     conditioning — this interaction is clinically
     important but not captured by either feature alone)
  → Missing value imputation using clinical logic
    (not mean imputation — use disease-specific medians)
              │
              ▼
Three-Model Ensemble:
  ┌─────────────┬─────────────┬─────────────┐
  │  XGBoost    │  CatBoost   │  LightGBM   │
  │             │             │             │
  │ Handles     │ Native cat  │ Fast, L1/L2 │
  │ mixed       │ encoding,   │ regularized │
  │ feature     │ great on    │ boosting,   │
  │ types well  │ high-card   │ low memory  │
  │             │ categoricals│             │
  └──────┬──────┴──────┬──────┴──────┬──────┘
         └─────────────┴─────────────┘
                       │
                       ▼
         Weighted average ensemble
         (weights optimized on validation set)
                       │
                       ▼
Fairness-Aware Evaluation:
  Standard metric: Concordance Index (C-index)
  → measures how often model correctly ranks
    higher-risk patient above lower-risk patient

  Fairness metric: Stratified C-index
  → compute C-index separately per racial group
  → Final score = Mean(per-group C-index)
                  − SD(per-group C-index)
  → Penalizes models that perform well on average
    but poorly for specific demographic groups
  → Forces the model to be consistent for EVERYONE
```

### Why Word2Vec for Medical Codes?

One-hot encoding treats every diagnosis code as completely independent. But "AML" (Acute Myeloid Leukemia) and "ALL" (Acute Lymphoblastic Leukemia) are both leukemias, treated similarly, with similar prognoses. Word2Vec learns a continuous representation where clinically similar codes have similar vectors — giving the model better signal than sparse one-hot encodings.

### Results
| Metric | Value |
|--------|-------|
| Competition Rank | **341 / 3,325 (Top 10%)** |
| RMSE | 0.2757 |
| Fairness Metric | Stratified Concordance Index |
| Ensemble | XGBoost + CatBoost + LightGBM |
| Competition Host | CIBMTR / Kaggle |

---
---

# 🤖 SECTION 2 — AGENTIC AI & LLMs

> I build AI systems that **act autonomously**, not just answer questions. These projects use LLMs as reasoning engines inside multi-step pipelines that plan, use tools, and deliver results with minimal human intervention. Each project below solves a real productivity problem that previously required hours of expert manual work.

---

## 📈 6. Stock Market Analyst Agent
**Personal Project | Autonomous Multi-Step ReAct Agent**

### The Problem — Research That Takes 8 Hours, Every Day

Professional stock analysis requires reading dozens of news articles, checking price history, analyzing technical indicators, and synthesizing everything into a coherent view. A skilled analyst spends **4–8 hours per stock**. For individual investors, small funds, and anyone without a dedicated research team, this level of analysis is simply out of reach.

**The goal:** An AI agent that performs the complete research workflow — from a single text query to a finished, professional PDF report — fully autonomously.

### What Makes This "Agentic" — Not Just a Chatbot

A chatbot answers one question at a time, only knows what it was trained on, and forgets everything between turns. An agent **plans a multi-step workflow, calls real tools, observes results, and decides what to do next**:

```
User: "Analyze NVIDIA stock this week"
              │
              ▼
ReAct Agent Loop (Reason → Act → Observe → Repeat):

  Iteration 1:
  THINK: "I need current news about NVIDIA"
    ACT: Web search tool → "NVIDIA stock news March 2026"
    OBSERVE: 12 articles returned, earnings beat estimates,
             new GPU partnership announced

  Iteration 2:
  THINK: "I need price and volume data"
    ACT: Financial API tool → NVDA OHLCV last 7 days
    OBSERVE: +8.3% weekly gain, volume 2.1× average

  Iteration 3:
  THINK: "I need analyst consensus"
    ACT: Web search → "NVIDIA analyst ratings price target"
    OBSERVE: 38 Buy, 4 Hold, 0 Sell | Mean target $1,240

  Iteration 4:
  THINK: "I have enough information to write the report"
    ACT: LLM synthesis → structured analysis
    ACT: PDF generator → formatted report
    STOP.
              │
              ▼
Output: Professional PDF report (~90 seconds total)
```

### Hallucination Prevention — The Most Critical Design Decision

LLMs sometimes invent financial data. A report with a fabricated stock price or a made-up earnings figure could cause real financial harm.

My solution: **strict separation of data retrieval and data interpretation**.
- All numbers (prices, volumes, returns, analyst ratings) come exclusively from API calls — never from LLM generation
- The LLM's role is only to interpret, synthesize, and explain the data
- Every numerical claim in the report is traceable to a specific tool call result

### Tech Stack
`Python` · `LangChain Agents` · `OpenAI GPT-4` · `ReAct Framework` · `Web Search Tool` · `Financial APIs` · `ReportLab PDF`

🔗 **[View Project](https://github.com/Qamar-usman-ai/stock-market-analyst)**

---

## 🔍 7. Product Research Agent
**Personal Project | Multi-Agent Product Intelligence Pipeline**

### The Problem — E-Commerce Research That Takes Hours

Before launching any product — whether on Amazon, Shopify, or a local e-commerce platform — a seller needs to answer dozens of questions: Who are the top competitors? What do customers complain about in reviews? What price points dominate? What features do top-ranking products share? What gap in the market exists?

Gathering all of this manually means hours of browsing competitor pages, reading hundreds of reviews, and synthesizing everything into a business decision. **Most small sellers skip this research entirely** — and launch products that fail because they didn't understand the competitive landscape.

**The goal:** An AI agent that conducts comprehensive product market research autonomously — delivering competitor analysis, review sentiment, pricing insights, and market gap identification in minutes.

### How the Multi-Agent Pipeline Works

```
User Input: "Research the wireless earbuds market
             for a product I want to launch"
              │
              ▼
┌─────────────────────────────────────────────────┐
│              Orchestrator Agent                 │
│  Receives goal, breaks into sub-tasks,          │
│  assigns to specialist agents, collects results │
└────┬──────────┬──────────┬──────────┬───────────┘
     │          │          │          │
     ▼          ▼          ▼          ▼
┌─────────┐ ┌────────┐ ┌────────┐ ┌──────────┐
│Competitor│ │ Review │ │Pricing │ │  Trend   │
│ Research │ │Analyst │ │Scanner │ │  Agent   │
│  Agent  │ │ Agent  │ │ Agent  │ │          │
│         │ │        │ │        │ │          │
│Finds top│ │Scrapes │ │Collects│ │Searches  │
│ products│ │& senti-│ │price   │ │Google    │
│ in niche│ │ment-   │ │distribu│ │Trends +  │
│         │ │analyzes│ │tion    │ │rising    │
│         │ │reviews │ │        │ │keywords  │
└────┬────┘ └───┬────┘ └───┬────┘ └────┬─────┘
     └──────────┴──────────┴───────────┘
                       │
                       ▼
              Synthesis Agent:
              Combines all findings into
              structured market intelligence report
                       │
                       ▼
Output: Market research report with:
  → Top 10 competitors + their key features
  → Most common customer complaints (= your opportunity)
  → Price distribution (where the market clusters)
  → Rising search trends (what customers want next)
  → Identified market gaps (where to position)
```

### The Key Insight — Complaints Are Opportunities

The most valuable output from this agent is the **complaint analysis**. When 40% of reviews for a product category mention "battery dies too fast" — that is a direct signal: build a product with better battery life and market it explicitly on that point. The agent surfaces these patterns automatically across hundreds of reviews.

### Tech Stack
`Python` · `LangChain` · `OpenAI` · `Web Search Tool` · `BeautifulSoup` · `Sentiment Analysis` · `Multi-Agent Architecture`

🔗 **[View Project](https://github.com/Qamar-usman-ai/Product-Research-Agent)**

---

## ✈️ 8. Smart AI Tour Planner
**Personal Project | Personalized Multi-Day Itinerary Agent**

### The Problem — Trip Planning That Wastes Days

Planning a trip to an unfamiliar city or country is exhausting. Dozens of browser tabs, conflicting TripAdvisor reviews, checking opening hours, figuring out what is close to what, balancing budget vs experience, accounting for travel time between attractions — a person easily spends **8–12 hours planning a 5-day trip**.

Travel agencies charge hundreds of dollars for itinerary planning. Generic travel blogs give the same recommendations to everyone regardless of their interests, budget, or travel style. Neither solution is personalized, up-to-date, or instant.

**The goal:** Tell the agent your destination, travel dates, interests, and budget — get a complete, logically ordered, day-by-day itinerary that accounts for geography, opening hours, travel time, and your personal preferences.

### How the Agent Builds a Personalized Itinerary

```
User Input:
  Destination: "Istanbul, Turkey"
  Dates: "5 days, April 2026"
  Interests: "History, food, photography, avoid crowds"
  Budget: "Mid-range ($100-150/day)"
              │
              ▼
Step 1 — Context Gathering:
  Agent searches for:
  → Current weather in April (packing advice)
  → Major events / festivals during dates
  → Current travel advisories
  → Visa requirements for user's nationality
              │
              ▼
Step 2 — Attraction Research:
  → Searches top attractions for stated interests
  → Retrieves opening hours, entry fees, crowd patterns
  → Identifies photography-friendly locations
  → Flags "avoid crowds" alternatives to tourist traps
              │
              ▼
Step 3 — Geographic Clustering:
  → Groups attractions by neighborhood
  → Ensures each day's itinerary is geographically logical
  → No day requires crossing the city back and forth
  → Walks estimated and included in schedule
              │
              ▼
Step 4 — Budget Validation:
  → Checks meal costs, entry fees, transport estimates
  → Ensures daily spend fits stated budget
  → Suggests free alternatives where needed
              │
              ▼
Step 5 — Itinerary Assembly + Formatting:
  → Day-by-day schedule with times
  → Each location: what it is, why it matches interests,
    practical info (entry fee, nearest metro, best time)
  → Restaurant recommendations near each area
  → Transportation instructions between stops
              │
              ▼
Output: Complete formatted travel itinerary
  → Downloadable / shareable
  → Logically ordered by geography
  → Budget-validated
  → Personalized to stated interests
```

### What Makes This Better Than Generic Travel Apps

Generic travel apps show the same "Top 10 in Istanbul" to everyone. This agent understands **your specific preferences**. A photography enthusiast gets different recommendations than a foodie. A budget traveler gets different routes than a luxury traveler. An introvert who "avoids crowds" gets early-morning or off-season alternatives to every tourist trap.

### Tech Stack
`Python` · `LangChain` · `OpenAI` · `Web Search Tool` · `Google Maps API` · `ReAct Framework` · `Streamlit`

🔗 **[View Project](https://github.com/Qamar-usman-ai/Smart-AI-Tour-Planner)**

---

## 📝 9. LinkedIn Viral Post Generator Agent
**Personal Project | Multi-Agent Content Creation Pipeline**

### The Problem — Great Ideas, No Time to Write

LinkedIn has become the most important professional networking platform in the world, with **over 1 billion users**. Professionals who post consistently grow their network, attract job opportunities, and build industry authority. But writing high-quality, engaging LinkedIn posts consistently is hard:

- Most people have interesting ideas and experiences but struggle to translate them into compelling text
- Writing one good post takes 30–60 minutes of thinking, drafting, and editing
- Understanding what writing style drives engagement (hooks, structure, hashtags) requires studying the platform extensively
- Most people post once, get 3 likes, and give up

**The goal:** Input a topic, idea, or raw notes — get a polished, engagement-optimized LinkedIn post ready to publish in seconds.

### The Multi-Agent Pipeline

This project uses multiple specialized agents working in sequence — each doing one job excellently, rather than one agent doing everything adequately:

```
User Input: "I want to post about my RNA folding
             silver medal on Kaggle"
              │
              ▼
┌─────────────────────────────────────────────────┐
│           Agent 1: Research Agent               │
│                                                 │
│  Searches LinkedIn and web for:                 │
│  → Current trending posts on AI/ML              │
│  → What hooks are performing well this week     │
│  → Related hashtags with current engagement     │
│  → Competitor posts on similar topics           │
│                                                 │
│  Output: Research brief with trends + context  │
└─────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│           Agent 2: Writer Agent                 │
│                                                 │
│  Uses research brief + user's topic to write:   │
│  → Strong opening hook (first line must stop    │
│    the scroll — most important line)            │
│  → Story or insight body (personal + valuable)  │
│  → Clear takeaway or lesson                     │
│  → Call to action (question to spark comments)  │
│  → Relevant hashtags (5–10, not 30)             │
│                                                 │
│  Output: Draft LinkedIn post                   │
└─────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│           Agent 3: Reviewer Agent               │
│                                                 │
│  Reviews draft against viral post criteria:     │
│  → Hook strength (would I stop scrolling?)      │
│  → Clarity and flow (easy to read on mobile?)   │
│  → Tone (authentic, not corporate?)             │
│  → Engagement triggers (does it invite replies?)│
│  → Length (LinkedIn sweet spot: 150–300 words)  │
│  → Emoji placement (enhances, doesn't distract) │
│                                                 │
│  Output: Revised, optimized post               │
└─────────────────────────────────────────────────┘
              │
              ▼
Final Output: Publication-ready LinkedIn post
  → Strong hook
  → Authentic voice
  → Optimized structure
  → Right hashtags
  → Ready to copy-paste and post
```

### The Science of LinkedIn Virality

The agent's Writer is prompted with real patterns from high-performing LinkedIn posts:
- Posts that start with a bold statement or surprising fact get 3× more engagement than posts starting with "I'm excited to announce..."
- Short paragraphs (1–2 sentences) perform better on mobile than dense text blocks
- Ending with a direct question dramatically increases comment rate
- Personal stories outperform generic advice by 5–10× in reach

### Tech Stack
`Python` · `LangChain` · `OpenAI / Groq` · `Multi-Agent Architecture (CrewAI)` · `Web Search Tool` · `Streamlit`

🔗 **[View Project](https://github.com/Qamar-usman-ai/LinkedIn-Viral-Post-Generator-AI-Agent)**

---

## 💬 10. Chat with PDFs & Websites (RAG System)
**Personal Project | Production-Ready Retrieval-Augmented Generation**

### The Problem — Your Knowledge Is Locked in Documents

The average knowledge worker spends **2.5 hours per day** searching for information they already have — buried in PDFs, reports, contracts, and websites. Reading a 200-page technical document to find one specific answer takes 30 minutes. Searching a website manually takes 10 minutes. Asking an LLM directly gives you a hallucinated answer that sounds confident but may be wrong.

**The goal:** Ask any question in plain English about any document or website — get a precise, accurate answer grounded in the actual source content, in seconds.

### Why RAG Solves the Hallucination Problem

Standard LLMs are trained on fixed data and cannot access your private documents. They also have a fundamental tendency to "hallucinate" — generating plausible-sounding but fabricated answers when they don't know something.

RAG (Retrieval-Augmented Generation) solves both problems:

```
Document Ingestion Phase (done once per document):

PDF / Website URL
              │
              ▼
Text Extraction
(PyPDF2 for PDFs, BeautifulSoup for websites)
              │
              ▼
Chunking:
  → Split into 500-word overlapping segments
  → 50-word overlap ensures no information
    is lost at chunk boundaries
              │
              ▼
Embedding:
  → Each chunk → 768-dimensional vector
  → Vector captures semantic meaning, not just keywords
  → "Revenue increased" and "sales went up"
    get similar vectors even with different words
              │
              ▼
FAISS Vector Store:
  → Stores all vectors
  → Enables similarity search in milliseconds
  → Even 100,000-page corpora search in <100ms

──────────────────────────────────────────────────

Query Phase (every question):

User: "What were the main risks mentioned in
       the Q3 earnings call?"
              │
              ▼
Embed the question → query vector
              │
              ▼
FAISS Similarity Search:
  → Find top-5 most semantically similar chunks
  → Returns the actual source text passages
              │
              ▼
Prompt Construction:
  "Answer based only on this context: [5 passages]
   Question: What were the main risks mentioned?"
              │
              ▼
LLM (Gemini / GPT-4):
  → Answers from retrieved context only
  → Cannot invent information not in the document
  → Includes citation: "According to page 12..."
              │
              ▼
Accurate, grounded answer in <3 seconds
```

### Tech Stack
`Streamlit` · `LangChain` · `FAISS` · `Google Gemini / OpenAI` · `PyPDF2` · `BeautifulSoup` · `Python`

🔗 **[Live Demo](your-demo-link-here)**

---

## 🗃️ 11. Chat with Your Data — NL2SQL Agent
**Personal Project | Natural Language to SQL to Business Insight**

### The Problem — Data Trapped Behind Technical Barriers

Businesses have enormous amounts of valuable data in spreadsheets, databases, and CSV files. Extracting insight from this data requires SQL knowledge — a skill most business users do not have. The result: analysts become bottlenecks, every business question requires a data ticket, and the people who most need data insights (sales managers, marketing leads, operations heads) wait days for answers they could act on immediately.

**The goal:** Ask a business question in plain English — get the answer directly from your data, no SQL required.

### How the NL2SQL Agent Works

```
User: "Which region had the fastest sales growth
       last quarter compared to the same quarter
       last year?"
              │
              ▼
Schema Reader:
  → Reads table names, column names, data types
  → Builds a schema description for the LLM
  → "Table: sales_data, Columns: region (text),
     date (date), revenue (float), units (int)..."
              │
              ▼
SQL Generator (LLM):
  → Understands the business question
  → Maps question concepts to schema columns
  → Writes correct, executable SQL:

     SELECT region,
       (SUM(CASE WHEN quarter='Q3_2025' THEN revenue END) -
        SUM(CASE WHEN quarter='Q3_2024' THEN revenue END)) /
        SUM(CASE WHEN quarter='Q3_2024' THEN revenue END) * 100
        AS growth_pct
     FROM sales_data
     WHERE quarter IN ('Q3_2025', 'Q3_2024')
     GROUP BY region
     ORDER BY growth_pct DESC
     LIMIT 1
              │
              ▼
Query Execution:
  → Run SQL against actual data (SQLite / Pandas)
  → Return raw results
              │
              ▼
Answer Generator (LLM):
  → Converts raw query results to natural language
  → "The Northeast region had the fastest growth
     at +23.4% vs Q3 2024, driven by..."
              │
              ▼
Human-readable business insight in plain English
```

### Safety: Preventing Destructive Queries

The agent includes a SQL safety layer that:
- Detects and blocks DELETE, DROP, UPDATE, INSERT statements
- Limits result sets to 10,000 rows maximum
- Runs in read-only database mode
- Logs all queries for audit trail

### Tech Stack
`Streamlit` · `LangChain` · `Google Generative AI` · `Pandas` · `SQLite` · `Python`

🔗 **[Live Demo](your-demo-link-here)**

---

## ⚙️ 12. AutoML Platform
**Personal Project | End-to-End Automated Machine Learning**

### The Problem — ML Expertise Shouldn't Be a Barrier

Building a production ML model requires 2–4 weeks of expert work: exploratory analysis, preprocessing, feature selection, model selection, hyperparameter tuning, evaluation, and documentation. Most small businesses, researchers, and domain experts cannot afford this expertise. They have the data and the business question — but not the technical path to an answer.

**The goal:** Upload your CSV, select your target column, get a production-ready trained model with full evaluation report — no ML expertise required.

### The Complete Automated Pipeline

```
User Input: CSV file + target column + task type
(classification or regression)
              │
              ▼
Stage 1 — Automated EDA Report:
  → Missing value analysis (column-by-column)
  → Distribution plots for each feature
  → Outlier detection (IQR + Z-score)
  → Correlation heatmap
  → Class balance check (for classification)
  → Skewness analysis (for regression targets)
              │
              ▼
Stage 2 — Automated Preprocessing:
  → Numeric: median imputation + StandardScaler
  → Categorical: mode imputation + target encoding
    (for high cardinality) or one-hot (for low)
  → Train / Validation / Test split (80/10/10, stratified)
              │
              ▼
Stage 3 — Baseline Model Comparison:
  Trains 6 models with default hyperparameters:
  → Logistic / Linear Regression
  → Random Forest
  → XGBoost
  → LightGBM
  → SVM
  → K-Nearest Neighbors
  Evaluates each with 5-fold CV
  Selects best model for optimization
              │
              ▼
Stage 4 — Bayesian Hyperparameter Optimization:
  → Optuna framework (100 trials)
  → Smarter than grid search — learns from each trial
    which hyperparameter regions are promising
  → Typically 5–15% improvement over baseline
              │
              ▼
Stage 5 — Final Evaluation + Explainability:
  → Test set evaluation (never seen during training)
  → SHAP values: which features drive predictions?
  → For classification: confusion matrix, ROC curve
  → For regression: residual plots, predicted vs actual
  → Downloadable model (.pkl) + PDF report
```

### Tech Stack
`Streamlit` · `Scikit-learn` · `XGBoost` · `LightGBM` · `Optuna` · `SHAP` · `Matplotlib` · `Python`

🔗 **[GitHub](https://github.com/Qamar-usman-ai/Automated-ml)** · **[Live Demo](your-demo-link-here)**

---
---

# 🥇 SECTION 3 — COMPETITION ML

> Full technical writeups for my highest-ranked Kaggle competitions.

---

## 🧬 13. RNA 3D Structure Prediction — Silver Medal
**Kaggle Featured Competition | Rank 57 / 1,516 Teams | Top 4% | 🥈 Silver Medal**

### The Problem — One of Biology's Grand Unsolved Challenges

RNA molecules carry genetic instructions and are the active ingredient in mRNA vaccines. Their 3D shape determines their function — and predicting that 3D shape from sequence alone has been **unsolved for decades**. Getting it right accelerates drug discovery, CRISPR design, and cancer immunotherapy.

### My Two-Stage Pipeline

```
RNA Sequence
      │
      ▼
Stage 1 — RNA2nd (18-layer Transformer):
  Predicts secondary structure (base-pair probabilities)
      │
      ▼
Stage 2 — MSA2XYZ:
  Predicts 3D X,Y,Z atomic coordinates
  using sequence + secondary structure as input
      │
      ▼
20-Model Ensemble (different seeds + augmentations)
      │
      ▼
OpenMM Energy Minimization
(physics-based refinement — forces structure to obey
 real molecular bond lengths, angles, and clash rules)
      │
      ▼
Final 3D Structure — evaluated by TM-score
```

**Key insight:** Breaking the problem into secondary → tertiary structure gave ~8% TM-score improvement over direct 3D prediction. The ensemble of 20 models reduced variance significantly. OpenMM physics refinement was a non-obvious post-processing step that many teams missed.

**Result: Rank 57 / 1,516 (Top 4%) 🥈 Silver Medal**

---

## 🧠 14. Problematic Internet Use in Children — Silver Medal
**Kaggle Featured Competition | Rank 76 / 3,559 Teams | Top 3% | 🥈 Silver Medal**

### The Problem — Catching Mental Health Risk Before It Becomes Visible

PIU in children is linked to depression, anxiety, and social isolation — but by the time questionnaire-based diagnosis occurs, harm has already happened. My insight: **physical inactivity appears in accelerometer data before behavioral symptoms become clinically visible**.

### Key Innovation — Nelder-Mead Threshold Optimization

The model outputs a continuous score. Converting it to discrete severity classes (0/1/2/3) requires threshold selection. Instead of guessing (0.5, 1.5, 2.5), I used **Nelder-Mead optimization to learn the thresholds that maximize QWK on out-of-fold validation data**.

Optimized thresholds: [0.47, 1.41, 2.28] delivered ~0.02 QWK improvement over naive thresholds — the margin separating rank 76 from rank 200+.

**Final QWK: 0.463 | Rank 76 / 3,559 (Top 3%) 🥈 Silver Medal**

---

## 🛡️ 15. PII Detection in Student Writing
**Kaggle Featured Competition | Rank 209 / 2,048 Teams | F5 Score: 0.953**

### The Problem — Student Privacy at Scale

Educational AI needs real student writing data. But that data contains names, emails, and phone numbers. Manual review cannot scale. This system automates privacy protection so large-scale educational research becomes legally and ethically possible.

### Three-Layer Detection Architecture

```
Student Essay Text
      │
      ▼
Layer 1: DeBERTa-v3-large Ensemble (3 models)
  Token classification → PII probability per word
      │
      ▼
Layer 2: Rule-Based Detection
  Regex patterns for emails, phones, URLs
  spaCy NER for names
      │
      ▼
Layer 3: Union Fusion
  Flag as PII if ANY layer detects it
  (F5 metric weights recall 5× over precision —
   missing real PII is far worse than a false positive)
```

**F5 Score: 0.953 | Rank 209 / 2,048 (Top 10%)**

---
---

# 📊 SECTION 4 — BUSINESS FORECASTING & APPLIED ML

> Applied ML solving real business and economic problems. Each project has a real stakeholder, a real economic impact, and a real dataset from an operating organization.

---

## 🛒 16. Rohlik Grocery Orders Forecasting
**Personal Project | MAPE: 3.37% | Interactive Streamlit Dashboard**

### The Problem — Warehouse Operations Live and Die by Forecasts

Rohlik is an online grocery retailer operating warehouses across Central Europe. Every day, warehouse managers must answer one question: **how many orders will arrive tomorrow?** The answer drives staffing levels, inventory positioning, cold storage activation, and delivery slot availability.

A 20% forecast error means either empty shelves and failed deliveries, or massive overstaffing and food waste. For a warehouse processing 10,000 orders per day, a 3% MAPE vs a 20% MAPE is the difference between profitable operations and chronic loss.

### Feature Engineering — Teaching the Model What a Human Forecaster Knows

The entire value of this model is in the features. Raw dates mean nothing to XGBoost — I had to encode every signal a human forecaster would intuitively consider:

```python
# Cyclical encoding — the most important transformation:
# Monday=0, Sunday=6 implies Sunday is "far from" Monday
# But they are adjacent days on the weekly cycle
# Cyclical encoding fixes this:

df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
# Now Monday and Sunday are correctly adjacent in feature space

# Holiday intelligence via TF-IDF:
# "Christmas" and "Christmas Eve" should be similar
# "Easter" and "Labour Day" should be different
# TF-IDF on holiday names encodes these semantic relationships:

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(ngram_range=(1,2))
holiday_features = tfidf.fit_transform(df['holiday_name'].fillna('none'))

# Lag features — recent history is the best predictor:
df['lag_1'] = df['orders'].shift(1)    # yesterday
df['lag_7'] = df['orders'].shift(7)    # same day last week
df['lag_28'] = df['orders'].shift(28)  # same day last month
df['rolling_7_mean'] = df['orders'].rolling(7).mean()
df['rolling_7_std'] = df['orders'].rolling(7).std()
```

### Results
| Metric | Value |
|--------|-------|
| **MAPE** | **3.37%** |
| Model | XGBoost |
| Features | Cyclical encoding + TF-IDF holidays + lag features |
| Dashboard | Interactive Streamlit EDA + live forecast |

🔗 **[GitHub](https://github.com/Qamar-usman-ai/Rohlik-Orders-Forecasting-Challenge)**

---

## 💰 17. data.org Financial Health Prediction — Zindi
**Zindi Competition | Rank 28 / 900 Teams (Top 3%)**

### The Problem — Billions of Small Businesses, No Access to Credit

Small and medium-sized businesses represent **90% of all businesses and over 50% of employment** in Africa and South Asia. Yet most have no access to formal financing — not because they are bad businesses, but because they operate informally and lack the formal financial records banks require for credit scoring.

**The result:** Entrepreneurs with viable businesses cannot get loans to grow. The economic potential of millions of small businesses is trapped by a data gap.

**This project predicts the financial health of small businesses using alternative data signals** — mobile money patterns, survey responses, registration data — enabling responsible lending to businesses that traditional banks cannot assess.

### My EDA Approach — Understanding the Data Before Modeling

Before building any model, I spent significant time understanding what the data actually contained:

```python
import pandas as pd
import numpy as np

# Complete missing value and unique values analysis
analysis_results = []

for col in train_df.columns:
    missing_count = train_df[col].isnull().sum()
    missing_pct = (missing_count / len(train_df)) * 100
    unique_values = train_df[col].dropna().unique()
    unique_count = len(unique_values)

    # Show sample for high-cardinality columns
    if unique_count > 20:
        unique_sample = str(sorted(unique_values)[:10]) + " ..."
    else:
        unique_sample = str(sorted(unique_values))

    analysis_results.append({
        'Column': col,
        'Data Type': train_df[col].dtype,
        'Missing %': round(missing_pct, 2),
        'Unique Count': unique_count,
        'Sample Values': unique_sample
    })

    # For numeric columns, understand the distribution
    if train_df[col].dtype in ['int64', 'float64']:
        non_null = train_df[col].dropna()
        print(f"{col}: range [{non_null.min():.1f}, {non_null.max():.1f}]"
              f" | mean: {non_null.mean():.1f} | missing: {missing_pct:.1f}%")

analysis_df = pd.DataFrame(analysis_results)
analysis_df = analysis_df.sort_values('Missing %', ascending=False)
```

**Key findings from EDA:**
- Several financial indicator columns had >30% missingness — many SMEs genuinely don't track these metrics. Missingness itself encodes information about business formality.
- The target (financial health score) was imbalanced — most businesses clustered at moderate health, very few at excellent or critical.
- Several categorical columns had high cardinality requiring target encoding rather than one-hot encoding.

### My Complete Modeling Pipeline

```
Raw SME Data:
(financial indicators + survey responses + registration info)
              │
              ▼
Feature Engineering:
  → Financial ratios (revenue per employee, debt/revenue)
  → Missingness indicator features (is_missing_revenue_col)
  → Target encoding for high-cardinality categoricals
  → Log transformation for skewed financial variables
  → Business age and registration features
              │
              ▼
XGBoost + LightGBM Ensemble
  → Stratified K-Fold cross-validation
  → Bayesian hyperparameter optimization (Optuna)
  → Ensemble weighted average
              │
              ▼
Output: Financial health score (ordinal classification)
```

### Results
| Metric | Value |
|--------|-------|
| **Competition Rank** | **28 / 900 (Top 3%)** |
| Platform | Zindi |
| Sponsor | data.org |
| Impact | Enables credit access for underserved SMEs |

---

## 🚦 18. Barbados Traffic Analysis — Zindi
**Zindi Competition | Rank 40 / 222 Teams (Top 18%)**

### The Problem — One Roundabout Paralyzing a Capital City

Traffic congestion costs the global economy **over $1 trillion per year**. In small island nations with limited road infrastructure, a single bottleneck can paralyze the entire urban network. Bridgetown, the capital of Barbados, has one critical roundabout where congestion regularly causes city-wide delays.

**The goal:** Predict traffic flow patterns accurately enough to identify the root causes of congestion — so the government can make data-driven infrastructure decisions.

### Key Insight From My Analysis

EDA revealed that traffic at this roundabout contains **two distinct populations with different patterns**:

- **Local commuters:** peak 7–9am, predictable weekly cycle, Monday–Friday
- **Tourist traffic:** peak 10am–12pm, higher on weekends, varies by season

A single model treating all traffic identically performed poorly. Creating separate features for each traffic pattern — and allowing the model to learn their different seasonalities — significantly improved prediction accuracy.

### My Solution

```
Sensor Data (vehicle counts by direction + timestamp)
              │
              ▼
Feature Engineering:
  → Cyclical time features (hour, day, week — sin/cos)
  → Commuter pattern features (rush hour windows)
  → Tourist pattern features (weekend + midday peaks)
  → Holiday calendar features
  → Weather interaction features
  → Direction-specific lag features
              │
              ▼
XGBoost Regressor
  → Time-based cross-validation
    (always train on past, validate on future —
     no future data leakage)
              │
              ▼
Output: Traffic flow per direction per hour
```

### Results
| Metric | Value |
|--------|-------|
| Competition Rank | **40 / 222 (Top 18%)** |
| Client | Government of Barbados |
| Platform | Zindi |

---

## 🐄 19. DigiCow Farmer Training Adoption — Zindi
**Zindi Competition | Rank 88 / 387 Teams (Top 23%)**

### The Problem — Training Programs That Don't Change Behavior

DigiCow provides digital farm management tools and training to dairy farmers in East Africa. They invest heavily in training programs — but completion rates are low. Most farmers start training and disengage before implementing new practices. The practices they never learn are the ones that would most improve their farm productivity and income.

**The business question:** Can we predict which farmers will complete training and put it into practice — so we can proactively coach at-risk farmers before they disengage?

This is a classic **churn prediction problem** applied to agricultural training — if we can identify likely dropouts early, intervention is possible and cheap. After dropout, it is too late.

### What Predicts Training Completion

Through feature importance analysis, the most predictive signals were:

1. **Early engagement velocity** — how quickly a farmer completed the first 2 steps. Slow starters almost never finish.
2. **Farm productivity proxy** — milk yield per cow. Higher-productivity farmers are more motivated to learn improvements.
3. **Technology access** — smartphone vs basic phone. Digital training engagement is much higher for smartphone users.
4. **Geographic region** — some regions had systematically higher engagement due to local DigiCow field officer activity.

### My Solution

```
Farmer Profile:
(registration data + early engagement + farm metrics + device + location)
              │
              ▼
Feature Engineering:
  → Engagement velocity score (early step completion speed)
  → Productivity proxy (milk/cow ratio)
  → Technology access indicator
  → Regional engagement rate features
              │
              ▼
LightGBM Classifier
  → Stratified K-Fold CV
  → Class weights for imbalanced completion rates
  → SHAP explainability for field officer guidance
              │
              ▼
Output: Dropout risk probability per farmer
High-risk farmers flagged for proactive outreach
```

### Results
| Metric | Value |
|--------|-------|
| Competition Rank | **88 / 387 (Top 23%)** |
| Client | DigiCow |
| Platform | Zindi |

---

## 🌽 20. agriBORA Maize Price Forecasting — Zindi
**Zindi Competition | Weekly Maize Price Forecasting — Kenya**

### The Problem — Hunger Caused by Unpredictable Prices

In Kenya, maize is the staple food for **90% of the population**. Maize price volatility directly causes food insecurity:
- Farmers sell at harvest when prices are lowest (they need cash) and buy back at planting when prices are highest
- Traders cannot optimize inventory without knowing where prices are heading
- Government intervention (releasing strategic grain reserves) happens too late because nobody predicted the price spike

**Accurate weekly price forecasting gives farmers, traders, and policymakers the information they need to make better decisions — directly protecting food security.**

### Why Agricultural Price Forecasting Is Hard

Unlike financial markets, agricultural prices are driven by:
- **Harvest calendars** — prices always fall at main harvest (October–November in Kenya)
- **Weather shocks** — a drought can triple prices within 6 weeks
- **Cross-border trade** — Ugandan and Tanzanian prices affect Kenyan prices
- **Global commodity markets** — world maize prices (CBOT) propagate locally
- **Structural breaks** — COVID, drought years, and the 2022 Ukraine war all created sudden regime changes

### My Solution — Walk-Forward Time Series Validation

```
Historical Weekly Prices (per market, per region)
              │
              ▼
Feature Engineering:
  → Lag features: t-1, t-4, t-8, t-13, t-26, t-52 weeks
  → Rolling statistics: 4-week and 13-week mean, std, trend
  → Harvest calendar indicators (pre/post harvest weeks)
  → Cross-market spread features (Nairobi vs Mombasa gap)
  → Seasonal decomposition components
              │
              ▼
XGBoost with Walk-Forward Validation:
  → Train on weeks 1–100, validate on week 101
  → Train on weeks 1–101, validate on week 102
  → ... (never look ahead, always predict future)
  → This is the only valid CV strategy for time series
              │
              ▼
Output: Weekly price forecast per market
  Evaluated by RMSE + directional accuracy
  (did we predict whether price went up or down?)
```

---
---

## 🛠️ Complete Tech Stack

```
Languages        Python, SQL
ML / Boosting    XGBoost, LightGBM, CatBoost, Scikit-learn
Deep Learning    PyTorch, TensorFlow, ResNet, EfficientNet, DeBERTa, Transformers
LLMs / Agents    LangChain, LangGraph, CrewAI, OpenAI, Google Gemini,
                 ReAct Framework, RAG, FAISS, NL2SQL, Multi-Agent Systems
MLOps / Cloud    Azure ML, Docker, MLflow, GitHub Actions, CI/CD, Streamlit
Data Science     Pandas, NumPy, Matplotlib, Seaborn, SHAP, Optuna, OpenCV
```

---

## 📜 Certifications

| Certification | Issuer |
|--------------|--------|
| 🏅 TensorFlow Developer Certificate | Google |
| 🏅 IBM Data Science Professional Certificate | IBM / Coursera |
| 🏅 Machine Learning Specialization | Stanford University / Andrew Ng |

---

## 📊 GitHub Activity

![Qamar's GitHub Stats](https://github-readme-stats.vercel.app/api?username=Qamar-usman-ai&show_icons=true&theme=default&hide_border=true&count_private=true)

![Top Languages](https://github-readme-stats.vercel.app/api/top-langs/?username=Qamar-usman-ai&layout=compact&theme=default&hide_border=true)

[![GitHub Streak](https://streak-stats.demolab.com?user=Qamar-usman-ai&theme=default&hide_border=true)](https://git.io/streak-stats)

---

## 📬 Let's Connect

I am open to remote ML engineering roles, research collaborations, and freelance consulting.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Qamar%20Usman-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/qamar-usman)
[![Kaggle](https://img.shields.io/badge/Kaggle-qamarmath-20BEFF?style=flat&logo=kaggle)](https://kaggle.com/qamarmath)
[![Zindi](https://img.shields.io/badge/Zindi-qamarcodes-1A1A2E?style=flat)](https://zindi.africa/users/qamarcodes)
[![Email](https://img.shields.io/badge/Email-Contact%20Me-D14836?style=flat&logo=gmail)](mailto:your.email@gmail.com)

---

*"The best model is the one that solves a real problem for a real person."*
