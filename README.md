# AI-Based Medical Disease Prediction System

A modern, extensible platform for predicting anemia types and generating medical reports using advanced machine learning and generative AI. Developed as part of the AI Lab AGH Kraków student scientific society.

---

## 🚀 Overview

This project predicts the probability of various anemia types based on laboratory blood parameters and generates professional Polish-language medical epicrises using Gemini 1.5 Flash. It is built with Python, scikit-learn, XGBoost, pandas, and deep neural networks, and features a Streamlit-based user interface.

---

## 🩸 Supported Diseases

- **Anemia** (fully implemented)
    - Aplastyczna
    - Hemolityczna
    - Makrocytarna
    - Mikrocytarna
    - Normocytarna
    - Healthy
- **Diabetes** (in development)
- **Infection** (in development)

---

## 🛠️ Features

- **Anemia type prediction** using XGBoost and PCA
- **Probability estimation** for each class
- **Automated, AI-generated medical epicrisis** (Polish) via Gemini 1.5 Flash
- **PDF export** of generated epicrisis with Unicode font support
- **Interactive web interface** (Streamlit)
- **Expandable architecture** for future disease modules

---

## 🖥️ Installation

**Clone the repository:**
**git clone https://github.com/antoniopater/LLM_anemia**

**Install dependencies (CPU version):**

**pip install -r requirements-cpu.txt**
*(For CUDA-enabled systems, use `requirements-cuda.txt`)*

---

## 🚦 Usage

**Start the application:**
**streamlit run app/app.py**


**How it works:**
- Select the disease from the sidebar (currently: Anemia)
- Enter laboratory values for the patient
- Click "Predict" to get:
    - The predicted anemia type and probability
    - An AI-generated medical epicrisis (in Polish)
    - Option to download the epicrisis as a PDF

---

## 🧬 Model Details

- **Features:** RBC, HGB, HCT, MCV, MCH, MCHC, RDW, PLT, WBC
- **Preprocessing:** Scaling and PCA (parameters saved in `preprocess/anemia/`)
- **Classifier:** XGBoost (model in `models/anemia/modelXGBoost.pkl`)
- **Performance:** ROC curves and confusion matrices included in `models/anemia/`
- **Synthetic data:** Generated with VAE (`trainingData/anemia/`)

---

## 📂 Project Structure
```
.
├── KNN_with_better_preprocess
│   └── __pycache__
│       └── preprocess.cpython-312.pyc
├── LLM_models
│   ├── Fine_tuning_generate_data.py
│   ├── main.py
│   ├── medicalBertModel.py
│   ├── model.py
│   └── model_one_epoch.py
├── Visualizer
│   └── Hematology
│       └── Hematology_visualiser.py
├── app
│   └── app.py
├── dataAnalysis
│   └── anemia
│       ├── PCA_analysis.py
│       ├── exmapleFilePCAtree.png
│       ├── roc_curve.py
│       └── visualize.py
├── fonts
│   ├── DejaVuSans.cw127.pkl
│   ├── DejaVuSans.pkl
│   └── DejaVuSans.ttf
├── generatingData
│   ├── anemia
│   │   └── generte_with_model.py
│   └── infection
│       └── generate_with_model.py
├── models
│   └── anemia
│       ├── confusion_matrix.png
│       ├── modelXGBoost.pkl
│       ├── modelXGBoost.py
│       └── roc_curve_multiclass.png
├── preprocess
│   ├── Indection
│   │   └── preprocess.py
│   └── anemia
│       ├── __pycache__
│       │   └── preprocess.cpython-310.pyc
│       ├── pca.pkl
│       ├── preprocess.py
│       └── scaler.pkl
├── requirements-cpu.txt
├── requirements-cuda.txt
├── requirements.txt
├── structure.txt
└── trainingData
    └── anemia
        ├── synthetic_data_vae.csv
        ├── synthetic_data_vae2.csv
        └── synthetic_data_vae3.csv
```
---

## 📊 Example Workflow

1. **User enters blood test results** in the Streamlit form.
2. **Model predicts** the most probable anemia type and shows the probability.
3. **Gemini 1.5 Flash** generates a professional medical epicrisis in Polish.
4. **User can download** the epicrisis as a PDF with proper font support.

---

## 📚 Data Source

Medical datasets were provided by collaborating researchers from Jagiellonian University (scientific society). 
Synthetic data is generated using deep generative models (VAE).

---

## 👨‍💻 Authors

- Jan Banasik
- Antoni Pater

Project developed as part of the AI Lab AGH Kraków student scientific society.

---

## 📄 License

MIT License (see LICENSE file for details)

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes** only. Model predictions are not a substitute for professional medical advice or diagnosis. Always consult a qualified healthcare provider.

