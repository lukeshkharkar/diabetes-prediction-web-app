# Diabetes Prediction Web Application

A machine learning-based web application designed to predict the likelihood of diabetes in individuals using diagnostic health measurements. The project features a complete machine learning pipeline—from data preprocessing and feature scaling to model training, REST API creation using Flask, and web deployment on Render.

**Live Demo:** [Diabetes Prediction Web App](https://diabetes-prediction-web-app-0n9b.onrender.com/)

---

##  Project Overview

Diabetes is a critical global health issue requiring early detection for effective management and prevention. This repository provides an end-to-end Machine Learning solution that analyzes key patient parameters (such as Glucose level, BMI, and Age) to accurately determine whether an individual is diabetic or non-diabetic.

---

##  Features

* **Interactive Web Interface:** User-friendly HTML forms to input health diagnostics.
* **RESTful API Backend:** Dedicated `/predict` endpoint to process predictions via JSON requests.
* **Data Standardization:** Integrates `StandardScaler` (`scaler.pkl`) to ensure input data is properly scaled before model prediction.
* **Cloud Deployment:** Fully deployed and active on Render web services.

---

##  Tech Stack

* **Language:** Python 3.12.1
* **Machine Learning & Data Processing:** Scikit-Learn, Pandas, NumPy
* **Web Framework:** Flask
* **Deployment:** Render

---

##  Dataset & Model Performance

### Dataset Overview
* **Source:** Pima Indians Diabetes Dataset (Kaggle)
* **Total Samples:** 768 (500 Non-Diabetic, 268 Diabetic)
* **Input Features (8):** `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`

### Machine Learning Model
* **Algorithm:** Support Vector Machine (SVM) with Linear Kernel
* **Train/Test Split:** 80% Training (614 samples) / 20% Testing (154 samples)

| Metric | Score |
| :--- | :--- |
| **Training Accuracy** | 78.26% |
| **Test Accuracy** | 77.27% |
| **Precision (Diabetic)** | 75.67% |
| **Recall (Diabetic)** | 51.85% |
| **F1-Score** | 61.54% |
| **ROC-AUC** | 0.7920 |

---

##  Repository Structure

```text
diabetes-prediction-web-app/
├── templates/            # HTML files for web UI
├── app.py                # Flask application code & API endpoints
├── model.pkl             # Saved Support Vector Machine model
├── scaler.pkl            # Pre-trained StandardScaler instance
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
