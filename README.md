# Diabetes Prediction Web Application

A lightweight web application that predicts the likelihood of diabetes based on diagnostic patient data.

---

## Overview

Diabetes is a widespread health condition that requires early detection for effective management. This project uses diagnostic health indicators—such as glucose levels, BMI, and age—to train a machine learning model capable of predicting whether an individual is diabetic or non-diabetic. The final model is integrated into a Flask backend and hosted online for public demonstration.

---

## Key Highlights

* **Input Data:** Trained on the Pima Indians Diabetes Dataset (768 records with 8 diagnostic features).
* **Machine Learning Model:** Support Vector Machine (SVM) classifier achieving ~77% testing accuracy.
* **Web Integration:** Built using Flask for processing requests and served via an interactive user interface.
* **Deployment:** Live on Render web services.

---

## Technical Stack

* **Language:** Python 3.12.1
* **Libraries:** Scikit-Learn, Pandas, NumPy
* **Framework:** Flask
* **Hosting Platform:** Render

---

## Project Structure

```text
diabetes-prediction-web-app/
├── templates/            # HTML user interface
├── app.py                # Flask server application
├── model.pkl             # Trained SVM model
├── scaler.pkl            # Pre-trained feature scaler
├── requirements.txt      # Project dependencies
└── README.md             # Repository documentation
```

---

