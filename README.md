# 🛳️ Titanic Survival Prediction App

A machine learning web app that predicts passenger survival on the Titanic using the XGBoost algorithm. Built with Python, Streamlit, and scikit-learn.

![screenshot](https://img.shields.io/badge/Machine%20Learning-XGBoost-blue?style=for-the-badge)
![screenshot](https://img.shields.io/badge/Web%20App-Streamlit-brightgreen?style=for-the-badge)

---

## 🚀 Features

- Preprocessed Titanic dataset with engineered features
- Model trained using **XGBoost Classifier**
- Interactive prediction UI using **Streamlit**
- Dynamic visualizations (confusion matrix, accuracy score)
- Model and scaler saved with **Joblib** for reusability

---

## 🧠 Machine Learning Pipeline

1. Data cleaning and feature engineering:
   - Imputation of missing values
   - Feature binning (age, fare)
   - Encoding of categorical variables

2. Model training:
   - XGBoost classifier with train/test split
   - Min-Max scaling of input features
   - Saved model with `joblib`

3. Web interface:
   - Uses Streamlit for UI
   - Predicts survival from user input
   - Visual feedback on model performance

---

## 🗂️ Project Structure

├── titanic.csv # Dataset (Kaggle Titanic)
├── streamlit_app.py # Streamlit UI script
├── xgb_titanic_model.joblib # Trained XGBoost model
├── scaler.joblib # MinMaxScaler for numeric features
├── features.joblib # Feature names saved from training
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## 🧪 Installation & Running the App

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/titanic-streamlit-xgboost.git
cd titanic-streamlit-xgboost

2. Create a Virtual Environment (Optional but Recommended)

3.  Install Dependencies

pip install -r requirements.txt

4.  Run the Streamlit App

streamlit run streamlit_app.py

SAMPLE OUTPUT

Model accuracy printed on screen

Confusion matrix heatmap

Prediction form for new passenger data

MODEL PERFORMANCE

The current XGBoost model achieves:

✅ ~82–88% accuracy (depending on test split)

✅ Balanced precision/recall

✅ Tuned via early stopping and learning rate

DEPENDENCIES

streamlit

xgboost

scikit-learn

pandas

seaborn

matplotlib

joblib

numpy


You can install them all via pip install -r requirements.txt

DEPLOY

You can deploy this app on Streamlit Cloud in 3 steps:

Push the repo to GitHub

Go to Streamlit Cloud

Link your GitHub and launch the streamlit_app.py

CONTRIBUTING

Feel free to fork the repo, improve the app, or tune the model. Pull requests are welcome!

LICENCE

This project is open-source and available under the MIT License.


Ed759


