# Imports
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix

from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib

# Load dataset
data = pd.read_csv('titanic.csv')
data.info()
print(data.isnull().sum())

# --- Helper functions ---

def fill_missing_ages(df):
    age_fill_map = {}
    for Pclass in df["Pclass"].unique():
        age_fill_map[Pclass] = df[df["Pclass"] == Pclass]["Age"].median()
    df["Age"] = df.apply(
        lambda row: age_fill_map[row["Pclass"]] if pd.isnull(row["Age"]) else row["Age"],
        axis=1
    )

def extract_title(df):
    df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',
        'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['Title'] = df['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3,
                                   "Master": 4, "Rare": 5})
    df['Title'].fillna(0, inplace=True)
    return df

def preprocess_data(df):
    # Extract titles before dropping 'Name'
    df = extract_title(df)

    # Drop unneeded columns
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

    # Fill missing values
    df["Embarked"] = df["Embarked"].fillna("S")
    df.drop(columns=["Embarked"], inplace=True)

    fill_missing_ages(df)

    # Encode Sex
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})

    # Feature engineering
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = np.where(df["FamilySize"] == 1, 1, 0)
    df["Farebin"] = pd.qcut(df["Fare"], 4, labels=False)
    df["Agebin"] = pd.cut(df["Age"], bins=[0, 12, 20, 40, 60, np.inf], labels=False)
    df["FarePerClass"] = df["Fare"] / df["Pclass"]

    return df

# --- Preprocess data ---
data = preprocess_data(data)

# Create features and target
X = data.drop(columns=["Survived"])
y = data["Survived"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Train XGBoost with GridSearch ---
def tune_model(X_train, y_train):
    params_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    grid_search = GridSearchCV(model, params_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

best_model = tune_model(X_train, y_train)

import joblib

# Save the trained model
joblib.dump(best_model, 'xgb_titanic_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(X.columns.tolist(), 'features.joblib') 

# --- Evaluate model ---
def evaluate_model(model, X_test, y_test):
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    matrix = confusion_matrix(y_test, prediction)
    return accuracy, matrix

accuracy, matrix = evaluate_model(best_model, X_test, y_test)

print(f"\n✅ Accuracy: {accuracy*100:.2f}%")
print("Confusion Matrix:")
print(matrix)

# --- Plot confusion matrix ---
def plot_model(matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d',
                xticklabels=['Not Survived', 'Survived'],
                yticklabels=['Not Survived', 'Survived'],
                cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

plot_model(matrix)

# Load model, scaler, and feature list
model = joblib.load("xgb_titanic_model.joblib")
scaler = joblib.load("scaler.joblib")
feature_order = joblib.load("features.joblib")

# UI
st.title("🚢 Titanic Survival Prediction (XGBoost)")

uploaded_file = st.file_uploader("Upload Titanic-style CSV", type=["csv"])

def preprocess_uploaded_data(df):
    # Extract titles
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',
        'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['Title'] = df['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})
    df['Title'].fillna(0, inplace=True)

    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

    df["Embarked"] = df["Embarked"].fillna("S")
    df.drop(columns=["Embarked"], inplace=True)

    age_fill_map = df.groupby("Pclass")["Age"].transform("median")
    df["Age"] = df["Age"].fillna(age_fill_map)

    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = np.where(df["FamilySize"] == 1, 1, 0)
    df["Farebin"] = pd.qcut(df["Fare"], 4, labels=False, duplicates='drop')
    df["Agebin"] = pd.cut(df["Age"], bins=[0, 12, 20, 40, 60, np.inf], labels=False)
    df["FarePerClass"] = df["Fare"] / df["Pclass"]

    return df

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Display uploaded data
    st.subheader("Raw Uploaded Data")
    st.dataframe(df.head())

    df_cleaned = preprocess_uploaded_data(df)

    # Ensure columns are in same order
    X_input = df_cleaned[feature_order]
    X_scaled = scaler.transform(X_input)

    # Predict
    y_pred = model.predict(X_scaled)

    if "Survived" in df.columns:
        y_true = df["Survived"]
        acc = accuracy_score(y_true, y_pred)
        st.success(f"Model Accuracy: {acc * 100:.2f}%")

        matrix = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=['Not Survived', 'Survived'],
                    yticklabels=['Not Survived', 'Survived'])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)

    # Show predictions
    st.subheader("Predictions")
    df["Prediction"] = y_pred
    st.dataframe(df[["Prediction"] + [col for col in df.columns if col != "Prediction"]])