import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_curve, confusion_matrix, auc
from streamlit_ydata_profiling import st_profile_report
import os
from ydata_profiling import ProfileReport

# Function to plot ROC curve and confusion matrix
def plot_roc_curve(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_confusion_matrix(model, X_test, y_test):
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    
    plt.figure(figsize=(10, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()



# Load preprocessed test data (X_test, y_test) and models
X_test = joblib.load('../data/processed/X_test.pkl')
y_test = joblib.load('../data/processed/y_test.pkl')

# Load models
rf_model = joblib.load('../models/random_forest_model.pkl')
# Load other models similarly

# Required for profiling report
if os.path.exists('../data/raw/web_app_dataset.csv'): 
    df = pd.read_csv('../data/raw/web_app_dataset.csv', index_col=None)

# Streamlit app layout
st.sidebar.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
choice = st.sidebar.radio("Navigation", ["Data", "Profiling", "Modelling", "Make Inference"])
st.info("This web application helps to showcase the results of a model that classifies whether a customer will complaint or not")


if choice == "Data":
    st.title("Preview the Dataset")
    file = '../data/raw/web_app_dataset.csv'
    df = pd.read_csv(file)        
    #df.to_csv('../data/interim/dataset.csv', index=None)
    st.dataframe(df)

if choice == "Profiling":
    st.title("Exploratory Data Analysis Report")
    #st.markdown(open("../reports/EDA_report.html").read(), unsafe_allow_html=True)
    #df = pd.read_csv('../data/raw/web_app_dataset.csv', index_col=None)
      # Generate profile report
    profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
    st_profile_report(profile)

if choice == "Modelling":
  st.title("Model Performance Metrics")
  model_choice = st.selectbox("Choose a model", ["Random Forest", "Logistic Regression", "XGBoost"])

  # Load model based on choice
  if model_choice == "Random Forest":
      selected_model = rf_model
  elif model_choice == "Logistic Regression":
      st.text("Logistic Regression Model")
      #selected_model = logistic_model  # Assuming logistic_model is defined
  elif model_choice == "XGBoost":
      st.text("XGBoost Model")
      #selected_model = xgb_model  # Assuming xgb_model is defined

  if st.button('Show Metrics'):
    # Display ROC Curve
    st.subheader(f"ROC Curve for {model_choice}")
    fig, ax = plt.subplots()
    plot_roc_curve(selected_model, X_test, y_test)  # Implement the plot inside the function
    st.pyplot(fig)

    # Confusion Matrix
    st.subheader(f"Confusion Matrix for {model_choice}")
    fig, ax = plt.subplots()
    plot_confusion_matrix(selected_model, X_test, y_test)  # Implement the plot inside the function
    st.pyplot(fig)

    # Note about Random Forest being the best model
    if model_choice != "Random Forest":
      st.markdown("Note: Random Forest was identified as the best model in this analysis.")


   
if choice == "Make Inference":
    st.title("Make an Inference")
    # Form for user input
    # Use rf_model.predict() for prediction

# Ensure all necessary packages are installed in your Streamlit environment
