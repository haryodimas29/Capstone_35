import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

# Function to load data
def load_data(file):
    data = pd.read_csv(file)
    return data

# Function to preprocess data
def preprocess_data(data):
    label_encoder = LabelEncoder()
    data['Type'] = label_encoder.fit_transform(data['Type'])
    X = data[['Type', 'Air temperature [K]', 'Process temperature [K]', 'Torque [Nm]']]
    y = label_encoder.fit_transform(data['Failure Type'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Function to evaluate model performance
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    X = pd.concat([X_train, X_test])
    y = np.append(y_train, y_test)
    cv_scores = cross_val_score(model, X, y, cv=5)
    mean_cv_score = np.mean(cv_scores)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return accuracy, f1, mean_cv_score, conf_matrix, y_pred

# Function to map predicted labels to failure types, status, and action
def map_predicted_labels(y_pred):
    failure_types = {
        0: 'No Failure',
        1: 'Power Failure',
        2: 'Tool Wear Failure',
        3: 'Overstrain Failure',
        4: 'Random Failures',
        5: 'Heat Dissipation Failure'
    }
    repair_status = []
    repair_action = []

    for label in y_pred:
        if label in failure_types:
            if failure_types[label] == "No Failure":
                repair_status.append("No Maintenance Needed")
                repair_action.append("Relax")
            else:
                repair_status.append("Maintenance Required")
                repair_action.append("Immediate Repair")
        else:
            repair_status.append("Maintenance Required")
            repair_action.append("Immediate Repair")

    return repair_status, repair_action

# Function to map actual labels to failure types
def map_actual_labels(y_actual):
    failure_types = {
        0: 'No Failure',
        1: 'Power Failure',
        2: 'Tool Wear Failure',
        3: 'Overstrain Failure',
        4: 'Random Failures',
        5: 'Heat Dissipation Failure'
    }
    actual_failure = [failure_types[label] for label in y_actual]
    return actual_failure

# Function to display evaluation results with actual failure types
def show_results(accuracy, f1, mean_cv_score, conf_matrix, y_pred, y_actual, repair_status, repair_action):
    failure_types = {
        0: 'No Failure',
        1: 'Power Failure',
        2: 'Tool Wear Failure',
        3: 'Overstrain Failure',
        4: 'Random Failures',
        5: 'Heat Dissipation Failure'
    }
    predicted_failure = [failure_types[label] for label in y_pred]
    actual_failure = map_actual_labels(y_actual)

    st.header("")
    st.subheader("Evaluation Results:")
    st.write(f"Accuracy: {accuracy}")
    st.write(f"F1 Score: {f1}")
    st.write(f"Cross-Validation Score: {mean_cv_score}")

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    st.pyplot(fig)

    st.write("Predicted Failure, Actual Failure, Repair Status, and Repair Action:")
    results_df = pd.DataFrame({
        'Predicted Failure': predicted_failure,
        'Actual Failure': actual_failure,
        'Repair Status': repair_status,
        'Repair Action': repair_action
    })
    st.write(results_df)

# Function to display the login page
def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Masuk"):
        if username == "abc" and password == "123":
            st.session_state.logged_in = True
            st.experimental_rerun()
        else:
            st.error("Username atau password salah")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    st.title("Visualisasi Data dan Prediksi Akurasi dan Kegagalan")
    logo_url = "logo.png"
    st.sidebar.image(logo_url, use_column_width=True)

    st.sidebar.title("Visualisasi Data")
    visualization_type = st.sidebar.selectbox("Pilih Tipe Visualisasi", ["Pie Chart", "Scatter Plot", "Simple Area Chart"])

    uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])

    st.success("MASUKKAN DATA DAN LIHAT HASILNYA DIBAWAH")
    
    st.sidebar.title("Pemilihan Model")
    model_name = st.sidebar.selectbox("Pilih Model", ["Random Forest", "Gradient Boosting", "SVM Classifier", "Neural Networks"])

    button_clicked = st.sidebar.button("Evaluasi")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        processed_data = data[['Air temperature [K]', 'Process temperature [K]', 'Torque [Nm]', 'Failure Type']]

        if visualization_type == "Pie Chart":
            st.subheader("Pie Chart")
            fig, ax = plt.subplots()
            data['Failure Type'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
            ax.set_aspect('equal')
            st.pyplot(fig)

        elif visualization_type == "Scatter Plot":
            st.subheader("Scatter Plot")
            x_variable = st.selectbox("Pilih Variabel X", processed_data.columns[:-1])
            y_variable = st.selectbox("Pilih Variabel Y", processed_data.columns[:-1])
            fig, ax = plt.subplots()
            sns.scatterplot(x=x_variable, y=y_variable, hue='Failure Type', data=processed_data, ax=ax)
            st.pyplot(fig)

        elif visualization_type in ["Simple Area Chart"]:
            st.subheader(visualization_type)
            x_variable = st.selectbox("Pilih Variabel X", processed_data.columns[:-1])
            y_variable = st.selectbox("Pilih Variabel Y", processed_data.columns[:-1])
            fig, ax = plt.subplots()
            sns.barplot(x=x_variable, y=y_variable, hue='Failure Type', data=processed_data, ax=ax)
            st.pyplot(fig)

        X_train, X_test, y_train, y_test = preprocess_data(data)

        if button_clicked:
            if model_name == "Random Forest":
                model = RandomForestClassifier(random_state=42)
            elif model_name == "Gradient Boosting":
                model = GradientBoostingClassifier(random_state=42)
            elif model_name == "SVM Classifier":
                model = SVC(random_state=42)
            elif model_name == "Neural Networks":
                model = MLPClassifier(random_state=42)

            accuracy, f1, mean_cv_score, conf_matrix, y_pred = evaluate_model(model, X_train, X_test, y_train, y_test)

            repair_status, repair_action = map_predicted_labels(y_pred)

            actual_failure = map_actual_labels(y_test)

            show_results(accuracy, f1, mean_cv_score, conf_matrix, y_pred, y_test, repair_status, repair_action)

    else:
        st.title("Visualisasi Data")
        st.sidebar.title("Visualisasi Data")
        st.sidebar.info("Silakan unggah file CSV untuk memulai visualisasi dan pemodelan.")
else:
    login_page()
