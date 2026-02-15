import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import time


# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Adult Income ML App",
    page_icon="📊",
    layout="wide"
)

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("📁 Dataset Downloads")

try:
    train_data = open("data/download_file/adult_test.csv", "rb").read()
    st.sidebar.download_button(
        label="⬇ Download Testing Dataset (2 MB)",
        data=train_data,
        file_name="adult.csv",
        mime="text/csv"
    )
except:
    st.sidebar.warning("Dataset file not found in data/ folder.")

st.sidebar.markdown("---")
st.sidebar.info(
    "Upload a CSV file with the same structure as the Adult dataset.\n\n"
    "Make sure column order matches training data."
)

# ---------------- MAIN TITLE ---------------- #

# Remove top padding
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>Adult Income Classification Dashboard</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; font-size:18px;'>Compare performance of multiple ML models</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- MODEL SELECTION ---------------- #
col1, col2 = st.columns(2)

with col1:
    model_choice = st.selectbox(
        "🔍 Select Model",
        ["Logistic_Regression",
         "Decision_Tree",
         "KNN",
         "Naive_Bayes",
         "Random_Forest",
         "XGBoost"]
    )

with col2:
    uploaded_file = st.file_uploader("📤 Upload Test CSV", type=["csv"])

# Visual feedback when model changes
st.success(f"Currently Selected Model: {model_choice}")

if uploaded_file:

    # Small inline upload progress
    upload_container = st.empty()

    with upload_container.container():
        col_a, col_b = st.columns([4, 1])
        with col_a:
            small_progress = st.progress(0)
        with col_b:
            tick_placeholder = st.empty()

    # Simulated upload progress animation
    for percent in range(0, 101, 20):
        small_progress.progress(percent)
        time.sleep(0.1)

    # Replace progress with success tick
    small_progress.empty()
    tick_placeholder.checkbox("Upload successful ✅", value=True, disabled=True)

    # -------- Processing -------- #
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]

    data = pd.read_csv(uploaded_file, names=column_names, skipinitialspace=True)

    if data['income'].dtype == 'object':
        data['income'] = data['income'].str.replace('.', '', regex=False)

    data = data.dropna()

    for col in data.select_dtypes(include=['object']).columns:
        data[col] = LabelEncoder().fit_transform(data[col])

    X = data.drop("income", axis=1)
    y = data["income"]

    model = joblib.load(f"model/{model_choice}.pkl")
    predictions = model.predict(X)

    st.markdown("## 📊 Model Evaluation Results")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### 📝 Classification Report")
        # st.text(classification_report(y, predictions))
        report = classification_report(y, predictions, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

    with col4:
        st.markdown("### 🔥 Confusion Matrix")
        cm = confusion_matrix(y, predictions)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    st.balloons()

else:
    st.info("Upload a test CSV file to see predictions.")