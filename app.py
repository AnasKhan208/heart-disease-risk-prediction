import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# =========================================================
# LOAD MODEL METRICS
# =========================================================

@st.cache_data
def load_metrics():

    with open("model_metrics.json", "r") as file:
        metrics = json.load(file)

    return metrics


metrics = load_metrics()

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* ================================
       MAIN APPLICATION
       ================================ */

    .stApp {
        background: #f5f7fb;
    }

    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ================================
       HEADINGS
       ================================ */

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #17324d !important;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #667085 !important;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 700;
        color: #17324d !important;
        margin-top: 25px;
        margin-bottom: 15px;
    }


    /* ================================
       STREAMLIT INPUT LABELS
       ================================ */

    [data-testid="stWidgetLabel"] {
        color: #17324d !important;
    }

    [data-testid="stWidgetLabel"] p {
        color: #17324d !important;
        font-weight: 600 !important;
    }

    [data-testid="stWidgetLabel"] label {
        color: #17324d !important;
    }


    /* ================================
       INPUT TEXT
       ================================ */

    input {
        color: #ffffff !important;
    }

    textarea {
        color: #ffffff !important;
    }


    /* ================================
       SELECTBOX
       ================================ */

    [data-baseweb="select"] {
        border-radius: 10px;
    }


    /* ================================
       BUTTON
       ================================ */

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 55px;
        font-size: 18px;
        font-weight: 700;
        border: none;
        background-color: #17324d;
        color: white;
        transition: 0.2s;
    }

    .stButton > button:hover {
        background-color: #254f73;
        color: white;
    }


    /* ================================
       INFO CARDS
       ================================ */

    .info-card {
        background: white;
        padding: 22px;
        border-radius: 15px;
        border: 1px solid #e6eaf0;
        box-shadow: 0 3px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }


    /* ================================
       HIGH RISK
       ================================ */

    .risk-high {
        background: #fff1f2;
        border-left: 7px solid #dc2626;
        padding: 25px;
        border-radius: 12px;
        margin-top: 20px;
    }


    /* ================================
       LOW RISK
       ================================ */

    .risk-low {
        background: #ecfdf3;
        border-left: 7px solid #16a34a;
        padding: 25px;
        border-radius: 12px;
        margin-top: 20px;
    }


    /* ================================
       RESULT TEXT
       ================================ */

    .result-title {
        font-size: 28px;
        font-weight: 800;
    }

    .probability {
        font-size: 38px;
        font-weight: 800;
    }


    /* ================================
       SIDEBAR
       ================================ */

    [data-testid="stSidebar"] {
        background-color: #242630;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: white !important;
    }

    [data-testid="stSidebar"] p {
        color: #f2f4f7 !important;
    }


    /* ================================
       FOOTER
       ================================ */

    .footer {
        text-align: center;
        color: #667085;
        padding: 30px;
        font-size: 14px;
    }

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "heart_disease_model.pkl"
    )

    return model


model = load_model()

# =========================================================
# LOAD MODEL METRICS
# =========================================================

@st.cache_data
def load_metrics():

    with open("model_metrics.json", "r") as file:
        metrics = json.load(file)

    return metrics


metrics = load_metrics()
# =========================================================
# LOAD FEATURE IMPORTANCE
# =========================================================

@st.cache_data
def load_feature_importance():

    with open("feature_importance.json", "r") as file:
        feature_importance = json.load(file)

    return feature_importance


feature_importance = load_feature_importance()
# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🫀 HeartCare AI")

    st.markdown(
        """
        ### About

        This application uses a machine learning
        model to estimate heart disease risk based
        on selected patient and lifestyle features.
        """
    )

    st.markdown("---")

    st.markdown("### 🤖 Model")

    st.info(
        "The application uses the best-performing "
        "classification model selected during the "
        "machine learning experiment."
    )

    st.markdown("---")

    st.markdown("### ⚠️ Disclaimer")

    st.warning(
        "This application is an educational machine "
        "learning project and must not be used as "
        "a substitute for professional medical advice."
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🫀 Heart Disease Risk Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered cardiovascular risk prediction using machine learning'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# PATIENT INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">👤 Patient Information</div>',
    unsafe_allow_html=True
)

with st.container():

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=50
        )

    with col2:

        sex = st.selectbox(
            "Sex",
            ["Male", "Female"]
        )

    with col3:

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=25.0,
            step=0.1
        )


# =========================================================
# BLOOD & CARDIOVASCULAR INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">🩸 Blood & Cardiovascular Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    resting_bp_systolic = st.number_input(
        "Resting Systolic BP",
        min_value=50,
        max_value=250,
        value=120
    )

with col2:

    resting_bp_diastolic = st.number_input(
        "Resting Diastolic BP",
        min_value=30,
        max_value=150,
        value=80
    )

with col3:

    resting_heart_rate = st.number_input(
        "Resting Heart Rate",
        min_value=30,
        max_value=220,
        value=70
    )


col1, col2, col3 = st.columns(3)

with col1:

    cholesterol_total = st.number_input(
        "Total Cholesterol",
        min_value=50,
        max_value=500,
        value=200
    )

with col2:

    hdl = st.number_input(
        "HDL",
        min_value=10,
        max_value=150,
        value=50
    )

with col3:

    ldl = st.number_input(
        "LDL",
        min_value=10,
        max_value=400,
        value=120
    )


col1, col2, col3 = st.columns(3)

with col1:

    triglycerides = st.number_input(
        "Triglycerides",
        min_value=20,
        max_value=800,
        value=150
    )

with col2:

    fasting_blood_sugar = st.number_input(
        "Fasting Blood Sugar",
        min_value=50,
        max_value=400,
        value=100
    )

with col3:

    hba1c = st.number_input(
        "HbA1c",
        min_value=3.0,
        max_value=15.0,
        value=5.5,
        step=0.1
    )


# =========================================================
# HEART TEST INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">❤️ Heart & Exercise Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    max_heart_rate_achieved = st.number_input(
        "Maximum Heart Rate",
        min_value=50,
        max_value=250,
        value=150
    )

with col2:

    chest_pain_type = st.selectbox(
        "Chest Pain Type",
        [
            "Typical angina",
            "Atypical angina",
            "Non-anginal pain",
            "Asymptomatic"
        ]
    )

with col3:

    exercise_induced_angina = st.selectbox(
        "Exercise-induced Angina",
        ["No", "Yes"]
    )


st_depression = st.number_input(
    "ST Depression",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)


# =========================================================
# LIFESTYLE INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">🏃 Lifestyle Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    smoker_status = st.selectbox(
        "Smoking Status",
        [
            "Never",
            "Former",
            "Current"
        ]
    )

with col2:

    alcohol_units_per_week = st.number_input(
        "Alcohol Units / Week",
        min_value=0.0,
        max_value=100.0,
        value=2.0,
        step=1.0
    )

with col3:

    exercise_minutes_per_week = st.number_input(
        "Exercise Minutes / Week",
        min_value=0,
        max_value=2000,
        value=150
    )


col1, col2, col3 = st.columns(3)

with col1:

    sleep_hours = st.number_input(
        "Sleep Hours / Day",
        min_value=1.0,
        max_value=15.0,
        value=7.0,
        step=0.5
    )

with col2:

    stress_score = st.number_input(
        "Stress Score",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1
    )

with col3:

    daily_steps = st.number_input(
        "Daily Steps",
        min_value=0,
        max_value=50000,
        value=7000
    )


col1, col2 = st.columns(2)

with col1:

    diet_quality_score = st.number_input(
        "Diet Quality Score",
        min_value=0.0,
        max_value=10.0,
        value=6.0,
        step=0.1
    )

with col2:

    family_history = st.selectbox(
        "Family History of Heart Disease",
        ["No", "Yes"]
    )


wearable_owner = st.selectbox(
    "Wearable Device Owner",
    ["No", "Yes"]
)


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.markdown("---")

predict_button = st.button(
    "🔍 Predict Heart Disease Risk",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    patient_data = {

        "age": age,
        "sex": sex,

        "resting_bp_systolic":
            resting_bp_systolic,

        "resting_bp_diastolic":
            resting_bp_diastolic,

        "cholesterol_total":
            cholesterol_total,

        "hdl":
            hdl,

        "ldl":
            ldl,

        "triglycerides":
            triglycerides,

        "fasting_blood_sugar":
            fasting_blood_sugar,

        "hba1c":
            hba1c,

        "bmi":
            bmi,

        "resting_heart_rate":
            resting_heart_rate,

        "max_heart_rate_achieved":
            max_heart_rate_achieved,

        "chest_pain_type":
            chest_pain_type,

        "exercise_induced_angina":
            exercise_induced_angina == "Yes",

        "st_depression":
            st_depression,

        "family_history":
            family_history == "Yes",

        "smoker_status":
            smoker_status,

        "alcohol_units_per_week":
            alcohol_units_per_week,

        "exercise_minutes_per_week":
            exercise_minutes_per_week,

        "sleep_hours":
            sleep_hours,

        "stress_score":
            stress_score,

        "wearable_owner":
            wearable_owner == "Yes",

        "daily_steps":
            daily_steps,

        "diet_quality_score":
            diet_quality_score
    }


    patient_df = pd.DataFrame(
        [patient_data]
    )


    # Prediction
    prediction = model.predict(
        patient_df
    )[0]

    probability = model.predict_proba(
        patient_df
    )[0][1]


    # =====================================================
    # RESULT
    # =====================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">📊 Prediction Result</div>',
        unsafe_allow_html=True
    )


    if prediction == 1:

        st.markdown(
            f"""
            <div class="risk-high">

                <div class="result-title">
                    ⚠️ Higher Heart Disease Risk
                </div>

                <p>
                    The machine learning model predicts
                    a higher likelihood of heart disease
                    based on the information provided.
                </p>

                <div class="probability">
                    {probability:.1%}
                </div>

                <p>
                    Estimated probability
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="risk-low">

                <div class="result-title">
                    ✅ Lower Heart Disease Risk
                </div>

                <p>
                    The machine learning model predicts
                    a lower likelihood of heart disease
                    based on the information provided.
                </p>

                <div class="probability">
                    {probability:.1%}
                </div>

                <p>
                    Estimated probability
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    # Progress bar
    st.markdown("### Risk Probability")

    st.progress(
        float(probability)
    )

# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.markdown("---")

st.markdown(
    "## 📈 Model Performance"
)

st.write(
    "The selected machine learning model was evaluated "
    "using multiple classification metrics on the test dataset."
)


# =========================================================
# SELECTED MODEL
# =========================================================

st.markdown("### 🤖 Selected Model")

st.info(
    f"**{metrics['model']}** was selected as the final model "
    "based on the machine learning evaluation."
)


# =========================================================
# PERFORMANCE METRIC CARDS
# =========================================================

st.markdown("### 📊 Evaluation Metrics")

col1, col2, col3, col4, col5 = st.columns(5)


with col1:
    st.metric(
        label="Accuracy",
        value=f"{metrics['accuracy']:.2%}"
    )


with col2:
    st.metric(
        label="Precision",
        value=f"{metrics['precision']:.2%}"
    )


with col3:
    st.metric(
        label="Recall",
        value=f"{metrics['recall']:.2%}"
    )


with col4:
    st.metric(
        label="F1 Score",
        value=f"{metrics['f1_score']:.2%}"
    )


with col5:
    st.metric(
        label="ROC-AUC",
        value=f"{metrics['roc_auc']:.2%}"
    )


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.markdown("---")

st.markdown(
    "### 🔎 Top Important Features"
)

st.write(
    "These features had the greatest influence on the "
    "predictions made by the selected machine learning model."
)


# Convert JSON data to DataFrame

importance_df = pd.DataFrame(
    feature_importance
)


# Take top 10 features

top_features = importance_df.head(10).copy()


# Clean feature names

def clean_feature_name(feature):

    feature = feature.replace("num__", "")
    feature = feature.replace("cat__", "")
    feature = feature.replace("_", " ")

    return feature.title()


top_features["Feature"] = (
    top_features["feature"]
    .apply(clean_feature_name)
)

top_features["Importance"] = (
    top_features["importance"]
)


# Prepare chart data

chart_data = top_features[
    ["Feature", "Importance"]
].set_index("Feature")


# Display chart

st.bar_chart(
    chart_data,
    horizontal=True
)
# =========================================================
# PROJECT INFORMATION
# =========================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">📚 About This Project</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        """
        <div class="info-card">

        ### 🤖 Machine Learning

        Multiple classification algorithms
        were evaluated to identify the best
        performing model.

        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        """
        <div class="info-card">

        ### 📈 Evaluation

        Models were evaluated using accuracy,
        precision, recall, F1-score and ROC-AUC.

        </div>
        """,
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        """
        <div class="info-card">

        ### 🎯 Objective

        To provide an educational ML-based
        heart disease risk prediction system.

        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">📈 Model Performance</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="color:#667085;">
    The selected machine learning model was evaluated using
    multiple classification metrics on the test dataset.
    </p>
    """,
    unsafe_allow_html=True
)


# Model name

st.markdown(
    f"""
    <div class="info-card">

    <h3>🤖 Selected Model</h3>

    <p style="font-size:22px; font-weight:700; color:#17324d;">
    {metrics["model"]}
    </p>

    </div>
    """,
    unsafe_allow_html=True
)


# Performance metrics

col1, col2, col3, col4, col5 = st.columns(5)

with col1:

    st.metric(
        "Accuracy",
        f"{final_accuracy:.2%}"
    )

with col2:

    st.metric(
        "Precision",
        f"{final_precision:.2%}"
    )

with col3:

    st.metric(
        "Recall",
        f"{final_recall:.2%}"
    )

with col4:

    st.metric(
        "F1 Score",
        f"{final_f1:.2%}"
    )

with col5:

    st.metric(
        "ROC-AUC",
        f"{final_roc_auc:.2%}"
    )
# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

    Heart Disease Risk Predictor • Machine Learning Project

    <br><br>

    ⚠️ Educational purpose only — not a medical diagnosis.

    </div>
    """,
    unsafe_allow_html=True
)
