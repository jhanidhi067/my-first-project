
"""Student Success AI - Interactive Streamlit Dashboard."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from train_model import main as train_model_main, FEATURES

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
METRICS_PATH = BASE_DIR / "results" / "test_metrics.csv"
DATA_PATH = BASE_DIR / "data" / "student_data.csv"

st.set_page_config(
    page_title="Student Success AI",
    page_icon="🎓",
    layout="wide",
)

# Train the model if this is the first application run.
if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    with st.spinner("Preparing the machine learning model..."):
        train_model_main()

model = joblib.load(MODEL_PATH)
metrics = pd.read_csv(METRICS_PATH).iloc[0]
dataset = pd.read_csv(DATA_PATH)

st.title("🎓 Student Success AI")
st.subheader("Academic Insights and Personalized Learning Support")

st.info(
    "Educational prototype using fictional synthetic data. "
    "Predictions are experimental estimates, not verified judgments "
    "about actual students."
)

# Model evaluation summary
st.header("Model Evaluation")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Selected model", str(metrics["model"]))
c2.metric("Test accuracy", f'{metrics["accuracy"]:.1%}')
c3.metric("Test precision", f'{metrics["precision"]:.1%}')
c4.metric("Test recall", f'{metrics["recall"]:.1%}')

st.caption(
    "Metrics are measured on a held-out test set generated from "
    "the same synthetic data process. They do not represent "
    "real-world student prediction performance."
)

st.divider()

# Student input form
st.header("Check a Student Learning Profile")
st.write(
    "Enter a student's learning details to receive an experimental "
    "model estimate and practical study suggestions."
)

with st.form("student_profile"):
    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.slider(
            "Study hours per day",
            min_value=0.5,
            max_value=8.0,
            value=3.0,
            step=0.5,
        )
        attendance = st.slider(
            "Attendance (%)",
            min_value=40,
            max_value=100,
            value=75,
        )
        assignments = st.slider(
            "Assignment completion (%)",
            min_value=30,
            max_value=100,
            value=70,
        )

    with col2:
        previous_score = st.slider(
            "Previous assessment score (%)",
            min_value=25,
            max_value=100,
            value=60,
        )
        sleep_hours = st.slider(
            "Sleep hours per night",
            min_value=4.0,
            max_value=10.0,
            value=7.0,
            step=0.5,
        )

    submitted = st.form_submit_button(
        "Analyze Learning Profile",
        type="primary",
    )

if submitted:
    student = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance": attendance,
        "assignment_completion": assignments,
        "previous_score": previous_score,
        "sleep_hours": sleep_hours,
    }], columns=FEATURES)

    prediction = int(model.predict(student)[0])

    st.header("Learning Support Insights")

    if prediction == 1:
        st.warning(
            "The experimental model flags this profile as potentially "
            "benefiting from additional academic support."
        )
    else:
        st.success(
            "The experimental model does not flag this profile for "
            "additional support under its learned pattern."
        )

    st.caption(
        "This is not a diagnosis or a reliable assessment of ability. "
        "A teacher or student should consider personal circumstances "
        "and additional evidence before deciding on any action."
    )

    recommendations = []

    if study_hours < 2:
        recommendations.append(
            "Try a manageable study routine, starting with 25-minute "
            "focused sessions and short breaks."
        )

    if attendance < 75:
        recommendations.append(
            "Review missed lessons and plan how to attend upcoming "
            "classes or access their learning materials."
        )

    if assignments < 70:
        recommendations.append(
            "Break assignments into smaller tasks and set deadlines "
            "for each part."
        )

    if previous_score < 50:
        recommendations.append(
            "Review earlier assessment feedback and practise the "
            "topics that need improvement."
        )

    if sleep_hours < 7:
        recommendations.append(
            "Consider a consistent sleep schedule that supports "
            "rest and concentration."
        )

    if not recommendations:
        recommendations.append(
            "Maintain a balanced routine and review learning goals "
            "regularly. Ask for feedback when a topic is unclear."
        )

    st.subheader("Personalized Suggestions")
    for item in recommendations:
        st.markdown(f"- {item}")

# Dataset overview
st.divider()
st.header("Explore the Sample Dataset")

left, right = st.columns(2)

with left:
    st.write("First 10 fictional student records")
    st.dataframe(dataset.head(10), use_container_width=True)

with right:
    st.write("Synthetic target distribution")
    counts = dataset["at_risk"].map({
        0: "Not flagged",
        1: "Support flag",
    }).value_counts()
    st.bar_chart(counts)

st.caption(
    "All sample records are synthetic. This application should not "
    "be used to rank, penalize, admit, or exclude students."
)
