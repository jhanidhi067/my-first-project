
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

st.set_page_config(
    page_title="Student Success AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CREATE SYNTHETIC DATA AND MODEL ----------

@st.cache_resource
def build_project():
    rng = np.random.default_rng(42)
    n = 1200

    df = pd.DataFrame({
        "study_hours": np.round(rng.uniform(0, 10, n), 1),
        "attendance": rng.integers(40, 101, n),
        "assignment_completion": rng.integers(20, 101, n),
        "previous_score": rng.integers(25, 101, n),
        "sleep_hours": np.round(rng.uniform(3, 10, n), 1)
    })

    risk_score = (
        0.30 * (10 - df["study_hours"]) / 10
        + 0.25 * (100 - df["attendance"]) / 100
        + 0.20 * (100 - df["assignment_completion"]) / 100
        + 0.25 * (100 - df["previous_score"]) / 100
    )

    df["at_risk"] = (risk_score > 0.48).astype(int)

    features = [
        "study_hours",
        "attendance",
        "assignment_completion",
        "previous_score",
        "sleep_hours"
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        df[features],
        df["at_risk"],
        test_size=0.20,
        random_state=42,
        stratify=df["at_risk"]
    )

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        )
    }

    return model, df, features, metrics


model, data, features, metrics = build_project()

# ---------- SIDEBAR ----------

st.sidebar.title("🎓 Student Success AI")
st.sidebar.caption("Academic Insights Platform")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "🧑‍🎓 Student Simulator",
        "📊 Data Explorer",
        "🧠 Model Insights"
    ]
)

st.sidebar.divider()
st.sidebar.info(
    "Portfolio prototype using synthetic data. "
    "Predictions are experimental and are not verified "
    "assessments of real students."
)

# ---------- HEADER ----------

st.title("🎓 Student Success AI")
st.write(
    "Explore academic indicators, simulate student profiles, "
    "and discover potential learning-support opportunities."
)

st.caption("Interactive machine-learning demonstration")

# ---------- OVERVIEW ----------

if page == "🏠 Overview":

    st.subheader("📌 Project Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sample records", f"{len(data):,}")
    c2.metric("Input features", len(features))
    c3.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
    c4.metric("Model", "Random Forest")

    st.caption(
        "Metrics are measured on a held-out synthetic test set. "
        "They do not indicate performance on real students."
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Student support flags")
        counts = (
            data["at_risk"]
            .map({0: "Not flagged", 1: "Flagged"})
            .value_counts()
        )
        st.bar_chart(counts)

    with right:
        st.subheader("Average learning indicators")
        averages = data[features].mean().rename({
            "study_hours": "Study hours",
            "attendance": "Attendance %",
            "assignment_completion": "Assignments %",
            "previous_score": "Previous score",
            "sleep_hours": "Sleep hours"
        })
        st.bar_chart(averages)

    st.divider()
    st.subheader("🚀 Explore the platform")

    c1, c2, c3 = st.columns(3)
    c1.markdown("**Student Simulator**")
    c1.write("Adjust inputs and inspect model predictions.")
    c2.markdown("**Data Explorer**")
    c2.write("Filter, inspect, and download sample records.")
    c3.markdown("**Model Insights**")
    c3.write("Review model metrics and feature importance.")

# ---------- STUDENT SIMULATOR ----------

elif page == "🧑‍🎓 Student Simulator":

    st.subheader("🧑‍🎓 Interactive Student Profile")
    st.write("Adjust the inputs below to update the model estimate.")

    col1, col2 = st.columns(2)

    with col1:
        study = st.slider(
            "Daily study hours", 0.0, 10.0, 4.0, 0.5
        )
        attendance = st.slider(
            "Attendance (%)", 0, 100, 70
        )
        assignments = st.slider(
            "Assignment completion (%)", 0, 100, 70
        )

    with col2:
        previous = st.slider(
            "Previous score (%)", 0, 100, 60
        )
        sleep = st.slider(
            "Sleep hours per night", 0.0, 12.0, 7.0, 0.5
        )

    student = pd.DataFrame([{
        "study_hours": study,
        "attendance": attendance,
        "assignment_completion": assignments,
        "previous_score": previous,
        "sleep_hours": sleep
    }])[features]

    prediction = int(model.predict(student)[0])
    probability = float(model.predict_proba(student)[0][1])

    st.divider()
    st.subheader("📍 Model Estimate")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Estimated flag probability",
        f"{probability:.1%}"
    )
    c2.metric(
        "Study hours",
        f"{study:.1f} hrs/day"
    )
    c3.metric(
        "Attendance",
        f"{attendance}%"
    )

    if prediction == 1:
        st.warning(
            "The model flagged this example for possible additional support."
        )
    else:
        st.success(
            "The model did not flag this example."
        )

    st.caption(
        "The probability is not calibrated or validated for real-world use. "
        "It is not the probability that a student will fail."
    )

    st.subheader("💡 Suggested learning-support actions")

    suggestions = []

    if study < 5:
        suggestions.append(
            "Try a consistent study schedule with small daily goals."
        )
    if attendance < 75:
        suggestions.append(
            "Explore attendance barriers and available academic support."
        )
    if assignments < 70:
        suggestions.append(
            "Break assignments into manageable tasks and ask for help early."
        )
    if previous < 60:
        suggestions.append(
            "Review challenging topics and practice with feedback."
        )
    if sleep < 7:
        suggestions.append(
            "Consider a regular sleep routine that supports rest."
        )

    if suggestions:
        for item in suggestions:
            st.markdown(f"- {item}")
    else:
        st.write(
            "Continue reviewing your learning goals and maintain helpful habits."
        )

    st.subheader("📊 Your profile compared with the sample average")

    comparison = pd.DataFrame({
        "Your profile": [
            study, attendance, assignments, previous, sleep
        ],
        "Sample average": [
            data["study_hours"].mean(),
            data["attendance"].mean(),
            data["assignment_completion"].mean(),
            data["previous_score"].mean(),
            data["sleep_hours"].mean()
        ]
    }, index=[
        "Study hours",
        "Attendance (%)",
        "Assignments (%)",
        "Previous score (%)",
        "Sleep hours"
    ])

    st.bar_chart(comparison)

    st.download_button(
        "⬇️ Download this profile as CSV",
        data=student.to_csv(index=False),
        file_name="student_profile.csv",
        mime="text/csv"
    )

# ---------- DATA EXPLORER ----------

elif page == "📊 Data Explorer":

    st.subheader("📊 Synthetic Student Data Explorer")
    st.write("Filter records to explore patterns in the generated dataset.")

    flagged_filter = st.selectbox(
        "Support-flag filter",
        ["All students", "Flagged", "Not flagged"]
    )

    filtered = data.copy()

    if flagged_filter == "Flagged":
        filtered = filtered[filtered["at_risk"] == 1]
    elif flagged_filter == "Not flagged":
        filtered = filtered[filtered["at_risk"] == 0]

    min_attendance = st.slider(
        "Minimum attendance (%)", 0, 100, 0
    )

    filtered = filtered[
        filtered["attendance"] >= min_attendance
    ]

    st.write(f"**Matching records:** {len(filtered)}")

    st.dataframe(
        filtered.head(200),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download filtered records",
        data=filtered.to_csv(index=False),
        file_name="student_data_filtered.csv",
        mime="text/csv"
    )

    st.subheader("Attendance and previous scores")

    st.scatter_chart(
        filtered,
        x="attendance",
        y="previous_score",
        color="at_risk"
    )

    st.caption(
        "All records are synthetic and do not represent actual students."
    )

# ---------- MODEL INSIGHTS ----------

elif page == "🧠 Model Insights":

    st.subheader("🧠 Model Performance")

    c1, c2, c3 = st.columns(3)
    c1.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
    c2.metric("Test precision", f"{metrics['precision']:.1%}")
    c3.metric("Test recall", f"{metrics['recall']:.1%}")

    st.caption(
        "These metrics describe a held-out test set generated from the same "
        "synthetic process used to create the training data."
    )

    st.divider()
    st.subheader("Feature Importance")

    importance = pd.Series(
        model.feature_importances_,
        index=features
    ).sort_values(ascending=True)

    st.bar_chart(importance)

    st.write(
        "Feature importance describes how the trained Random Forest uses "
        "features across its trees. It does not establish causation, fairness, "
        "or the reason behind an individual prediction."
    )

    st.divider()
    st.subheader("🛡️ Responsible AI Checklist")

    st.markdown(
        "- **Data:** Synthetic records only.\n"
        "- **Validation:** Real-world performance has not been established.\n"
        "- **Fairness:** Demographic fairness has not been evaluated.\n"
        "- **Privacy:** Do not enter identifying student information.\n"
        "- **Human oversight:** Do not use predictions to grade, exclude, "
        "or penalize students.\n"
        "- **Next step:** Validate with appropriately authorized data and "
        "evaluate errors, calibration, and fairness before real use."
    )

st.divider()
st.caption(
    "Student Success AI | Portfolio prototype | "
    "Synthetic data only"
)

