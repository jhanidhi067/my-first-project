
# Student Success AI 🎓

An experimental machine learning project that explores student
learning patterns and provides personalized study suggestions.

## Project Overview

Student Success AI combines machine learning and data analysis
to explore study hours, attendance, assignment completion,
previous assessment scores, and sleep duration.

It compares multiple classification models and presents an
interactive dashboard for exploring fictional student profiles.

## Features

- Synthetic dataset generation
- Data quality checks
- Logistic Regression, Decision Tree, and Random Forest
- Validation-based model selection
- Held-out test evaluation
- Accuracy, precision, recall, and F1-score
- Confusion matrix and classification report
- Personalized rule-based learning suggestions
- Interactive Streamlit dashboard

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Joblib

## Project Structure

    my-first-project/
    ├── index.html
    ├── train_model.py
    ├── app.py
    ├── requirements.txt
    ├── README.md
    ├── data/
    ├── models/
    └── results/

The data, models, and results folders are generated when the
Python application runs. They do not need to be created manually
in the GitHub editor.

## Installation

Install Python 3.10 or newer, then run:

    pip install -r requirements.txt

## Run the Dashboard

From the project directory, run:

    streamlit run app.py

The application will open in your browser.

## How It Works

1. Generate fictional sample student records.
2. Split the data into development and test sets.
3. Compare three classification models using validation data.
4. Evaluate the selected model on held-out test data.
5. Use the dashboard to explore a fictional student profile.
6. Display practical study suggestions based on entered values.

## Limitations and Responsible Use

This is an educational prototype that uses synthetic data.
Its results do not establish real-world accuracy, fairness,
or usefulness for actual students.

The application must not be used to rank, penalize, admit,
or exclude students. Any future real-world use would require
appropriate data permissions, rigorous validation, privacy
protection, fairness evaluation, and human oversight.

## Author

Developed as a Python machine learning portfolio project.
