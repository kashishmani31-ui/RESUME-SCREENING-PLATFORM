# 📄 AI Resume Screening System

An ML-based Resume Screening System that analyzes resumes and predicts the likelihood of a candidate being shortlisted for a job.

## 📌 Project Overview

The AI Resume Screening System is a Machine Learning-based application developed using Python and Streamlit.

The system allows users to:

- Enter a Job Description
- Upload one or multiple resume PDFs
- Extract important candidate information
- Detect technical skills
- Compare resume skills with the Job Description
- Predict shortlisting probability using a trained Machine Learning model
- Compare multiple candidates
- Visualize candidate features using charts

This project is developed for educational and demonstration purposes.

## 🚀 Features

### 📋 Job Description Analysis
Users can enter a Job Description and the system identifies relevant technical skills from it.

### 📄 Resume Upload
The application supports uploading multiple PDF resumes at the same time.

### 🔍 Resume Feature Extraction

The system extracts:

- Years of Experience
- Education Level
- Project Count
- GitHub Activity
- Resume Length
- Technical Skills

### 🛠️ Skill Detection

The system detects predefined technical skills such as:

- Python
- Machine Learning
- SQL
- HTML
- CSS
- JavaScript
- Java
- C++
- React
- Node.js
- Git
- GitHub
- Django
- Flask
- TensorFlow
- PyTorch

### 🎯 Job Description Skill Matching

The skills detected in the resume are compared with the technical skills required in the Job Description.

A skill match percentage is calculated for each resume.

### 🤖 Machine Learning Prediction

The trained Machine Learning model predicts whether a candidate is likely to be shortlisted.

The application also displays a predicted shortlisting probability.

### 🔄 Multiple Resume Comparison

Multiple resumes can be uploaded and compared using:

- Experience
- Education
- Projects
- GitHub Activity
- Skills Match
- Shortlisting Probability

### 📊 Data Visualization

The application provides charts for:

- Candidate profile
- Feature comparison
- Shortlisting probability
- Feature importance

## 🧠 Machine Learning

The project uses a trained Machine Learning pipeline for candidate screening.

The model uses features such as:

- Years of Experience
- Skills Match Score
- Education Level
- Project Count
- Resume Length
- GitHub Activity

The trained model is saved as:

`resume_screening_model.pkl`

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- Joblib
- PyPDF

## 📁 Project Structure

```text
AI-Resume-Screening-System/
│
├── app.py
├── resume_screening_model.pkl
├── requirements.txt
├── README.md
└── .gitignore