import streamlit as st
import pandas as pd
import joblib
from pypdf import PdfReader
import re
import math


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = joblib.load("resume_screening_model.pkl")
except Exception as e:
    st.error("Could not load resume_screening_model.pkl")
    st.error(f"Error: {e}")
    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("📄 AI Resume Screening System")

st.write(
    "An ML-based system that analyzes resumes against a "
    "Job Description and predicts the likelihood of shortlisting."
)


# ============================================================
# JOB DESCRIPTION
# ============================================================

st.header("📋 Job Description")

job_description = st.text_area(
    "Paste the Job Description here",
    height=180,
    placeholder=(
        "Example:\n"
        "We are looking for a Software Developer with skills "
        "in Python, SQL, Machine Learning, Git and React."
    )
)

if job_description.strip():
    st.success("✅ Job Description added successfully.")
else:
    st.info(
        "ℹ️ Add a Job Description to calculate JD-specific "
        "skill matching."
    )


# ============================================================
# SKILLS LIST
# ============================================================

ALL_SKILLS = [
    "python",
    "machine learning",
    "sql",
    "html",
    "css",
    "javascript",
    "java",
    "c++",
    "react",
    "node.js",
    "git",
    "github",
    "django",
    "flask",
    "tensorflow",
    "pytorch"
]


# ============================================================
# HELPER: SKILL REGEX
# ============================================================

def skill_pattern(skill):

    if skill == "c++":
        return r"(?<!\w)c\+\+(?!\w)"

    if skill == "node.js":
        return r"(?<!\w)node\.js(?!\w)"

    if skill == "machine learning":
        return r"(?<!\w)machine\s+learning(?!\w)"

    return (
        r"(?<!\w)"
        + re.escape(skill)
        + r"(?!\w)"
    )


# ============================================================
# EXTRACT EXPERIENCE
# ============================================================

def extract_experience(text):

    patterns = [
        r"(\d+)\s*(?:years?|yrs?)\s*(?:of)?\s*experience",
        r"experience\s*[:\-]?\s*(\d+)\s*(?:years?|yrs?)",
        r"(\d+)\s*\+\s*(?:years?|yrs?)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            return int(match.group(1))

    return 0


# ============================================================
# EXTRACT EDUCATION
# ============================================================

def extract_education(text):

    text_lower = text.lower()

    if any(
        word in text_lower
        for word in [
            "phd",
            "ph.d",
            "doctorate"
        ]
    ):
        return "PhD"

    if any(
        word in text_lower
        for word in [
            "master",
            "mca",
            "m.tech",
            "mtech",
            "mba",
            "msc",
            "m.sc"
        ]
    ):
        return "Masters"

    if any(
        word in text_lower
        for word in [
            "bachelor",
            "b.tech",
            "btech",
            "bca",
            "bsc",
            "b.sc"
        ]
    ):
        return "Bachelors"

    if any(
        word in text_lower
        for word in [
            "high school",
            "12th",
            "higher secondary"
        ]
    ):
        return "High School"

    return "High School"


# ============================================================
# EXTRACT PROJECT COUNT
# ============================================================

def extract_projects(text):

    patterns = [
        r"projects?\s*[:\-]?\s*(\d+)",
        r"(\d+)\s*projects?"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            return int(match.group(1))

    return 0


# ============================================================
# EXTRACT GITHUB ACTIVITY
# ============================================================

def extract_github_activity(text):

    patterns = [
        r"github\s*activity\s*[:\-]?\s*(\d+)",
        r"(\d+)\s*(?:github\s*)?contributions?"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            return int(match.group(1))

    return 0


# ============================================================
# DETECT SKILLS
# ============================================================

def detect_skills(text):

    text_lower = text.lower()

    detected = []

    for skill in ALL_SKILLS:

        pattern = skill_pattern(skill)

        if re.search(
            pattern,
            text_lower
        ):
            detected.append(skill)

    return detected


# ============================================================
# JD SKILLS
# ============================================================

def extract_jd_skills(job_description):

    if not job_description.strip():
        return []

    text_lower = job_description.lower()

    jd_skills = []

    for skill in ALL_SKILLS:

        pattern = skill_pattern(skill)

        if re.search(
            pattern,
            text_lower
        ):
            jd_skills.append(skill)

    return jd_skills


# ============================================================
# JD SKILL MATCH SCORE
# ============================================================

def calculate_jd_skill_score(
    detected_skills,
    jd_skills
):

    if not jd_skills:
        return 0.0

    matched = 0

    for skill in jd_skills:

        if skill in detected_skills:
            matched += 1

    return (
        matched
        / len(jd_skills)
    ) * 100


# ============================================================
# GENERAL SKILL SCORE
# ============================================================

def calculate_general_skill_score(
    detected_skills
):

    if not ALL_SKILLS:
        return 0.0

    return (
        len(detected_skills)
        / len(ALL_SKILLS)
    ) * 100


# ============================================================
# RESUME UPLOAD
# ============================================================

st.header("📤 Upload Resumes")

uploaded_files = st.file_uploader(
    "Select one or more resume PDFs",
    type=["pdf"],
    accept_multiple_files=True
)


# ============================================================
# PROCESS RESUMES
# ============================================================

if uploaded_files:

    st.success(
        f"✅ {len(uploaded_files)} resume(s) uploaded."
    )

    comparison_results = []

    for file_number, uploaded_file in enumerate(
        uploaded_files,
        start=1
    ):

        st.divider()

        st.header(
            f"📄 Resume {file_number}: "
            f"{uploaded_file.name}"
        )

        # ----------------------------------------------------
        # FILE SIZE
        # ----------------------------------------------------

        file_size_mb = (
            uploaded_file.size
            / (1024 * 1024)
        )

        if file_size_mb > 5:

            st.error(
                "❌ File is larger than 5 MB. "
                "This resume was skipped."
            )

            continue

        # ----------------------------------------------------
        # PDF EXTRACTION
        # ----------------------------------------------------

        try:

            pdf_reader = PdfReader(
                uploaded_file
            )

            resume_text = ""

            for page in pdf_reader.pages:

                page_text = page.extract_text()

                if page_text:
                    resume_text += (
                        page_text + "\n"
                    )

        except Exception as e:

            st.error(
                f"❌ Could not read "
                f"{uploaded_file.name}"
            )

            st.error(
                f"Error: {e}"
            )

            continue

        # ----------------------------------------------------
        # TEXT CHECK
        # ----------------------------------------------------

        if len(
            resume_text.strip()
        ) < 30:

            st.error(
                "❌ Not enough text could be "
                "extracted from this PDF."
            )

            continue

        st.success(
            "✅ Resume text extracted successfully."
        )

        # ----------------------------------------------------
        # RESUME TEXT
        # ----------------------------------------------------

        with st.expander(
            "📄 View Resume Text"
        ):

            st.write(
                resume_text
            )

        # ----------------------------------------------------
        # FEATURE EXTRACTION
        # ----------------------------------------------------

        years_experience = (
            extract_experience(
                resume_text
            )
        )

        education_level = (
            extract_education(
                resume_text
            )
        )

        project_count = (
            extract_projects(
                resume_text
            )
        )

        github_activity = (
            extract_github_activity(
                resume_text
            )
        )

        detected_skills = (
            detect_skills(
                resume_text
            )
        )

        general_skill_score = (
            calculate_general_skill_score(
                detected_skills
            )
        )

        jd_skills = extract_jd_skills(
            job_description
        )

        if jd_skills:

            jd_skill_score = (
                calculate_jd_skill_score(
                    detected_skills,
                    jd_skills
                )
            )

            skills_match_score = (
                jd_skill_score
            )

        else:

            jd_skill_score = 0.0

            skills_match_score = (
                general_skill_score
            )

        resume_length = len(
            resume_text
        )

        # ----------------------------------------------------
        # EXTRACTED INFORMATION
        # ----------------------------------------------------

        st.subheader(
            "🔍 Extracted Information"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Experience",
                f"{years_experience} years"
            )

        with col2:

            st.metric(
                "Education",
                education_level
            )

        with col3:

            st.metric(
                "Projects",
                project_count
            )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "GitHub Activity",
                github_activity
            )

        with col2:

            st.metric(
                "Resume Length",
                f"{resume_length}"
            )

        with col3:

            st.metric(
                "Skills Match",
                f"{skills_match_score:.2f}%"
            )

        # ----------------------------------------------------
        # SKILL ANALYSIS
        # ----------------------------------------------------

        st.subheader(
            "🧠 Resume Skill Analysis"
        )

        if jd_skills:

            st.markdown(
                "### 🎯 Skills Required by Job Description"
            )

            st.write(
                ", ".join(
                    skill.title()
                    for skill in jd_skills
                )
            )

            matched_jd_skills = [
                skill
                for skill in jd_skills
                if skill in detected_skills
            ]

            missing_jd_skills = [
                skill
                for skill in jd_skills
                if skill not in detected_skills
            ]

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    "### ✅ JD Skills Matched"
                )

                if matched_jd_skills:

                    for skill in matched_jd_skills:

                        st.write(
                            f"🟢 {skill.title()}"
                        )

                else:

                    st.write(
                        "No JD skills matched."
                    )

            with col2:

                st.markdown(
                    "### ⚪ JD Skills Missing"
                )

                if missing_jd_skills:

                    for skill in missing_jd_skills:

                        st.write(
                            f"⚪ {skill.title()}"
                        )

                else:

                    st.write(
                        "All JD skills detected."
                    )

            st.markdown(
                "### 📊 JD Skill Match"
            )

            st.progress(
                min(
                    int(jd_skill_score),
                    100
                )
            )

            st.write(
                f"JD Skill Match Score: "
                f"**{jd_skill_score:.2f}%**"
            )

        else:

            st.warning(
                "No Job Description was provided. "
                "The general predefined skill list "
                "is being used."
            )

            st.markdown(
                "### ✅ Detected Skills"
            )

            if detected_skills:

                for skill in detected_skills:

                    st.write(
                        f"🟢 {skill.title()}"
                    )

            else:

                st.write(
                    "No predefined skills detected."
                )

            st.write(
                f"General Skills Score: "
                f"**{general_skill_score:.2f}%**"
            )

        # ----------------------------------------------------
        # CANDIDATE DETAILS
        # ----------------------------------------------------

        st.subheader(
            "👤 Candidate Details"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.text_input(
                "Years of Experience",
                value=str(
                    years_experience
                ),
                disabled=True,
                key=f"exp_{file_number}"
            )

            st.text_input(
                "Education Level",
                value=education_level,
                disabled=True,
                key=f"edu_{file_number}"
            )

            st.text_input(
                "Project Count",
                value=str(
                    project_count
                ),
                disabled=True,
                key=f"projects_{file_number}"
            )

        with col2:

            st.text_input(
                "Skills Match Score",
                value=(
                    f"{skills_match_score:.2f}%"
                ),
                disabled=True,
                key=f"skills_{file_number}"
            )

            st.text_input(
                "Resume Length",
                value=str(
                    resume_length
                ),
                disabled=True,
                key=f"length_{file_number}"
            )

            st.text_input(
                "GitHub Activity",
                value=str(
                    github_activity
                ),
                disabled=True,
                key=f"github_{file_number}"
            )

        # ----------------------------------------------------
        # MODEL INPUT
        # ----------------------------------------------------

        candidate = pd.DataFrame(
            {
                "years_experience": [
                    years_experience
                ],
                "skills_match_score": [
                    skills_match_score
                ],
                "education_level": [
                    education_level
                ],
                "project_count": [
                    project_count
                ],
                "resume_length": [
                    resume_length
                ],
                "github_activity": [
                    github_activity
                ]
            }
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        try:

            prediction = model.predict(
                candidate
            )[0]

            probabilities = (
                model.predict_proba(
                    candidate
                )[0]
            )

            classes = list(
                model.classes_
            )

            if 1 in classes:

                positive_index = (
                    classes.index(1)
                )

                probability = float(
                    probabilities[
                        positive_index
                    ]
                )

            else:

                probability = float(
                    max(probabilities)
                )

        except Exception as e:

            st.error(
                "❌ Model prediction failed."
            )

            st.error(
                f"Error: {e}"
            )

            continue

        # ----------------------------------------------------
        # SCREENING RESULT
        # ----------------------------------------------------

        st.subheader(
            "🔮 Screening Result"
        )

        if prediction == 1:

            st.success(
                "✅ Candidate is likely to be "
                "SHORTLISTED"
            )

            result_text = "Shortlisted"

        else:

            st.error(
                "❌ Candidate is likely NOT "
                "to be shortlisted"
            )

            result_text = "Not Shortlisted"

        st.metric(
            "Shortlisting Probability",
            f"{probability * 100:.2f}%"
        )

        st.progress(
            max(
                0.0,
                min(
                    probability,
                    1.0
                )
            )
        )

        # ----------------------------------------------------
        # RESULT EXPLANATION
        # ----------------------------------------------------

        if probability >= 0.75:

            st.info(
                "The trained model gives a high "
                "predicted probability based on "
                "the extracted features."
            )

        elif probability >= 0.50:

            st.warning(
                "The trained model gives a moderate "
                "predicted probability based on "
                "the extracted features."
            )

        else:

            st.info(
                "The trained model gives a lower "
                "predicted probability based on "
                "the extracted features."
            )

        # ----------------------------------------------------
        # CANDIDATE SUMMARY
        # ----------------------------------------------------

        st.subheader(
            "📋 Candidate Summary"
        )

        summary = pd.DataFrame(
            {
                "Feature": [
                    "Years of Experience",
                    "Skills Match Score",
                    "Education Level",
                    "Project Count",
                    "Resume Length",
                    "GitHub Activity"
                ],
                "Value": [
                    years_experience,
                    f"{skills_match_score:.2f}%",
                    education_level,
                    project_count,
                    resume_length,
                    github_activity
                ]
            }
        )

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # DASHBOARD
        # ----------------------------------------------------

        st.subheader(
            "📊 Candidate Dashboard"
        )

        experience_score = min(
            (years_experience / 10) * 100,
            100
        )

        project_score = min(
            (project_count / 10) * 100,
            100
        )

        github_score = min(
            (github_activity / 500) * 100,
            100
        )

        resume_score = min(
            (resume_length / 1000) * 100,
            100
        )

        skills_score = min(
            skills_match_score,
            100
        )

        dashboard_data = pd.DataFrame(
            {
                "Feature": [
                    "Experience",
                    "Projects",
                    "GitHub Activity",
                    "Resume Length",
                    "Skills Match"
                ],
                "Score": [
                    round(
                        experience_score,
                        2
                    ),
                    round(
                        project_score,
                        2
                    ),
                    round(
                        github_score,
                        2
                    ),
                    round(
                        resume_score,
                        2
                    ),
                    round(
                        skills_score,
                        2
                    )
                ]
            }
        )

        st.bar_chart(
            dashboard_data.set_index(
                "Feature"
            )
        )

        # ----------------------------------------------------
        # SAVE RESULT FOR COMPARISON
        # ----------------------------------------------------

        comparison_results.append(
            {
                "Resume": uploaded_file.name,
                "Experience": years_experience,
                "Education": education_level,
                "Projects": project_count,
                "GitHub": github_activity,
                "Resume Length": resume_length,
                "Skills Match": round(
                    skills_match_score,
                    2
                ),
                "Probability": round(
                    probability * 100,
                    2
                ),
                "Result": result_text
            }
        )

    # ========================================================
    # MULTI-RESUME COMPARISON
    # ========================================================

    if comparison_results:

        st.divider()

        st.header(
            "🔄 Compare Uploaded Resumes"
        )

        comparison_df = pd.DataFrame(
            comparison_results
        )

        # ----------------------------------------------------
        # COMPARISON TABLE
        # ----------------------------------------------------

        st.subheader(
            "📋 Resume Comparison"
        )

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # PROBABILITY COMPARISON
        # ----------------------------------------------------

        st.subheader(
            "📊 Shortlisting Probability Comparison"
        )

        probability_chart = (
            comparison_df[
                [
                    "Resume",
                    "Probability"
                ]
            ]
            .set_index("Resume")
        )

        st.bar_chart(
            probability_chart
        )

        # ----------------------------------------------------
        # FEATURE COMPARISON
        # ----------------------------------------------------

        st.subheader(
            "📈 Candidate Feature Comparison"
        )

        feature_columns = [
            "Experience",
            "Projects",
            "GitHub",
            "Skills Match"
        ]

        feature_chart = (
            comparison_df[
                ["Resume"] + feature_columns
            ]
            .set_index("Resume")
        )

        st.bar_chart(
            feature_chart
        )

        # ----------------------------------------------------
        # NORMALIZED PROFILE COMPARISON
        # ----------------------------------------------------

        st.subheader(
            "📊 Normalized Candidate Profile"
        )

        normalized_rows = []

        for row in comparison_results:

            normalized_rows.append(
                {
                    "Resume": row["Resume"],
                    "Experience": min(
                        row["Experience"] / 10 * 100,
                        100
                    ),
                    "Projects": min(
                        row["Projects"] / 10 * 100,
                        100
                    ),
                    "GitHub": min(
                        row["GitHub"] / 500 * 100,
                        100
                    ),
                    "Skills Match": min(
                        row["Skills Match"],
                        100
                    )
                }
            )

        normalized_df = pd.DataFrame(
            normalized_rows
        )

        st.dataframe(
            normalized_df.round(2),
            use_container_width=True,
            hide_index=True
        )

        normalized_chart = (
            normalized_df
            .set_index("Resume")
        )

        st.bar_chart(
            normalized_chart
        )

        st.info(
            "The comparison shows extracted resume "
            "features and model predictions for all "
            "uploaded resumes. It is intended for "
            "educational demonstration."
        )


# ============================================================
# MODEL COMPARISON
# ============================================================

st.divider()

st.header(
    "📊 Model Comparison"
)

model_comparison = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest"
        ],
        "Accuracy": [
            "90.55%",
            "86.42%",
            "89.87%"
        ]
    }
)

st.dataframe(
    model_comparison,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.header(
    "📈 Important Features"
)

try:

    if not hasattr(
        model,
        "named_steps"
    ):

        st.info(
            "Feature importance is available "
            "for the trained pipeline model."
        )

    else:

        preprocessor = (
            model.named_steps.get(
                "preprocessor"
            )
        )

        classifier = (
            model.named_steps.get(
                "classifier"
            )
        )

        if (
            preprocessor is not None
            and classifier is not None
            and hasattr(
                classifier,
                "coef_"
            )
        ):

            feature_names = (
                preprocessor
                .get_feature_names_out()
            )

            coefficients = (
                classifier.coef_[0]
            )

            coefficient_data = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Importance": abs(
                        coefficients
                    )
                }
            )

            grouped_features = []

            education_values = []

            for _, row in (
                coefficient_data.iterrows()
            ):

                feature_name = row[
                    "Feature"
                ]

                importance_value = row[
                    "Importance"
                ]

                if feature_name.startswith(
                    "cat__education_level_"
                ):

                    education_values.append(
                        importance_value
                    )

                else:

                    clean_name = (
                        feature_name
                    )

                    if clean_name.startswith(
                        "num__"
                    ):

                        clean_name = (
                            clean_name
                            .replace(
                                "num__",
                                ""
                            )
                        )

                    clean_name = (
                        clean_name
                        .replace(
                            "_",
                            " "
                        )
                        .title()
                    )

                    grouped_features.append(
                        {
                            "Feature": clean_name,
                            "Importance": (
                                importance_value
                            )
                        }
                    )

            if education_values:

                education_importance = math.sqrt(
                    sum(
                        value ** 2
                        for value
                        in education_values
                    )
                )

                grouped_features.append(
                    {
                        "Feature":
                            "Education Level",
                        "Importance":
                            education_importance
                    }
                )

            actual_importance = pd.DataFrame(
                grouped_features
            )

            if not actual_importance.empty:

                total_importance = (
                    actual_importance[
                        "Importance"
                    ].sum()
                )

                if total_importance > 0:

                    actual_importance[
                        "Importance (%)"
                    ] = (
                        actual_importance[
                            "Importance"
                        ]
                        / total_importance
                    ) * 100

                else:

                    actual_importance[
                        "Importance (%)"
                    ] = 0

                actual_importance = (
                    actual_importance
                    .sort_values(
                        by="Importance (%)",
                        ascending=False
                    )
                    .reset_index(
                        drop=True
                    )
                )

                chart_data = (
                    actual_importance[
                        [
                            "Feature",
                            "Importance (%)"
                        ]
                    ]
                    .set_index(
                        "Feature"
                    )
                )

                st.bar_chart(
                    chart_data
                )

                st.write(
                    "Feature importance is calculated "
                    "from the absolute Logistic Regression "
                    "coefficients. Education categories "
                    "are combined into one Education Level "
                    "feature."
                )

                display_importance = (
                    actual_importance[
                        [
                            "Feature",
                            "Importance (%)"
                        ]
                    ].copy()
                )

                display_importance[
                    "Importance (%)"
                ] = (
                    display_importance[
                        "Importance (%)"
                    ].round(2)
                )

                st.dataframe(
                    display_importance,
                    use_container_width=True,
                    hide_index=True
                )

        else:

            st.info(
                "Feature importance is not available "
                "for this model configuration."
            )

except Exception as e:

    st.warning(
        "Feature importance could not be calculated."
    )

    st.caption(
        f"Technical details: {e}"
    )


# ============================================================
# PROJECT WORKFLOW
# ============================================================

st.header(
    "🔄 Project Workflow"
)

workflow = [
    "📋 Enter Job Description",
    "📄 Upload One or More Resume PDFs",
    "📝 Extract Resume Text",
    "🔍 Extract Candidate Features",
    "🛠️ Detect Technical Skills",
    "🎯 Compare Resume Skills with Job Description",
    "🤖 Apply Machine Learning Model",
    "📊 Calculate Shortlisting Probability",
    "🔄 Compare Multiple Resumes",
    "📈 Display Candidate Dashboard",
    "✅ Show Screening Results"
]

for step_number, step in enumerate(
    workflow,
    start=1
):

    st.write(
        f"{step_number}. {step}"
    )


# ============================================================
# ABOUT PROJECT
# ============================================================

st.header(
    "ℹ️ About the Project"
)

st.write(
    """
This AI Resume Screening System uses Machine Learning
to analyze resume features and generate a predicted
shortlisting probability.

The system extracts:

• Years of Experience
• Education Level
• Technical Skills
• Project Count
• Resume Length
• GitHub Activity

A Job Description can also be entered. The system
detects technical skills mentioned in the Job Description
and calculates how many of those skills are present
in each resume.

Multiple resumes can be uploaded together and compared
using extracted features and model predictions.

This project is intended for educational and demonstration
purposes. It does not make actual hiring decisions.
"""
)