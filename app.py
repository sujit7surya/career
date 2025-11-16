import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Load Courses & Build TF-IDF
# -------------------------
df_courses = pd.read_csv("courses.csv")

course_texts = (df_courses["title"] + " " +
                df_courses["skill_tags"] + " " +
                df_courses["level"])

vectorizer = TfidfVectorizer(stop_words="english")
course_tfidf = vectorizer.fit_transform(course_texts.tolist())


def build_user_profile_text(education_level, major, tech_skills, soft_skills,
                            target_domain=None, preferred_duration=None):
    tech_skills_clean = [s.strip().lower() for s in tech_skills]
    soft_skills_clean = [s.strip().lower() for s in soft_skills]

    parts = [
        f"Education level: {education_level}",
        f"Major: {major}",
        f"Technical skills: {', '.join(tech_skills_clean)}",
        f"Soft skills: {', '.join(soft_skills_clean)}"
    ]
    if target_domain:
        parts.append(f"Target domain: {target_domain}")
    if preferred_duration:
        parts.append(f"Preferred study duration: {preferred_duration}")
    return ". ".join(parts)


def estimate_user_level(tech_skills):
    n = len(tech_skills)
    if n <= 2:
        return "beginner"
    elif n <= 5:
        return "intermediate"
    else:
        return "advanced"


def level_compatible(user_level, course_level):
    order = {"beginner": 0, "intermediate": 1, "advanced": 2}
    u = order.get(user_level, 0)
    c = order.get(course_level, 0)
    return c <= u + 1


def recommend_courses_ui(education_level, major, tech_skills_input, soft_skills_input,
                         target_domain, preferred_duration, top_n=7):
    tech_skills = [s.strip().lower() for s in tech_skills_input.split(",") if s.strip()]
    soft_skills = [s.strip().lower() for s in soft_skills_input.split(",") if s.strip()]

    user_profile_text = build_user_profile_text(
        education_level, major, tech_skills, soft_skills,
        target_domain, preferred_duration
    )

    user_tfidf = vectorizer.transform([user_profile_text])
    sims = cosine_similarity(user_tfidf, course_tfidf)[0]

    df = df_courses.copy()
    df["fit_score"] = (sims * 100).round(2)

    user_level = estimate_user_level(tech_skills)
    df["level_ok"] = df["level"].apply(lambda lvl: level_compatible(user_level, lvl))
    df = df[df["level_ok"]].sort_values(by="fit_score", ascending=False)
    df = df.head(top_n).reset_index(drop=True)

    tech_skills_set = set(tech_skills)
    timelines = []
    explanations = []

    for _, row in df.iterrows():
        level = row["level"]
        tags = [t.strip().lower() for t in row["skill_tags"].split(",")]

        matched = list(set(tags) & tech_skills_set)
        missing = [t for t in tags if t not in matched]

        if level == "beginner":
            timeline = "short-term"
        elif level == "intermediate":
            if row["fit_score"] > 70:
                timeline = "short-term"
            else:
                timeline = "long-term"
        else:
            timeline = "long-term"

        if matched and missing:
            expl = (
                f"This course leverages your skills in {', '.join(matched)} "
                f"and helps you build {', '.join(missing)}."
            )
        elif matched and not missing:
            expl = (
                f"This course strongly matches your current skills ({', '.join(matched)}) "
                f"and deepens your expertise."
            )
        elif not matched and missing:
            expl = (
                f"This course introduces new skills such as {', '.join(missing)}, "
                f"aligned with your target domain."
            )
        else:
            expl = "This course is relevant to your profile based on overall similarity."

        timelines.append(timeline)
        explanations.append(expl)

    df["timeline"] = timelines
    df["explanation"] = explanations
    return df


# -------------------------
# Streamlit UI
# -------------------------
st.title("SmartCareer - AI Course & Certification Recommender")

st.subheader("Enter your profile")

education_level = st.selectbox(
    "Education Level",
    ["High School", "Diploma", "B.Com", "B.Sc", "BCA", "B.Tech", "M.Sc", "MCA", "M.Tech", "Other"],
    index=6 if "B.Tech" in ["High School","Diploma","B.Com","B.Sc","BCA","B.Tech","M.Sc","MCA","M.Tech","Other"] else 0
)

major = st.text_input("Major / Degree (e.g., Information Technology, CSE, ECE)")

tech_skills_input = st.text_input("Technical Skills (comma-separated)", "python, sql, excel")
soft_skills_input = st.text_input("Soft Skills (comma-separated)", "communication, teamwork")

target_domain = st.text_input("Target Domain / Career Goal (e.g., data science, web development)", "data science")

preferred_duration = st.selectbox(
    "Preferred Study Duration",
    ["No preference", "1-3 months", "3-6 months", "6-12 months"]
)

if st.button("Get Recommendations"):
    if not major.strip():
        st.warning("Please enter your major/degree.")
    else:
        recs = recommend_courses_ui(
            education_level, major, tech_skills_input, soft_skills_input,
            target_domain, preferred_duration, top_n=7
        )

        if recs.empty:
            st.write("No suitable courses found based on your profile.")
        else:
            st.subheader("Recommended Courses")
            for _, row in recs.iterrows():
                st.markdown(f"### {row['title']} ({row['provider']})")
                st.markdown(f"- **Level:** {row['level'].title()}")
                st.markdown(f"- **Duration:** {row['duration']}")
                st.markdown(f"- **Fit Score:** {row['fit_score']}/100")
                st.markdown(f"- **Timeline:** {row['timeline'].title()}")
                st.markdown(f"- **Why:** {row['explanation']}")
                st.markdown(f"- [Course Link]({row['link']})")
                st.markdown("---")
