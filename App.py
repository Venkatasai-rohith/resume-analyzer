import streamlit as st
import pdfplumber
import spacy
import re
import json
from spacy.matcher import PhraseMatcher
from sentence_transformers import SentenceTransformer, util


# CUSTOM STYLING

st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}
</style>
""", unsafe_allow_html=True)



# LOAD MODELS


nlp = spacy.load("en_core_web_sm")

@st.cache_resource
def load_similarity_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

sim_model = load_similarity_model()



# LOAD DATABASE


with open("skills_database.json") as f:
    SKILL_DB = json.load(f)

with open("skill_synonyms.json") as f:
    SYNONYMS = json.load(f)



# SYNONYM NORMALIZATION


def normalize_text(text):
    text = text.lower()
    for main_skill, variants in SYNONYMS.items():
        for variant in variants:
            text = re.sub(rf"\b{variant}\b", main_skill, text)
    return text



# BUILD MATCHER


def build_matcher(nlp, skill_list):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill) for skill in skill_list]
    matcher.add("SKILLS", patterns)
    return matcher



# SECTION KEYWORDS


SKILL_SECTIONS = ["skills", "technical skills", "core skills", "key skills", "technologies", "tech stack"]
EXPERIENCE_SECTIONS = ["work experience", "experience", "professional experience", "employment history", "internship experience"]
PROJECT_SECTIONS = ["projects", "personal projects", "academic projects", "project experience"]
STOP_SECTIONS = ["skills", "technical skills", "experience", "work experience", "projects", "education", "certifications", "achievements"]



# TEXT EXTRACTION


def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text



# SECTION EXTRACTOR


def extract_section(text, section_keywords, stop_keywords):
    section_pattern = r"|".join(section_keywords)
    stop_pattern = r"|".join(stop_keywords)
    pattern = rf"({section_pattern})(.*?)(?={stop_pattern}|$)"
    match = re.search(pattern, text, re.S | re.I)
    return match.group(2).strip() if match else ""



# SKILL EXTRACTION


def extract_skills_phrase(text, matcher):
    doc = nlp(text)
    matches = matcher(doc)

    skills = set()
    matched_tokens = set()

    for _, start, end in matches:
        span = doc[start:end]
        skill = span.text.lower().strip()
        skills.add(skill)

        # track tokens used in phrase
        for token in span:
            matched_tokens.add(token.text.lower())

    return skills, matched_tokens


def extract_job_skills(text, matcher):
    # Normalize text first
    text = normalize_text(text)

    # Extract phrase-based skills
    phrase_skills, matched_tokens = extract_skills_phrase(text, matcher)

    doc = nlp(text)
    fallback_skills = set()

    for token in doc:
        word = token.text.lower().strip()

        # Skip tokens already part of a phrase
        if word in matched_tokens:
            continue

        if (
            not token.is_stop and
            not token.is_punct and
            token.pos_ in ["PROPN"] and   # only proper nouns
            len(word) >= 3
        ):
            fallback_skills.add(word)

    # Combine
    combined = phrase_skills.union(fallback_skills)

    return sorted(list(combined))





# SIMILARITY


def compute_similarity(text1, text2):
    embeddings = sim_model.encode([text1, text2], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return max(0, float(similarity))



# DOMAIN SKILL COMBINER


def get_combined_skills(domains):
    combined = set()
    for domain in domains:
        combined.update(SKILL_DB[domain]["skills"])
    return sorted(list(combined))



# DOMAIN-BASED SUGGESTIONS


def generate_suggestions(missing_skills, domains):
    suggestions = []
    domain = domains[0] if domains else None

    for skill in missing_skills[:5]:

        if domain == "AIML":
            suggestions.append(
                f"Add a strong AI/ML project using {skill} with measurable results."
            )

        elif domain == "Web Development":
            suggestions.append(
                f"Build a responsive project using {skill} and deploy it online."
            )

        elif domain == "Backend":
            suggestions.append(
                f"Develop a backend project using {skill} with database integration."
            )

        elif domain == "DevOps":
            suggestions.append(
                f"Gain hands-on experience with {skill} in deployment or CI/CD."
            )

        else:
            suggestions.append(
                f"Try to gain hands-on experience with {skill} and include it in your resume."
            )

    return suggestions



# STREAMLIT UI


st.markdown(
    "<h1 style='text-align: center;'>AI Resume Analyzer</h1>",
    unsafe_allow_html=True
)


resume = st.file_uploader("Upload Your Resume", type=["pdf"])
job_desc = st.text_area("Enter the skills required for the job")

st.subheader("Select relevant Domains")
available_domains = list(SKILL_DB.keys())

selected_domains = st.multiselect(
    "Select any 3 domains relevant to the job:",
    available_domains,
    max_selections=4
)


if st.button("Analyze"):

    if resume is None or job_desc.strip() == "":
        st.error("Please upload resume and job description.")
        st.stop()

    # Extract & normalize text
    resume_text = normalize_text(extract_text(resume))
    job_desc = normalize_text(job_desc)

    # Domain skill setup
    domain_skills = get_combined_skills(selected_domains)
    matcher = build_matcher(nlp, domain_skills)

    # Extract sections
    skills_section = extract_section(resume_text, SKILL_SECTIONS, STOP_SECTIONS)
    experience_section = extract_section(resume_text, EXPERIENCE_SECTIONS, STOP_SECTIONS)
    projects_section = extract_section(resume_text, PROJECT_SECTIONS, STOP_SECTIONS)

    # Extract skills
    resume_skills, _ = extract_skills_phrase(skills_section, matcher)
    resume_skills = sorted(list(resume_skills))

    job_skills = extract_job_skills(job_desc, matcher)


    # Skill comparison
    matched_skills = list(set(resume_skills) & set(job_skills))
    missing_skills = list(set(job_skills) - set(resume_skills))

    skill_score = len(matched_skills) / len(job_skills) if job_skills else 0

    # Similarity scores
    semantic_similarity = compute_similarity(resume_text, job_desc)
    experience_similarity = compute_similarity(experience_section, job_desc) if experience_section else 0
    projects_similarity = compute_similarity(projects_section, job_desc) if projects_section else 0

    # Final weighted score
    final_score = int(max(0, min(1, (
        0.7 * skill_score +
        0.2 * semantic_similarity +
        0.1 * experience_similarity
    ))) * 100)

    # Suggestions
    suggestions = generate_suggestions(missing_skills, selected_domains)

    
    # DASHBOARD OUTPUT
    

    st.markdown("##  Resume Match Analysis")

    st.markdown("### 🎯 Overall Match Score")
    st.progress(min(final_score / 100, 1.0))
    st.markdown(f"### **{final_score}% Match**")

    st.divider()

    st.markdown("###  Section Relevance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Resume", f"{int(semantic_similarity*100)}%")
    col2.metric("Experience", f"{int(experience_similarity*100)}%")
    col3.metric("Projects", f"{int(projects_similarity*100)}%")

    st.divider()

    st.markdown("###  Matched Skills")
    if matched_skills:
        st.markdown("<br>".join([f"🟢 {skill}" for skill in matched_skills]), unsafe_allow_html=True)
    else:
        st.write("No direct skill matches found.")

    st.divider()

    st.markdown("###  Missing Skills")
    if missing_skills:
        st.markdown("<br>".join([f"🔴 {skill}" for skill in missing_skills]), unsafe_allow_html=True)
    else:
        st.write("No missing skills 🎉")

    st.divider()

    st.markdown("### 💡 Suggestions to Improve")
    if suggestions:
        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")
    else:
        st.write("Your resume already matches the job well.")
