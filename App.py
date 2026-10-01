import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import streamlit as st
import pdfplumber
import spacy
import re
import json
from sentence_transformers import SentenceTransformer, util

from analyzer_engine import (
    load_knowledge_bases,
    build_universal_matcher,
    normalize_text_with_synonyms,
    extract_sections,
    extract_skills_multi_section,
    extract_job_skills_smart,
    auto_detect_domains,
    calculate_semantic_skill_matches,
    audit_resume_quality_and_ats,
    generate_dynamic_recommendations,
    generate_markdown_report,
    compute_match_score
)

# ---------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Resume Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Modern Gradient Background */
.stApp {
    background: linear-gradient(135deg, #0d1b2a, #1b263b, #212d40);
    color: #f0f4f8;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Custom Metric Card */
.metric-box {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 12px;
    padding: 18px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(8px);
}
.metric-box h2 {
    margin: 0;
    font-size: 2.2rem;
    font-weight: 700;
}
.metric-box p {
    margin: 4px 0 0 0;
    color: #94a3b8;
    font-size: 0.95rem;
}

/* Skill Badges */
.skill-badge-direct {
    display: inline-block;
    background: rgba(16, 185, 129, 0.18);
    color: #34d399;
    border: 1px solid #10b981;
    border-radius: 20px;
    padding: 4px 12px;
    margin: 4px;
    font-size: 0.88rem;
    font-weight: 500;
}
.skill-badge-transferable {
    display: inline-block;
    background: rgba(59, 130, 246, 0.18);
    color: #60a5fa;
    border: 1px solid #3b82f6;
    border-radius: 20px;
    padding: 4px 12px;
    margin: 4px;
    font-size: 0.88rem;
    font-weight: 500;
}
.skill-badge-missing {
    display: inline-block;
    background: rgba(239, 68, 68, 0.18);
    color: #f87171;
    border: 1px solid #ef4444;
    border-radius: 20px;
    padding: 4px 12px;
    margin: 4px;
    font-size: 0.88rem;
    font-weight: 500;
}

/* Card Container */
.insight-card {
    background: rgba(255, 255, 255, 0.04);
    border-left: 4px solid #38bdf8;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 16px;
}
.insight-card-warning {
    background: rgba(245, 158, 11, 0.08);
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 16px;
}
.insight-card-success {
    background: rgba(16, 185, 129, 0.08);
    border-left: 4px solid #10b981;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# RESOURCE CACHING & MODELS
# ---------------------------------------------------------

@st.cache_resource(show_spinner="Loading spaCy linguistic model...")
def get_spacy_nlp():
    return spacy.load("en_core_web_sm")

@st.cache_resource(show_spinner="Loading Sentence Transformer embeddings...")
def get_similarity_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def get_knowledge_base():
    return load_knowledge_bases(".")

nlp = get_spacy_nlp()
sim_model = get_similarity_model()
SKILLS_DB, SYNONYMS, UNIVERSAL_SKILLS = get_knowledge_base()

# Build universal matcher once
@st.cache_resource
def get_cached_matcher():
    return build_universal_matcher(nlp, UNIVERSAL_SKILLS)

universal_matcher = get_cached_matcher()


# ---------------------------------------------------------
# TEXT EXTRACTION HELPER
# ---------------------------------------------------------

def extract_pdf_text(file) -> str:
    """Extracts raw text from uploaded PDF file using pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text.strip()


def compute_semantic_sim(text1: str, text2: str) -> float:
    """Computes cosine similarity between two text passages."""
    if not text1 or not text2:
        return 0.0
    # Truncate to reasonable token length for miniLM
    t1 = text1[:2500]
    t2 = text2[:2500]
    e1 = sim_model.encode(t1, convert_to_tensor=True)
    e2 = sim_model.encode(t2, convert_to_tensor=True)
    sim = util.cos_sim(e1, e2)
    return max(0.0, float(sim))


# ---------------------------------------------------------
# SAMPLE DATA PRESETS
# ---------------------------------------------------------

SAMPLE_AIML_RESUME = """P. Venkata Sai Rohith
Email: rohith.ai@example.com | Phone: +1-555-019-2834 | linkedin.com/in/sai-rohith | github.com/rohith-ai

PROFESSIONAL SUMMARY
Machine Learning & Software Engineer with experience developing deep learning architectures, transformer pipelines, and high-throughput REST APIs. Proven record of optimizing inference latency and deploying scalable AI microservices.

TECHNICAL SKILLS
- Programming Languages: Python, C++, SQL, Bash
- Frameworks & Libraries: PyTorch, Scikit-learn, HuggingFace, FastAPI, Pandas, NumPy, OpenCV
- Cloud & Tools: Docker, Git, Linux, PostgreSQL, Redis, GitHub Actions

WORK EXPERIENCE
Machine Learning Engineer Intern | NeuralTech Innovations (2025 - Present)
- Architected and trained convolutional and transformer neural networks using PyTorch, reducing inference latency by 34% on GPU clusters.
- Engineered 12+ REST API endpoints with FastAPI to serve real-time model inferences to over 40,000 daily active users.
- Automated data preprocessing pipelines with Pandas and SQL, processing 2.5M records daily with zero downtime.
- Worked on bug fixes and assisted team with documentation.

PROJECTS
Vision & NLP Search Engine
- Built a multi-modal semantic search engine combining sentence embeddings and vector similarity search.
- Containerized application services using Docker and automated CI/CD test workflows via GitHub Actions.
- Refactored backend database queries in PostgreSQL, improving response throughput by 28%.

EDUCATION
Bachelor of Technology in Artificial Intelligence & Machine Learning (Expected 2026)
GPA: 3.8 / 4.0
"""

SAMPLE_AIML_JD = """We are looking for a Machine Learning Engineer to join our growing Applied AI team.

Key Responsibilities:
- Build, optimize, and deploy production machine learning and deep learning models.
- Develop scalable backend RESTful APIs using Python (FastAPI or Flask) for serving models.
- Work with containerization and cloud infrastructure (Docker, Kubernetes, AWS).
- Collaborate on database architecture (SQL, PostgreSQL, Redis).
- Experience with TensorFlow, Keras, or PyTorch is required.
- Knowledge of MLOps, CI/CD pipelines, and model monitoring is a strong plus.
"""

SAMPLE_FULLSTACK_RESUME = """Alex Taylor
Email: alex.taylor@example.com | Phone: +1-555-482-9901 | linkedin.com/in/alextaylor | github.com/alextaylor

PROFESSIONAL SUMMARY
Full Stack Software Developer with 3+ years building responsive web platforms and scalable cloud microservices.

TECHNICAL SKILLS
Languages: TypeScript, JavaScript, Python, HTML5, CSS3, SQL
Frontend: React, Next.js, Tailwind CSS, Redux, Responsive Design
Backend: Node.js, Express, PostgreSQL, MongoDB, REST API, JWT
DevOps: Docker, AWS, Git, CI/CD, Jest

WORK EXPERIENCE
Full Stack Developer | CloudScale Solutions (2024 - Present)
- Engineered responsive client-facing web application with React and Tailwind CSS, increasing user engagement by 45%.
- Architected RESTful microservices in Node.js and Express, supporting 150,000+ monthly active requests.
- Optimized PostgreSQL database schema and indexing, cutting median query time by 52%.
- Responsible for maintaining code repositories and participating in code reviews.

PROJECTS
Real-Time Collaboration Platform
- Developed an interactive dashboard using Next.js and Socket.io with JWT authentication.
- Deployed multi-container services with Docker to AWS ECS with automated CI/CD pipelines.

EDUCATION
B.S. in Computer Science | 2024
"""

SAMPLE_FULLSTACK_JD = """Senior Full Stack Developer (React & Node.js)

We need an experienced Full Stack Developer to lead development on our core SaaS products.

Requirements:
- Strong expertise in React, Next.js, and TypeScript for front-end architecture.
- Deep experience with Node.js, Express or NestJS, and REST API design.
- Hands-on experience with databases: PostgreSQL, MongoDB, and Redis caching.
- Familiarity with Cloud & DevOps: AWS or GCP, Docker, Kubernetes, CI/CD.
- Experience with automated testing using Jest or Cypress.
"""


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("Settings & Controls")
    
    st.markdown("### Analysis Parameters")
    semantic_threshold = st.slider(
        "Semantic Transferability Sensitivity",
        min_value=0.40,
        max_value=0.75,
        value=0.50,
        step=0.05,
        help="Controls how aggressively related skills are identified as transferable (e.g. PyTorch to TensorFlow, FastAPI to Flask)."
    )

    domain_mode = st.radio(
        "Domain Strategy",
        ["Auto-Detect (Recommended)", "Manual Selection"],
        help="Auto-Detect analyzes the Job Description and Resume automatically."
    )

    selected_domains = []
    if domain_mode == "Manual Selection":
        available_domains = list(SKILLS_DB.keys())
        selected_domains = st.multiselect(
            "Select target domains:",
            available_domains,
            default=available_domains[:2]
        )

    st.divider()
    st.markdown("### Quick Test Presets")
    st.caption("Load pre-configured sample data to test instantly:")
    
    col_p1, col_p2 = st.columns(2)
    load_aiml = col_p1.button("AI / ML", use_container_width=True)
    load_fs = col_p2.button("Full Stack", use_container_width=True)

    if load_aiml:
        st.session_state["resume_input_text"] = SAMPLE_AIML_RESUME
        st.session_state["jd_input_text"] = SAMPLE_AIML_JD
        st.session_state["input_method"] = "Paste Resume Text"
        st.rerun()

    if load_fs:
        st.session_state["resume_input_text"] = SAMPLE_FULLSTACK_RESUME
        st.session_state["jd_input_text"] = SAMPLE_FULLSTACK_JD
        st.session_state["input_method"] = "Paste Resume Text"
        st.rerun()

    st.divider()
    st.caption("**100% Offline & Private**: Powered by local SpaCy NLP, universal skill taxonomy, and SentenceTransformer semantic embeddings.")


# ---------------------------------------------------------
# MAIN INTERFACE HEADER
# ---------------------------------------------------------

st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h1 style="font-size: 2.8rem; font-weight: 800; background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        AI Resume Analyzer
    </h1>
    <p style="color: #94a3b8; font-size: 1.15rem; max-width: 750px; margin: 0 auto;">
        Next-generation resume intelligence: 3-tier semantic skill matching, transferable competency discovery, ATS audit, and dynamic personalized improvement blueprints.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------------

default_method = st.session_state.get("input_method", "Upload PDF Resume")
input_method = st.radio(
    "Select Resume Input Format:",
    ["Upload PDF Resume", "Paste Resume Text"],
    horizontal=True,
    index=0 if default_method == "Upload PDF Resume" else 1
)

col_input_left, col_input_right = st.columns([1, 1], gap="large")

with col_input_left:
    st.markdown("### Candidate Resume")
    resume_raw_text = ""
    resume_file_name = "Uploaded_Resume.pdf"

    if input_method == "Upload PDF Resume":
        uploaded_file = st.file_uploader("Upload your resume in PDF format", type=["pdf"])
        if uploaded_file is not None:
            resume_raw_text = extract_pdf_text(uploaded_file)
            resume_file_name = uploaded_file.name
            st.success(f"Loaded '{uploaded_file.name}' ({len(resume_raw_text.split())} words extracted)")
    else:
        preset_resume = st.session_state.get("resume_input_text", "")
        resume_raw_text = st.text_area(
            "Paste your complete resume text:",
            value=preset_resume,
            height=300,
            placeholder="Paste raw resume text including summary, technical skills, experience, projects, and education..."
        )
        resume_file_name = "Pasted_Resume.txt"

with col_input_right:
    st.markdown("### Target Job Description")
    preset_jd = st.session_state.get("jd_input_text", "")
    job_description_text = st.text_area(
        "Enter Job Description or Required Role Competencies:",
        value=preset_jd,
        height=300,
        placeholder="Paste target job description, responsibilities, and required tech stack here..."
    )

analyze_button = st.button("Analyze & Generate Intelligent Insights", type="primary", use_container_width=True)


# ---------------------------------------------------------
# ANALYSIS EXECUTION PIPELINE
# ---------------------------------------------------------

if analyze_button:
    if not resume_raw_text.strip():
        st.error("Please provide a resume by uploading a PDF or pasting resume text.")
        st.stop()
    if not job_description_text.strip():
        st.error("Please enter the target Job Description or required technical skills.")
        st.stop()

    with st.spinner("Analyzing resume content, mapping semantic competencies, and auditing ATS compliance..."):
        # 1. Normalize text
        normalized_resume = normalize_text_with_synonyms(resume_raw_text, SYNONYMS)
        normalized_jd = normalize_text_with_synonyms(job_description_text, SYNONYMS)

        # 2. Extract sections
        sections = extract_sections(normalized_resume)

        # 3. Multi-section skill extraction
        resume_skills_dict = extract_skills_multi_section(
            normalized_resume,
            sections,
            universal_matcher,
            nlp,
            UNIVERSAL_SKILLS
        )
        resume_skill_names = set(resume_skills_dict.keys())

        # 4. Job skills extraction
        job_skills = extract_job_skills_smart(
            job_description_text,
            universal_matcher,
            nlp,
            UNIVERSAL_SKILLS,
            SYNONYMS
        )
        job_skills_set = set(job_skills)

        # 5. Domain detection
        detected_domains = auto_detect_domains(job_skills, job_description_text, SKILLS_DB)
        primary_domain = detected_domains[0][0] if detected_domains else "Full Stack Development"
        if selected_domains:
            primary_domain = selected_domains[0]

        # 6. 3-Tier Semantic Matching
        direct_matches, semantic_matches, missing_skills = calculate_semantic_skill_matches(
            resume_skill_names,
            job_skills_set,
            sim_model,
            semantic_threshold=semantic_threshold
        )

        # 7. Section & Overall Semantic Similarities
        overall_sim = compute_semantic_sim(normalized_resume, normalized_jd)
        exp_text = sections.get("experience", "")
        proj_text = sections.get("projects", "")
        exp_sim = compute_semantic_sim(exp_text, normalized_jd) if exp_text else overall_sim * 0.85
        proj_sim = compute_semantic_sim(proj_text, normalized_jd) if proj_text else overall_sim * 0.80

        # 8. Holistic Match Score
        final_match_score = compute_match_score(
            semantic_similarity=overall_sim,
            experience_similarity=exp_sim,
            projects_similarity=proj_sim,
            direct_count=len(direct_matches),
            transferable_count=len(semantic_matches),
            total_job_skills=len(job_skills_set)
        )

        # 9. Quality & ATS Audit
        audit_data = audit_resume_quality_and_ats(resume_raw_text, sections)

        # 10. Dynamic Contextual Recommendations
        recommendations = generate_dynamic_recommendations(
            missing_skills=missing_skills,
            semantic_matches=semantic_matches,
            direct_matches=direct_matches,
            primary_domain=primary_domain,
            audit_data=audit_data,
            job_text=job_description_text
        )

        # 11. Markdown Export Report
        markdown_report = generate_markdown_report(
            resume_name=resume_file_name,
            overall_score=final_match_score,
            semantic_similarity=overall_sim,
            experience_similarity=exp_sim,
            projects_similarity=proj_sim,
            direct_matches=direct_matches,
            semantic_matches=semantic_matches,
            missing_skills=missing_skills,
            audit_data=audit_data,
            recommendations=recommendations,
            primary_domain=primary_domain
        )

    # ---------------------------------------------------------
    # DASHBOARD TABS
    # ---------------------------------------------------------

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Match Overview",
        "Skill Gap & Transferability",
        "ATS & Content Health Audit",
        "Dynamic Action Plan",
        "Parsed Data & Export"
    ])

    # -----------------------------------------------------
    # TAB 1: MATCH OVERVIEW
    # -----------------------------------------------------
    with tab1:
        st.markdown(f"### Target Role Alignment: **{primary_domain}**")

        # Top 4 Metric Cards
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        with m_col1:
            st.markdown(f"""
            <div class="metric-box">
                <h2 style="color: {'#34d399' if final_match_score >= 70 else '#f59e0b' if final_match_score >= 50 else '#f87171'};">
                    {final_match_score}%
                </h2>
                <p>Overall Match Score</p>
            </div>
            """, unsafe_allow_html=True)

        with m_col2:
            ats_val = audit_data.get("ats_score", 0)
            st.markdown(f"""
            <div class="metric-box">
                <h2 style="color: {'#34d399' if ats_val >= 75 else '#f59e0b' if ats_val >= 55 else '#f87171'};">
                    {ats_val}/100
                </h2>
                <p>ATS Health Score</p>
            </div>
            """, unsafe_allow_html=True)

        with m_col3:
            st.markdown(f"""
            <div class="metric-box">
                <h2 style="color: #60a5fa;">{int(exp_sim * 100)}%</h2>
                <p>Experience Relevance</p>
            </div>
            """, unsafe_allow_html=True)

        with m_col4:
            st.markdown(f"""
            <div class="metric-box">
                <h2 style="color: #c084fc;">{int(proj_sim * 100)}%</h2>
                <p>Projects Relevance</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Progress bar
        st.progress(min(1.0, max(0.0, final_match_score / 100.0)))

        # Executive Verdict Box
        if final_match_score >= 75:
            st.markdown("""
            <div class="insight-card-success">
                <h4 style="margin:0 0 6px 0; color: #34d399;">High Match Potential</h4>
                Candidate demonstrates strong domain alignment, covering core technical requirements with relevant project and work experience. Focus on bridging the remaining nice-to-have competencies.
            </div>
            """, unsafe_allow_html=True)
        elif final_match_score >= 50:
            st.markdown("""
            <div class="insight-card-warning">
                <h4 style="margin:0 0 6px 0; color: #fbbf24;">Moderate Match - Bridgeable Gaps</h4>
                Candidate has foundational technical competencies and several transferable skills. Tailoring resume bullet points to emphasize direct impact with required tools will significantly raise interview callback rates.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="insight-card">
                <h4 style="margin:0 0 6px 0; color: #38bdf8;">Significant Skill Gap</h4>
                The target job description emphasizes technical tools and methodologies not prominently represented in the resume. Review the Action Plan tab for targeted project blueprints.
            </div>
            """, unsafe_allow_html=True)

        # Summary Breakdown Columns
        col_ov_left, col_ov_right = st.columns(2)
        with col_ov_left:
            st.markdown("#### Skill Fulfillment Breakdown")
            total_reqs = len(job_skills_set) if job_skills_set else 1
            st.write(f"- **Direct Skill Matches**: `{len(direct_matches)}` / `{total_reqs}` ({round((len(direct_matches)/total_reqs)*100)}%)")
            st.write(f"- **Transferable / Semantic Matches**: `{len(semantic_matches)}` / `{total_reqs}` ({round((len(semantic_matches)/total_reqs)*100)}%)")
            st.write(f"- **Missing Skill Gaps**: `{len(missing_skills)}` / `{total_reqs}` ({round((len(missing_skills)/total_reqs)*100)}%)")
            st.write(f"- **Candidate Total Skills Identified**: `{len(resume_skill_names)}` skills")

        with col_ov_right:
            st.markdown("#### Semantic Context Alignment")
            st.write(f"- **Global Resume to JD Semantic Similarity**: `{int(overall_sim * 100)}%`")
            st.write(f"- **Work Experience Alignment**: `{int(exp_sim * 100)}%`")
            st.write(f"- **Personal Projects Alignment**: `{int(proj_sim * 100)}%`")
            st.write(f"- **Detected Domain Fit**: `{primary_domain}`")

    # -----------------------------------------------------
    # TAB 2: SKILL GAP & TRANSFERABILITY
    # -----------------------------------------------------
    with tab2:
        st.markdown("### 3-Tier Technical Skill Analysis")
        st.caption("Unlike rigid keyword matchers, our engine identifies exact matches, semantic equivalents, and transferable frameworks.")

        # Direct Matches
        st.markdown(f"#### Direct Matches (`{len(direct_matches)}`)")
        if direct_matches:
            badge_html = "".join([f'<span class="skill-badge-direct">{s}</span>' for s in direct_matches])
            st.markdown(badge_html, unsafe_allow_html=True)
        else:
            st.info("No exact matching skills found in the resume.")

        st.divider()

        # Semantic & Transferable Matches
        st.markdown(f"#### Semantic & Transferable Matches (`{len(semantic_matches)}`)")
        st.caption("Skills where your existing experience translates directly to the required tool (e.g. PyTorch translates to TensorFlow, Docker to Kubernetes, FastAPI to Flask):")
        
        if semantic_matches:
            for sm in semantic_matches:
                with st.container():
                    st.markdown(f"""
                    <div class="insight-card">
                        <strong style="color: #60a5fa; font-size: 1.05rem;">Job Requires: {sm['job_skill'].upper()}</strong>
                        <span style="background: rgba(59, 130, 246, 0.2); color: #93c5fd; padding: 2px 8px; border-radius: 12px; font-size: 0.85rem; margin-left: 10px;">
                            {sm['similarity']}% Transferability
                        </span>
                        <p style="margin: 6px 0 2px 0; color: #cbd5e1;">
                            <strong>Your Related Skill:</strong> <code>{sm['resume_skill']}</code> ({sm.get('cluster', 'Semantic Alignment')})
                        </p>
                        <p style="margin: 0; color: #94a3b8; font-size: 0.9rem;">
                            <em>{sm['explanation']}</em>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.write("No transferable skill equivalents identified.")

        st.divider()

        # Missing Skills
        st.markdown(f"#### True Missing Skills (`{len(missing_skills)}`)")
        st.caption("Technical competencies required by the job that have no direct or transferable evidence in the resume:")
        if missing_skills:
            badge_html = "".join([f'<span class="skill-badge-missing">{s}</span>' for s in missing_skills])
            st.markdown(badge_html, unsafe_allow_html=True)
        else:
            st.success("Outstanding! You satisfy all extracted technical requirements for this position.")

    # -----------------------------------------------------
    # TAB 3: ATS & CONTENT HEALTH AUDIT
    # -----------------------------------------------------
    with tab3:
        st.markdown("### ATS Readiness & Content Strength Audit")

        col_ats1, col_ats2 = st.columns(2, gap="large")

        with col_ats1:
            st.markdown("#### Contact Information Check")
            ci = audit_data.get("contact_info", {})
            st.write(f"[{'Found' if ci.get('email') else 'Missing'}] **Email Address Detected**")
            st.write(f"[{'Found' if ci.get('phone') else 'Missing'}] **Phone Number Detected**")
            st.write(f"[{'Found' if ci.get('linkedin') else 'Not Found'}] **LinkedIn Profile URL**")
            st.write(f"[{'Found' if ci.get('github') else 'Not Found'}] **GitHub / Portfolio Link**")

            st.markdown("#### Standard Section Architecture")
            ds = audit_data.get("detected_sections", {})
            for sec_name, present in ds.items():
                st.write(f"{'[Found]' if present else '[Missing]'} **{sec_name}**")

            st.markdown("#### Resume Length & Density")
            st.write(f"- **Total Words**: `{audit_data.get('word_count', 0)}` words")
            st.info(f"**Diagnostic**: {audit_data.get('length_status', '')}")

        with col_ats2:
            st.markdown("#### Measurable Impact & Quantification")
            q_rate = audit_data.get("quant_rate", 0)
            q_count = audit_data.get("quantified_count", 0)
            t_count = audit_data.get("total_bullets", 1)

            st.write(f"**Quantification Rate**: `{q_rate}%` ({q_count} of {t_count} bullets have metrics)")
            st.progress(min(1.0, max(0.0, q_rate / 100.0)))
            
            if q_rate < 30:
                st.warning("**Low Quantification**: Under 30% of your experience bullets contain measurable metrics (%, $, scale, users, latency). Recruiters strongly favor quantified outcomes.")
            else:
                st.success("**Good Quantification**: Your bullet points effectively use metrics to demonstrate impact.")

            st.markdown("#### Action Verb Strength Audit")
            st.write(f"- **Strong Power Verbs Found**: `{audit_data.get('strong_verbs_count', 0)}` verbs")
            if audit_data.get("strong_verbs_sample"):
                st.caption(f"Sample strong verbs: {', '.join([f'*{v}*' for v in audit_data['strong_verbs_sample'][:8]])}")

            weak_verbs = audit_data.get("weak_verbs_detected", [])
            if weak_verbs:
                st.markdown("**Weak / Passive Verbs Detected to Replace:**")
                for wv in weak_verbs:
                    st.markdown(f"- Replace `\"{wv['weak']}\"` with: **{wv['suggested']}**")
            else:
                st.success("No common weak verbs detected in your resume!")

    # -----------------------------------------------------
    # TAB 4: DYNAMIC ACTION PLAN & RECOMMENDATIONS
    # -----------------------------------------------------
    with tab4:
        st.markdown("### Tailored Action Plan & Next Steps")
        st.caption("Context-aware recommendations dynamically formulated to close your exact skill gaps.")

        # 1. Project Blueprints
        st.markdown("#### Custom Project Blueprints (Bridging Missing Skills)")
        bps = recommendations.get("project_blueprints", [])
        if bps:
            for bp in bps:
                st.markdown(f"""
                <div class="insight-card">
                    <h4 style="margin: 0 0 6px 0; color: #38bdf8;">{bp['title']}</h4>
                    <p style="margin: 0; color: #e2e8f0; font-size: 0.95rem; line-height: 1.5;">
                        {bp['description']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No missing skills to bridge! Your project background aligns with the job profile.")

        st.divider()

        # 2. STAR Bullet Rewriter
        st.markdown("#### STAR Method Resume Bullet Rewriter")
        st.caption("Transform unquantified or passive bullet points into high-impact recruiter magnets:")

        star = recommendations.get("star_rewrite", {})
        col_star1, col_star2 = st.columns(2, gap="medium")
        
        with col_star1:
            st.markdown("""
            <div style="background: rgba(239, 68, 68, 0.08); border-left: 3px solid #ef4444; border-radius: 6px; padding: 12px 16px;">
                <strong style="color: #f87171;">Before (Unquantified / Passive):</strong><br>
                <p style="color: #cbd5e1; margin-top: 6px; font-style: italic;">
                    "{}"
                </p>
            </div>
            """.format(star.get("original", "Worked on developing backend APIs and fixing bugs.")), unsafe_allow_html=True)

        with col_star2:
            st.markdown("""
            <div style="background: rgba(16, 185, 129, 0.08); border-left: 3px solid #10b981; border-radius: 6px; padding: 12px 16px;">
                <strong style="color: #34d399;">After (STAR Method - Action & Measurable Impact):</strong><br>
                <p style="color: #f1f5f9; margin-top: 6px; font-weight: 500;">
                    "{}"
                </p>
            </div>
            """.format(star.get("example_rewrite", "")), unsafe_allow_html=True)

        st.divider()

        # 3. Targeted Interview Questions
        st.markdown("#### Targeted Interview Preparation")
        st.caption("Anticipate deep technical inquiries and bridge questions in your upcoming interviews:")

        qs = recommendations.get("interview_questions", [])
        if qs:
            for i, q in enumerate(qs, 1):
                st.markdown(f"""
                <div class="insight-card">
                    <span style="background: rgba(129, 140, 248, 0.2); color: #a5b4fc; padding: 2px 8px; border-radius: 10px; font-size: 0.8rem; font-weight: 600;">
                        {q['type'].upper()} - {q['skill'].upper()}
                    </span>
                    <p style="margin: 8px 0 0 0; color: #f8fafc; font-size: 0.95rem;">
                        <strong>Q{i}:</strong> {q['question']}
                    </p>
                </div>
                """, unsafe_allow_html=True)

    # -----------------------------------------------------
    # TAB 5: PARSED DATA & EXPORT
    # -----------------------------------------------------
    with tab5:
        st.markdown("### Parsed Sections & Export Report")

        # Download Report Button
        st.download_button(
            label="Download Comprehensive Analysis Report (Markdown)",
            data=markdown_report,
            file_name=f"Resume_Analysis_{primary_domain.replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True
        )

        st.markdown("#### Parsed Resume Sections")
        for sec_name, sec_content in sections.items():
            with st.expander(f"Section: {sec_name.upper()} ({len(sec_content.split())} words)"):
                if sec_content:
                    st.text(sec_content)
                else:
                    st.write("*No content isolated for this section header.*")

        with st.expander("Raw Extracted Resume Skills & Evidence"):
            for s, data in sorted(resume_skills_dict.items()):
                st.write(f"- **{s}**: Found in {', '.join(data['sections'])} (Frequency: {data['frequency']})")
