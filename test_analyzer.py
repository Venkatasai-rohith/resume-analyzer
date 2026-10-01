"""
test_analyzer.py
Sanity and integration tests for analyzer_engine.py
"""
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import spacy
from sentence_transformers import SentenceTransformer
from analyzer_engine import (
    load_knowledge_bases,
    build_universal_matcher,
    extract_sections,
    extract_skills_multi_section,
    extract_job_skills_smart,
    auto_detect_domains,
    calculate_semantic_skill_matches,
    audit_resume_quality_and_ats,
    generate_dynamic_recommendations,
    generate_markdown_report
)

def run_tests():
    print("Step 1: Loading models and knowledge base...")
    nlp = spacy.load("en_core_web_sm")
    sim_model = SentenceTransformer("all-MiniLM-L6-v2")
    skills_db, synonyms, universal_skills = load_knowledge_bases(".")
    
    assert len(universal_skills) > 50, f"Expected >50 skills, got {len(universal_skills)}"
    print(f"Loaded {len(universal_skills)} universal skills across {len(skills_db)} domains.")

    matcher = build_universal_matcher(nlp, universal_skills)

    sample_resume = """
    P. Venkata Sai Rohith
    Email: rohith@example.com | Phone: +1-234-567-8901 | linkedin.com/in/rohith | github.com/rohith

    PROFESSIONAL SUMMARY
    Passionate AIML and Software Engineer with 2+ years of experience developing deep learning architectures and REST APIs.

    TECHNICAL SKILLS
    Languages: Python, C++, SQL
    Frameworks & Tools: PyTorch, Scikit-learn, FastAPI, Git, Docker, Pandas, NumPy

    WORK EXPERIENCE
    AI Engineer Intern - Tech Corp (Jan 2025 - Present)
    - Architected and trained convolutional neural networks using PyTorch, reducing inference latency by 32% on production clusters.
    - Worked on developing REST APIs with FastAPI to serve ML models for 25,000+ daily active users.
    - Responsible for data cleaning and pipeline maintenance.

    PROJECTS
    Intelligent Search & Recommendation Engine
    - Developed a semantic search application using sentence embeddings and vector search.
    - Containerized application services using Docker and automated test deployment.

    EDUCATION
    B.Tech in Artificial Intelligence & Machine Learning, 2026
    """

    sample_job_description = """
    We are seeking a Machine Learning Engineer to join our AI team.
    Requirements:
    - Strong proficiency in Python and SQL
    - Hands-on experience with TensorFlow or Keras for deep learning models
    - Experience building and deploying APIs using FastAPI or Flask
    - Working knowledge of Kubernetes and AWS cloud deployment
    - Familiarity with CI/CD and Docker
    """

    print("\nStep 2: Testing section extraction...")
    sections = extract_sections(sample_resume)
    assert bool(sections["skills"]), "Skills section should be detected!"
    assert bool(sections["experience"]), "Experience section should be detected!"
    assert bool(sections["projects"]), "Projects section should be detected!"
    print("Detected sections:", [k for k, v in sections.items() if v])

    print("\nStep 3: Testing multi-section skill extraction...")
    resume_skills_dict = extract_skills_multi_section(
        sample_resume, sections, matcher, nlp, universal_skills
    )
    resume_skill_names = set(resume_skills_dict.keys())
    print("Skills extracted from resume:", sorted(list(resume_skill_names)))
    assert "pytorch" in resume_skill_names
    assert "docker" in resume_skill_names

    print("\nStep 4: Testing smart job skills extraction...")
    job_skills = extract_job_skills_smart(sample_job_description, matcher, nlp, universal_skills, synonyms)
    print("Skills extracted from JD:", job_skills)
    assert "python" in job_skills
    assert "tensorflow" in job_skills or "keras" in job_skills

    print("\nStep 5: Testing auto-domain detection...")
    domains = auto_detect_domains(job_skills, sample_job_description, skills_db)
    print("Detected domains for JD:", domains[:2])
    assert domains[0][0] in ["AIML", "Data Science", "Backend Development"]

    print("\nStep 6: Testing 3-tier semantic skill matching...")
    direct, semantic, missing = calculate_semantic_skill_matches(
        resume_skill_names, set(job_skills), sim_model, semantic_threshold=0.65
    )
    print("Direct matches:", direct)
    print("Semantic / Transferable matches:")
    for sm in semantic:
        print(f"  * JD: {sm['job_skill']} <-> Resume: {sm['resume_skill']} ({sm['similarity']}%)")
    print("Missing skills:", missing)

    # Note: TensorFlow should match with PyTorch semantically!
    transferable_job_skills = [sm["job_skill"] for sm in semantic]
    print("Transferable job skills detected:", transferable_job_skills)

    print("\nStep 7: Testing ATS & resume audit...")
    audit = audit_resume_quality_and_ats(sample_resume, sections)
    print(f"ATS Score: {audit['ats_score']}/100")
    print(f"Contact Info: {audit['contact_info']}")
    print(f"Quantification Rate: {audit['quant_rate']}% ({audit['quantified_count']}/{audit['total_bullets']})")
    print(f"Strong Verbs Count: {audit['strong_verbs_count']}, Sample: {audit['strong_verbs_sample'][:5]}")
    print(f"Weak Verbs Detected: {audit['weak_verbs_detected']}")

    print("\nStep 8: Testing dynamic recommendations...")
    recommendations = generate_dynamic_recommendations(
        missing, semantic, direct, domains[0][0], audit, sample_job_description
    )
    print("Project Blueprint:", recommendations["project_blueprints"][0]["title"] if recommendations["project_blueprints"] else "None")
    print("STAR Rewrite Example:", recommendations["star_rewrite"]["example_rewrite"])
    print("Interview Questions Count:", len(recommendations["interview_questions"]))

    print("\nStep 9: Testing markdown report generation...")
    report = generate_markdown_report(
        "Candidate_Resume.pdf",
        85,
        0.82,
        0.78,
        0.75,
        direct,
        semantic,
        missing,
        audit,
        recommendations,
        domains[0][0]
    )
    assert len(report) > 300
    print("Markdown report generated successfully (length:", len(report), "chars).")
    print("\nALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
