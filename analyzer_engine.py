"""
analyzer_engine.py
Core Intelligence Engine for AI Resume Analyzer
Includes:
- Robust multi-section extraction
- Universal skill database & synonym normalization
- Whole-resume multi-section skill extraction
- 3-tier matching: Exact, Semantic / Transferable (SentenceTransformer), and Missing
- Resume quality & ATS audit (Contact info, Sections, Action verbs, Quantification rate)
- Dynamic contextual recommendations:
    * Custom project blueprints based on missing skills
    * STAR method bullet point rewriter
    * Targeted interview questions
    * Exportable analysis report
"""

import re
import json
import os
from typing import Dict, List, Tuple, Set, Any
import spacy
from spacy.matcher import PhraseMatcher
from sentence_transformers import SentenceTransformer, util

# Core Action Verbs Taxonomy
STRONG_ACTION_VERBS = {
    "accelerated", "accomplished", "achieved", "acquired", "adapted", "administered",
    "advanced", "advocated", "aligned", "allocated", "analyzed", "appraised",
    "architected", "assembled", "asserted", "assessed", "audited", "authored",
    "automated", "boosted", "built", "centralized", "championed", "coached",
    "collaborated", "commercialized", "composed", "computed", "conceptualized",
    "consolidated", "constructed", "consulted", "contracted", "converted",
    "coordinated", "crafted", "created", "customized", "debugged", "decreased",
    "delegated", "delivered", "deployed", "designed", "determined", "developed",
    "devised", "diagnosed", "directed", "discovered", "dispatched", "diversified",
    "documented", "drafted", "drove", "earned", "elevated", "eliminated",
    "enabled", "enacted", "engineered", "enhanced", "enlarged", "established",
    "evaluated", "examined", "executed", "expanded", "expedited", "fabricated",
    "facilitated", "forecasted", "formulated", "fostered", "founded", "generated",
    "governed", "guided", "halted", "headed", "identified", "implemented",
    "improved", "improvised", "increased", "indexed", "influenced", "initiated",
    "innovated", "inspected", "installed", "instituted", "instructed", "integrated",
    "intensified", "intercepted", "interpreted", "introduced", "invented",
    "investigated", "isolated", "launched", "lead", "led", "leveraged",
    "licensed", "localized", "managed", "maximized", "measured", "mediated",
    "mentored", "migrated", "minimized", "modeled", "modernized", "monitored",
    "motivated", "navigated", "negotiated", "nurtured", "obtained", "operated",
    "optimized", "orchestrated", "organized", "originated", "outperformed",
    "overhauled", "oversaw", "partnered", "performed", "pioneered", "planned",
    "prepared", "presented", "prioritized", "produced", "programmed", "promoted",
    "published", "purchased", "quantified", "raised", "realigned", "rebuilt",
    "reclaimed", "reconciled", "redesigned", "reduced", "reevaluated", "refactored",
    "refined", "reformed", "regained", "regulated", "rehabilitated", "reinforced",
    "rejuvenated", "remodeled", "reorganized", "replaced", "researched", "resolved",
    "restored", "restructured", "retrieved", "revamped", "reviewed", "revitalized",
    "revolutionized", "routed", "safeguarded", "scaled", "scheduled", "screened",
    "secured", "selected", "separated", "settled", "shaped", "simplified",
    "simulated", "slashed", "solidified", "solved", "spearheaded", "specialized",
    "specified", "standardized", "stimulated", "streamlined", "strengthened",
    "structured", "succeeded", "supervised", "supplied", "supported", "surpassed",
    "sustained", "synthesized", "systematized", "tabulated", "targeted", "tested",
    "tracked", "trained", "transformed", "translated", "triaged", "triggered",
    "troubleshot", "unified", "unlocked", "updated", "upgraded", "utilized",
    "validated", "visualized", "won", "yielded"
}

WEAK_ACTION_VERBS = {
    "worked on": "engineered / developed / implemented",
    "worked with": "collaborated with / leveraged / utilized",
    "helped with": "facilitated / supported / contributed to",
    "helped to": "enabled / assisted in",
    "responsible for": "spearheaded / managed / owned",
    "handled": "executed / directed / managed",
    "participated in": "contributed to / engaged in",
    "assisted": "facilitated / accelerated",
    "tried to": "initiated / spearheaded",
    "involved in": "drove / collaborated on",
    "did": "executed / implemented",
    "made": "architected / produced / designed",
    "looked after": "monitored / administered"
}

# Domain knowledge mapping for project ideas and blueprints
DOMAIN_PROJECT_TEMPLATES = {
    "AIML": {
        "title": "Autonomous End-to-End ML Pipeline",
        "description": "Develop a reproducible machine learning or deep learning workflow incorporating data ingestion, model fine-tuning or training with {skills}, evaluation tracking, and packaging with containerized inference endpoints (e.g. FastAPI / Docker)."
    },
    "Generative AI & LLMs": {
        "title": "Production RAG & Multi-Agent Intelligence System",
        "description": "Architect a Retrieval-Augmented Generation (RAG) system utilizing {skills}, hybrid vector search, dynamic prompt chaining, and evaluation metrics (faithfulness and latency benchmarks)."
    },
    "Data Science": {
        "title": "Predictive Analytics & Executive Insight Dashboard",
        "description": "Design an analytical pipeline processing large-scale datasets with {skills}, featuring automated statistical hypothesis testing, feature importance extraction, and an interactive business dashboard."
    },
    "Frontend Development": {
        "title": "High-Performance Modern Web Platform",
        "description": "Build an accessible, responsive web application using {skills} featuring state management, optimized client-side caching, sub-second LCP/FID metrics, and automated end-to-end component testing."
    },
    "Backend Development": {
        "title": "Distributed Scalable Microservices Architecture",
        "description": "Design and implement high-throughput REST/gRPC microservices using {skills}, with robust database connection pooling, distributed caching, JWT authentication, and structured error observability."
    },
    "Cloud DevOps": {
        "title": "GitOps CI/CD Infrastructure as Code Deployment",
        "description": "Establish an automated cloud infrastructure pipeline utilizing {skills}, incorporating infrastructure provisioning, container orchestration, blue-green deployment strategies, and centralized monitoring."
    },
    "Full Stack Development": {
        "title": "Full-Stack Cloud-Native SaaS Platform",
        "description": "Create an end-to-end full-stack web application integrating {skills} across the frontend interface, secure API layer, transactional database, and containerized cloud deployment."
    },
    "Cybersecurity": {
        "title": "Automated Security Vulnerability Assessment Suite",
        "description": "Develop a security scanning and auditing toolkit incorporating {skills} to detect OWASP vulnerabilities, audit network configurations, and generate automated compliance reports."
    },
    "Mobile Development": {
        "title": "Cross-Platform Feature-Rich Mobile Application",
        "description": "Engineer a native-performing mobile app using {skills} with offline-first local data synchronization, smooth UI transitions, push notifications, and biometric authentication."
    }
}

# Transferable tech competency clusters
TRANSFERABLE_CLUSTERS = [
    {"name": "Deep Learning Frameworks", "skills": {"tensorflow", "pytorch", "keras", "jax", "caffe", "mxnet"}},
    {"name": "Python Web APIs", "skills": {"fastapi", "flask", "django", "tornado", "bottle", "pyramid", "rest api"}},
    {"name": "Node/JS Backend", "skills": {"node.js", "express", "nestjs", "fastify", "koa", "rest api"}},
    {"name": "Relational Databases", "skills": {"postgresql", "mysql", "sqlite", "oracle", "sql server", "mariadb", "sql"}},
    {"name": "NoSQL / Document DBs", "skills": {"mongodb", "couchdb", "dynamodb", "firestore", "cassandra", "firebase"}},
    {"name": "In-Memory Caching", "skills": {"redis", "memcached"}},
    {"name": "Cloud Platforms", "skills": {"amazon web services", "google cloud platform", "microsoft azure", "aws", "gcp", "azure"}},
    {"name": "Containerization & Orchestration", "skills": {"docker", "kubernetes", "podman", "openshift", "helm"}},
    {"name": "CI/CD & DevOps", "skills": {"jenkins", "github actions", "gitlab ci", "circleci", "travis ci", "argo cd", "ci/cd"}},
    {"name": "Frontend Frameworks", "skills": {"react", "vue", "angular", "svelte", "next.js"}},
    {"name": "State Management", "skills": {"redux", "zustand", "mobx", "vuex", "pinia"}},
    {"name": "Vector Databases", "skills": {"pinecone", "chromadb", "weaviate", "milvus", "qdrant"}},
    {"name": "LLM Orchestration & RAG", "skills": {"langchain", "llamaindex", "haystack", "rag", "large language models", "generative ai"}},
    {"name": "Data Analysis & ML", "skills": {"pandas", "numpy", "scikit-learn", "scipy", "r"}},
    {"name": "CSS & Styling", "skills": {"tailwind", "bootstrap", "material ui", "chakra ui", "sass", "css"}},
    {"name": "Testing Frameworks", "skills": {"pytest", "unittest", "jest", "mocha", "cypress", "playwright", "selenium"}}
]


def load_knowledge_bases(base_path: str = ".") -> Tuple[Dict[str, Any], Dict[str, List[str]], Set[str]]:
    """Loads skills_database.json and skill_synonyms.json, returning universal skills and mappings."""
    db_path = os.path.join(base_path, "skills_database.json")
    syn_path = os.path.join(base_path, "skill_synonyms.json")

    skills_db = {}
    if os.path.exists(db_path):
        with open(db_path, "r", encoding="utf-8") as f:
            skills_db = json.load(f)

    synonyms = {}
    if os.path.exists(syn_path):
        with open(syn_path, "r", encoding="utf-8") as f:
            synonyms = json.load(f)

    universal_skills = set()
    for domain_info in skills_db.values():
        if isinstance(domain_info, dict) and "skills" in domain_info:
            for s in domain_info["skills"]:
                universal_skills.add(s.lower().strip())

    # Add canonical synonym keys as well
    for key in synonyms.keys():
        universal_skills.add(key.lower().strip())

    return skills_db, synonyms, universal_skills


def normalize_text_with_synonyms(text: str, synonyms: Dict[str, List[str]]) -> str:
    """Normalizes text by mapping known abbreviations and synonyms to canonical skill names."""
    if not text:
        return ""
    normalized = text.lower()
    for canonical, variants in synonyms.items():
        for variant in variants:
            # Match whole words only, handling special chars like c++, c#, .net
            escaped_variant = re.escape(variant.lower())
            normalized = re.sub(rf"(?<!\w){escaped_variant}(?!\w)", canonical.lower(), normalized)
    return normalized


def build_universal_matcher(nlp: spacy.language.Language, skills: Set[str]) -> PhraseMatcher:
    """Builds a fast SpaCy PhraseMatcher covering all known skills."""
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = []
    for skill in skills:
        cleaned = skill.strip()
        if cleaned:
            patterns.append(nlp.make_doc(cleaned))
    if patterns:
        matcher.add("ALL_SKILLS", patterns)
    return matcher


def extract_sections(text: str) -> Dict[str, str]:
    """
    Intelligently splits resume text into standardized sections using common header patterns.
    Returns a dictionary of section names to extracted text content.
    """
    sections = {
        "summary": "",
        "skills": "",
        "experience": "",
        "projects": "",
        "education": "",
        "certifications": ""
    }

    if not text:
        return sections

    # Header regex patterns
    header_patterns = [
        ("summary", r"(?:summary|professional\s+summary|profile|about\s+me|objective|career\s+objective)"),
        ("skills", r"(?:technical\s+skills|core\s+skills|skills\s*(?:&|and)?\s*technologies|skills\s*(?:&|and)?\s*tools|skills|technologies|tech\s+stack|proficiencies)"),
        ("experience", r"(?:work\s+experience|professional\s+experience|experience|employment\s+history|internship\s+experience|internships)"),
        ("projects", r"(?:personal\s+projects|academic\s+projects|project\s+experience|projects|key\s+projects)"),
        ("education", r"(?:education|academic\s+background|academics|qualifications|academic\s+qualifications)"),
        ("certifications", r"(?:certifications|certificates|licenses|achievements|honors\s*(?:&|and)?\s*awards)")
    ]

    # Combine into a unified delimiter pattern
    combined_pattern = r"(?im)^(?:\s*[\#\*\-]*\s*)(?P<header>" + "|".join(
        [f"(?P<{name}>{pat})" for name, pat in header_patterns]
    ) + r")\s*[:\-\u2014]?\s*(?:\n|\r\n)"

    matches = list(re.finditer(combined_pattern, text))
    if not matches:
        # Fallback: scan without requiring newline immediately after header
        fallback_pattern = r"(?im)^(?:\s*[\#\*\-]*\s*)(?P<header>" + "|".join(
            [f"(?P<{name}>{pat})" for name, pat in header_patterns]
        ) + r")\s*[:\-\u2014]\s*"
        matches = list(re.finditer(fallback_pattern, text))

    if not matches:
        # If no explicit sections found, allocate everything to experience & projects
        sections["experience"] = text
        return sections

    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()

        # Find which named group matched
        for name, _ in header_patterns:
            if match.group(name):
                sections[name] = section_text
                break

    return sections


def extract_skills_multi_section(
    full_text: str,
    sections: Dict[str, str],
    matcher: PhraseMatcher,
    nlp: spacy.language.Language,
    universal_skills: Set[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Extracts skills across the entire resume and tags their contextual evidence
    (found in Skills, Experience, Projects, Summary).
    Returns: {skill_name: {"sections": [...], "frequency": int, "evidence_weight": float}}
    """
    skills_found: Dict[str, Dict[str, Any]] = {}

    def scan_text_segment(segment_text: str, section_name: str, weight: float):
        if not segment_text:
            return
        doc = nlp(segment_text[:50000])  # safeguard length
        matches = matcher(doc)
        for _, start, end in matches:
            skill = doc[start:end].text.lower().strip()
            if not skill:
                continue
            if skill not in skills_found:
                skills_found[skill] = {
                    "sections": set(),
                    "frequency": 0,
                    "evidence_weight": 0.0
                }
            skills_found[skill]["sections"].add(section_name)
            skills_found[skill]["frequency"] += 1
            skills_found[skill]["evidence_weight"] += weight

    # Scan dedicated sections with custom weights
    scan_text_segment(sections.get("skills", ""), "Skills Section", 1.0)
    scan_text_segment(sections.get("experience", ""), "Work Experience", 1.5)
    scan_text_segment(sections.get("projects", ""), "Projects", 1.3)
    scan_text_segment(sections.get("summary", ""), "Summary", 0.8)

    # Also scan full text for anything outside parsed sections
    scan_text_segment(full_text, "General Text", 0.5)

    # Convert sets to sorted lists
    for skill, data in skills_found.items():
        data["sections"] = sorted(list(data["sections"]))

    return skills_found


def extract_job_skills_smart(
    job_text: str,
    matcher: PhraseMatcher,
    nlp: spacy.language.Language,
    universal_skills: Set[str],
    synonyms: Dict[str, List[str]]
) -> List[str]:
    """
    Smart extraction of skills required by a job description without falsely capturing
    generic proper nouns (e.g. company names, cities, titles).
    """
    normalized_jd = normalize_text_with_synonyms(job_text, synonyms)
    doc = nlp(normalized_jd[:50000])

    matches = matcher(doc)
    extracted = set()

    for _, start, end in matches:
        span = doc[start:end]
        skill = span.text.lower().strip()
        if skill:
            extracted.add(skill)

    # Additional regex scan for universal skills (especially single-token or punctuated terms)
    for skill in universal_skills:
        escaped = re.escape(skill)
        if re.search(rf"(?<!\w){escaped}(?!\w)", normalized_jd):
            extracted.add(skill)

    return sorted(list(extracted))


def auto_detect_domains(
    skills: List[str],
    text: str,
    skills_db: Dict[str, Any]
) -> List[Tuple[str, float]]:
    """
    Detects the primary domain(s) of the job description or resume based on skill overlap and frequency.
    Returns a sorted list of (domain_name, score).
    """
    scores = {}
    lower_text = text.lower()

    for domain_name, data in skills_db.items():
        domain_skills = set(s.lower() for s in data.get("skills", []))
        if not domain_skills:
            continue

        overlap = sum(1 for s in skills if s in domain_skills)
        keyword_hits = sum(1 for s in domain_skills if re.search(rf"\b{re.escape(s)}\b", lower_text))
        
        total_score = overlap * 2.0 + keyword_hits * 0.5
        if total_score > 0:
            scores[domain_name] = total_score

    sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_domains if sorted_domains else [("Full Stack Development", 1.0)]


def calculate_semantic_skill_matches(
    resume_skills: Set[str],
    job_skills: Set[str],
    sim_model: SentenceTransformer,
    semantic_threshold: float = 0.50
) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """
    3-Tier Matching Algorithm:
    - Tier 1: Direct matches (100%)
    - Tier 2: Semantic / Transferable matches (via competency clusters & contextual embeddings)
    - Tier 3: True missing skills
    """
    direct_matches = sorted(list(resume_skills & job_skills))
    unmatched_job_skills = list(job_skills - resume_skills)

    if not unmatched_job_skills or not resume_skills:
        missing_skills = sorted(unmatched_job_skills)
        return direct_matches, [], missing_skills

    resume_skills_list = list(resume_skills)
    semantic_matches = []
    still_unmatched = []

    # 1. Check known transferable clusters first
    matched_job_skills_set = set()
    for job_skill in unmatched_job_skills:
        cluster_found = False
        for cluster in TRANSFERABLE_CLUSTERS:
            c_skills = cluster["skills"]
            if job_skill in c_skills:
                # Find which resume skills are in the same cluster
                overlapping_resume_skills = [rs for rs in resume_skills_list if rs in c_skills]
                if overlapping_resume_skills:
                    best_match = overlapping_resume_skills[0]
                    semantic_matches.append({
                        "job_skill": job_skill,
                        "resume_skill": best_match,
                        "similarity": 85.0,
                        "cluster": cluster["name"],
                        "explanation": f"Candidate's proficiency in '{best_match}' demonstrates transferable knowledge in {cluster['name']} relevant to '{job_skill}'."
                    })
                    matched_job_skills_set.add(job_skill)
                    cluster_found = True
                    break
        if not cluster_found:
            still_unmatched.append(job_skill)

    # 2. Contextual embedding similarity for remaining skills
    if still_unmatched and resume_skills_list:
        job_prompts = [f"Proficiency and practical engineering experience in {s}" for s in still_unmatched]
        resume_prompts = [f"Proficiency and practical engineering experience in {s}" for s in resume_skills_list]

        job_embeddings = sim_model.encode(job_prompts, convert_to_tensor=True)
        resume_embeddings = sim_model.encode(resume_prompts, convert_to_tensor=True)

        sim_matrix = util.cos_sim(job_embeddings, resume_embeddings)

        missing_skills = []
        for i, job_skill in enumerate(still_unmatched):
            best_idx = int(sim_matrix[i].argmax())
            raw_score = float(sim_matrix[i][best_idx])
            best_resume_skill = resume_skills_list[best_idx]

            if raw_score >= semantic_threshold and best_resume_skill != job_skill:
                sim_pct = round(min(95.0, max(70.0, 70.0 + (raw_score - semantic_threshold) * 60.0)), 1)
                semantic_matches.append({
                    "job_skill": job_skill,
                    "resume_skill": best_resume_skill,
                    "similarity": sim_pct,
                    "cluster": "Semantic Alignment",
                    "explanation": f"Strong conceptual and operational similarity between '{best_resume_skill}' and '{job_skill}'."
                })
            else:
                missing_skills.append(job_skill)
    else:
        missing_skills = [s for s in unmatched_job_skills if s not in matched_job_skills_set]

    semantic_matches.sort(key=lambda x: x["similarity"], reverse=True)
    missing_skills.sort()

    return direct_matches, semantic_matches, missing_skills


def audit_resume_quality_and_ats(
    resume_text: str,
    sections: Dict[str, str]
) -> Dict[str, Any]:
    """
    Performs a thorough ATS and content quality audit:
    - Contact information verification (Email, Phone, LinkedIn, GitHub)
    - Section completeness
    - Quantification rate (% of bullet points with measurable impact metrics)
    - Action verb strength audit (Strong power verbs vs Weak/Passive verbs)
    - Word count & readability assessment
    """
    # 1. Contact Information
    email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", resume_text)
    phone_match = re.search(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", resume_text)
    linkedin_match = re.search(r"linkedin\.com/in/[\w\-_]+", resume_text, re.IGNORECASE)
    github_match = re.search(r"github\.com/[\w\-_]+", resume_text, re.IGNORECASE)

    contact_info = {
        "email": bool(email_match),
        "phone": bool(phone_match),
        "linkedin": bool(linkedin_match),
        "github": bool(github_match)
    }

    # 2. Section Completeness Check
    detected_sections = {
        "Summary / Objective": bool(sections.get("summary")),
        "Technical Skills": bool(sections.get("skills")),
        "Work Experience": bool(sections.get("experience")),
        "Projects": bool(sections.get("projects")),
        "Education": bool(sections.get("education")),
        "Certifications / Awards": bool(sections.get("certifications"))
    }

    # 3. Bullet Point Extraction (strictly target experience/projects or bullet syntax)
    candidate_bullet_source = (
        (sections.get("experience", "") + "\n" + sections.get("projects", ""))
        if (sections.get("experience") or sections.get("projects"))
        else resume_text
    )

    lines = [line.strip() for line in candidate_bullet_source.splitlines() if line.strip()]
    bullet_lines = []
    date_regex = re.compile(r"(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{4}|\b\d{4}\s*[\-\u2013\u2014]\s*(?:\d{4}|present))\b", re.IGNORECASE)

    for l in lines:
        # Ignore contact lines, degrees, GPA, and pure date/company headers
        lower_l = l.lower()
        if any(token in lower_l for token in ["@", "linkedin", "github", "phone", "http", "education", "gpa", "b.tech", "btech", "degree"]):
            continue
        if date_regex.search(l) and len(l) < 80:
            continue
        # True bullet or descriptive responsibility sentence
        if l.startswith(("-", "\u2022", "*", "\u2013", "\u2014")) or (len(l) > 40 and not l.isupper() and not l.endswith(":")):
            bullet_lines.append(l)

    quant_pattern = re.compile(
        r"(\b\d+(\.\d+)?%|\$\d+[\d,]*|\b\d+x\b|\b\d+\s*(?:k|m|million|billion|users|customers|queries|requests|ms|seconds|hours|percent)\b)",
        re.IGNORECASE
    )

    quantified_bullets = []
    unquantified_bullets = []

    for b in bullet_lines:
        if quant_pattern.search(b):
            quantified_bullets.append(b)
        else:
            unquantified_bullets.append(b)

    total_bullets = len(bullet_lines) if bullet_lines else 1
    quant_rate = round((len(quantified_bullets) / total_bullets) * 100, 1)

    # 4. Action Verb Audit
    lower_text = resume_text.lower()
    strong_verbs_detected = set()
    for verb in STRONG_ACTION_VERBS:
        if re.search(rf"\b{re.escape(verb)}\b", lower_text):
            strong_verbs_detected.add(verb)

    weak_verbs_detected = []
    for weak_verb, fix in WEAK_ACTION_VERBS.items():
        if re.search(rf"\b{re.escape(weak_verb)}\b", lower_text):
            weak_verbs_detected.append({"weak": weak_verb, "suggested": fix})

    # 5. Length & Word Count Check
    words = resume_text.split()
    word_count = len(words)
    if word_count < 250:
        length_status = "Too short (Under 250 words) - Expand on your responsibilities and projects."
        length_score = 50
    elif 350 <= word_count <= 850:
        length_status = "Optimal length (1-2 pages standard density)."
        length_score = 100
    elif 850 < word_count <= 1300:
        length_status = "Comprehensive length - Ensure all details are high-signal."
        length_score = 85
    else:
        length_status = "Very lengthy (Over 1300 words) - Consider trimming filler content."
        length_score = 70

    # ATS Health Score Calculation (0 - 100)
    contact_score = sum(25 for v in contact_info.values() if v)
    section_score = (sum(1 for v in detected_sections.values() if v) / len(detected_sections)) * 100
    action_verb_score = min(100, len(strong_verbs_detected) * 10) - (len(weak_verbs_detected) * 8)
    action_verb_score = max(20, min(100, action_verb_score))

    ats_score = int(
        0.25 * contact_score +
        0.25 * section_score +
        0.20 * min(100, quant_rate * 1.5) +
        0.15 * action_verb_score +
        0.15 * length_score
    )
    ats_score = max(10, min(100, ats_score))

    return {
        "ats_score": ats_score,
        "contact_info": contact_info,
        "detected_sections": detected_sections,
        "word_count": word_count,
        "length_status": length_status,
        "quant_rate": quant_rate,
        "quantified_count": len(quantified_bullets),
        "total_bullets": total_bullets,
        "strong_verbs_count": len(strong_verbs_detected),
        "strong_verbs_sample": sorted(list(strong_verbs_detected))[:12],
        "weak_verbs_detected": weak_verbs_detected,
        "sample_unquantified_bullet": unquantified_bullets[0] if unquantified_bullets else ""
    }


def generate_dynamic_recommendations(
    missing_skills: List[str],
    semantic_matches: List[Dict[str, Any]],
    direct_matches: List[str],
    primary_domain: str,
    audit_data: Dict[str, Any],
    job_text: str
) -> Dict[str, Any]:
    """
    Generates intelligent, context-aware suggestions:
    - Tailored project blueprints combining missing skills
    - Concrete STAR-method bullet rewrites
    - Targeted interview questions for matched and missing skills
    """
    project_blueprints = []

    if missing_skills:
        # Group missing skills into 2-3 key skills per project idea
        template = DOMAIN_PROJECT_TEMPLATES.get(
            primary_domain,
            DOMAIN_PROJECT_TEMPLATES["Full Stack Development"]
        )
        grouped_skills = ", ".join(missing_skills[:4])
        project_blueprints.append({
            "title": f"{template['title']} ({primary_domain})",
            "skills": missing_skills[:4],
            "description": template["description"].format(skills=f"**{grouped_skills}**")
        })

        if len(missing_skills) > 4:
            alt_domain = "Cloud DevOps" if primary_domain != "Cloud DevOps" else "Backend Development"
            alt_template = DOMAIN_PROJECT_TEMPLATES.get(alt_domain, DOMAIN_PROJECT_TEMPLATES["Backend Development"])
            remaining_skills = ", ".join(missing_skills[4:8])
            project_blueprints.append({
                "title": f"{alt_template['title']} ({alt_domain})",
                "skills": missing_skills[4:8],
                "description": alt_template["description"].format(skills=f"**{remaining_skills}**")
            })

    # STAR Bullet Rewrite Example
    sample_bullet = audit_data.get("sample_unquantified_bullet", "")
    if sample_bullet:
        cleaned_bullet = re.sub(r"^[\*\-\u2022\d\.\s]+", "", sample_bullet).strip()
        star_rewrite = {
            "original": cleaned_bullet,
            "star_framework": {
                "situation": "Identify the project context and high-level business or technical challenge.",
                "task": "Specify your distinct responsibility and tools used.",
                "action": "Detail the technical implementation using strong action verbs.",
                "result": "Highlight measurable impact (e.g. % performance increase, latency reduction, user growth)."
            },
            "example_rewrite": f"Architected and deployed {cleaned_bullet[:50]}... utilizing best practices, reducing execution latency by 35% and improving team release velocity."
        }
    else:
        star_rewrite = {
            "original": "Responsible for developing backend APIs and fixing bugs.",
            "star_framework": {
                "situation": "Enterprise microservices revamp.",
                "task": "Develop resilient, low-latency REST endpoints.",
                "action": "Built modular endpoints with automated unit testing and caching.",
                "result": "Increased endpoint throughput by 42% across 100k+ daily requests."
            },
            "example_rewrite": "Engineered 14+ high-throughput REST API endpoints in Python/FastAPI with Redis caching, reducing average response latency by 42% for 100k+ daily active users."
        }

    # Targeted Interview Questions
    interview_questions = []

    # 1. Questions on Matched Skills (to demonstrate technical depth)
    for skill in direct_matches[:2]:
        interview_questions.append({
            "type": "Technical Depth",
            "skill": skill,
            "question": f"Can you describe an architectural challenge or edge case you encountered when utilizing {skill.capitalize()} in production and how you resolved it?"
        })

    # 2. Bridge Questions on Transferable Skills
    for sm in semantic_matches[:2]:
        job_s = sm["job_skill"]
        res_s = sm["resume_skill"]
        interview_questions.append({
            "type": "Skill Bridge / Transferability",
            "skill": f"{res_s} -> {job_s}",
            "question": f"The position requires {job_s.capitalize()}, while your resume highlights {res_s.capitalize()}. How does your expertise in {res_s} translate, and what steps would you take to achieve immediate velocity with {job_s}?"
        })

    # 3. Questions on Critical Missing Skills
    for skill in missing_skills[:2]:
        interview_questions.append({
            "type": "Competency Growth",
            "skill": skill,
            "question": f"This role emphasizes {skill.capitalize()}. Have you explored or built any prototypes using {skill}, or how do your existing engineering competencies allow you to ramp up rapidly?"
        })

    return {
        "project_blueprints": project_blueprints,
        "star_rewrite": star_rewrite,
        "interview_questions": interview_questions
    }


def generate_markdown_report(
    resume_name: str,
    overall_score: int,
    semantic_similarity: float,
    experience_similarity: float,
    projects_similarity: float,
    direct_matches: List[str],
    semantic_matches: List[Dict[str, Any]],
    missing_skills: List[str],
    audit_data: Dict[str, Any],
    recommendations: Dict[str, Any],
    primary_domain: str
) -> str:
    """Generates a comprehensive markdown report for export/download."""
    md = []
    md.append(f"# AI Resume Analysis Report: {resume_name}\n")
    md.append(f"**Target Role Domain**: {primary_domain}  ")
    md.append(f"**Overall Match Score**: {overall_score}%  ")
    md.append(f"**ATS Health Score**: {audit_data.get('ats_score', 0)}/100  \n")
    md.append("---\n")

    md.append("## 1. Relevance Breakdown")
    md.append(f"- **Overall Semantic Alignment**: {int(semantic_similarity * 100)}%")
    md.append(f"- **Experience Relevance**: {int(experience_similarity * 100)}%")
    md.append(f"- **Projects Relevance**: {int(projects_similarity * 100)}%\n")

    md.append("## 2. Skill Alignment Breakdown")
    md.append("### Direct Skill Matches")
    if direct_matches:
        md.append(", ".join([f"`{s}`" for s in direct_matches]))
    else:
        md.append("No exact skill matches detected.")

    md.append("\n### Transferable / Semantic Matches")
    if semantic_matches:
        for sm in semantic_matches:
            md.append(f"- **{sm['job_skill']}** (Matched via `{sm['resume_skill']}` - {sm['similarity']}% similarity)")
    else:
        md.append("None detected.")

    md.append("\n### Missing Key Skills")
    if missing_skills:
        for ms in missing_skills:
            md.append(f"- `{ms}`")
    else:
        md.append("All key skills satisfied!")

    md.append("\n---\n")
    md.append("## 3. ATS & Content Health Audit")
    md.append(f"- **Word Count**: {audit_data.get('word_count', 0)} words ({audit_data.get('length_status', '')})")
    md.append(f"- **Quantified Bullets**: {audit_data.get('quant_rate', 0)}% of bullets contain measurable metrics")
    md.append(f"- **Strong Action Verbs Count**: {audit_data.get('strong_verbs_count', 0)}")

    if audit_data.get("weak_verbs_detected"):
        md.append("- **Weak Verbs to Replace**:")
        for wv in audit_data["weak_verbs_detected"]:
            md.append(f"  * `{wv['weak']}` -> Replace with: *{wv['suggested']}*")

    md.append("\n---\n")
    md.append("## 4. Action Plan & Project Recommendations")
    for bp in recommendations.get("project_blueprints", []):
        md.append(f"### {bp['title']}")
        md.append(f"{bp['description']}\n")

    md.append("## 5. STAR Method Bullet Enhancement")
    star = recommendations.get("star_rewrite", {})
    md.append(f"**Original Bullet**: *\"{star.get('original', '')}\"*  ")
    md.append(f"**Optimized Rewrite**: **\"{star.get('example_rewrite', '')}\"**\n")

    md.append("## 6. Tailored Interview Prep Questions")
    for q in recommendations.get("interview_questions", []):
        md.append(f"- **[{q['type']}]** ({q['skill']}): {q['question']}")

    return "\n".join(md)


def compute_match_score(
    semantic_similarity: float,
    experience_similarity: float,
    projects_similarity: float,
    direct_count: int,
    transferable_count: int,
    total_job_skills: int
) -> int:
    """
    Computes a balanced, holistic match score:
    - Direct skills: 1.0 weight
    - Transferable / semantic skills: 0.75 weight
    - Overall semantic embedding alignment: 20%
    - Section relevance (experience & projects): 25%
    """
    if total_job_skills == 0:
        skill_score = semantic_similarity
    else:
        effective_matches = direct_count + (0.75 * transferable_count)
        skill_score = min(1.0, effective_matches / total_job_skills)

    section_relevance = (
        (experience_similarity * 0.6 + projects_similarity * 0.4)
        if (experience_similarity > 0 or projects_similarity > 0)
        else semantic_similarity
    )

    final = (
        0.55 * skill_score +
        0.20 * semantic_similarity +
        0.25 * section_relevance
    )
    return int(max(5, min(99, round(final * 100))))

