# AI Resume Analyzer 

An advanced, offline-first AI resume analysis and intelligence platform that moves beyond rigid keyword matching to provide **3-tier semantic skill matching**, **transferable competency discovery**, **ATS readiness auditing**, and **dynamic context-aware improvement plans**.

---

## Key Features

### 1. 3-Tier Technical Skill & Competency Matching
- **Tier 1: Direct & Normalized Matches**: Canonical tech aliases and abbreviations (e.g. `React.js` <-> `React`, `k8s` <-> `Kubernetes`, `Postgres` <-> `PostgreSQL`).
- **Tier 2: Semantic & Transferable Matches**: Leverages `SentenceTransformer` embeddings and technical competency clusters to identify transferable frameworks (e.g., recognizing `PyTorch` as transferable to `TensorFlow`, `FastAPI` to `Flask`, or `Docker` to `Kubernetes` with similarity percentages).
- **Tier 3: True Missing Gaps**: Accurately isolates critical missing skills without false penalties for transferable equivalents.

### 2. Multi-Section Whole-Resume Skill Extraction
- Scans all sections (Summary, Technical Skills, Work Experience, Projects, Education) and maps **evidence depth** (e.g., whether a tool was just listed in skills or actively used in production work experience).

### 3. Dynamic ATS & Content Health Audit
- **Contact Info Verification**: Detects Email, Phone, LinkedIn, and GitHub links.
- **Section Architecture Check**: Audits the presence of standard ATS sections.
- **Measurable Impact & Quantification**: Scans bullets for metrics (`%`, `$`, `k`, `ms`, scale, latency) and computes your quantifiable achievement rate.
- **Action Verb Strength**: Highlights strong power verbs vs weak/passive phrases (`"worked on"`, `"responsible for"`) with concrete replacements.

### 4. Tailored Action Plan & Next Steps
- **Custom Project Blueprints**: Synthesizes missing skills into realistic architectural project ideas tailored to your target domain.
- **STAR Method Bullet Rewriter**: Selects actual bullet points from your resume and provides an actionable Before vs. After rewrite using the STAR framework.
- **Targeted Interview Preparation**: Formulates technical depth questions for matched skills, transition/bridge questions for transferable skills, and prep questions for missing technologies.

### 5. Flexible Input & Export
- Dual resume input: Upload PDF or Paste raw text directly.
- Instant pre-configured sample presets (AI/ML Engineer, Full Stack Developer).
- One-click Downloadable Comprehensive Analysis Report (`.md`).

---

## Project Architecture

```
resume-analyzer/
|
|-- App.py                     # Streamlit frontend with modern multi-tab dashboard
|-- analyzer_engine.py         # Modular core NLP, semantic matching & dynamic audit engine
|-- skill_synonyms.json        # Expanded tech synonym and alias mappings
|-- skills_database.json       # Comprehensive multi-domain skill taxonomy
|-- test_analyzer.py           # Verification and integration test suite
|-- requirements.txt           # Project dependencies
`-- README.md                  # Documentation
```

---

## Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Venkatasai-rohith/resume-analyzer.git
cd resume-analyzer
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download spaCy Language Model
```bash
python -m spacy download en_core_web_sm
```

### 4. Run the Application
```bash
streamlit run App.py
```

### 5. Open in Browser
Navigate to: `http://localhost:8501`

---

## Tech Stack
- **Frontend**: Streamlit
- **Document Parsing**: pdfplumber
- **NLP & Linguistics**: spaCy (`en_core_web_sm`)
- **Semantic Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)
- **Machine Learning / Vector Ops**: PyTorch, Scikit-learn, NumPy

---

## Author
**P. Venkata Sai Rohith**  
AIML Student & Software Engineer  
GitHub: [Venkatasai-rohith](https://github.com/Venkatasai-rohith)

---

## License
This project is licensed under the MIT License.
