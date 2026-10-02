import os
import re
import json
import sqlite3
import random
from pathlib import Path
import numpy as np
import requests
from flask import Flask, jsonify, render_template, request, session
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

load_dotenv()
# ============================================================
# 1. BASIC CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "learner.db"
VECTOR_PATH = DATA_DIR / "vectors.json"
# ============================================================
# 2. LOCAL AI CONFIGURATION
# ============================================================
# Free local embedding model.
#
# On the first run, Sentence Transformers may download
# this model. After that it is stored locally.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
print("Loading local embedding model...")
embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)
print("Embedding model loaded.")
# Optional Ollama configuration.
#
# The app works without Ollama, but chat and quiz quality
# are better when Ollama is running.
OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate"
)
OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:3b"
)

# OpenAI configuration. Keep the API key on the SERVER only.
# Never put OPENAI_API_KEY in index.html or browser JavaScript.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna").strip()
AI_PROVIDER = os.getenv("AI_PROVIDER", "auto").strip().lower()

openai_client = None
if OpenAI is not None and OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as exc:
        print("OpenAI client initialization failed:", exc)

# ============================================================
# 3. FLASK
# ============================================================
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("ADAPTLEARN_SECRET_KEY", "adaptlearn-local-demo-secret")
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024
# ============================================================
# 4. DATABASE INITIALIZATION
# ============================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # --------------------------------------------------------
    # Student accounts and user-specific learning data
    # --------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_learner_profile (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            topic TEXT NOT NULL,
            mastery REAL DEFAULT 0.0,
            attempts INTEGER DEFAULT 0,
            correct INTEGER DEFAULT 0,
            incorrect INTEGER DEFAULT 0,
            UNIQUE(user_id, topic)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_quiz_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            topic TEXT NOT NULL,
            question TEXT NOT NULL,
            user_answer TEXT,
            correct_answer TEXT,
            is_correct INTEGER DEFAULT 0,
            difficulty TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_study_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            topic TEXT NOT NULL,
            session_type TEXT DEFAULT 'chat',
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # --------------------------------------------------------
    # Learner profile
    # --------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS learner_profile (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT UNIQUE NOT NULL,
            mastery REAL DEFAULT 0.0,
            attempts INTEGER DEFAULT 0,
            correct INTEGER DEFAULT 0,
            incorrect INTEGER DEFAULT 0
        )
    """)
    # --------------------------------------------------------
    # Quiz attempt history
    # --------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS study_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            session_type TEXT DEFAULT 'chat',
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quiz_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            question TEXT NOT NULL,
            user_answer TEXT,
            correct_answer TEXT,
            is_correct INTEGER DEFAULT 0,
            difficulty TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
# Initialize database when program starts.
init_db()
# ============================================================
# 5. DOCUMENT TEXT EXTRACTION
# ============================================================
def extract_text_from_pdf(path):
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)
def extract_text_from_docx(path):
    document = Document(path)
    paragraphs = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)
def extract_text_from_txt(path):
    return Path(path).read_text(
        encoding="utf-8",
        errors="ignore"
    )
def extract_text(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    elif suffix == ".docx":
        return extract_text_from_docx(path)
    elif suffix == ".txt":
        return extract_text_from_txt(path)
    else:
        raise ValueError(
            "Unsupported file format."
        )
# ============================================================
# 6. CLEAN TEXT
# ============================================================
def clean_text(text):
    text = text.replace(
        "\x00",
        " "
    )
    # Preserve sentence endings while cleaning
    # excessive spaces.
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )
    text = re.sub(
        r"\n+",
        "\n",
        text
    )
    return text.strip()
# ============================================================
# 7. TEXT CHUNKING
# ============================================================
def chunk_text(
    text,
    chunk_size=1000,
    overlap=150
):
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[
            start:end
        ].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks
# ============================================================
# 8. LOCAL EMBEDDINGS
# ============================================================
def get_embedding(text):
    vector = embedding_model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return vector.tolist()
def get_embeddings(texts):
    if not texts:
        return []
    vectors = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return vectors.tolist()
# ============================================================
# 9. VECTOR DATABASE
# ============================================================
def load_vectors():
    if not VECTOR_PATH.exists():
        return []
    try:
        with open(
            VECTOR_PATH,
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)
    except Exception:
        return []
def save_vectors(vectors):
    with open(
        VECTOR_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            vectors,
            file,
            ensure_ascii=False
        )
# ============================================================
# 10. COSINE SIMILARITY
# ============================================================
def cosine_similarity(vector_a, vector_b):
    a = np.asarray(
        vector_a,
        dtype=np.float32
    )
    b = np.asarray(
        vector_b,
        dtype=np.float32
    )
    denominator = (
        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )
    if denominator == 0:
        return 0.0
    return float(
        np.dot(a, b)
        /
        denominator
    )
# ============================================================
# 11. RAG RETRIEVAL
# ============================================================
def retrieve_context(
    query,
    topic=None,
    top_k=5
):
    """Return only chunks that are relevant enough to answer the question."""
    vectors = load_vectors()
    if not vectors:
        return []
    query_embedding = get_embedding(query)
    similarity_threshold = 0.40
    scored_results = []
    for item in vectors:
        if not item.get("embedding") or not item.get("text"):
            continue
        raw_score = cosine_similarity(
            query_embedding,
            item["embedding"]
        )
        # Reject unrelated chunks BEFORE applying any topic/file boost.
        if raw_score < similarity_threshold:
            continue
        score = raw_score
        if topic:
            source = item.get("source", "").lower()
            if topic.lower() in source:
                score += 0.10
        scored_results.append((score, item))
    scored_results.sort(
        key=lambda result: result[0],
        reverse=True
    )
    return [
        {
            "source": item.get("source", "Unknown"),
            "text": item.get("text", ""),
            "score": round(score, 4)
        }
        for score, item in scored_results[:top_k]
    ]
# 12. OPEN SOURCE FALLBACK - WIKIPEDIA / WIKIMEDIA
# ============================================================
WIKIPEDIA_SEARCH_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page"
WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"

def search_wikipedia(query):
    """Search Wikimedia and return a useful article summary, or None."""
    try:
        headers = {
            "User-Agent": "AdaptLearnAI/1.0 educational-tutor"
        }
        response = requests.get(
            WIKIPEDIA_SEARCH_URL,
            params={
                "q": query,
                "limit": 3
            },
            headers=headers,
            timeout=8
        )
        response.raise_for_status()
        data = response.json()
        pages = data.get("pages", [])
        if not pages:
            return None
        for page in pages:
            title = page.get("title", "").strip()
            if not title:
                continue
            safe_title = requests.utils.quote(title, safe="")
            summary_response = requests.get(
                WIKIPEDIA_SUMMARY_URL + safe_title,
                headers=headers,
                timeout=8
            )
            if not summary_response.ok:
                continue
            summary = summary_response.json()
            extract = (
                summary.get("extract")
                or ""
            ).strip()
            if extract:
                return {
                    "source": "Wikipedia",
                    "title": title,
                    "text": extract,
                    "url": summary.get("content_urls", {})
                        .get("desktop", {})
                        .get("page", "")
                }
    except Exception as exc:
        print("Wikipedia lookup failed:", exc)
    return None

def generate_external_answer(message, topic):
    """Answer from the open-source Wikimedia fallback."""
    result = search_wikipedia(message)
    if not result:
        return None
    prompt = f"""
You are AdaptLearn AI, an engineering tutor.
STUDENT QUESTION:
{message}
TOPIC:
{topic}
OPEN-SOURCE SOURCE: Wikipedia
ARTICLE:
{result['title']}
SOURCE CONTENT:
{result['text']}

Instructions:
1. Answer the student's question using only the supplied source content.
2. Do not invent facts that are not supported by the source.
3. Explain the answer clearly for an engineering student.
4. Keep the answer concise.
5. State that the information came from Wikipedia.
"""
    try:
        if AI_PROVIDER == "ollama":
            answer = ask_ollama(prompt)
        elif AI_PROVIDER == "openai":
            answer = ask_openai(prompt)
        elif openai_available():
            answer = ask_openai(prompt)
        elif ollama_available():
            answer = ask_ollama(prompt)
        else:
            answer = ""
        if answer:
            return answer.strip() + f"\n\nSource: Wikipedia — {result['title']}"
    except Exception as exc:
        print("External AI answer failed:", exc)
    return (
        f"According to Wikipedia ({result['title']}):\n\n"
        f"{result['text']}\n\n"
        "Source: Wikipedia"
    )


# ============================================================
# 13. AUTHENTICATION / CURRENT STUDENT
# ============================================================
def current_user_id():
    value = session.get("user_id")
    return int(value) if value else None

def current_user():
    user_id = current_user_id()
    if not user_id:
        return None
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT id, name, username FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    if not row:
        session.clear()
        return None
    return {"id": row[0], "name": row[1], "username": row[2]}

# ============================================================
# 14. LEARNER PROFILE
# ============================================================
def get_mastery(topic):
    user_id = current_user_id()
    if not user_id:
        return 0.0
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT mastery FROM user_learner_profile WHERE user_id = ? AND topic = ?",
        (user_id, topic)
    ).fetchone()
    conn.close()
    return float(row[0]) if row else 0.0
# ============================================================
# 13. DIFFICULTY SELECTION
# ============================================================
def get_difficulty(topic):
    mastery = get_mastery(
        topic
    )
    if mastery < 0.35:
        return "easy"
    elif mastery < 0.70:
        return "medium"
    else:
        return "hard"
# ============================================================
# 14. UPDATE MASTERY
# ============================================================
def update_mastery(topic, is_correct):
    user_id = current_user_id()
    if not user_id:
        return 0.0
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    row = cursor.execute(
        "SELECT mastery, attempts, correct, incorrect FROM user_learner_profile WHERE user_id = ? AND topic = ?",
        (user_id, topic)
    ).fetchone()
    if row is None:
        mastery = 0.15 if is_correct else 0.0
        attempts = 1
        correct = 1 if is_correct else 0
        incorrect = 0 if is_correct else 1
        cursor.execute(
            "INSERT INTO user_learner_profile (user_id, topic, mastery, attempts, correct, incorrect) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, topic, mastery, attempts, correct, incorrect)
        )
    else:
        mastery, attempts, correct, incorrect = row
        attempts = int(attempts) + 1
        correct = int(correct) + (1 if is_correct else 0)
        incorrect = int(incorrect) + (0 if is_correct else 1)
        mastery = float(mastery) + (0.15 if is_correct else -0.10)
        mastery = max(0.0, min(1.0, mastery))
        cursor.execute(
            "UPDATE user_learner_profile SET mastery = ?, attempts = ?, correct = ?, incorrect = ? WHERE user_id = ? AND topic = ?",
            (mastery, attempts, correct, incorrect, user_id, topic)
        )
    conn.commit()
    conn.close()
    return mastery
# ============================================================
# 15. SAVE QUIZ ATTEMPT
# ============================================================
def save_quiz_attempt(topic, question, user_answer, correct_answer, is_correct, difficulty):
    user_id = current_user_id()
    if not user_id:
        return
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO user_quiz_attempts (user_id, topic, question, user_answer, correct_answer, is_correct, difficulty) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (user_id, topic, question, user_answer, correct_answer, 1 if is_correct else 0, difficulty)
    )
    conn.commit()
    conn.close()
# ============================================================
# 16. OPENAI API
# ============================================================
def openai_available():
    return openai_client is not None and bool(OPENAI_API_KEY)

def ask_openai(prompt):
    if not openai_available():
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        input=prompt
    )
    text = getattr(response, "output_text", "") or ""
    return text.strip()

def ai_provider_name():
    if AI_PROVIDER == "openai":
        return "openai" if openai_available() else "openai-not-configured"
    if AI_PROVIDER == "ollama":
        return "ollama" if ollama_available() else "ollama-not-available"
    if openai_available():
        return "openai"
    if ollama_available():
        return "ollama"
    return "retrieval-fallback"

# ============================================================
# 17. OLLAMA STATUS
# ============================================================
def ollama_available():
    try:
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=2
        )
        return (
            response.status_code
            == 200
        )
    except requests.RequestException:
        return False
# ============================================================
# 17. LOCAL LLM CALL
# ============================================================
def ask_ollama(
    prompt,
    temperature=0.3
):
    payload = {
        "model":
            OLLAMA_MODEL,
        "prompt":
            prompt,
        "stream":
            False,
        "options": {
            "temperature":
                temperature
        }
    }
    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=180
    )
    response.raise_for_status()
    data = response.json()
    return data.get(
        "response",
        ""
    ).strip()
# ============================================================
# 18. FALLBACK RAG ANSWER
# ============================================================
def fallback_rag_answer(
    question,
    context_items
):
    if not context_items:
        return (
            "I could not find relevant information "
            "in the uploaded study materials."
        )
    # Without a local LLM, show the strongest
    # retrieved pieces of evidence.
    passages = []
    for item in context_items[:3]:
        text = item["text"]
        if len(text) > 600:
            text = (
                text[:600]
                + "..."
            )
        passages.append(
            f"From {item['source']}:\n{text}"
        )
    return (
        "I found these relevant sections in your "
        "uploaded study material:\n\n"
        +
        "\n\n".join(passages)
    )
# ============================================================
# 19. LOCAL RAG TUTOR
# ============================================================
def generate_tutor_answer(
    message,
    topic,
    mastery,
    difficulty,
    context_items
):
    if not context_items:
        external_answer = generate_external_answer(
            message=message,
            topic=topic
        )
        if external_answer:
            return external_answer
        return "Couldn't find answer in the uploaded study material or the open-source knowledge source."

    context_text = "\n\n".join(
        [
            f"SOURCE: {item['source']}\n{item['text']}"
            for item in context_items
        ]
    )
    prompt = f"""
You are AdaptLearn AI, an adaptive tutor for engineering students.

SUBJECT:
{topic}
LEARNER MASTERY:
{mastery:.2f}
CURRENT DIFFICULTY:
{difficulty}
STUDENT QUESTION:
{message}

UPLOADED STUDY MATERIAL:
{context_text}

Instructions:
1. Use the uploaded material as the primary source.
2. Do not invent facts that are not supported by the material.
3. Explain the concept according to the student's current mastery level.
4. If mastery is low, use simple explanations.
5. If mastery is medium, include conceptual reasoning.
6. If mastery is high, include deeper technical reasoning.
7. Prefer Socratic guidance when appropriate.
8. Mention the relevant uploaded source when useful.
9. Keep the response concise and suitable for an engineering student.
"""
    try:
        if AI_PROVIDER == "ollama":
            if not ollama_available():
                raise RuntimeError("Ollama is not available.")
            return ask_ollama(prompt)
        if AI_PROVIDER == "openai":
            return ask_openai(prompt)
        if openai_available():
            return ask_openai(prompt)
        if ollama_available():
            return ask_ollama(prompt)
    except Exception as exc:
        print("Tutor AI call failed:", exc)
    return fallback_rag_answer(message, context_items)

# ============================================================
# 20. JSON EXTRACTION FROM LOCAL LLM
# ============================================================
def extract_json(text):
    if not text:
        return None
    text = text.strip()
    # Remove markdown fences.
    text = re.sub(
        r"^\`\`\`json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(
        r"^\`\`\`\s*",
        "",
        text
    )
    text = re.sub(
        r"\s*\`\`\`$",
        "",
        text
    )
    # First try direct parsing.
    try:
        return json.loads(
            text
        )
    except Exception:
        pass
    # Then attempt to find the first
    # JSON object.
    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL
    )
    if not match:
        return None
    try:
        return json.loads(
            match.group(0)
        )
    except Exception:
        return None
# ============================================================
# 21. FALLBACK QUIZ GENERATOR
# ============================================================
def generate_fallback_quiz(
    topic,
    difficulty,
    context_items,
    focus=""
):
    if not context_items:
        return {
            "question":
                (
                    f"Which topic is the student currently practicing "
                    f"within {topic}?"
                ),
            "options": [
                topic,
                "Unrelated topic A",
                "Unrelated topic B",
                "Unrelated topic C"
            ],
            "correct_answer":
                topic,
            "explanation":
                (
                    "Review the selected topic using "
                    "your uploaded study materials."
                )
        }
    text = context_items[0]["text"]
    # Break retrieved text into sentences.
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )
    sentences = [
        sentence.strip()
        for sentence
        in sentences
        if len(
            sentence.split()
        ) >= 8
    ]
    if not sentences:
        sentence = (
            text[:300]
            .strip()
        )
    else:
        sentence = sentences[0]
    # Create a simple comprehension question.
    focus_label = focus or topic
    question = (
        f"Which statement is supported by the uploaded material "
        f"about {focus_label} within {topic}?"
    )
    correct_answer = sentence
    # Use other sentences from retrieved chunks
    # as distractors where possible.
    candidates = []
    for item in context_items[1:]:
        other_sentences = re.split(
            r"(?<=[.!?])\s+",
            item["text"]
        )
        for other in other_sentences:
            other = other.strip()
            if (
                len(other.split()) >= 8
                and other != correct_answer
            ):
                candidates.append(
                    other
                )
            if len(candidates) >= 3:
                break
        if len(candidates) >= 3:
            break
    # Fill missing choices.
    while len(candidates) < 3:
        candidates.append(
            (
                f"This statement is not supported by "
                f"the retrieved material for {topic} "
                f"({len(candidates) + 1})."
            )
        )
    options = [
        correct_answer
    ] + candidates[:3]
    random.shuffle(
        options
    )
    return {
        "question":
            question,
        "options":
            options,
        "correct_answer":
            correct_answer,
        "explanation":
            (
                "The correct statement comes directly "
                "from the retrieved study material."
            )
    }
# ============================================================
# 22. ADAPTIVE QUIZ GENERATION
# ============================================================
def generate_quiz_with_local_ai(
    topic,
    mastery,
    difficulty,
    context_items,
    focus=""
):
    if not context_items:
        return generate_fallback_quiz(topic, difficulty, context_items, focus)

    context_text = "\n\n".join(
        [f"SOURCE: {item['source']}\n{item['text']}" for item in context_items]
    )
    prompt = f"""
You are generating an adaptive engineering quiz for a student who is actively studying a specific subject.
SUBJECT:
{topic}
CURRENT LEARNING FOCUS:
{focus or topic}
LEARNER MASTERY:
{mastery:.2f}
DIFFICULTY:
{difficulty}
SOURCE MATERIAL:
{context_text}

Create ONE multiple choice question that directly tests the CURRENT LEARNING FOCUS within the SUBJECT.
Rules:
- The question must be specifically about the SUBJECT and CURRENT LEARNING FOCUS, not generic engineering knowledge.
- Prefer the exact concept, law, formula, process, definition, or application the student was just asking about.
- Use only information supported by the provided study material.
- Match the question difficulty to {difficulty}.
- Provide exactly four options.
- Exactly one option must be correct.
- Provide a short explanation.
- Do not include markdown.
- Return ONLY valid JSON.
Required format:
{{
  "question": "question text",
  "options": ["option 1", "option 2", "option 3", "option 4"],
  "correct_answer": "exact option text",
  "explanation": "short explanation"
}}
"""
    try:
        if AI_PROVIDER == "ollama":
            if not ollama_available():
                raise RuntimeError("Ollama is not available.")
            response = ask_ollama(prompt, temperature=0.2)
        elif AI_PROVIDER == "openai":
            response = ask_openai(prompt)
        elif openai_available():
            response = ask_openai(prompt)
        elif ollama_available():
            response = ask_ollama(prompt, temperature=0.2)
        else:
            raise RuntimeError("No AI provider configured.")
        quiz = extract_json(response)
        if not quiz:
            raise ValueError("Invalid JSON from AI model.")
        required_fields = ["question", "options", "correct_answer", "explanation"]
        for field in required_fields:
            if field not in quiz:
                raise ValueError(f"Missing field: {field}")
        if not isinstance(quiz["options"], list) or len(quiz["options"]) != 4:
            raise ValueError("Quiz must contain four options.")
        if quiz["correct_answer"] not in quiz["options"]:
            raise ValueError("Correct answer must exactly match one option.")
        return quiz
    except Exception as exc:
        print("Quiz AI call failed:", exc)
        return generate_fallback_quiz(topic, difficulty, context_items, focus)

# ============================================================
# 23. HOME PAGE
# ============================================================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/me")
def me():
    user = current_user()
    return jsonify({"logged_in": bool(user), "user": user})

@app.route("/api/register", methods=["POST"])
def register():
    from werkzeug.security import generate_password_hash
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    username = data.get("username", "").strip().lower()
    password = data.get("password", "")
    if not name or not username or not password:
        return jsonify({"error": "Name, username, and password are required."}), 400
    if len(password) < 4:
        return jsonify({"error": "Password must contain at least 4 characters."}), 400
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute(
            "INSERT INTO users (name, username, password_hash) VALUES (?, ?, ?)",
            (name, username, generate_password_hash(password))
        )
        conn.commit()
        user_id = cur.lastrowid
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"error": "That username is already registered."}), 409
    conn.close()
    session["user_id"] = user_id
    return jsonify({"success": True, "user": {"id": user_id, "name": name, "username": username}})

@app.route("/api/login", methods=["POST"])
def login():
    from werkzeug.security import check_password_hash
    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip().lower()
    password = data.get("password", "")
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT id, name, username, password_hash FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    if not row or not check_password_hash(row[3], password):
        return jsonify({"error": "Invalid username or password."}), 401
    session["user_id"] = row[0]
    return jsonify({"success": True, "user": {"id": row[0], "name": row[1], "username": row[2]}})

@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"success": True})

# ============================================================
# 24. HEALTH CHECK
# ============================================================
@app.route("/api/health")
def health():
    vectors = load_vectors()
    return jsonify({
        "status":
            "ok",
        "message":
            "AdaptLearn AI backend is running.",
        "embedding":
            "local",
        "embedding_model":
            EMBEDDING_MODEL_NAME,
        "stored_chunks":
            len(vectors),
        "ollama_available":
            ollama_available(),
        "ollama_model":
            OLLAMA_MODEL,
        "openai_available":
            openai_available(),
        "openai_model":
            OPENAI_MODEL,
        "ai_provider":
            ai_provider_name()
    })
# ============================================================
# 25. UPLOAD + LOCAL INDEXING
# ============================================================
@app.route(
    "/api/upload",
    methods=["POST"]
)
def upload_material():
    if "file" not in request.files:
        return jsonify({
            "error":
                "No file uploaded."
        }), 400
    uploaded_file = request.files[
        "file"
    ]
    if not uploaded_file.filename:
        return jsonify({
            "error":
                "No filename provided."
        }), 400
    filename = Path(
        uploaded_file.filename
    ).name
    extension = Path(
        filename
    ).suffix.lower()
    allowed_extensions = {
        ".pdf",
        ".docx",
        ".txt"
    }
    if extension not in allowed_extensions:
        return jsonify({
            "error":
                (
                    "Only PDF, DOCX and TXT "
                    "files are supported."
                )
        }), 400
    save_path = (
        UPLOAD_DIR
        /
        filename
    )
    uploaded_file.save(
        save_path
    )
    try:
        # --------------------------------------------
        # Extract document text
        # --------------------------------------------
        text = extract_text(
            save_path
        )
        text = clean_text(
            text
        )
        if not text:
            return jsonify({
                "error":
                    (
                        "No readable text could be "
                        "extracted from this document."
                    )
            }), 400
        # --------------------------------------------
        # Split into RAG chunks
        # --------------------------------------------
        chunks = chunk_text(
            text
        )
        if not chunks:
            return jsonify({
                "error":
                    "No document chunks were created."
            }), 400
        # --------------------------------------------
        # Generate ALL embeddings locally
        # --------------------------------------------
        embeddings = get_embeddings(
            chunks
        )
        vectors = load_vectors()
        # Remove old chunks from the same document,
        # preventing duplicates after re-upload.
        vectors = [
            item
            for item
            in vectors
            if item.get("source")
            != filename
        ]
        # --------------------------------------------
        # Store local vectors
        # --------------------------------------------
        for index, (
            chunk,
            embedding
        ) in enumerate(
            zip(
                chunks,
                embeddings
            )
        ):
            vectors.append({
                "source":
                    filename,
                "chunk_id":
                    index,
                "text":
                    chunk,
                "embedding":
                    embedding
            })
        save_vectors(
            vectors
        )
        return jsonify({
            "success":
                True,
            "filename":
                filename,
            "chunks":
                len(chunks),
            "embedding":
                "local",
            "message":
                (
                    f"{filename} was indexed "
                    f"successfully using local embeddings."
                )
        })
    except Exception as error:
        return jsonify({
            "error":
                str(error)
        }), 500
# ============================================================
# 26. MATERIAL LIST
# ============================================================
@app.route("/api/materials")
def materials():
    files = []
    for path in UPLOAD_DIR.iterdir():
        if path.is_file():
            files.append({
                "name":
                    path.name,
                "size":
                    path.stat().st_size
            })
    return jsonify({
        "materials":
            files
    })
# ============================================================
# 27. RAG SEARCH TEST ENDPOINT
# ============================================================
@app.route(
    "/api/search",
    methods=["POST"]
)
def search_material():
    data = request.get_json(
        silent=True
    ) or {}
    query = data.get(
        "query",
        ""
    ).strip()
    topic = data.get(
        "topic",
        ""
    ).strip()
    if not query:
        return jsonify({
            "error":
                "Query is required."
        }), 400
    results = retrieve_context(
        query=query,
        topic=topic,
        top_k=5
    )
    return jsonify({
        "query":
            query,
        "results":
            results
    })
# ============================================================
# 28. ADAPTIVE TUTOR CHAT
# ============================================================
@app.route(
    "/api/chat",
    methods=["POST"]
)
def chat():
    data = request.get_json(
        silent=True
    ) or {}
    message = data.get(
        "message",
        ""
    ).strip()
    topic = data.get(
        "topic",
        "General Engineering"
    ).strip()
    if not message:
        return jsonify({
            "error":
                "Message is required."
        }), 400
    record_study_session(topic, "chat")
    mastery = get_mastery(
        topic
    )
    difficulty = get_difficulty(
        topic
    )
    context_items = retrieve_context(
        query=message,
        topic=topic,
        top_k=5
    )
    answer = generate_tutor_answer(
        message=message,
        topic=topic,
        mastery=mastery,
        difficulty=difficulty,
        context_items=context_items
    )
    sources = list(
        dict.fromkeys(
            [
                item["source"]
                for item
                in context_items
            ]
        )
    )
    return jsonify({
        "answer":
            answer,
        "topic":
            topic,
        "mastery":
            mastery,
        "difficulty":
            difficulty,
        "sources":
            sources,
        "local_llm":
            ollama_available()
    })
# ============================================================
# 29. ADAPTIVE QUIZ GENERATION
# ============================================================
@app.route(
    "/api/quiz/generate",
    methods=["POST"]
)
def generate_quiz():
    data = request.get_json(
        silent=True
    ) or {}
    focus = data.get(
        "focus",
        ""
    ).strip()
    topic = data.get(
        "topic",
        "General Engineering"
    ).strip()
    if not topic:
        return jsonify({
            "error":
                "Topic is required."
        }), 400
    mastery = get_mastery(
        topic
    )
    difficulty = get_difficulty(
        topic
    )
    retrieval_query = focus or topic
    context_items = retrieve_context(
        query=retrieval_query,
        topic=topic,
        top_k=5
    )
    quiz = generate_quiz_with_local_ai(
        topic=topic,
        mastery=mastery,
        difficulty=difficulty,
        context_items=context_items,
        focus=focus
    )
    sources = list(
        dict.fromkeys(
            [
                item["source"]
                for item
                in context_items
            ]
        )
    )
    return jsonify({
        "success":
            True,
        "topic":
            topic,
        "mastery":
            mastery,
        "difficulty":
            difficulty,
        "focus":
            focus or topic,
        "quiz":
            quiz,
        "sources":
            sources,
        "local_llm":
            ollama_available()
    })
# ============================================================
# 30. QUIZ ATTEMPT
# ============================================================
@app.route(
    "/api/quiz/attempt",
    methods=["POST"]
)
def quiz_attempt():
    data = request.get_json(
        silent=True
    ) or {}
    topic = data.get(
        "topic",
        ""
    ).strip()
    question = data.get(
        "question",
        ""
    ).strip()
    user_answer = data.get(
        "user_answer",
        ""
    ).strip()
    correct_answer = data.get(
        "correct_answer",
        ""
    ).strip()
    difficulty = data.get(
        "difficulty",
        "easy"
    ).strip()
    if not topic:
        return jsonify({
            "error":
                "Topic is required."
        }), 400
    if not question:
        return jsonify({
            "error":
                "Question is required."
        }), 400
    if not correct_answer:
        return jsonify({
            "error":
                "Correct answer is required."
        }), 400
    # Case-insensitive comparison.
    is_correct = (
        user_answer.casefold()
        ==
        correct_answer.casefold()
    )
    # --------------------------------------------
    # Adaptive learner update
    # --------------------------------------------
    new_mastery = update_mastery(
        topic,
        is_correct
    )
    # --------------------------------------------
    # Save attempt
    # --------------------------------------------
    save_quiz_attempt(
        topic=topic,
        question=question,
        user_answer=user_answer,
        correct_answer=correct_answer,
        is_correct=is_correct,
        difficulty=difficulty
    )
    next_difficulty = get_difficulty(
        topic
    )
    return jsonify({
        "success":
            True,
        "is_correct":
            is_correct,
        "mastery":
            round(
                new_mastery,
                3
            ),
        "mastery_percent":
            round(
                new_mastery * 100,
                1
            ),
        "next_difficulty":
            next_difficulty,
        "message":
            (
                "Correct! Your mastery increased."
                if is_correct
                else
                (
                    "That answer was incorrect. "
                    "The system will adjust your "
                    "learning path."
                )
            )
    })
# ============================================================
# 31. STUDY SESSION TRACKING
# ============================================================
def record_study_session(topic, session_type="chat"):
    user_id = current_user_id()
    if not user_id:
        return
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO user_study_sessions (user_id, topic, session_type) VALUES (?, ?, ?)", (user_id, topic or "General Engineering", session_type))
    conn.commit()
    conn.close()

# ============================================================
# 31. DASHBOARD
# ============================================================
@app.route("/api/dashboard")
def dashboard():
    user_id = current_user_id()
    if not user_id:
        return jsonify({"error": "Login required."}), 401
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    rows = cursor.execute(
        "SELECT topic, mastery, attempts, correct, incorrect FROM user_learner_profile WHERE user_id = ? ORDER BY topic",
        (user_id,)
    ).fetchall()
    topics = []
    total_attempts = total_correct = total_incorrect = 0
    for topic, mastery, attempts, correct, incorrect in rows:
        total_attempts += int(attempts); total_correct += int(correct); total_incorrect += int(incorrect)
        topics.append({"topic": topic, "mastery": round(float(mastery) * 100, 1), "attempts": int(attempts), "correct": int(correct), "incorrect": int(incorrect), "difficulty": get_difficulty(topic)})
    quiz_attempts = int(cursor.execute("SELECT COUNT(*) FROM user_quiz_attempts WHERE user_id = ?", (user_id,)).fetchone()[0])
    study_sessions = int(cursor.execute("SELECT COUNT(*) FROM user_study_sessions WHERE user_id = ?", (user_id,)).fetchone()[0])
    completed_topics = int(cursor.execute("SELECT COUNT(DISTINCT topic) FROM user_learner_profile WHERE user_id = ? AND attempts > 0", (user_id,)).fetchone()[0])
    mistakes = int(cursor.execute("SELECT COUNT(*) FROM user_quiz_attempts WHERE user_id = ? AND is_correct = 0", (user_id,)).fetchone()[0])
    weak_areas = [{"topic": r[0], "mistakes": int(r[1])} for r in cursor.execute("SELECT topic, COUNT(*) FROM user_quiz_attempts WHERE user_id = ? AND is_correct = 0 GROUP BY topic ORDER BY COUNT(*) DESC LIMIT 5", (user_id,)).fetchall()]
    recent_sessions = [{"topic": r[0], "type": r[1], "created_at": r[2]} for r in cursor.execute("SELECT topic, session_type, started_at FROM user_study_sessions WHERE user_id = ? ORDER BY id DESC LIMIT 10", (user_id,)).fetchall()]
    conn.close()
    accuracy = round(total_correct / total_attempts * 100, 1) if total_attempts else 0
    return jsonify({"user": current_user(), "topics": topics, "summary": {"total_attempts": total_attempts, "correct": total_correct, "incorrect": total_incorrect, "accuracy": accuracy, "quiz_attempts": quiz_attempts, "study_sessions": study_sessions, "completed_topics": completed_topics, "mistakes": mistakes}, "weak_areas": weak_areas, "recent_sessions": recent_sessions})
@app.route("/api/quiz/history")
def quiz_history():
    user_id = current_user_id()
    if not user_id:
        return jsonify({"error": "Login required."}), 401
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT topic, question, user_answer, correct_answer, is_correct, difficulty, created_at FROM user_quiz_attempts WHERE user_id = ? ORDER BY id DESC LIMIT 50", (user_id,)).fetchall()
    conn.close()
    return jsonify({"attempts": [{"topic": r[0], "question": r[1], "user_answer": r[2], "correct_answer": r[3], "is_correct": bool(r[4]), "difficulty": r[5], "created_at": r[6]} for r in rows]})
# ============================================================
# 33. START FLASK
# ============================================================
if __name__ == "__main__":
    print()
    print("=" * 60)
    print("AdaptLearn AI - API Ready Backend")
    print("=" * 60)
    print(f"Embeddings: LOCAL ({EMBEDDING_MODEL_NAME})")
    print(f"AI provider: {ai_provider_name()}")
    if openai_available():
        print(f"OpenAI model: {OPENAI_MODEL}")
    if ollama_available():
        print(f"Ollama model: {OLLAMA_MODEL}")
    print("API key: configured on server" if openai_available() else "OpenAI API key: not configured")
    print("=" * 60)
    app.run(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true"
    )
