import os

import re

import json

import sqlite3

import random

from pathlib import Path



import numpy as np

import requests



from flask import Flask, jsonify, render_template, request

from pypdf import PdfReader

from docx import Document

from sentence_transformers import SentenceTransformer





\# ============================================================

\# 1. BASIC CONFIGURATION

\# ============================================================



BASE_DIR = Path(\_\_file\_\_).resolve().parent



DATA_DIR = BASE_DIR / "data"

UPLOAD_DIR = BASE_DIR / "uploads"



DATA_DIR.mkdir(exist_ok=True)

UPLOAD_DIR.mkdir(exist_ok=True)



DB_PATH = DATA_DIR / "learner.db"

VECTOR_PATH = DATA_DIR / "vectors.json"





\# ============================================================

\# 2. LOCAL AI CONFIGURATION

\# ============================================================



\# Free local embedding model.

\#

\# On the first run, Sentence Transformers may download

\# this model. After that it is stored locally.

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"



print("Loading local embedding model...")



embedding_model = SentenceTransformer(

    EMBEDDING_MODEL_NAME

)



print("Embedding model loaded.")





\# Optional Ollama configuration.

\#

\# The app works without Ollama, but chat and quiz quality

\# are better when Ollama is running.



OLLAMA_URL = os.getenv(

    "OLLAMA_URL",

    "http\://localhost:11434/api/generate"

)



OLLAMA_MODEL = os.getenv(

    "OLLAMA_MODEL",

    "llama3.2:3b"

)





\# ============================================================

\# 3. FLASK

\# ============================================================



app = Flask(\_\_name\_\_)





\# ============================================================

\# 4. DATABASE INITIALIZATION

\# ============================================================



def init_db():



    conn = sqlite3.connect(DB_PATH)



    cursor = conn.cursor()



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





\# Initialize database when program starts.

init_db()





\# ============================================================

\# 5. DOCUMENT TEXT EXTRACTION

\# ============================================================



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





\# ============================================================

\# 6. CLEAN TEXT

\# ============================================================



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





\# ============================================================

\# 7. TEXT CHUNKING

\# ============================================================



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





\# ============================================================

\# 8. LOCAL EMBEDDINGS

\# ============================================================



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





\# ============================================================

\# 9. VECTOR DATABASE

\# ============================================================



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





\# ============================================================

\# 10. COSINE SIMILARITY

\# ============================================================



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

        \*

        np.linalg.norm(b)

    )



    if denominator == 0:



        return 0.0



    return float(

        np.dot(a, b)

        /

        denominator

    )





\# ============================================================

\# 11. RAG RETRIEVAL

\# ============================================================



def retrieve_context(

    query,

    topic=None,

    top_k=5

):



    vectors = load_vectors()



    if not vectors:



        return []



    query_embedding = get_embedding(

        query

    )



    scored_results = []



    for item in vectors:



        score = cosine_similarity(

            query_embedding,

            item["embedding"]

        )



        # Give small preference to documents whose

        # filename contains the topic.



        if topic:



            source = item.get(

                "source",

                ""

            ).lower()



            if topic.lower() in source:



                score += 0.10



        scored_results.append(

            (

                score,

                item

            )

        )



    scored_results.sort(

        key=lambda result: result[0],

        reverse=True

    )



    results = []



    for score, item in scored_results[:top_k]:



        results.append({



            "source":

                item.get(

                    "source",

                    "Unknown"

                ),



            "text":

                item.get(

                    "text",

                    ""

                ),



            "score":

                round(score, 4)

        })



    return results





\# ============================================================

\# 12. LEARNER PROFILE

\# ============================================================



def get_mastery(topic):



    conn = sqlite3.connect(

        DB_PATH

    )



    cursor = conn.cursor()



    cursor.execute(

        """

        SELECT mastery

        FROM learner_profile

        WHERE topic = ?

        """,

        (topic,)

    )



    row = cursor.fetchone()



    conn.close()



    if row is None:



        return 0.0



    return float(

        row[0]

    )





\# ============================================================

\# 13. DIFFICULTY SELECTION

\# ============================================================



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





\# ============================================================

\# 14. UPDATE MASTERY

\# ============================================================



def update_mastery(

    topic,

    is_correct

):



    conn = sqlite3.connect(

        DB_PATH

    )



    cursor = conn.cursor()



    cursor.execute(

        """

        SELECT

            mastery,

            attempts,

            correct,

            incorrect

        FROM learner_profile

        WHERE topic = ?

        """,

        (topic,)

    )



    row = cursor.fetchone()



    if row is None:



        mastery = (

            0.15

            if is_correct

            else 0.0

        )



        attempts = 1



        correct = (

            1

            if is_correct

            else 0

        )



        incorrect = (

            0

            if is_correct

            else 1

        )



        cursor.execute(

            """

            INSERT INTO learner_profile

            (

                topic,

                mastery,

                attempts,

                correct,

                incorrect

            )

            VALUES (?, ?, ?, ?, ?)

            """,

            (

                topic,

                mastery,

                attempts,

                correct,

                incorrect

            )

        )



    else:



        mastery = float(

            row[0]

        )



        attempts = int(

            row[1]

        )



        correct = int(

            row[2]

        )



        incorrect = int(

            row[3]

        )



        attempts += 1



        if is_correct:



            correct += 1



            mastery += 0.15



        else:



            incorrect += 1



            mastery -= 0.10



        # Keep mastery between 0 and 1.



        mastery = max(

            0.0,

            min(

                1.0,

                mastery

            )

        )



        cursor.execute(

            """

            UPDATE learner_profile



            SET

                mastery = ?,

                attempts = ?,

                correct = ?,

                incorrect = ?



            WHERE topic = ?

            """,

            (

                mastery,

                attempts,

                correct,

                incorrect,

                topic

            )

        )



    conn.commit()



    conn.close()



    return mastery





\# ============================================================

\# 15. SAVE QUIZ ATTEMPT

\# ============================================================



def save_quiz_attempt(

    topic,

    question,

    user_answer,

    correct_answer,

    is_correct,

    difficulty

):



    conn = sqlite3.connect(

        DB_PATH

    )



    cursor = conn.cursor()



    cursor.execute(

        """

        INSERT INTO quiz_attempts

        (

            topic,

            question,

            user_answer,

            correct_answer,

            is_correct,

            difficulty

        )



        VALUES (?, ?, ?, ?, ?, ?)

        """,

        (

            topic,

            question,

            user_answer,

            correct_answer,

            1 if is_correct else 0,

            difficulty

        )

    )



    conn.commit()



    conn.close()





\# ============================================================

\# 16. OLLAMA STATUS

\# ============================================================



def ollama_available():



    try:



        response = requests.get(

            "http\://localhost:11434/api/tags",

            timeout=2

        )



        return (

            response.status_code

            == 200

        )



    except requests.RequestException:



        return False





\# ============================================================

\# 17. LOCAL LLM CALL

\# ============================================================



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





\# ============================================================

\# 18. FALLBACK RAG ANSWER

\# ============================================================



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





\# ============================================================

\# 19. LOCAL RAG TUTOR

\# ============================================================



def generate_tutor_answer(

    message,

    topic,

    mastery,

    difficulty,

    context_items

):



    if not context_items:



        return (

            "I do not yet have enough information "

            "from your uploaded study materials. "

            "Upload notes or a PDF related to this topic."

        )



    context_text = "\n\n".join(



        [

            (

                f"SOURCE: {item['source']}\n"

                f"{item['text']}"

            )



            for item

            in context_items

        ]

    )



    # If Ollama is unavailable, RAG retrieval

    # still works.



    if not ollama_available():



        return fallback_rag_answer(

            message,

            context_items

        )



    prompt = f"""

You are AdaptLearn AI,

an adaptive tutor for engineering students.



TOPIC:

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



1\. Use the uploaded material as the primary source.

2\. Do not invent facts not supported by the material.

3\. Explain the concept according to the student's

   current mastery level.

4\. If mastery is low, use simple explanations.

5\. If mastery is medium, include conceptual reasoning.

6\. If mastery is high, include deeper technical reasoning.

7\. Prefer Socratic guidance instead of simply giving

   an answer when appropriate.

8\. Mention which uploaded source supports the explanation.

9\. Keep the response concise and suitable for an

   engineering student.

"""



    try:



        return ask_ollama(

            prompt

        )



    except Exception:



        return fallback_rag_answer(

            message,

            context_items

        )





\# ============================================================

\# 20. JSON EXTRACTION FROM LOCAL LLM

\# ============================================================



def extract_json(text):



    if not text:



        return None



    text = text.strip()



    # Remove markdown fences.



    text = re.sub(

        r"^\`\`\`json\s\*",

        "",

        text,

        flags=re.IGNORECASE

    )



    text = re.sub(

        r"^\`\`\`\s\*",

        "",

        text

    )



    text = re.sub(

        r"\s\*\`\`\`$",

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

        r"\\{.\*\\}",

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





\# ============================================================

\# 21. FALLBACK QUIZ GENERATOR

\# ============================================================



def generate_fallback_quiz(

    topic,

    difficulty,

    context_items

):



    if not context_items:



        return {



            "question":

                (

                    f"What topic should you review "

                    f"before continuing with {topic}?"

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



    question = (

        f"Which statement is supported by the "

        f"uploaded material about {topic}?"

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





\# ============================================================

\# 22. ADAPTIVE QUIZ GENERATION

\# ============================================================



def generate_quiz_with_local_ai(

    topic,

    mastery,

    difficulty,

    context_items

):



    if not context_items:



        return generate_fallback_quiz(

            topic,

            difficulty,

            context_items

        )



    if not ollama_available():



        return generate_fallback_quiz(

            topic,

            difficulty,

            context_items

        )



    context_text = "\n\n".join(



        [

            (

                f"SOURCE: {item['source']}\n"

                f"{item['text']}"

            )



            for item

            in context_items

        ]

    )



    prompt = f"""

You are generating an adaptive engineering quiz.



TOPIC:

{topic}



LEARNER MASTERY:

{mastery:.2f}



DIFFICULTY:

{difficulty}



SOURCE MATERIAL:

{context_text}



Create ONE multiple choice question.



Rules:



\- Use only information supported by the provided

  study material.

\- Match the question difficulty to:

  {difficulty}

\- Provide exactly four options.

\- Exactly one option must be correct.

\- Provide a short explanation.

\- Do not include markdown.

\- Return ONLY JSON.



Required format:



{{

    "question": "question text",

    "options": [

        "option 1",

        "option 2",

        "option 3",

        "option 4"

    ],

    "correct_answer": "exact option text",

    "explanation": "short explanation"

}}

"""



    try:



        response = ask_ollama(

            prompt,

            temperature=0.2

        )



        quiz = extract_json(

            response

        )



        if not quiz:



            raise ValueError(

                "Invalid JSON from local model."

            )



        required_fields = [

            "question",

            "options",

            "correct_answer",

            "explanation"

        ]



        for field in required_fields:



            if field not in quiz:



                raise ValueError(

                    f"Missing field: {field}"

                )



        if len(

            quiz["options"]

        ) != 4:



            raise ValueError(

                "Quiz must contain four options."

            )



        return quiz



    except Exception:



        return generate_fallback_quiz(

            topic,

            difficulty,

            context_items

        )





\# ============================================================

\# 23. HOME PAGE

\# ============================================================



@app.route("/")

def home():



    return render_template(

        "index.html"

    )





\# ============================================================

\# 24. HEALTH CHECK

\# ============================================================



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

            OLLAMA_MODEL

    })





\# ============================================================

\# 25. UPLOAD + LOCAL INDEXING

\# ============================================================



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





\# ============================================================

\# 26. MATERIAL LIST

\# ============================================================



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





\# ============================================================

\# 27. RAG SEARCH TEST ENDPOINT

\# ============================================================



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





\# ============================================================

\# 28. ADAPTIVE TUTOR CHAT

\# ============================================================



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





\# ============================================================

\# 29. ADAPTIVE QUIZ GENERATION

\# ============================================================



@app.route(

    "/api/quiz/generate",

    methods=["POST"]

)

def generate_quiz():



    data = request.get_json(

        silent=True

    ) or {}



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



    context_items = retrieve_context(

        query=topic,

        topic=topic,

        top_k=5

    )



    quiz = generate_quiz_with_local_ai(

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



        "success":

            True,



        "topic":

            topic,



        "mastery":

            mastery,



        "difficulty":

            difficulty,



        "quiz":

            quiz,



        "sources":

            sources,



        "local_llm":

            ollama_available()

    })





\# ============================================================

\# 30. QUIZ ATTEMPT

\# ============================================================



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

                new_mastery \* 100,

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





\# ============================================================

\# 31. DASHBOARD

\# ============================================================



@app.route("/api/dashboard")

def dashboard():



    conn = sqlite3.connect(

        DB_PATH

    )



    cursor = conn.cursor()



    cursor.execute(

        """

        SELECT

            topic,

            mastery,

            attempts,

            correct,

            incorrect



        FROM learner_profile



        ORDER BY topic

        """

    )



    rows = cursor.fetchall()



    topics = []



    total_attempts = 0

    total_correct = 0

    total_incorrect = 0



    for row in rows:



        topic = row[0]



        mastery = float(

            row[1]

        )



        attempts = int(

            row[2]

        )



        correct = int(

            row[3]

        )



        incorrect = int(

            row[4]

        )



        total_attempts += attempts

        total_correct += correct

        total_incorrect += incorrect



        topics.append({



            "topic":

                topic,



            "mastery":

                round(

                    mastery \* 100,

                    1

                ),



            "attempts":

                attempts,



            "correct":

                correct,



            "incorrect":

                incorrect,



            "difficulty":

                get_difficulty(

                    topic

                )

        })



    cursor.execute(

        """

        SELECT COUNT(\*)

        FROM quiz_attempts

        """

    )



    stored_quiz_attempts = (

        cursor.fetchone()[0]

    )



    conn.close()



    accuracy = 0



    if total_attempts > 0:



        accuracy = (

            total_correct

            /

            total_attempts

            \*

            100

        )



    return jsonify({



        "topics":

            topics,



        "summary": {



            "total_attempts":

                total_attempts,



            "correct":

                total_correct,



            "incorrect":

                total_incorrect,



            "accuracy":

                round(

                    accuracy,

                    1

                ),



            "stored_quiz_attempts":

                stored_quiz_attempts

        }

    })





\# ============================================================

\# 32. QUIZ HISTORY

\# ============================================================



@app.route("/api/quiz/history")

def quiz_history():



    conn = sqlite3.connect(

        DB_PATH

    )



    cursor = conn.cursor()



    cursor.execute(

        """

        SELECT

            topic,

            question,

            user_answer,

            correct_answer,

            is_correct,

            difficulty,

            created_at



        FROM quiz_attempts



        ORDER BY id DESC



        LIMIT 50

        """

    )



    rows = cursor.fetchall()



    conn.close()



    attempts = []



    for row in rows:



        attempts.append({



            "topic":

                row[0],



            "question":

                row[1],



            "user_answer":

                row[2],



            "correct_answer":

                row[3],



            "is_correct":

                bool(

                    row[4]

                ),



            "difficulty":

                row[5],



            "created_at":

                row[6]

        })



    return jsonify({



        "attempts":

            attempts

    })





\# ============================================================

\# 33. START FLASK

\# ============================================================



if \_\_name\_\_ == "\_\_main\_\_":



    print()

    print("=" \* 60)



    print(

        "AdaptLearn AI - Local RAG Backend"

    )



    print("=" \* 60)



    print(

        "Embeddings: LOCAL"

    )



    print(

        f"Model: {EMBEDDING_MODEL_NAME}"

    )



    print(

        "OpenAI API: NOT REQUIRED"

    )



    if ollama_available():



        print(

            f"Local LLM: {OLLAMA_MODEL}"

        )



    else:



        print(

            "Local LLM: Ollama not detected"

        )



        print(

            "RAG retrieval and adaptive "

            "learning will still work."

        )



    print("=" \* 60)

    print()



    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )