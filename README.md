# AdaptLearn AI

AdaptLearn AI is a personalised AI tutor for engineering students.

This first milestone contains:

- Flask backend
- Real LLM-powered tutor chat through the OpenAI Responses API
- Topic-aware tutoring
- Short conversation history
- PDF/DOCX/TXT upload endpoint
- Existing AdaptLearn-style frontend
- A clean base for the next RAG, adaptive-learning and dashboard milestones

## 1. Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Configure the API key

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

On Windows, you can simply duplicate the file manually.

Then put your API key in:

```text
OPENAI_API_KEY=...
```

Never put the API key in `index.html` or JavaScript.

## 4. Start AdaptLearn AI

```bash
python app.py
```

Open:

http://127.0.0.1:5000

## 5. What is intentionally NOT finished yet

The upload endpoint currently stores files but does not extract/chunk/embed them.
That is deliberate: the next milestone will add the RAG pipeline.

Likewise, the progress values are not yet connected to a database. The next
milestone will add learner events, mastery scores and adaptive difficulty.


## RAG milestone

AdaptLearn AI now has a local retrieval pipeline:

1. Upload PDF, DOCX or TXT.
2. Extract readable text.
3. Split it into overlapping chunks.
4. Generate embeddings with `text-embedding-3-small`.
5. Store embeddings and chunks in `data/vectors.json`.
6. Embed each tutor question.
7. Retrieve the most similar chunks.
8. Send those passages to the LLM as study context.
9. Show which uploaded file(s) were retrieved.

The local JSON vector store is intentionally simple for the hackathon MVP.
It can later be replaced by Chroma, FAISS, Qdrant, Pinecone, or another
vector database without changing the overall RAG architecture.

### Test RAG

Start the server:

```bash
python app.py
```

Upload a small TXT/PDF/DOCX containing a fact you know.

Then ask a question specifically about that fact.

The chat should show a `Retrieved from:` line under the answer when a relevant
chunk is found.

### Important

The first run needs internet access because embeddings and chat responses are
generated through the OpenAI API. Keep `.env` private and never commit it.
