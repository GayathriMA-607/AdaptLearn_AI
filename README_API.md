# AdaptLearn AI — API-ready deployment build

This build uses the OpenAI Responses API for tutor answers and adaptive quiz generation when `OPENAI_API_KEY` is configured. Uploaded documents are still indexed with the local Sentence Transformers embedding model, so no OpenAI embedding key is required.

## 1. Local setup

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your OpenAI API key:

```text
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-6-luna
AI_PROVIDER=auto
ADAPTLEARN_SECRET_KEY=use-a-long-random-secret
```

Do not commit `.env` or expose the API key in HTML/JavaScript.

Start:

```bash
python app.py
```

Open `http://127.0.0.1:5000`.

## 2. Provider behavior

- `AI_PROVIDER=auto`: OpenAI if the key is configured; otherwise Ollama; otherwise retrieval fallback.
- `AI_PROVIDER=openai`: use OpenAI. If the key is missing or the API request fails, the app falls back safely.
- `AI_PROVIDER=ollama`: use the local Ollama model.

## 3. Deployment

Set environment variables in your hosting provider rather than putting the key in source code:

```text
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-6-luna
AI_PROVIDER=openai
ADAPTLEARN_SECRET_KEY=...
FLASK_DEBUG=false
HOST=0.0.0.0
PORT=<provider port>
```

The included `Procfile` runs:

```bash
gunicorn app:app
```

For production, use a deployment service with persistent storage if you want the SQLite database and uploaded files to survive restarts. A managed database/object-storage setup is preferable for a larger multi-user deployment.

## 4. What uses the OpenAI key

- Tutor responses grounded in retrieved study material.
- Adaptive multiple-choice quiz generation.
- Explanation of Wikipedia fallback results.

The browser never receives `OPENAI_API_KEY`; requests go to Flask, and Flask calls OpenAI server-side.

## 5. Voice tutor

The current voice tutor uses the browser's built-in speech synthesis, so it does not require another API key. It reads the tutor's returned answer aloud.
