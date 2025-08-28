### Minimal Chatbot (FastAPI + Transformers)

This is a lightweight web chatbot using a Hugging Face Transformers text-generation pipeline on a FastAPI backend, with a minimal HTML/CSS frontend.

#### Files
- `app.py`: FastAPI backend exposing `/chat` and serving the static page
- `index.html`: Single-page chat UI
- `style.css`: Minimal, responsive styling
- `requirements.txt`: Python dependencies

#### Quick Start (Recommended: Virtual Environment)

1) Create a virtual environment (if `venv` is missing, install OS package `python3-venv`).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

2) Install dependencies. For CPU-only installs on Linux, you may prefer the CPU wheel index for PyTorch:

```bash
pip install --upgrade -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

3) Run the server:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

4) Open the app:
- Navigate to `http://127.0.0.1:8000/`

#### Model Configuration
- Default model: `distilgpt2` (lightweight). Override via env var:

```bash
export MODEL_NAME=distilgpt2
```

You can also set `HF_DEVICE_MAP_AUTO=0` to disable `device_map="auto"` if desired.

#### API
- `POST /chat`

Request body:

```json
{
  "message": "Hello!",
  "max_new_tokens": 64,
  "temperature": 0.7,
  "top_p": 0.9
}
```

Response body:

```json
{ "reply": "Hi! How can I help you today?" }
```

#### Notes
- First run may download the model; keep the internet connection active.
- Frontend uses `fetch('/chat', ...)` to call the backend.
- The layout is responsive and mobile-friendly.

# cursorCodeRepo