import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Lazily created at startup
_text_generation_pipeline = None


def load_text_generation_pipeline(model_name: Optional[str] = None):
    """Load and return a text-generation pipeline.

    Uses a lightweight default model. Allows override via MODEL_NAME env var.
    """
    from transformers import pipeline  # Imported here to avoid slow import at module load

    selected_model_name = model_name or os.getenv("MODEL_NAME", "distilgpt2")

    # Using GPT-2 family requires specifying pad_token_id for generation to avoid warnings
    generator = pipeline(
        task="text-generation",
        model=selected_model_name,
        device_map="auto" if os.getenv("HF_DEVICE_MAP_AUTO", "1") == "1" else None,
    )
    return generator


class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's input text")
    max_new_tokens: int = Field(64, ge=1, le=256, description="Max tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling p-value")


class ChatResponse(BaseModel):
    reply: str


app = FastAPI(title="Minimal Chatbot (Transformers + FastAPI)")

# CORS: Allow local dev from different origins if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup_load_pipeline() -> None:
    global _text_generation_pipeline
    _text_generation_pipeline = load_text_generation_pipeline()


@app.get("/")
def serve_index() -> FileResponse:
    """Serve the single-page frontend."""
    return FileResponse("index.html")


@app.get("/style.css")
def serve_styles() -> FileResponse:
    """Serve the CSS file for the frontend."""
    return FileResponse("style.css")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Generate a response from the language model.

    The prompt format is simple: a short role-styled prefix to steer the model.
    """
    if _text_generation_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not ready yet. Try again shortly.")

    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message must not be empty.")

    # Build a very lightweight prompt to nudge assistant-like responses
    prompt = f"User: {user_message}\nAssistant:"

    try:
        outputs = _text_generation_pipeline(
            prompt,
            max_new_tokens=request.max_new_tokens,
            do_sample=True,
            temperature=request.temperature,
            top_p=request.top_p,
            # GPT-2 family uses 50256 as EOS/pad; avoids warnings
            pad_token_id=50256,
        )
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Generation error: {exc}")

    generated_text = outputs[0]["generated_text"]

    # Extract only the assistant's part after the last "Assistant:" marker
    reply_text = generated_text
    if "Assistant:" in generated_text:
        reply_text = generated_text.split("Assistant:")[-1]
    # Trim any trailing artifacts
    reply_text = reply_text.strip()

    # Basic safety: fall back if somehow empty
    if not reply_text:
        reply_text = "I'm here and ready to help! Could you please rephrase your question?"

    return ChatResponse(reply=reply_text)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )

