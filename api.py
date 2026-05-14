# api.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import tempfile
import os
import secrets
import time
import logging
import re
from pathlib import Path
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from chatbot_core import build_chat_graph, ask

load_dotenv()

logger = logging.getLogger(__name__)

# ── App Setup ─────────────────────────────────────────────────
app = FastAPI(
    title="PDF Chatbot API",
    description="Chat with your PDF and Word documents using AI",
    version="1.0.0"
)

# ── CORS ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session Storage ───────────────────────────────────────────
sessions: Dict[str, dict] = {}

# ── Security ──────────────────────────────────────────────────
INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"forget (you are|your|all)",
    r"you are now",
    r"reveal (your|the) (system |)prompt",
    r"jailbreak",
    r"bypass (your|all) (restrictions|rules)",
]

def detect_injection(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in INJECTION_PATTERNS)

def validate_file(filename: str, content: bytes) -> tuple[bool, str]:
    suffix = Path(filename).suffix.lower()
    if suffix not in [".pdf", ".docx"]:
        return False, f"File type '{suffix}' not allowed. Use PDF or DOCX."
    if len(content) > 10 * 1024 * 1024:
        return False, "File too large (max 10MB)"
    if len(content) == 0:
        return False, "File is empty"
    if suffix == ".pdf" and not content.startswith(b"%PDF"):
        return False, "Invalid PDF file"
    if suffix == ".docx" and not content.startswith(b"PK"):
        return False, "Invalid DOCX file"
    return True, ""

# ── Models ────────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question:   str
    session_id: str

class QuestionResponse(BaseModel):
    answer:     str
    session_id: str
    error:      Optional[str] = None

class SessionInfo(BaseModel):
    session_id: str
    num_pages:  int
    filenames:  List[str]
    message:    str

class HealthResponse(BaseModel):
    status:          str
    active_sessions: int

# ── Routes ────────────────────────────────────────────────────
@app.get("/", response_model=HealthResponse)
async def home():
    return HealthResponse(
        status="running",
        active_sessions=len(sessions)
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        active_sessions=len(sessions)
    )

@app.post("/upload", response_model=SessionInfo)
async def upload_documents(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Max 5 files per upload")

    pages     = []
    filenames = []
    tmp_paths = []

    try:
        for file in files:
            content = await file.read()

            is_valid, error = validate_file(file.filename, content)
            if not is_valid:
                raise HTTPException(status_code=400, detail=error)

            suffix = Path(file.filename).suffix.lower()
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as tmp:
                tmp.write(content)
                tmp_paths.append(tmp.name)

            loader = (PyPDFLoader(tmp_paths[-1])
                      if suffix == ".pdf"
                      else Docx2txtLoader(tmp_paths[-1]))

            pages.extend(loader.load())
            filenames.append(file.filename)

        if not pages:
            raise HTTPException(
                status_code=400,
                detail="No content found in uploaded files"
            )

        splitter   = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        )
        chunks     = splitter.split_documents(pages)
        chat_graph = build_chat_graph(pages, chunks)
        session_id = secrets.token_urlsafe(32)

        sessions[session_id] = {
            "chat_graph": chat_graph,
            "history":    ChatMessageHistory(),
            "num_pages":  len(pages),
            "filenames":  filenames,
            "created_at": time.time()
        }

        return SessionInfo(
            session_id=session_id,
            num_pages=len(pages),
            filenames=filenames,
            message=f"Successfully processed {len(files)} file(s)"
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process files. Please try again."
        )

    finally:
        for path in tmp_paths:
            if os.path.exists(path):
                os.unlink(path)


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    if request.session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Upload documents first via /upload"
        )

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if len(request.question) > 1000:
        raise HTTPException(status_code=400, detail="Question too long (max 1000 chars)")

    if detect_injection(request.question):
        raise HTTPException(status_code=400, detail="Invalid request")

    session = sessions[request.session_id]

    try:
        answer = ask(
            session["chat_graph"],
            request.question.strip(),
            session["history"]
        )
        return QuestionResponse(
            answer=answer,
            session_id=request.session_id
        )

    except Exception as e:
        logger.error(f"Ask error: {e}")
        return QuestionResponse(
            answer="",
            session_id=request.session_id,
            error="Something went wrong. Please try again."
        )


@app.get("/session/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[session_id]
    return SessionInfo(
        session_id=session_id,
        num_pages=session["num_pages"],
        filenames=session["filenames"],
        message="Session active"
    )


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del sessions[session_id]
    return {"message": f"Session {session_id} deleted"}


@app.delete("/session/{session_id}/history")
async def clear_history(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    sessions[session_id]["history"] = ChatMessageHistory()
    return {"message": "Chat history cleared"}