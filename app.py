from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
from datetime import datetime
import logging
import threading
import requests
from chatbot import RAGChatbot
# from dotenv import load_dotenv
#
# load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Chatbot API - Multi-User",
    description="HR Assistant Chatbot with Per-User Session Management",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global base chatbot instance
base_chatbot = None

# Per-user session storage
user_sessions = {}
session_lock = threading.Lock()

# Configuration
MAX_SESSIONS = 100
SESSION_TIMEOUT = 3600  # 1 hour


class UserSession:
    """Isolated session for each user"""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.chat_history = []
        self.conversation_context = {
            'current_employee': None,
            'last_mentioned_entities': []
        }
        self.last_activity = datetime.now()

    def update_activity(self):
        self.last_activity = datetime.now()


def cleanup_old_sessions():
    """Remove inactive sessions"""
    with session_lock:
        current_time = datetime.now()
        to_remove = []

        for user_id, session in user_sessions.items():
            time_diff = (current_time - session.last_activity).total_seconds()
            if time_diff > SESSION_TIMEOUT:
                to_remove.append(user_id)

        for user_id in to_remove:
            del user_sessions[user_id]
            logger.info(f"Cleaned up session for user: {user_id}")


def get_or_create_session(user_id: str) -> UserSession:
    """Get existing session or create new one"""
    with session_lock:
        if len(user_sessions) > MAX_SESSIONS:
            cleanup_old_sessions()

        if user_id not in user_sessions:
            user_sessions[user_id] = UserSession(user_id)
            logger.info(f"Created new session for user: {user_id}")

        session = user_sessions[user_id]
        session.update_activity()
        return session


# Pydantic models
class ChatRequest(BaseModel):
    question: str
    user_id: str


class ChatResponse(BaseModel):
    question: str
    answer: str
    timestamp: str
    user_id: str
    session_info: Dict


@app.on_event("startup")
async def startup_event():
    global base_chatbot

    logger.info("=== Starting RAG Chatbot Initialization ===")

    try:
        PDF_PATH = os.getenv("PDF_PATH", "./data/policies.pdf")
        HF_TOKEN = os.getenv("HF_TOKEN")

        # if not HF_TOKEN:
        #     raise ValueError("HF_TOKEN environment variable not set")

        logger.info(f"PDF Path: {PDF_PATH}")
        logger.info(f"File exists: {os.path.exists(PDF_PATH)}")

        if not os.path.exists(PDF_PATH):
            raise ValueError(f"PDF file not found at {PDF_PATH}")

        base_chatbot = RAGChatbot(PDF_PATH, HF_TOKEN)
        logger.info("=== Base chatbot initialized successfully! ===")

    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {e}")
        raise


@app.get("/")
async def root():
    return {
        "service": "RAG Chatbot API",
        "version": "2.0.0",
        "status": "healthy",
        "active_sessions": len(user_sessions),
        "chatbot_loaded": base_chatbot is not None,
        "endpoints": {
            "docs": "/docs",
            "chat": "POST /api/chat",
            "history": "GET /api/history/{user_id}",
            "reset": "POST /api/reset?user_id=xxx",
            "sessions": "GET /api/sessions"
        }
    }


@app.get("/api/health")
async def health_check():
    if base_chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "chatbot_ready": True,
        "active_sessions": len(user_sessions)
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a question to the chatbot with user session isolation"""
    if base_chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    if not request.user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    
    try:
        logger.info(f"User {request.user_id}: {request.question[:50]}...")
        
        # Get user session
        session = get_or_create_session(request.user_id)
        
        # Resolve pronouns
        resolved_question = base_chatbot._resolve_pronouns_for_session(
            request.question, 
            session.conversation_context
        )
        
        # Retrieve relevant chunks
        retrieved_data = base_chatbot._retrieve(resolved_question, k=20)
        
        # Search user's chat history
        relevant_past_chats = base_chatbot._search_session_history(
            resolved_question,
            session.chat_history,
            k=5
        )
        
        # Build prompt
        prompt = base_chatbot._build_prompt_for_session(
            resolved_question,
            retrieved_data,
            relevant_past_chats,
            session.chat_history,
            session.conversation_context
        )
        
        # ✅ NEW: Call Hugging Face Router API
        payload = {
            "model": base_chatbot.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 512,
            "temperature": 0.3
        }
        
        response = requests.post(
            base_chatbot.api_url,
            headers=base_chatbot.headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        
        # Extract answer
        answer = result["choices"][0]["message"]["content"]
        
        # Update user's conversation context
        base_chatbot._update_conversation_context_for_session(
            request.question,
            answer,
            session.conversation_context
        )
        
        # Store in user's history
        chat_entry = {
            'timestamp': datetime.now().isoformat(),
            'question': request.question,
            'answer': answer,
            'used_past_context': len(relevant_past_chats) > 0
        }
        session.chat_history.append(chat_entry)
        
        response_data = ChatResponse(
            question=request.question,
            answer=answer,
            timestamp=datetime.now().isoformat(),
            user_id=request.user_id,
            session_info={
                'total_messages': len(session.chat_history),
                'current_context': session.conversation_context.get('current_employee')
            }
        )
        
        logger.info(f"User {request.user_id}: Question processed successfully")
        return response_data
        
    except Exception as e:
        logger.error(f"Error for user {request.user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/api/reset")
async def reset_chat(user_id: str):
    """Reset chat history for specific user"""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    with session_lock:
        if user_id in user_sessions:
            del user_sessions[user_id]
            logger.info(f"Reset session for user: {user_id}")
            return {"message": f"Chat history reset for user {user_id}", "status": "success"}
        else:
            return {"message": f"No session found for user {user_id}", "status": "success"}


@app.get("/api/history/{user_id}")
async def get_history(user_id: str):
    """Get chat history for specific user"""
    session = get_or_create_session(user_id)

    return {
        "user_id": user_id,
        "total_conversations": len(session.chat_history),
        "current_context": session.conversation_context.get('current_employee'),
        "history": session.chat_history
    }


@app.get("/api/sessions")
async def get_active_sessions():
    """Get list of active sessions"""
    with session_lock:
        return {
            "total_sessions": len(user_sessions),
            "max_sessions": MAX_SESSIONS,
            "session_timeout_seconds": SESSION_TIMEOUT,
            "sessions": [
                {
                    "user_id": user_id,
                    "messages": len(session.chat_history),
                    "last_activity": session.last_activity.isoformat(),
                    "current_context": session.conversation_context.get('current_employee')
                }
                for user_id, session in user_sessions.items()
            ]
        }


@app.post("/api/cleanup")
async def manual_cleanup():
    """Manually trigger session cleanup"""
    cleanup_old_sessions()
    return {
        "message": "Cleanup completed",
        "active_sessions": len(user_sessions)
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)