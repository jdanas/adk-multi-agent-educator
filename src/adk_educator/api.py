"""
Web API for the ADK Multi-Agent Educator system.

This module provides a RESTful API that allows web applications and other
services to interact with the multi-agent educational system.
"""

import asyncio
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from .core.session_manager import SessionManager
from .core.coordinator import AgentCoordinator
from .agents.math_agent import MathAgent
from .agents.science_agent import ScienceAgent
from .agents.music_agent import MusicAgent
from .config import StudentRequest, SubjectType, DifficultyLevel, SystemConfig


# Request/Response Models for the API
class QuestionRequest(BaseModel):
    """Request model for asking questions."""
    question: str = Field(..., description="The student's question")
    subject: Optional[SubjectType] = Field(None, description="Preferred subject area")
    topic: Optional[str] = Field(None, description="Specific topic")
    difficulty: Optional[DifficultyLevel] = Field(None, description="Difficulty level")
    context: Optional[str] = Field(None, description="Additional context")
    student_name: Optional[str] = Field(None, description="Student name for session tracking")


class AgentResponseModel(BaseModel):
    """Response model for agent answers."""
    agent_name: str
    subject: str
    response_text: str
    confidence_score: float
    suggested_follow_ups: List[str]
    resources: List[str]
    requires_collaboration: bool
    collaboration_subjects: List[str]
    timestamp: datetime = Field(default_factory=datetime.now)


class SessionInfo(BaseModel):
    """Session information model."""
    session_id: str
    student_id: str
    duration_minutes: float
    total_interactions: int
    active_subjects: List[str]
    preferred_agents: List[str]
    last_activity: float


class AgentInfo(BaseModel):
    """Agent capability information."""
    name: str
    subject: str
    specializations: List[str]
    difficulty_levels: List[str]
    can_collaborate: bool
    collaboration_partners: List[str]


# Global variables for the session manager and coordinator
session_manager: Optional[SessionManager] = None
coordinator: Optional[AgentCoordinator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global session_manager, coordinator
    
    # Startup
    try:
        session_manager = SessionManager()
        coordinator = AgentCoordinator(session_manager)
        
        # Initialize and register agents
        math_agent = MathAgent()
        science_agent = ScienceAgent()
        music_agent = MusicAgent()
        
        session_manager.register_agent(math_agent)
        session_manager.register_agent(science_agent)
        session_manager.register_agent(music_agent)
        
        print("✅ ADK Multi-Agent Educator API started successfully!")
        
    except Exception as e:
        print(f"❌ Error starting API: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    if session_manager:
        session_manager.shutdown()


# Create FastAPI app
app = FastAPI(
    title="ADK Multi-Agent Educator API",
    description="A REST API for the ADK Multi-Agent Educational system featuring specialized AI agents for Math, Science, and Music",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "ADK Multi-Agent Educator API",
        "version": "0.1.0",
        "description": "Multi-agent educational system with AI experts in Math, Science, and Music",
        "endpoints": {
            "ask": "/ask - Ask educational questions",
            "agents": "/agents - Get available agents information",
            "sessions": "/sessions - Manage learning sessions",
            "health": "/health - API health status"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if not session_manager or not coordinator:
        raise HTTPException(status_code=503, detail="Service not properly initialized")
    
    agents_count = len(session_manager.registered_agents)
    active_sessions = len(session_manager.sessions)
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agents_available": agents_count,
        "active_sessions": active_sessions,
        "configuration": {
            "collaboration_enabled": SystemConfig.ENABLE_COLLABORATION,
            "max_agents_per_session": SystemConfig.MAX_AGENTS_PER_SESSION,
            "session_timeout": SystemConfig.SESSION_TIMEOUT
        }
    }


@app.post("/ask", response_model=AgentResponseModel)
async def ask_question(request: QuestionRequest, background_tasks: BackgroundTasks):
    """
    Ask an educational question to the AI agents.
    
    This endpoint processes educational questions and routes them to the
    appropriate specialist agents, handling collaboration when beneficial.
    """
    if not session_manager or not coordinator:
        raise HTTPException(status_code=503, detail="Service not available")
    
    try:
        # Create or get session
        student_id = request.student_name or f"anonymous_{uuid.uuid4().hex[:8]}"
        session_id = session_manager.create_session(student_id)
        
        # Determine if this is a multi-subject question
        relevant_subjects = coordinator.identify_relevant_subjects(request.question)
        
        if len(relevant_subjects) > 1 and SystemConfig.ENABLE_COLLABORATION:
            # Multi-subject question - use coordinator
            student_request = StudentRequest(
                session_id=session_id,
                student_id=student_id,
                subject=request.subject or relevant_subjects[0],
                topic=request.topic or "interdisciplinary",
                difficulty=request.difficulty,
                specific_question=request.question,
                context=request.context
            )
            
            response = await coordinator.process_complex_query(student_request, relevant_subjects)
        else:
            # Single subject question
            subject = request.subject or (relevant_subjects[0] if relevant_subjects else SubjectType.MATH)
            
            student_request = StudentRequest(
                session_id=session_id,
                student_id=student_id,
                subject=subject,
                topic=request.topic or "general",
                difficulty=request.difficulty,
                specific_question=request.question,
                context=request.context
            )
            
            response = await session_manager.process_student_request(student_request)
        
        # Convert to API response model
        return AgentResponseModel(
            agent_name=response.agent_name,
            subject=response.subject.value,
            response_text=response.response_text,
            confidence_score=response.confidence_score,
            suggested_follow_ups=response.suggested_follow_ups,
            resources=response.resources,
            requires_collaboration=response.requires_collaboration,
            collaboration_subjects=[s.value for s in response.collaboration_subjects]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.get("/agents", response_model=Dict[str, AgentInfo])
async def get_agents():
    """Get information about all available educational agents."""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Service not available")
    
    try:
        agents_data = session_manager.get_available_agents()
        
        result = {}
        for agent_name, info in agents_data.items():
            result[agent_name] = AgentInfo(
                name=info["name"],
                subject=info["subject"],
                specializations=info["specializations"],
                difficulty_levels=info["difficulty_levels"],
                can_collaborate=info["can_collaborate"],
                collaboration_partners=info["collaboration_partners"]
            )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting agents info: {str(e)}")


@app.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """Get information about a specific learning session."""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Service not available")
    
    try:
        summary = session_manager.get_session_summary(session_id)
        
        if "error" in summary:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return SessionInfo(
            session_id=summary["session_id"],
            student_id=summary["student_id"],
            duration_minutes=summary["duration_minutes"],
            total_interactions=summary["total_interactions"],
            active_subjects=summary["active_subjects"],
            preferred_agents=summary["preferred_agents"],
            last_activity=summary["last_activity"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting session info: {str(e)}")


@app.post("/sessions")
async def create_session(student_name: str):
    """Create a new learning session."""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Service not available")
    
    try:
        session_id = session_manager.create_session(student_name)
        return {
            "session_id": session_id,
            "student_name": student_name,
            "created_at": datetime.now().isoformat(),
            "message": "Session created successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")


@app.get("/subjects")
async def get_subjects():
    """Get available subjects and their details."""
    return {
        "math": {
            "name": "Mathematics",
            "description": "Algebra, geometry, calculus, statistics, and more",
            "specializations": ["algebra", "geometry", "calculus", "statistics", "trigonometry"],
            "agent": "Professor Mathematics"
        },
        "science": {
            "name": "Science",
            "description": "Physics, chemistry, biology, earth science, and more",
            "specializations": ["physics", "chemistry", "biology", "earth_science", "astronomy"],
            "agent": "Dr. Science Explorer"
        },
        "music": {
            "name": "Music",
            "description": "Music theory, history, composition, and performance",
            "specializations": ["music_theory", "composition", "performance", "music_history"],
            "agent": "Maestro Harmony"
        }
    }


@app.get("/performance")
async def get_performance_metrics():
    """Get performance metrics for all agents."""
    if not coordinator:
        raise HTTPException(status_code=503, detail="Service not available")
    
    try:
        metrics = coordinator.get_performance_metrics()
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "summary": {
                "total_agents": len(metrics),
                "average_confidence": sum(m.get("average_confidence", 0) for m in metrics.values()) / len(metrics) if metrics else 0,
                "total_requests": sum(m.get("total_requests", 0) for m in metrics.values())
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting performance metrics: {str(e)}")


@app.post("/recommend")
async def recommend_learning_path(
    session_id: str,
    interests: List[str] = Field(..., description="Student interests for recommendations")
):
    """Get personalized learning recommendations based on student interests."""
    if not coordinator:
        raise HTTPException(status_code=503, detail="Service not available")
    
    try:
        recommendations = await coordinator.recommend_learning_path(session_id, interests)
        
        return {
            "session_id": session_id,
            "interests": interests,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")


def start_server(
    host: str = SystemConfig.HOST,
    port: int = SystemConfig.PORT,
    reload: bool = SystemConfig.RELOAD
):
    """Start the FastAPI server."""
    print(f"🚀 Starting ADK Multi-Agent Educator API on {host}:{port}")
    uvicorn.run("adk_educator.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    start_server()
