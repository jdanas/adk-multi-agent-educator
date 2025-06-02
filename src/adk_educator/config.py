"""
Base configuration and types for the ADK Multi-Agent Educator system.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class SubjectType(str, Enum):
    """Supported educational subjects."""
    MATH = "math"
    SCIENCE = "science"
    MUSIC = "music"


class DifficultyLevel(str, Enum):
    """Difficulty levels for educational content."""
    ELEMENTARY = "elementary"
    MIDDLE = "middle"
    HIGH = "high"
    COLLEGE = "college"


class MessageType(str, Enum):
    """Types of messages in the multi-agent system."""
    STUDENT_QUERY = "student_query"
    AGENT_RESPONSE = "agent_response"
    COLLABORATION_REQUEST = "collaboration_request"
    SYSTEM_MESSAGE = "system_message"


class AgentConfig(BaseModel):
    """Configuration for individual agents."""
    name: str
    subject: SubjectType
    model: str = Field(default_factory=lambda: os.getenv("DEFAULT_MODEL", "gemini-1.5-pro"))
    temperature: float = Field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.7")))
    max_tokens: int = Field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "2048")))
    specializations: List[str] = Field(default_factory=list)
    difficulty_levels: List[DifficultyLevel] = Field(default_factory=list)


class Message(BaseModel):
    """Represents a message in the multi-agent system."""
    id: str
    type: MessageType
    sender: str
    recipient: Optional[str] = None
    subject: Optional[SubjectType] = None
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=lambda: __import__('time').time())


class StudentRequest(BaseModel):
    """Represents a student's educational request."""
    session_id: str
    student_id: str
    subject: Optional[SubjectType] = None  # None for interdisciplinary questions
    topic: str
    difficulty: Optional[DifficultyLevel] = None
    specific_question: str
    context: Optional[str] = None
    preferred_learning_style: Optional[str] = None


class AgentResponse(BaseModel):
    """Represents an agent's response to a student request."""
    agent_name: str
    subject: Optional[SubjectType] = None  # None for multi-subject responses
    response_text: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    suggested_follow_ups: List[str] = Field(default_factory=list)
    resources: List[str] = Field(default_factory=list)
    requires_collaboration: bool = False
    collaboration_subjects: List[SubjectType] = Field(default_factory=list)


class SystemConfig:
    """System-wide configuration."""
    
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-1.5-pro")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
    
    # Multi-agent settings
    ENABLE_COLLABORATION = os.getenv("ENABLE_COLLABORATION", "true").lower() == "true"
    MAX_AGENTS_PER_SESSION = int(os.getenv("MAX_AGENTS_PER_SESSION", "3"))
    SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", "1800"))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.getenv("LOG_FORMAT", "json")
    
    # Server
    HOST = os.getenv("HOST", "localhost")
    PORT = int(os.getenv("PORT", "8000"))
    RELOAD = os.getenv("RELOAD", "true").lower() == "true"


# Subject-specific configurations
MATH_SPECIALIZATIONS = [
    "algebra", "geometry", "calculus", "statistics", "trigonometry",
    "linear_algebra", "differential_equations", "number_theory"
]

SCIENCE_SPECIALIZATIONS = [
    "physics", "chemistry", "biology", "earth_science", "astronomy",
    "environmental_science", "anatomy", "genetics", "quantum_mechanics"
]

MUSIC_SPECIALIZATIONS = [
    "music_theory", "composition", "performance", "music_history",
    "ear_training", "harmony", "rhythm", "instruments", "genres"
]
