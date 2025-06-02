"""
ADK Multi-Agent Educator Package

A sophisticated multi-agent system for educational support using Google's ADK,
featuring specialized agents for Math, Science, and Music education.
"""

__version__ = "0.1.0"
__author__ = "ADK Educator Team"

from .core.agent_base import BaseAgent
from .core.coordinator import AgentCoordinator
from .agents.math_agent import MathAgent
from .agents.science_agent import ScienceAgent
from .agents.music_agent import MusicAgent
from .core.session_manager import SessionManager

__all__ = [
    "BaseAgent",
    "AgentCoordinator", 
    "MathAgent",
    "ScienceAgent",
    "MusicAgent",
    "SessionManager",
]
