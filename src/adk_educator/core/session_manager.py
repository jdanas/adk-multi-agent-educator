"""
Session Manager for the ADK Multi-Agent Educator system.

This module manages student learning sessions, tracks progress,
and coordinates interactions between students and multiple agents.
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from loguru import logger

from ..config import (
    StudentRequest, AgentResponse, Message, MessageType, 
    SubjectType, SystemConfig
)
from .agent_base import BaseAgent


@dataclass
class LearningSession:
    """Represents an active learning session for a student."""
    session_id: str
    student_id: str
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    active_subjects: Set[SubjectType] = field(default_factory=set)
    total_interactions: int = 0
    preferred_agents: List[str] = field(default_factory=list)
    learning_style: Optional[str] = None
    difficulty_preference: Optional[str] = None


class SessionManager:
    """
    Manages student learning sessions and coordinates multi-agent interactions.
    
    The SessionManager is responsible for:
    - Creating and managing student sessions
    - Routing requests to appropriate agents
    - Facilitating collaboration between agents
    - Tracking learning progress and preferences
    """
    
    def __init__(self):
        """Initialize the session manager."""
        self.sessions: Dict[str, LearningSession] = {}
        self.registered_agents: Dict[SubjectType, BaseAgent] = {}
        self.cleanup_task: Optional[asyncio.Task] = None
        
        logger.info("SessionManager initialized")
    
    def register_agent(self, agent: BaseAgent) -> None:
        """
        Register an agent with the session manager.
        
        Args:
            agent: The agent to register
        """
        self.registered_agents[agent.subject] = agent
        logger.info(f"Registered {agent.name} for {agent.subject.value}")
        
        # Set up collaboration relationships with other agents
        for other_agent in self.registered_agents.values():
            if other_agent != agent:
                agent.add_collaboration_partner(other_agent)
                other_agent.add_collaboration_partner(agent)
    
    def create_session(self, student_id: str) -> str:
        """
        Create a new learning session for a student.
        
        Args:
            student_id: Unique identifier for the student
            
        Returns:
            Session ID for the new session
        """
        session_id = str(uuid.uuid4())
        session = LearningSession(
            session_id=session_id,
            student_id=student_id
        )
        
        self.sessions[session_id] = session
        logger.info(f"Created session {session_id} for student {student_id}")
        
        # Start cleanup task if not already running
        if self.cleanup_task is None or self.cleanup_task.done():
            self.cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[LearningSession]:
        """
        Get a learning session by ID.
        
        Args:
            session_id: The session identifier
            
        Returns:
            LearningSession if found, None otherwise
        """
        return self.sessions.get(session_id)
    
    async def process_student_request(self, request: StudentRequest) -> AgentResponse:
        """
        Process a student's learning request.
        
        This method routes the request to the appropriate agent(s) and
        handles collaboration when necessary.
        
        Args:
            request: The student's educational request
            
        Returns:
            Response from the agent(s)
        """
        try:
            # Update session activity
            session = self.sessions.get(request.session_id)
            if not session:
                logger.warning(f"Session {request.session_id} not found, creating new session")
                new_session_id = self.create_session(request.student_id)
                session = self.sessions[new_session_id]
                # Update the request session_id to match the created session
                request.session_id = new_session_id
            
            session.last_activity = time.time()
            session.active_subjects.add(request.subject)
            session.total_interactions += 1
            
            # Get the primary agent for the subject
            primary_agent = self.registered_agents.get(request.subject)
            if not primary_agent:
                return AgentResponse(
                    agent_name="SystemManager",
                    subject=request.subject,
                    response_text=f"Sorry, we don't currently have an expert available for {request.subject.value}. Please try a different subject.",
                    confidence_score=0.0
                )
            
            # Process the request with the primary agent
            response = await primary_agent.process_request(request)
            
            # Handle collaboration if needed
            if response.requires_collaboration and SystemConfig.ENABLE_COLLABORATION:
                response = await self._handle_collaboration(request, response, primary_agent)
            
            # Update session preferences based on response quality
            if response.confidence_score > 0.8:
                if primary_agent.name not in session.preferred_agents:
                    session.preferred_agents.append(primary_agent.name)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing student request: {str(e)}")
            return AgentResponse(
                agent_name="SystemManager",
                subject=request.subject,
                response_text="I'm experiencing technical difficulties. Please try again in a moment.",
                confidence_score=0.0
            )
    
    async def _handle_collaboration(
        self, 
        request: StudentRequest, 
        initial_response: AgentResponse, 
        primary_agent: BaseAgent
    ) -> AgentResponse:
        """
        Handle collaboration between multiple agents.
        
        Args:
            request: The original student request
            initial_response: Response from the primary agent
            primary_agent: The primary agent handling the request
            
        Returns:
            Enhanced response incorporating collaboration
        """
        try:
            collaboration_responses = []
            
            # Get responses from relevant collaboration partners
            for subject in initial_response.collaboration_subjects:
                if subject in self.registered_agents:
                    collaborating_agent = self.registered_agents[subject]
                    collab_response = await collaborating_agent.collaborate_with(
                        primary_agent, 
                        request.specific_question
                    )
                    collaboration_responses.append(
                        f"\n**{collaborating_agent.name} adds:** {collab_response}"
                    )
            
            # Combine responses
            if collaboration_responses:
                enhanced_response = initial_response.response_text + "\n\n" + "\n".join(collaboration_responses)
                
                return AgentResponse(
                    agent_name=f"{primary_agent.name} (with collaboration)",
                    subject=initial_response.subject,
                    response_text=enhanced_response,
                    confidence_score=min(1.0, initial_response.confidence_score + 0.2),
                    suggested_follow_ups=initial_response.suggested_follow_ups,
                    resources=initial_response.resources,
                    requires_collaboration=False
                )
            
            return initial_response
            
        except Exception as e:
            logger.error(f"Error in collaboration: {str(e)}")
            return initial_response
    
    def get_session_summary(self, session_id: str) -> Dict[str, any]:
        """
        Get a summary of a learning session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            Dictionary containing session summary information
        """
        session = self.sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}
        
        # Calculate session duration
        duration_minutes = (time.time() - session.created_at) / 60
        
        # Get agent activities
        agent_activities = {}
        for subject in session.active_subjects:
            if subject in self.registered_agents:
                agent = self.registered_agents[subject]
                agent_activities[agent.name] = {
                    "subject": subject.value,
                    "interactions": len([
                        msg for msg in agent.session_history.get(session_id, [])
                        if msg.type == MessageType.STUDENT_QUERY
                    ])
                }
        
        return {
            "session_id": session_id,
            "student_id": session.student_id,
            "duration_minutes": round(duration_minutes, 1),
            "total_interactions": session.total_interactions,
            "active_subjects": [subject.value for subject in session.active_subjects],
            "preferred_agents": session.preferred_agents,
            "agent_activities": agent_activities,
            "last_activity": session.last_activity
        }
    
    def get_available_agents(self) -> Dict[str, Dict[str, any]]:
        """
        Get information about all available agents.
        
        Returns:
            Dictionary mapping agent names to their capabilities
        """
        return {
            agent.name: agent.get_capabilities() 
            for agent in self.registered_agents.values()
        }
    
    async def _cleanup_expired_sessions(self) -> None:
        """Background task to clean up expired sessions."""
        while True:
            try:
                current_time = time.time()
                expired_sessions = []
                
                for session_id, session in self.sessions.items():
                    if current_time - session.last_activity > SystemConfig.SESSION_TIMEOUT:
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    del self.sessions[session_id]
                    logger.info(f"Cleaned up expired session {session_id}")
                
                # Sleep for 5 minutes before next cleanup
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in session cleanup: {str(e)}")
                await asyncio.sleep(60)  # Shorter sleep on error
    
    def shutdown(self) -> None:
        """Shutdown the session manager and cleanup resources."""
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
        
        logger.info("SessionManager shut down")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.shutdown()
