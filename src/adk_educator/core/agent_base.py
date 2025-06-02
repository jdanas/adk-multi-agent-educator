"""
Base Agent class for the ADK Multi-Agent Educator system.

This module provides the foundational BaseAgent class that all specialized
educational agents inherit from. It handles common functionality like
Google AI integration, message processing, and collaboration protocols.
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger
import google.generativeai as genai

from ..config import (
    AgentConfig, Message, MessageType, StudentRequest, AgentResponse,
    SubjectType, SystemConfig
)


class BaseAgent(ABC):
    """
    Abstract base class for all educational agents in the system.
    
    This class provides common functionality for interacting with Google's
    Generative AI, processing student requests, and collaborating with other agents.
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize the base agent.
        
        Args:
            config: Configuration object containing agent-specific settings
        """
        self.config = config
        self.name = config.name
        self.subject = config.subject
        self.session_history: Dict[str, List[Message]] = {}
        self.collaboration_partners: List['BaseAgent'] = []
        
        # Initialize Google AI
        if SystemConfig.GOOGLE_API_KEY:
            genai.configure(api_key=SystemConfig.GOOGLE_API_KEY)
            self.model = genai.GenerativeModel(config.model)
        else:
            logger.warning("Google API key not configured. Agent will run in mock mode.")
            self.model = None
            
        logger.info(f"Initialized {self.name} agent for {self.subject.value}")
    
    @abstractmethod
    async def get_system_prompt(self) -> str:
        """
        Get the system prompt that defines the agent's role and expertise.
        
        This method must be implemented by each specialized agent to define
        their unique educational focus and teaching style.
        
        Returns:
            System prompt string defining the agent's role
        """
        pass
    
    @abstractmethod
    async def process_subject_specific_request(
        self, 
        request: StudentRequest
    ) -> AgentResponse:
        """
        Process a subject-specific student request.
        
        This method contains the core logic for handling requests within
        the agent's area of expertise.
        
        Args:
            request: The student's educational request
            
        Returns:
            Detailed response from the agent
        """
        pass
    
    async def process_request(self, request: StudentRequest) -> AgentResponse:
        """
        Main entry point for processing student requests.
        
        This method orchestrates the request processing pipeline, including
        validation, subject-specific processing, and response formatting.
        
        Args:
            request: The student's educational request
            
        Returns:
            Formatted agent response
        """
        try:
            logger.info(f"{self.name} processing request for topic: {request.topic}")
            
            # Validate that this agent can handle the request
            if request.subject != self.subject:
                return AgentResponse(
                    agent_name=self.name,
                    subject=self.subject,
                    response_text=f"I specialize in {self.subject.value}, but this question is about {request.subject.value}. Let me redirect you to the appropriate expert.",
                    confidence_score=0.0,
                    requires_collaboration=True,
                    collaboration_subjects=[request.subject]
                )
            
            # Store the request in session history
            self._store_message(request.session_id, Message(
                id=str(uuid.uuid4()),
                type=MessageType.STUDENT_QUERY,
                sender=request.student_id,
                subject=request.subject,
                content=request.specific_question,
                metadata={
                    "topic": request.topic,
                    "difficulty": request.difficulty.value if request.difficulty else None,
                    "context": request.context
                }
            ))
            
            # Process the request using subject-specific logic
            response = await self.process_subject_specific_request(request)
            
            # Store the response in session history
            self._store_message(request.session_id, Message(
                id=str(uuid.uuid4()),
                type=MessageType.AGENT_RESPONSE,
                sender=self.name,
                recipient=request.student_id,
                subject=self.subject,
                content=response.response_text,
                metadata={
                    "confidence_score": response.confidence_score,
                    "requires_collaboration": response.requires_collaboration
                }
            ))
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing request in {self.name}: {str(e)}")
            return AgentResponse(
                agent_name=self.name,
                subject=self.subject,
                response_text=f"I encountered an error while processing your request. Please try rephrasing your question or contact support.",
                confidence_score=0.0
            )
    
    async def generate_response(
        self, 
        prompt: str, 
        context: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Generate a response using Google's Generative AI.
        
        Args:
            prompt: The input prompt for the AI model
            context: Optional context to include in the prompt
            
        Returns:
            Tuple of (response_text, confidence_score)
        """
        try:
            if not self.model:
                # Mock response for testing when API key is not available
                return f"Mock response for: {prompt[:100]}...", 0.5
            
            # Prepare the full prompt with system instructions
            system_prompt = await self.get_system_prompt()
            full_prompt = f"{system_prompt}\n\n"
            
            if context:
                full_prompt += f"Context: {context}\n\n"
            
            full_prompt += f"Student Question: {prompt}\n\nResponse:"
            
            # Generate response
            response = await asyncio.to_thread(
                self.model.generate_content,
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_tokens
                )
            )
            
            response_text = response.text if response.text else "I apologize, but I couldn't generate a response. Please try again."
            
            # Calculate a simple confidence score based on response length and content
            confidence_score = min(1.0, len(response_text) / 500) * 0.8 + 0.2
            
            return response_text, confidence_score
            
        except Exception as e:
            logger.error(f"Error generating response in {self.name}: {str(e)}")
            return "I'm having trouble generating a response right now. Please try again later.", 0.1
    
    async def collaborate_with(self, other_agent: 'BaseAgent', message: str) -> str:
        """
        Collaborate with another agent to provide a comprehensive response.
        
        Args:
            other_agent: The agent to collaborate with
            message: The collaboration message/request
            
        Returns:
            Collaborative response
        """
        try:
            logger.info(f"{self.name} collaborating with {other_agent.name}")
            
            collaboration_prompt = f"""
            I am collaborating with {other_agent.name} (a {other_agent.subject.value} expert) 
            to answer this cross-disciplinary question: {message}
            
            Please provide insights from my {self.subject.value} perspective that would 
            complement {other_agent.name}'s expertise in {other_agent.subject.value}.
            """
            
            response, _ = await self.generate_response(collaboration_prompt)
            return response
            
        except Exception as e:
            logger.error(f"Error in collaboration between {self.name} and {other_agent.name}: {str(e)}")
            return "I'm having trouble collaborating right now."
    
    def _store_message(self, session_id: str, message: Message) -> None:
        """Store a message in the session history."""
        if session_id not in self.session_history:
            self.session_history[session_id] = []
        self.session_history[session_id].append(message)
        
        # Keep only the last 50 messages per session to manage memory
        if len(self.session_history[session_id]) > 50:
            self.session_history[session_id] = self.session_history[session_id][-50:]
    
    def get_session_context(self, session_id: str, max_messages: int = 10) -> str:
        """
        Get recent conversation context for a session.
        
        Args:
            session_id: The session identifier
            max_messages: Maximum number of recent messages to include
            
        Returns:
            Formatted context string
        """
        if session_id not in self.session_history:
            return ""
        
        recent_messages = self.session_history[session_id][-max_messages:]
        context_parts = []
        
        for msg in recent_messages:
            if msg.type == MessageType.STUDENT_QUERY:
                context_parts.append(f"Student: {msg.content}")
            elif msg.type == MessageType.AGENT_RESPONSE:
                context_parts.append(f"{msg.sender}: {msg.content[:200]}...")
        
        return "\n".join(context_parts)
    
    def add_collaboration_partner(self, agent: 'BaseAgent') -> None:
        """Add another agent as a collaboration partner."""
        if agent not in self.collaboration_partners:
            self.collaboration_partners.append(agent)
            logger.info(f"{self.name} now collaborating with {agent.name}")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get a summary of this agent's capabilities."""
        return {
            "name": self.name,
            "subject": self.subject.value,
            "specializations": self.config.specializations,
            "difficulty_levels": [level.value for level in self.config.difficulty_levels],
            "can_collaborate": len(self.collaboration_partners) > 0,
            "collaboration_partners": [agent.name for agent in self.collaboration_partners]
        }
