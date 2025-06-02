"""
Root Agent for ADK Multi-Agent Educator

This module provides the main entry point for the ADK framework to access
our multi-agent educational system.
"""

from typing import Dict, List, Any, Optional
import asyncio
import sys
import os
from loguru import logger

# Ensure the correct path is available for imports
current_dir = os.path.dirname(os.path.abspath(__file__))

# Try different import strategies to handle different ADK import scenarios
try:
    # First try: assume we're imported from project root (ADK scenario)
    from adk_educator.core.coordinator import AgentCoordinator
    from adk_educator.core.session_manager import SessionManager
    from adk_educator.agents.math_agent import MathAgent
    from adk_educator.agents.science_agent import ScienceAgent
    from adk_educator.agents.music_agent import MusicAgent
    from adk_educator.config import StudentRequest, SubjectType, DifficultyLevel
except ImportError:
    # Second try: add src directory to path and import
    src_dir = current_dir
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    try:
        from adk_educator.core.coordinator import AgentCoordinator
        from adk_educator.core.session_manager import SessionManager
        from adk_educator.agents.math_agent import MathAgent
        from adk_educator.agents.science_agent import ScienceAgent
        from adk_educator.agents.music_agent import MusicAgent
        from adk_educator.config import StudentRequest, SubjectType, DifficultyLevel
    except ImportError:
        # Third try: add parent directory to path (when src is imported as module)
        parent_dir = os.path.dirname(src_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        from src.adk_educator.core.coordinator import AgentCoordinator
        from src.adk_educator.core.session_manager import SessionManager
        from src.adk_educator.agents.math_agent import MathAgent
        from src.adk_educator.agents.science_agent import ScienceAgent
        from src.adk_educator.agents.music_agent import MusicAgent
        from src.adk_educator.config import StudentRequest, SubjectType, DifficultyLevel


class RootAgent:
    """
    Root agent that coordinates all specialized educational agents.
    
    This agent serves as the main entry point for the ADK framework,
    managing the multi-agent educational system and routing requests
    to appropriate specialized agents.
    """
    
    def __init__(self):
        """Initialize the root agent with all specialized agents."""
        logger.info("Initializing ADK Multi-Agent Educator Root Agent")
        
        # Create session manager
        self.session_manager = SessionManager()
        
        # Create specialized agents
        self.math_agent = MathAgent()
        self.science_agent = ScienceAgent()
        self.music_agent = MusicAgent()
        
        # Register agents with session manager
        self.session_manager.register_agent(self.math_agent)
        self.session_manager.register_agent(self.science_agent)
        self.session_manager.register_agent(self.music_agent)
        
        # Create coordinator
        self.coordinator = AgentCoordinator(self.session_manager)
        
        logger.info("Root agent initialized with Math, Science, and Music agents")
    
    async def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Process a message through the multi-agent system.
        
        Args:
            message: The student's question or request
            context: Optional context including session info, preferences, etc.
            
        Returns:
            Response from the appropriate agent(s)
        """
        try:
            # Extract context information
            session_id = context.get("session_id", "default") if context else "default"
            student_id = context.get("student_id", "student") if context else "student"
            subject = context.get("subject") if context else None
            difficulty = context.get("difficulty", "MIDDLE") if context else "MIDDLE"
            
            # Convert subject string to enum if provided
            subject_enum = None
            if subject:
                try:
                    subject_enum = SubjectType(subject.upper())
                except ValueError:
                    # If subject is specified but invalid, try to infer from message
                    subject_enum = self._infer_subject_from_message(message)
            else:
                # Try to infer subject from the message content
                subject_enum = self._infer_subject_from_message(message)
            
            # Convert difficulty string to enum
            try:
                difficulty_enum = DifficultyLevel(difficulty.upper())
            except ValueError:
                difficulty_enum = DifficultyLevel.MIDDLE
            
            # Create student request
            request = StudentRequest(
                session_id=session_id,
                student_id=student_id,
                subject=subject_enum,
                topic="general",
                difficulty=difficulty_enum,
                specific_question=message
            )
            
            # Process the request
            if subject_enum:
                # Route to specific agent
                logger.info(f"🎯 Routing {subject_enum.value} question to specialized agent")
                response = await self.session_manager.process_student_request(request)
                
                # Enhanced response with clear agent identification
                agent_emoji = {
                    "Professor Mathematics": "🔢",
                    "Dr. Science Explorer": "🔬", 
                    "Maestro Harmony": "🎵"
                }.get(response.agent_name, "🤖")
                
                return f"{agent_emoji} **{response.agent_name}** (Confidence: {response.confidence_score:.1%})\n\n{response.response_text}\n\n---\n*Specialized {subject_enum.value.title()} Agent Response*"
            else:
                # Use coordinator for interdisciplinary questions - involve all subjects
                logger.info("🔄 Interdisciplinary question detected - consulting all specialized agents")
                involving_subjects = [SubjectType.MATH, SubjectType.SCIENCE, SubjectType.MUSIC]
                response = await self.coordinator.process_complex_query(request, involving_subjects)
                
                return f"🎓 **{response.agent_name}** (Confidence: {response.confidence_score:.1%})\n\n{response.response_text}\n\n---\n*Multi-Agent Collaborative Response (Math 🔢 + Science 🔬 + Music 🎵)*"
                    
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    def _infer_subject_from_message(self, message: str) -> Optional[SubjectType]:
        """
        Infer the subject from the message content.
        
        Args:
            message: The student's message
            
        Returns:
            Inferred subject type or None for interdisciplinary
        """
        message_lower = message.lower()
        
        # Math keywords
        math_keywords = ['math', 'equation', 'solve', 'calculate', 'algebra', 'geometry', 
                        'calculus', 'trigonometry', 'statistics', 'number', 'formula',
                        'fraction', 'decimal', 'percentage', 'graph', 'derivative',
                        '+', '-', '*', '/', '=', 'plus', 'minus', 'times', 'divided',
                        'add', 'subtract', 'multiply', 'divide', 'sum', 'difference',
                        'x^', 'squared', 'cubed', 'root', 'logarithm', 'integral']
        
        # Science keywords  
        science_keywords = ['science', 'physics', 'chemistry', 'biology', 'experiment',
                           'molecule', 'atom', 'energy', 'force', 'gravity', 'photosynthesis',
                           'evolution', 'genetics', 'periodic', 'reaction', 'hypothesis']
        
        # Music keywords
        music_keywords = ['music', 'note', 'scale', 'chord', 'melody', 'harmony', 'rhythm',
                         'instrument', 'composition', 'theory', 'pitch', 'tempo', 'beat',
                         'major', 'minor', 'symphony', 'jazz', 'classical']
        
        math_score = sum(1 for keyword in math_keywords if keyword in message_lower)
        science_score = sum(1 for keyword in science_keywords if keyword in message_lower)
        music_score = sum(1 for keyword in music_keywords if keyword in message_lower)
        
        # Return the subject with the highest score, or None if tied
        max_score = max(math_score, science_score, music_score)
        if max_score == 0:
            return None  # No clear subject indication
        
        if math_score == max_score and math_score > science_score and math_score > music_score:
            return SubjectType.MATH
        elif science_score == max_score and science_score > math_score and science_score > music_score:
            return SubjectType.SCIENCE
        elif music_score == max_score and music_score > math_score and music_score > science_score:
            return SubjectType.MUSIC
        else:
            return None  # Tied scores or ambiguous
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get information about the agent's capabilities.
        
        Returns:
            Dictionary describing the multi-agent system capabilities
        """
        return {
            "name": "ADK Multi-Agent Educator",
            "description": "Specialized educational agents for Math, Science, and Music",
            "agents": [
                {
                    "name": self.math_agent.config.name,
                    "subject": self.math_agent.config.subject.value,
                    "specializations": self.math_agent.config.specializations[:5]
                },
                {
                    "name": self.science_agent.config.name,
                    "subject": self.science_agent.config.subject.value,
                    "specializations": self.science_agent.config.specializations[:5]
                },
                {
                    "name": self.music_agent.config.name,
                    "subject": self.music_agent.config.subject.value,
                    "specializations": self.music_agent.config.specializations[:5]
                }
            ],
            "features": [
                "Multi-agent collaboration",
                "Adaptive difficulty adjustment",
                "Session progress tracking",
                "Cross-subject knowledge integration",
                "Personalized learning recommendations"
            ]
        }


# Import ADK components for proper integration
try:
    from google.adk import Agent
    ADK_AVAILABLE = True
except ImportError:
    # Fallback if ADK is not available
    logger.warning("ADK not available, using fallback implementation")
    Agent = None
    ADK_AVAILABLE = False


# Create custom tool function for ADK integration
def process_educational_query(query: str) -> str:
    """
    Tool function for processing educational queries through our multi-agent system.
    This function serves as the bridge between ADK and our educational agents.
    
    Args:
        query: The student's educational question or request
        
    Returns:
        Response from the appropriate specialized agent(s) with clear agent identification
    """
    import asyncio
    
    if not query or not query.strip():
        return "🎓 **Multi-Agent Educational System Ready**\n\nI'm ready to help with:\n• **Math** (Professor Mathematics)\n• **Science** (Dr. Science Explorer)\n• **Music** (Maestro Harmony)\n\nWhat would you like to learn about?"
    
    # Create our internal agent if not already created
    if not hasattr(process_educational_query, '_internal_agent'):
        logger.info("🔧 Initializing internal multi-agent system...")
        process_educational_query._internal_agent = RootAgent()
    
    # Process the message through our multi-agent system
    # Run the async function in the current event loop or create a new one
    try:
        logger.info(f"📝 Processing query: '{query[:50]}...'")
        
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, we need to handle this differently
            # Create a new event loop for this operation
            import concurrent.futures
            import uuid
            session_id = str(uuid.uuid4())[:8]  # Short unique session ID
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    process_educational_query._internal_agent.process_message(
                        query.strip(),
                        {"user_id": "adk_user", "session_id": session_id}
                    )
                )
                response = future.result(timeout=30)
                logger.info("✅ Multi-agent processing completed successfully")
                return f"🤖 **Multi-Agent Response**\n\n{response}"
        else:
            import uuid
            session_id = str(uuid.uuid4())[:8]  # Short unique session ID
            response = loop.run_until_complete(
                process_educational_query._internal_agent.process_message(
                    query.strip(),
                    {"user_id": "adk_user", "session_id": session_id}
                )
            )
            logger.info("✅ Multi-agent processing completed successfully")
            return f"🤖 **Multi-Agent Response**\n\n{response}"
            
    except Exception as e:
        logger.error(f"❌ Error in multi-agent processing: {e}")
        return f"🚨 **Multi-Agent System Error**\n\nI apologize, but I encountered an error while processing your educational query through our specialized agents:\n\n`{str(e)}`\n\nPlease try rephrasing your question or try again in a moment."


def create_adk_agent():
    """Create an ADK-compatible agent instance."""
    if not ADK_AVAILABLE:
        # Return our custom agent if ADK is not available
        logger.info("ADK not available, returning custom RootAgent")
        return RootAgent()
    
    logger.info("Creating ADK-compatible agent")
    return Agent(
        name="multi_agent_educator",  # Valid identifier without spaces
        description="Multi-agent educational coordinator that routes questions to specialized Math, Science, and Music agents",
        instruction="""You are a COORDINATOR for a multi-agent educational system. Your ONLY job is to route student questions to the appropriate specialized agents.

DO NOT attempt to answer educational questions yourself. ALWAYS use the process_educational_query tool for ANY educational question.

Your specialized agents are:
- **Professor Mathematics**: Handles all math questions (algebra, geometry, calculus, statistics, etc.)
- **Dr. Science Explorer**: Handles all science questions (physics, chemistry, biology, etc.)  
- **Maestro Harmony**: Handles all music questions (theory, composition, performance, etc.)

For ANY educational question, immediately call the process_educational_query tool. The tool will:
1. Analyze the question to determine the subject area
2. Route to the appropriate specialist agent(s)
3. Handle interdisciplinary questions by coordinating multiple agents
4. Return the expert response

Your response should ONLY be the result from the process_educational_query tool - do not add additional commentary.""",
        model="gemini-2.0-flash-exp",
        tools=[process_educational_query]
    )


# Create the root agent instance that ADK will use
# First create our internal agent
_internal_root_agent = RootAgent()

# Then create the ADK-compatible agent
root_agent = create_adk_agent()

# Also create an 'agent' attribute for alternative discovery
agent = root_agent
