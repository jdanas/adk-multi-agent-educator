"""
Agent Coordinator for the ADK Multi-Agent Educator system.

This module provides the central coordination layer that manages
the interaction between multiple educational agents and handles
complex multi-subject queries.
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from loguru import logger

from ..config import StudentRequest, AgentResponse, SubjectType, SystemConfig
from .agent_base import BaseAgent
from .session_manager import SessionManager


class AgentCoordinator:
    """
    Central coordinator for managing multiple educational agents.
    
    The AgentCoordinator is responsible for:
    - Orchestrating interactions between agents
    - Handling complex multi-subject queries  
    - Load balancing and performance optimization
    - Providing unified responses from multiple agents
    """
    
    def __init__(self, session_manager: SessionManager):
        """
        Initialize the agent coordinator.
        
        Args:
            session_manager: The session manager instance
        """
        self.session_manager = session_manager
        self.performance_metrics: Dict[str, Dict[str, float]] = {}
        
        logger.info("AgentCoordinator initialized")
    
    async def process_complex_query(
        self, 
        request: StudentRequest, 
        involving_subjects: List[SubjectType]
    ) -> AgentResponse:
        """
        Process a complex query that involves multiple subjects.
        
        This method coordinates responses from multiple agents to provide
        a comprehensive answer to interdisciplinary questions.
        
        Args:
            request: The student's educational request
            involving_subjects: List of subjects involved in the query
            
        Returns:
            Coordinated response from multiple agents
        """
        try:
            logger.info(f"Processing complex query involving subjects: {[s.value for s in involving_subjects]}")
            
            # Get responses from all relevant agents
            agent_responses = await self._gather_agent_responses(request, involving_subjects)
            
            if not agent_responses:
                return AgentResponse(
                    agent_name="Coordinator",
                    subject=request.subject,
                    response_text="I couldn't find appropriate experts for your question. Please try rephrasing or breaking it into smaller questions.",
                    confidence_score=0.0
                )
            
            # Synthesize responses into a coherent answer
            synthesized_response = await self._synthesize_responses(
                request, agent_responses, involving_subjects
            )
            
            return synthesized_response
            
        except Exception as e:
            logger.error(f"Error processing complex query: {str(e)}")
            return AgentResponse(
                agent_name="Coordinator",
                subject=request.subject,
                response_text="I encountered an error while coordinating the response. Please try again.",
                confidence_score=0.0
            )
    
    async def _gather_agent_responses(
        self, 
        request: StudentRequest, 
        subjects: List[SubjectType]
    ) -> List[Tuple[SubjectType, AgentResponse]]:
        """
        Gather responses from multiple agents concurrently.
        
        Args:
            request: The student's request
            subjects: List of subjects to get responses from
            
        Returns:
            List of tuples containing (subject, response) pairs
        """
        tasks = []
        
        for subject in subjects:
            if subject in self.session_manager.registered_agents:
                agent = self.session_manager.registered_agents[subject]
                
                # Create a subject-specific request
                subject_request = StudentRequest(
                    session_id=request.session_id,
                    student_id=request.student_id,
                    subject=subject,
                    topic=request.topic,
                    difficulty=request.difficulty,
                    specific_question=f"From a {subject.value} perspective: {request.specific_question}",
                    context=request.context,
                    preferred_learning_style=request.preferred_learning_style
                )
                
                task = asyncio.create_task(
                    self._get_agent_response_with_timeout(agent, subject_request, subject)
                )
                tasks.append(task)
        
        if not tasks:
            return []
        
        # Wait for all responses with a timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=30.0  # 30 second timeout
            )
            
            valid_responses = []
            for result in results:
                if isinstance(result, tuple) and not isinstance(result, Exception):
                    valid_responses.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Agent response error: {str(result)}")
            
            return valid_responses
            
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for agent responses")
            return []
    
    async def _get_agent_response_with_timeout(
        self, 
        agent: BaseAgent, 
        request: StudentRequest, 
        subject: SubjectType
    ) -> Tuple[SubjectType, AgentResponse]:
        """
        Get response from a single agent with error handling.
        
        Args:
            agent: The agent to query
            request: The request to send
            subject: The subject type
            
        Returns:
            Tuple of (subject, response)
        """
        try:
            response = await agent.process_request(request)
            
            # Track performance metrics
            if agent.name not in self.performance_metrics:
                self.performance_metrics[agent.name] = {
                    "total_requests": 0,
                    "average_confidence": 0.0,
                    "success_rate": 0.0
                }
            
            metrics = self.performance_metrics[agent.name]
            metrics["total_requests"] += 1
            metrics["average_confidence"] = (
                (metrics["average_confidence"] * (metrics["total_requests"] - 1) + response.confidence_score) 
                / metrics["total_requests"]
            )
            
            return (subject, response)
            
        except Exception as e:
            logger.error(f"Error getting response from {agent.name}: {str(e)}")
            return (subject, AgentResponse(
                agent_name=agent.name,
                subject=subject,
                response_text=f"Error occurred while processing with {agent.name}",
                confidence_score=0.0
            ))
    
    async def _synthesize_responses(
        self, 
        request: StudentRequest, 
        responses: List[Tuple[SubjectType, AgentResponse]], 
        subjects: List[SubjectType]
    ) -> AgentResponse:
        """
        Synthesize multiple agent responses into a coherent answer.
        
        Args:
            request: The original student request
            responses: List of agent responses
            subjects: List of involved subjects
            
        Returns:
            Synthesized response
        """
        if len(responses) == 1:
            # Single response - just return it
            return responses[0][1]
        
        # Multiple responses - create a coordinated response
        response_sections = []
        all_resources = []
        all_follow_ups = []
        total_confidence = 0.0
        
        for subject, response in responses:
            if response.confidence_score > 0.3:  # Only include reasonably confident responses
                response_sections.append(
                    f"**{subject.value.title()} Perspective ({response.agent_name}):**\n{response.response_text}"
                )
                all_resources.extend(response.resources)
                all_follow_ups.extend(response.suggested_follow_ups)
                total_confidence += response.confidence_score
        
        if not response_sections:
            return AgentResponse(
                agent_name="Coordinator",
                subject=request.subject,
                response_text="I couldn't generate a confident response from any of our experts. Please try rephrasing your question.",
                confidence_score=0.0
            )
        
        # Create coordinated response
        coordinated_text = f"""I've consulted with our experts in {', '.join([s.value for s in subjects])} to give you a comprehensive answer:

{chr(10).join(response_sections)}

**Summary:** This question beautifully demonstrates how {', '.join([s.value for s in subjects])} are interconnected. Each perspective provides valuable insights that complement the others."""
        
        # Add interdisciplinary follow-up questions
        interdisciplinary_follow_ups = [
            f"How might {subjects[0].value} and {subjects[1].value} work together in real-world applications?",
            f"What other subjects connect to this topic?",
            "Can you give me a specific example where these fields intersect?"
        ]
        
        return AgentResponse(
            agent_name="Multi-Agent Coordinator",
            subject=request.subject,
            response_text=coordinated_text,
            confidence_score=min(1.0, total_confidence / len(responses)),
            suggested_follow_ups=list(set(all_follow_ups + interdisciplinary_follow_ups))[:5],
            resources=list(set(all_resources))[:10],
            requires_collaboration=False
        )
    
    def identify_relevant_subjects(self, query: str) -> List[SubjectType]:
        """
        Identify which subjects are relevant to a given query.
        
        This method uses keyword matching and heuristics to determine
        which educational domains are involved in a question.
        
        Args:
            query: The student's question text
            
        Returns:
            List of relevant subject types
        """
        query_lower = query.lower()
        relevant_subjects = []
        
        # Math keywords
        math_keywords = [
            "calculate", "equation", "formula", "solve", "mathematics", "math",
            "algebra", "geometry", "calculus", "statistics", "probability",
            "number", "fraction", "decimal", "percentage", "graph", "function"
        ]
        
        # Science keywords  
        science_keywords = [
            "experiment", "hypothesis", "theory", "physics", "chemistry", "biology",
            "molecule", "atom", "cell", "organism", "energy", "force", "reaction",
            "evolution", "genetics", "ecosystem", "planet", "element", "compound"
        ]
        
        # Music keywords
        music_keywords = [
            "music", "song", "melody", "harmony", "rhythm", "chord", "note",
            "instrument", "composer", "symphony", "scale", "key", "tempo",
            "beat", "sound", "audio", "acoustic", "frequency", "pitch"
        ]
        
        # Check for subject relevance
        if any(keyword in query_lower for keyword in math_keywords):
            relevant_subjects.append(SubjectType.MATH)
        
        if any(keyword in query_lower for keyword in science_keywords):
            relevant_subjects.append(SubjectType.SCIENCE)
        
        if any(keyword in query_lower for keyword in music_keywords):
            relevant_subjects.append(SubjectType.MUSIC)
        
        # If no specific subjects identified, default to the primary subject
        if not relevant_subjects:
            # Use simple heuristics or default to first available agent
            available_subjects = list(self.session_manager.registered_agents.keys())
            if available_subjects:
                relevant_subjects.append(available_subjects[0])
        
        return relevant_subjects
    
    def get_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance metrics for all agents.
        
        Returns:
            Dictionary of agent performance metrics
        """
        return self.performance_metrics.copy()
    
    async def recommend_learning_path(
        self, 
        session_id: str, 
        student_interests: List[str]
    ) -> Dict[str, List[str]]:
        """
        Recommend a learning path based on student interests and session history.
        
        Args:
            session_id: The student's session ID
            student_interests: List of topics the student is interested in
            
        Returns:
            Dictionary mapping subjects to recommended topics
        """
        try:
            session = self.session_manager.get_session(session_id)
            if not session:
                return {}
            
            recommendations = {}
            
            # Get recommendations from each agent based on interests
            for subject, agent in self.session_manager.registered_agents.items():
                subject_recommendations = []
                
                # Basic recommendations based on subject
                if subject == SubjectType.MATH:
                    if any(interest in ["problem solving", "logic", "patterns"] for interest in student_interests):
                        subject_recommendations.extend(["algebra puzzles", "geometric patterns", "mathematical modeling"])
                
                elif subject == SubjectType.SCIENCE:
                    if any(interest in ["nature", "experiments", "discovery"] for interest in student_interests):
                        subject_recommendations.extend(["hands-on experiments", "natural phenomena", "scientific method"])
                
                elif subject == SubjectType.MUSIC:
                    if any(interest in ["creativity", "art", "expression"] for interest in student_interests):
                        subject_recommendations.extend(["music composition", "rhythm exercises", "music theory basics"])
                
                if subject_recommendations:
                    recommendations[subject.value] = subject_recommendations
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating learning path recommendations: {str(e)}")
            return {}
