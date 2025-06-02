"""
Math Agent for the ADK Multi-Agent Educator system.

This agent specializes in mathematics education, covering topics from
elementary arithmetic to advanced calculus and beyond. It provides
step-by-step explanations, problem-solving strategies, and mathematical insights.
"""

from typing import List, Optional
from loguru import logger

from ..config import (
    AgentConfig, StudentRequest, AgentResponse, SubjectType, 
    DifficultyLevel, MATH_SPECIALIZATIONS
)
from ..core.agent_base import BaseAgent


class MathAgent(BaseAgent):
    """
    Specialized agent for mathematics education.
    
    The MathAgent provides comprehensive mathematics support including:
    - Step-by-step problem solving
    - Conceptual explanations
    - Mathematical reasoning
    - Practice problem generation
    - Connections to real-world applications
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize the Math Agent.
        
        Args:
            config: Optional configuration. If not provided, uses defaults.
        """
        if config is None:
            config = AgentConfig(
                name="Professor Mathematics",
                subject=SubjectType.MATH,
                specializations=MATH_SPECIALIZATIONS,
                difficulty_levels=[
                    DifficultyLevel.ELEMENTARY,
                    DifficultyLevel.MIDDLE, 
                    DifficultyLevel.HIGH,
                    DifficultyLevel.COLLEGE
                ]
            )
        
        super().__init__(config)
    
    async def get_system_prompt(self) -> str:
        """Get the system prompt defining the Math Agent's role and expertise."""
        return """You are Professor Mathematics, an expert mathematics educator with a passion for making math accessible and engaging. Your role is to:

1. **Explain Clearly**: Break down complex mathematical concepts into understandable steps
2. **Show Your Work**: Always provide step-by-step solutions with clear reasoning
3. **Build Understanding**: Focus on helping students understand the 'why' behind mathematical procedures
4. **Encourage Problem-Solving**: Guide students to discover solutions rather than just giving answers
5. **Make Connections**: Show how mathematical concepts relate to real-world applications
6. **Adapt to Level**: Adjust your explanations based on the student's difficulty level

**Your Specializations Include:**
- Arithmetic and Number Theory
- Algebra (from basic to advanced)
- Geometry and Trigonometry  
- Calculus and Analysis
- Statistics and Probability
- Linear Algebra
- Differential Equations
- Mathematical Modeling

**Teaching Style:**
- Start with what the student knows
- Use visual and conceptual explanations when helpful
- Provide multiple approaches to problems when possible
- Encourage questions and exploration
- Be patient and supportive
- Connect abstract concepts to concrete examples

**When Solving Problems:**
1. State what we're trying to find
2. Identify what information we have
3. Choose the appropriate method/formula
4. Work through each step clearly
5. Check the answer for reasonableness
6. Explain what the answer means in context

Remember: Every student can learn mathematics with the right support and approach!"""

    async def process_subject_specific_request(self, request: StudentRequest) -> AgentResponse:
        """
        Process a mathematics-specific student request.
        
        Args:
            request: The student's mathematics question
            
        Returns:
            Detailed mathematical response with explanations
        """
        try:
            # Get session context for continuity
            context = self.get_session_context(request.session_id)
            
            # Enhance the prompt with math-specific guidance
            enhanced_prompt = self._create_math_prompt(request, context)
            
            # Generate response using the AI model
            response_text, confidence = await self.generate_response(enhanced_prompt, context)
            
            # Generate math-specific follow-ups and resources
            follow_ups = self._generate_math_follow_ups(request)
            resources = self._generate_math_resources(request)
            
            # Check if collaboration with other subjects might be helpful
            requires_collaboration, collab_subjects = self._assess_collaboration_needs(request)
            
            return AgentResponse(
                agent_name=self.name,
                subject=self.subject,
                response_text=response_text,
                confidence_score=confidence,
                suggested_follow_ups=follow_ups,
                resources=resources,
                requires_collaboration=requires_collaboration,
                collaboration_subjects=collab_subjects
            )
            
        except Exception as e:
            logger.error(f"Error in MathAgent processing: {str(e)}")
            return AgentResponse(
                agent_name=self.name,
                subject=self.subject,
                response_text="I encountered an error while solving this math problem. Could you please rephrase your question or try a different approach?",
                confidence_score=0.1
            )
    
    def _create_math_prompt(self, request: StudentRequest, context: str) -> str:
        """Create an enhanced prompt for mathematical problem solving."""
        difficulty_guidance = ""
        if request.difficulty:
            difficulty_map = {
                DifficultyLevel.ELEMENTARY: "Use simple language and basic concepts. Include visual aids descriptions when helpful.",
                DifficultyLevel.MIDDLE: "Use grade-appropriate terminology. Show connections between concepts.",
                DifficultyLevel.HIGH: "Include more advanced techniques and theory. Prepare for college-level math.",
                DifficultyLevel.COLLEGE: "Use formal mathematical notation and rigorous explanations."
            }
            difficulty_guidance = difficulty_map.get(request.difficulty, "")
        
        prompt = f"""
Mathematical Question: {request.specific_question}
Topic Area: {request.topic}
{f"Difficulty Level: {request.difficulty.value} - {difficulty_guidance}" if request.difficulty else ""}
{f"Additional Context: {request.context}" if request.context else ""}

Please provide a comprehensive mathematical response that includes:
1. A clear restatement of the problem
2. Step-by-step solution with explanations
3. Verification of the answer
4. Real-world connection or application if relevant
5. Common mistakes to avoid for this type of problem

{f"Recent conversation context: {context}" if context else ""}
"""
        return prompt.strip()
    
    def _generate_math_follow_ups(self, request: StudentRequest) -> List[str]:
        """Generate mathematics-specific follow-up questions."""
        topic_lower = request.topic.lower()
        follow_ups = []
        
        # General follow-ups
        follow_ups.extend([
            "Would you like to see this problem solved using a different method?",
            "Can I show you a similar practice problem?",
            "Would you like me to explain any of the steps in more detail?"
        ])
        
        # Topic-specific follow-ups
        if any(keyword in topic_lower for keyword in ["algebra", "equation", "solve"]):
            follow_ups.extend([
                "Would you like to practice solving similar equations?",
                "How about we explore graphing this equation?",
                "Should we discuss how to check our solution?"
            ])
        
        elif any(keyword in topic_lower for keyword in ["geometry", "triangle", "circle", "area"]):
            follow_ups.extend([
                "Would you like to see the geometric visualization?",
                "Should we explore related geometric properties?",
                "How about calculating the perimeter as well?"
            ])
        
        elif any(keyword in topic_lower for keyword in ["calculus", "derivative", "integral"]):
            follow_ups.extend([
                "Would you like to see the graphical interpretation?", 
                "Should we verify this using a different calculus rule?",
                "How about exploring the real-world application of this concept?"
            ])
        
        elif any(keyword in topic_lower for keyword in ["statistics", "probability", "data"]):
            follow_ups.extend([
                "Would you like to see how to interpret these results?",
                "Should we explore what this means in practical terms?",
                "How about calculating the confidence interval?"
            ])
        
        return follow_ups[:5]  # Limit to 5 follow-ups
    
    def _generate_math_resources(self, request: StudentRequest) -> List[str]:
        """Generate mathematics-specific learning resources."""
        topic_lower = request.topic.lower()
        resources = [
            "Khan Academy - Comprehensive math lessons with practice problems",
            "Desmos Graphing Calculator - Interactive mathematical visualization",
            "Wolfram Alpha - Step-by-step mathematical solutions"
        ]
        
        # Topic-specific resources
        if any(keyword in topic_lower for keyword in ["algebra", "equation"]):
            resources.extend([
                "Algebra.com - Practice problems and tutorials",
                "IXL Math - Interactive algebra exercises"
            ])
        
        elif any(keyword in topic_lower for keyword in ["geometry", "triangle", "circle"]):
            resources.extend([
                "GeoGebra - Interactive geometry software",
                "Math Open Reference - Geometric definitions and properties"
            ])
        
        elif any(keyword in topic_lower for keyword in ["calculus", "derivative", "integral"]):
            resources.extend([
                "Paul's Online Math Notes - Comprehensive calculus tutorial",
                "MIT OpenCourseWare - Free calculus courses"
            ])
        
        elif any(keyword in topic_lower for keyword in ["statistics", "probability"]):
            resources.extend([
                "Statistics How To - Clear statistical explanations",
                "R Project - Statistical computing software"
            ])
        
        return resources[:7]  # Limit to 7 resources
    
    def _assess_collaboration_needs(self, request: StudentRequest) -> tuple[bool, List[SubjectType]]:
        """Assess if collaboration with other subjects would be beneficial."""
        query_lower = request.specific_question.lower()
        collaboration_subjects = []
        
        # Science connections
        if any(keyword in query_lower for keyword in [
            "physics", "velocity", "acceleration", "force", "energy", "wave", "frequency",
            "chemistry", "concentration", "reaction rate", "pH", "molarity",
            "biology", "population", "growth", "genetics", "evolution"
        ]):
            collaboration_subjects.append(SubjectType.SCIENCE)
        
        # Music connections  
        if any(keyword in query_lower for keyword in [
            "frequency", "wave", "sound", "acoustics", "rhythm", "beat",
            "harmony", "music", "audio", "pitch", "octave"
        ]):
            collaboration_subjects.append(SubjectType.MUSIC)
        
        return len(collaboration_subjects) > 0, collaboration_subjects
    
    async def generate_practice_problem(self, topic: str, difficulty: DifficultyLevel) -> str:
        """
        Generate a practice problem for a specific topic and difficulty level.
        
        Args:
            topic: The mathematical topic
            difficulty: The desired difficulty level
            
        Returns:
            Generated practice problem
        """
        try:
            prompt = f"""
Generate a {difficulty.value}-level practice problem for {topic}. 
Include the problem statement and provide the solution with step-by-step explanation.
Make sure the problem is engaging and relevant to students at this level.
"""
            
            response, _ = await self.generate_response(prompt)
            return response
            
        except Exception as e:
            logger.error(f"Error generating practice problem: {str(e)}")
            return f"I'm having trouble generating a practice problem for {topic} right now. Please try again later."
