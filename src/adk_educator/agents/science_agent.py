"""
Science Agent for the ADK Multi-Agent Educator system.

This agent specializes in science education, covering physics, chemistry, 
biology, earth science, and related fields. It provides scientific explanations,
experimental insights, and connections to real-world phenomena.
"""

from typing import List, Optional
from loguru import logger

from ..config import (
    AgentConfig, StudentRequest, AgentResponse, SubjectType, 
    DifficultyLevel, SCIENCE_SPECIALIZATIONS
)
from ..core.agent_base import BaseAgent


class ScienceAgent(BaseAgent):
    """
    Specialized agent for science education.
    
    The ScienceAgent provides comprehensive science support including:
    - Scientific explanations and theories
    - Experimental design and methodology
    - Real-world applications and phenomena
    - Cross-disciplinary connections
    - Laboratory safety and procedures
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize the Science Agent.
        
        Args:
            config: Optional configuration. If not provided, uses defaults.
        """
        if config is None:
            config = AgentConfig(
                name="Dr. Science Explorer",
                subject=SubjectType.SCIENCE,
                specializations=SCIENCE_SPECIALIZATIONS,
                difficulty_levels=[
                    DifficultyLevel.ELEMENTARY,
                    DifficultyLevel.MIDDLE,
                    DifficultyLevel.HIGH,
                    DifficultyLevel.COLLEGE
                ]
            )
        
        super().__init__(config)
    
    async def get_system_prompt(self) -> str:
        """Get the system prompt defining the Science Agent's role and expertise."""
        return """You are Dr. Science Explorer, a passionate science educator who makes scientific concepts accessible and exciting. Your mission is to:

1. **Explain Scientific Phenomena**: Make complex scientific concepts understandable through clear explanations
2. **Connect Theory to Practice**: Show how scientific principles apply to everyday life and real-world situations
3. **Encourage Scientific Thinking**: Foster curiosity, critical thinking, and the scientific method
4. **Promote Discovery**: Guide students to explore and investigate scientific questions
5. **Ensure Safety**: Always emphasize safety considerations in scientific activities
6. **Bridge Disciplines**: Show connections between different scientific fields

**Your Areas of Expertise:**
- **Physics**: Mechanics, energy, waves, electricity, magnetism, quantum physics
- **Chemistry**: Atoms, molecules, reactions, bonding, organic chemistry, biochemistry
- **Biology**: Cells, genetics, evolution, ecology, anatomy, physiology
- **Earth Science**: Geology, meteorology, oceanography, astronomy, environmental science
- **Applied Sciences**: Engineering principles, technology applications, research methods

**Teaching Approach:**
- Start with observable phenomena and work toward underlying principles
- Use analogies and real-world examples to explain abstract concepts
- Encourage hands-on thinking and experimentation
- Address common misconceptions directly
- Emphasize the process of scientific inquiry
- Show the beauty and wonder of natural phenomena

**When Explaining Scientific Concepts:**
1. Begin with what students can observe or experience
2. Introduce the underlying scientific principle
3. Provide clear, step-by-step explanations
4. Use analogies when helpful
5. Connect to real-world applications
6. Suggest safe, simple experiments when appropriate
7. Address potential misconceptions

**Safety First**: Always emphasize safety precautions for any experimental suggestions or laboratory activities.

Remember: Science is all around us, and every question is an opportunity for discovery!"""

    async def process_subject_specific_request(self, request: StudentRequest) -> AgentResponse:
        """
        Process a science-specific student request.
        
        Args:
            request: The student's science question
            
        Returns:
            Detailed scientific response with explanations
        """
        try:
            # Get session context for continuity
            context = self.get_session_context(request.session_id)
            
            # Enhance the prompt with science-specific guidance
            enhanced_prompt = self._create_science_prompt(request, context)
            
            # Generate response using the AI model
            response_text, confidence = await self.generate_response(enhanced_prompt, context)
            
            # Generate science-specific follow-ups and resources
            follow_ups = self._generate_science_follow_ups(request)
            resources = self._generate_science_resources(request)
            
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
            logger.error(f"Error in ScienceAgent processing: {str(e)}")
            return AgentResponse(
                agent_name=self.name,
                subject=self.subject,
                response_text="I encountered an error while exploring this scientific concept. Could you please rephrase your question or provide more specific details?",
                confidence_score=0.1
            )
    
    def _create_science_prompt(self, request: StudentRequest, context: str) -> str:
        """Create an enhanced prompt for scientific explanations."""
        difficulty_guidance = ""
        if request.difficulty:
            difficulty_map = {
                DifficultyLevel.ELEMENTARY: "Use simple, age-appropriate language. Focus on observable phenomena and basic concepts.",
                DifficultyLevel.MIDDLE: "Include more detailed explanations. Introduce scientific vocabulary appropriately.",
                DifficultyLevel.HIGH: "Use proper scientific terminology. Include more complex relationships and theories.",
                DifficultyLevel.COLLEGE: "Provide rigorous scientific explanations with appropriate technical detail."
            }
            difficulty_guidance = difficulty_map.get(request.difficulty, "")
        
        prompt = f"""
Scientific Question: {request.specific_question}
Topic Area: {request.topic}
{f"Difficulty Level: {request.difficulty.value} - {difficulty_guidance}" if request.difficulty else ""}
{f"Additional Context: {request.context}" if request.context else ""}

Please provide a comprehensive scientific response that includes:
1. Clear explanation of the scientific concept or phenomenon
2. Underlying scientific principles involved
3. Real-world examples or applications
4. Any relevant scientific processes or mechanisms
5. Safety considerations if experimental work is mentioned
6. Common misconceptions to avoid

{f"Recent conversation context: {context}" if context else ""}
"""
        return prompt.strip()
    
    def _generate_science_follow_ups(self, request: StudentRequest) -> List[str]:
        """Generate science-specific follow-up questions."""
        topic_lower = request.topic.lower()
        follow_ups = []
        
        # General follow-ups
        follow_ups.extend([
            "Would you like to explore a related scientific phenomenon?",
            "Can I suggest a safe experiment you could try?",
            "Would you like to know more about the scientists who discovered this?"
        ])
        
        # Physics follow-ups
        if any(keyword in topic_lower for keyword in ["physics", "force", "energy", "motion", "electricity", "magnetism"]):
            follow_ups.extend([
                "How does this principle apply to everyday objects?",
                "Would you like to see the mathematical relationship?",
                "Can we explore how this connects to other physics concepts?"
            ])
        
        # Chemistry follow-ups
        elif any(keyword in topic_lower for keyword in ["chemistry", "reaction", "molecule", "atom", "compound"]):
            follow_ups.extend([
                "Would you like to see what happens at the molecular level?",
                "Can we explore how this reaction occurs in nature?",
                "Should we discuss the energy changes involved?"
            ])
        
        # Biology follow-ups
        elif any(keyword in topic_lower for keyword in ["biology", "cell", "organism", "genetics", "evolution"]):
            follow_ups.extend([
                "How does this process vary across different species?",
                "Would you like to explore the evolutionary significance?",
                "Can we look at this from a cellular perspective?"
            ])
        
        # Earth Science follow-ups
        elif any(keyword in topic_lower for keyword in ["earth", "geology", "weather", "climate", "ocean"]):
            follow_ups.extend([
                "How does this process affect our daily lives?",
                "Would you like to explore the geological timeline?",
                "Can we discuss the environmental implications?"
            ])
        
        return follow_ups[:5]
    
    def _generate_science_resources(self, request: StudentRequest) -> List[str]:
        """Generate science-specific learning resources."""
        topic_lower = request.topic.lower()
        resources = [
            "NASA Education - Space and earth science resources",
            "National Geographic Kids - Engaging science content",
            "TED-Ed Science - Educational science videos",
            "Smithsonian National Museum of Natural History - Scientific exhibits and information"
        ]
        
        # Physics resources
        if any(keyword in topic_lower for keyword in ["physics", "force", "energy", "motion"]):
            resources.extend([
                "PhET Interactive Simulations - Physics simulations",
                "The Physics Classroom - Comprehensive physics tutorials"
            ])
        
        # Chemistry resources
        elif any(keyword in topic_lower for keyword in ["chemistry", "reaction", "molecule", "atom"]):
            resources.extend([
                "Royal Society of Chemistry - Chemistry education resources",
                "ChemSpider - Chemical database and information"
            ])
        
        # Biology resources
        elif any(keyword in topic_lower for keyword in ["biology", "cell", "organism", "genetics"]):
            resources.extend([
                "National Human Genome Research Institute - Genetics resources",
                "BioInteractive - Interactive biology materials"
            ])
        
        # Earth Science resources
        elif any(keyword in topic_lower for keyword in ["earth", "geology", "weather", "climate"]):
            resources.extend([
                "USGS Education - Geological and environmental resources",
                "NOAA Education - Weather and climate information"
            ])
        
        return resources[:7]
    
    def _assess_collaboration_needs(self, request: StudentRequest) -> tuple[bool, List[SubjectType]]:
        """Assess if collaboration with other subjects would be beneficial."""
        query_lower = request.specific_question.lower()
        collaboration_subjects = []
        
        # Math connections
        if any(keyword in query_lower for keyword in [
            "calculate", "equation", "formula", "graph", "measurement", "statistics",
            "probability", "rate", "ratio", "proportion", "exponential", "logarithm"
        ]):
            collaboration_subjects.append(SubjectType.MATH)
        
        # Music connections
        if any(keyword in query_lower for keyword in [
            "sound", "wave", "frequency", "pitch", "acoustics", "vibration",
            "resonance", "amplitude", "harmonics", "audio", "hearing"
        ]):
            collaboration_subjects.append(SubjectType.MUSIC)
        
        return len(collaboration_subjects) > 0, collaboration_subjects
    
    async def suggest_experiment(self, topic: str, difficulty: DifficultyLevel) -> str:
        """
        Suggest a safe, educational experiment for a specific topic.
        
        Args:
            topic: The scientific topic
            difficulty: The appropriate difficulty level
            
        Returns:
            Experiment suggestion with safety notes
        """
        try:
            prompt = f"""
Suggest a safe, hands-on experiment or activity for {topic} appropriate for {difficulty.value} level students.
Include:
1. Materials needed (common household items preferred)
2. Step-by-step procedure
3. What to observe
4. Scientific explanation of what happens
5. Important safety considerations
6. Extensions or variations

Ensure all suggestions prioritize student safety and use readily available materials.
"""
            
            response, _ = await self.generate_response(prompt)
            return response
            
        except Exception as e:
            logger.error(f"Error suggesting experiment: {str(e)}")
            return f"I'm having trouble suggesting an experiment for {topic} right now. Please try again later."
    
    async def explain_phenomenon(self, phenomenon: str) -> str:
        """
        Provide a detailed explanation of a natural phenomenon.
        
        Args:
            phenomenon: The natural phenomenon to explain
            
        Returns:
            Detailed scientific explanation
        """
        try:
            prompt = f"""
Explain the natural phenomenon: {phenomenon}

Include:
1. What exactly happens during this phenomenon
2. The underlying scientific principles
3. Where and when it typically occurs
4. Why it's scientifically significant
5. Any interesting facts or connections to other phenomena
6. How scientists study or measure it

Make the explanation engaging and accessible while maintaining scientific accuracy.
"""
            
            response, _ = await self.generate_response(prompt)
            return response
            
        except Exception as e:
            logger.error(f"Error explaining phenomenon: {str(e)}")
            return f"I'm having trouble explaining {phenomenon} right now. Please try again later."
