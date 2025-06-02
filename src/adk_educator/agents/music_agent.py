"""
Music Agent for the ADK Multi-Agent Educator system.

This agent specializes in music education, covering music theory, history,
composition, performance, and the science of sound. It provides musical
insights, creative guidance, and connections to cultural contexts.
"""

from typing import List, Optional
from loguru import logger

from ..config import (
    AgentConfig, StudentRequest, AgentResponse, SubjectType, 
    DifficultyLevel, MUSIC_SPECIALIZATIONS
)
from ..core.agent_base import BaseAgent


class MusicAgent(BaseAgent):
    """
    Specialized agent for music education.
    
    The MusicAgent provides comprehensive music support including:
    - Music theory and analysis
    - Historical and cultural contexts
    - Composition and creative techniques
    - Performance guidance and techniques
    - Acoustic and scientific aspects of music
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize the Music Agent.
        
        Args:
            config: Optional configuration. If not provided, uses defaults.
        """
        if config is None:
            config = AgentConfig(
                name="Maestro Harmony",
                subject=SubjectType.MUSIC,
                specializations=MUSIC_SPECIALIZATIONS,
                difficulty_levels=[
                    DifficultyLevel.ELEMENTARY,
                    DifficultyLevel.MIDDLE,
                    DifficultyLevel.HIGH,
                    DifficultyLevel.COLLEGE
                ]
            )
        
        super().__init__(config)
    
    async def get_system_prompt(self) -> str:
        """Get the system prompt defining the Music Agent's role and expertise."""
        return """You are Maestro Harmony, an inspiring music educator who brings the world of music to life for students of all levels. Your passion is to:

1. **Demystify Music Theory**: Make musical concepts accessible and practical
2. **Celebrate Musical Diversity**: Explore music from all cultures and time periods
3. **Foster Creativity**: Encourage original musical expression and composition
4. **Connect Music to Life**: Show how music relates to emotions, culture, and other subjects
5. **Develop Musical Skills**: Guide students in developing listening, performance, and analytical abilities
6. **Inspire Musical Discovery**: Cultivate lifelong appreciation and engagement with music

**Your Areas of Expertise:**
- **Music Theory**: Scales, chords, harmony, rhythm, form, analysis
- **Music History**: Classical, jazz, folk, world music, contemporary genres
- **Composition**: Songwriting, arranging, creative techniques, musical storytelling
- **Performance**: Instrumental and vocal techniques, ensemble playing, stage presence
- **Acoustics**: The science of sound, instruments, recording, audio technology
- **Music Technology**: Digital audio, notation software, electronic instruments
- **Cultural Context**: Music's role in society, cultural traditions, musical meaning

**Teaching Philosophy:**
- Every person has musical potential waiting to be discovered
- Music learning should be joyful and personally meaningful
- Theory serves creativity, not the other way around
- Listen first, analyze second
- Connect new concepts to familiar musical experiences
- Encourage experimentation and personal expression
- Value all musical styles and traditions equally

**When Teaching Music:**
1. Start with the student's musical interests and experiences
2. Use familiar songs and styles as examples when possible
3. Encourage active listening and musical exploration
4. Break complex concepts into manageable steps
5. Provide both theoretical understanding and practical application
6. Connect musical elements to emotions and personal expression
7. Suggest listening examples and hands-on activities

**Remember**: Music is a universal language that connects us all. Every musical question is an opportunity to explore this beautiful art form!"""

    async def process_subject_specific_request(self, request: StudentRequest) -> AgentResponse:
        """
        Process a music-specific student request.
        
        Args:
            request: The student's music question
            
        Returns:
            Detailed musical response with explanations
        """
        try:
            # Get session context for continuity
            context = self.get_session_context(request.session_id)
            
            # Enhance the prompt with music-specific guidance
            enhanced_prompt = self._create_music_prompt(request, context)
            
            # Generate response using the AI model
            response_text, confidence = await self.generate_response(enhanced_prompt, context)
            
            # Generate music-specific follow-ups and resources
            follow_ups = self._generate_music_follow_ups(request)
            resources = self._generate_music_resources(request)
            
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
            logger.error(f"Error in MusicAgent processing: {str(e)}")
            return AgentResponse(
                agent_name=self.name,
                subject=self.subject,
                response_text="I encountered an error while exploring this musical concept. Could you please rephrase your question or provide more details about what you'd like to learn?",
                confidence_score=0.1
            )
    
    def _create_music_prompt(self, request: StudentRequest, context: str) -> str:
        """Create an enhanced prompt for musical explanations."""
        difficulty_guidance = ""
        if request.difficulty:
            difficulty_map = {
                DifficultyLevel.ELEMENTARY: "Use simple musical terms. Focus on listening and basic concepts like loud/soft, fast/slow, high/low.",
                DifficultyLevel.MIDDLE: "Introduce standard musical vocabulary. Include basic theory concepts and familiar songs as examples.",
                DifficultyLevel.HIGH: "Use proper musical terminology. Include more advanced theory and analysis techniques.",
                DifficultyLevel.COLLEGE: "Provide sophisticated musical analysis with advanced theoretical concepts and historical context."
            }
            difficulty_guidance = difficulty_map.get(request.difficulty, "")
        
        prompt = f"""
Musical Question: {request.specific_question}
Topic Area: {request.topic}
{f"Difficulty Level: {request.difficulty.value} - {difficulty_guidance}" if request.difficulty else ""}
{f"Additional Context: {request.context}" if request.context else ""}

Please provide a comprehensive musical response that includes:
1. Clear explanation of the musical concept or question
2. Relevant music theory or historical information
3. Practical examples using familiar songs or pieces when possible
4. Connections to musical expression and creativity
5. Listening suggestions or musical activities when appropriate
6. Cultural or historical context if relevant

{f"Recent conversation context: {context}" if context else ""}
"""
        return prompt.strip()
    
    def _generate_music_follow_ups(self, request: StudentRequest) -> List[str]:
        """Generate music-specific follow-up questions."""
        topic_lower = request.topic.lower()
        follow_ups = []
        
        # General follow-ups
        follow_ups.extend([
            "Would you like to explore this concept with a specific song example?",
            "Can I suggest some listening activities to practice this?",
            "Would you like to learn about the cultural background of this music?"
        ])
        
        # Theory follow-ups
        if any(keyword in topic_lower for keyword in ["theory", "chord", "scale", "harmony", "melody"]):
            follow_ups.extend([
                "Would you like to hear how this sounds in different musical styles?",
                "Should we explore how to use this in composition?",
                "Can I show you how to identify this in songs you know?"
            ])
        
        # History follow-ups
        elif any(keyword in topic_lower for keyword in ["history", "classical", "jazz", "folk", "composer"]):
            follow_ups.extend([
                "Would you like to learn about other composers from this era?",
                "Should we explore how this style influenced modern music?",
                "Can we listen to some representative pieces together?"
            ])
        
        # Performance follow-ups
        elif any(keyword in topic_lower for keyword in ["performance", "instrument", "singing", "technique"]):
            follow_ups.extend([
                "Would you like some practice exercises for this technique?",
                "Should we discuss how to prepare for performance?",
                "Can I suggest ways to overcome performance anxiety?"
            ])
        
        # Composition follow-ups
        elif any(keyword in topic_lower for keyword in ["composition", "songwriting", "creativity", "arrangement"]):
            follow_ups.extend([
                "Would you like to try a composition exercise?",
                "Should we explore different song forms and structures?",
                "Can I suggest some creative constraints to inspire new ideas?"
            ])
        
        return follow_ups[:5]
    
    def _generate_music_resources(self, request: StudentRequest) -> List[str]:
        """Generate music-specific learning resources."""
        topic_lower = request.topic.lower()
        resources = [
            "Spotify/Apple Music - Listen to examples and create playlists",
            "YouTube - Music theory tutorials and performance videos",
            "IMSLP - Free sheet music library",
            "Coursera Music Courses - Structured music education"
        ]
        
        # Theory resources
        if any(keyword in topic_lower for keyword in ["theory", "chord", "scale", "harmony"]):
            resources.extend([
                "Tenuto (app) - Music theory practice and ear training",
                "musictheory.net - Interactive music theory lessons"
            ])
        
        # History resources
        elif any(keyword in topic_lower for keyword in ["history", "classical", "composer"]):
            resources.extend([
                "Classical Music Archives - Extensive classical music database",
                "AllMusic - Comprehensive music history and biographies"
            ])
        
        # Performance resources
        elif any(keyword in topic_lower for keyword in ["performance", "instrument", "technique"]):
            resources.extend([
                "SmartMusic - Interactive music practice software",
                "Metronome apps - Tempo and rhythm practice"
            ])
        
        # Composition resources
        elif any(keyword in topic_lower for keyword in ["composition", "songwriting"]):
            resources.extend([
                "MuseScore - Free music notation software",
                "GarageBand/Logic Pro - Digital audio workstations"
            ])
        
        return resources[:7]
    
    def _assess_collaboration_needs(self, request: StudentRequest) -> tuple[bool, List[SubjectType]]:
        """Assess if collaboration with other subjects would be beneficial."""
        query_lower = request.specific_question.lower()
        collaboration_subjects = []
        
        # Math connections
        if any(keyword in query_lower for keyword in [
            "frequency", "ratio", "proportion", "pattern", "sequence", "calculation",
            "measure", "time signature", "rhythm", "mathematical", "geometry", "fibonacci"
        ]):
            collaboration_subjects.append(SubjectType.MATH)
        
        # Science connections
        if any(keyword in query_lower for keyword in [
            "sound", "wave", "vibration", "acoustics", "physics", "frequency",
            "amplitude", "resonance", "instrument construction", "hearing", "brain"
        ]):
            collaboration_subjects.append(SubjectType.SCIENCE)
        
        return len(collaboration_subjects) > 0, collaboration_subjects
    
    async def analyze_song(self, song_title: str, artist: Optional[str] = None) -> str:
        """
        Provide musical analysis of a specific song.
        
        Args:
            song_title: The title of the song to analyze
            artist: Optional artist name for more specific identification
            
        Returns:
            Musical analysis of the song
        """
        try:
            artist_info = f" by {artist}" if artist else ""
            prompt = f"""
Provide a musical analysis of "{song_title}"{artist_info}. Include:

1. Key signature and basic harmonic structure
2. Song form and structure (verse, chorus, bridge, etc.)
3. Rhythmic elements and time signature
4. Melodic characteristics
5. Instrumentation and arrangement
6. Style and genre characteristics
7. Notable musical techniques or innovations
8. Cultural or historical significance

Make the analysis accessible to music students while maintaining musical accuracy.
"""
            
            response, _ = await self.generate_response(prompt)
            return response
            
        except Exception as e:
            logger.error(f"Error analyzing song: {str(e)}")
            return f"I'm having trouble analyzing '{song_title}' right now. Please try again later."
    
    async def suggest_composition_exercise(self, style: str, difficulty: DifficultyLevel) -> str:
        """
        Suggest a composition or creative exercise.
        
        Args:
            style: The musical style or genre
            difficulty: The appropriate difficulty level
            
        Returns:
            Composition exercise suggestion
        """
        try:
            prompt = f"""
Create a {difficulty.value}-level composition exercise in the {style} style. Include:

1. Clear objectives for the exercise
2. Musical parameters and constraints
3. Step-by-step guidance
4. Examples of techniques to use
5. Tips for getting started
6. Ways to expand or modify the exercise
7. Listening examples for inspiration

Make it engaging and achievable for the specified difficulty level.
"""
            
            response, _ = await self.generate_response(prompt)
            return response
            
        except Exception as e:
            logger.error(f"Error suggesting composition exercise: {str(e)}")
            return f"I'm having trouble creating a {style} composition exercise right now. Please try again later."
    
    async def explain_musical_concept(self, concept: str, use_examples: bool = True) -> str:
        """
        Provide a detailed explanation of a musical concept.
        
        Args:
            concept: The musical concept to explain
            use_examples: Whether to include musical examples
            
        Returns:
            Detailed explanation of the musical concept
        """
        try:
            examples_text = "Include specific musical examples from well-known songs." if use_examples else ""
            
            prompt = f"""
Explain the musical concept: {concept}

Include:
1. Clear definition and explanation
2. How it functions in music
3. Different types or variations
4. Historical development or origin
5. How to recognize it when listening
6. Its role in different musical styles
{examples_text}

Make the explanation clear and engaging for music students.
"""
            
            response, _ = await self.generate_response(prompt)
            return response
            
        except Exception as e:
            logger.error(f"Error explaining musical concept: {str(e)}")
            return f"I'm having trouble explaining '{concept}' right now. Please try again later."
