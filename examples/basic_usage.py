"""
Example usage and demonstration of the ADK Multi-Agent Educator system.

This module provides examples of how to use the multi-agent educational system
programmatically, showcasing different types of questions and interactions.
"""

import asyncio
from typing import List
from src.adk_educator.core.session_manager import SessionManager
from src.adk_educator.core.coordinator import AgentCoordinator
from src.adk_educator.agents.math_agent import MathAgent
from src.adk_educator.agents.science_agent import ScienceAgent
from src.adk_educator.agents.music_agent import MusicAgent
from src.adk_educator.config import StudentRequest, SubjectType, DifficultyLevel


async def demonstrate_single_subject_questions():
    """Demonstrate asking questions to individual subject experts."""
    print("🎓 === Single Subject Question Examples ===\n")
    
    # Create session manager and register agents
    session_manager = SessionManager()
    math_agent = MathAgent()
    science_agent = ScienceAgent()
    music_agent = MusicAgent()
    
    session_manager.register_agent(math_agent)
    session_manager.register_agent(science_agent) 
    session_manager.register_agent(music_agent)
    
    # Create a student session
    session_id = session_manager.create_session("Emma")
    
    # Math question
    print("📐 MATH QUESTION:")
    math_request = StudentRequest(
        session_id=session_id,
        student_id="Emma",
        subject=SubjectType.MATH,
        topic="quadratic equations",
        difficulty=DifficultyLevel.HIGH,
        specific_question="How do I solve x² + 5x + 6 = 0 using the quadratic formula?"
    )
    
    math_response = await session_manager.process_student_request(math_request)
    print(f"Agent: {math_response.agent_name}")
    print(f"Response: {math_response.response_text[:200]}...")
    print(f"Confidence: {math_response.confidence_score:.1%}")
    print(f"Follow-ups: {math_response.suggested_follow_ups[:2]}")
    print()
    
    # Science question
    print("🔬 SCIENCE QUESTION:")
    science_request = StudentRequest(
        session_id=session_id,
        student_id="Emma",
        subject=SubjectType.SCIENCE,
        topic="photosynthesis",
        difficulty=DifficultyLevel.MIDDLE,
        specific_question="How do plants convert sunlight into energy through photosynthesis?"
    )
    
    science_response = await session_manager.process_student_request(science_request)
    print(f"Agent: {science_response.agent_name}")
    print(f"Response: {science_response.response_text[:200]}...")
    print(f"Confidence: {science_response.confidence_score:.1%}")
    print(f"Follow-ups: {science_response.suggested_follow_ups[:2]}")
    print()
    
    # Music question
    print("🎵 MUSIC QUESTION:")
    music_request = StudentRequest(
        session_id=session_id,
        student_id="Emma",
        subject=SubjectType.MUSIC,
        topic="music theory",
        difficulty=DifficultyLevel.ELEMENTARY,
        specific_question="What's the difference between major and minor scales in music?"
    )
    
    music_response = await session_manager.process_student_request(music_request)
    print(f"Agent: {music_response.agent_name}")
    print(f"Response: {music_response.response_text[:200]}...")
    print(f"Confidence: {music_response.confidence_score:.1%}")
    print(f"Follow-ups: {music_response.suggested_follow_ups[:2]}")
    print()


async def demonstrate_interdisciplinary_questions():
    """Demonstrate cross-subject collaboration between agents."""
    print("🌐 === Interdisciplinary Question Examples ===\n")
    
    # Setup
    session_manager = SessionManager()
    coordinator = AgentCoordinator(session_manager)
    
    math_agent = MathAgent()
    science_agent = ScienceAgent()
    music_agent = MusicAgent()
    
    session_manager.register_agent(math_agent)
    session_manager.register_agent(science_agent)
    session_manager.register_agent(music_agent)
    
    session_id = session_manager.create_session("Alex")
    
    # Math + Science question
    print("📐🔬 MATH + SCIENCE COLLABORATION:")
    request1 = StudentRequest(
        session_id=session_id,
        student_id="Alex",
        subject=SubjectType.MATH,
        topic="physics calculations",
        difficulty=DifficultyLevel.HIGH,
        specific_question="If I drop a ball from a height of 100 meters, how fast will it be going when it hits the ground? Include both the physics explanation and the mathematical calculation."
    )
    
    # Let coordinator handle the multi-subject question
    relevant_subjects = coordinator.identify_relevant_subjects(request1.specific_question)
    print(f"Identified subjects: {[s.value for s in relevant_subjects]}")
    
    response1 = await coordinator.process_complex_query(request1, relevant_subjects)
    print(f"Coordinated Response: {response1.response_text[:300]}...")
    print(f"Confidence: {response1.confidence_score:.1%}")
    print()
    
    # Music + Science question
    print("🎵🔬 MUSIC + SCIENCE COLLABORATION:")
    request2 = StudentRequest(
        session_id=session_id,
        student_id="Alex",
        subject=SubjectType.MUSIC,
        topic="acoustics",
        difficulty=DifficultyLevel.MIDDLE,
        specific_question="How does the frequency of sound waves relate to musical pitch? Can you explain both the science and the musical aspects?"
    )
    
    relevant_subjects = coordinator.identify_relevant_subjects(request2.specific_question)
    print(f"Identified subjects: {[s.value for s in relevant_subjects]}")
    
    response2 = await coordinator.process_complex_query(request2, relevant_subjects)
    print(f"Coordinated Response: {response2.response_text[:300]}...")
    print(f"Confidence: {response2.confidence_score:.1%}")
    print()


async def demonstrate_adaptive_difficulty():
    """Demonstrate how agents adapt to different difficulty levels."""
    print("📊 === Adaptive Difficulty Examples ===\n")
    
    session_manager = SessionManager()
    math_agent = MathAgent()
    session_manager.register_agent(math_agent)
    
    session_id = session_manager.create_session("Jordan")
    
    # Same concept at different difficulty levels
    base_question = "What is a derivative in calculus?"
    
    difficulty_levels = [
        DifficultyLevel.MIDDLE,
        DifficultyLevel.HIGH,
        DifficultyLevel.COLLEGE
    ]
    
    for difficulty in difficulty_levels:
        print(f"📈 {difficulty.value.upper()} LEVEL:")
        request = StudentRequest(
            session_id=session_id,
            student_id="Jordan",
            subject=SubjectType.MATH,
            topic="calculus",
            difficulty=difficulty,
            specific_question=base_question
        )
        
        response = await session_manager.process_student_request(request)
        print(f"Response: {response.response_text[:250]}...")
        print(f"Confidence: {response.confidence_score:.1%}")
        print()


async def demonstrate_session_tracking():
    """Demonstrate session tracking and learning analytics."""
    print("📈 === Session Tracking Examples ===\n")
    
    session_manager = SessionManager()
    math_agent = MathAgent()
    science_agent = ScienceAgent()
    music_agent = MusicAgent()
    
    session_manager.register_agent(math_agent)
    session_manager.register_agent(science_agent)
    session_manager.register_agent(music_agent)
    
    # Create session and ask several questions
    session_id = session_manager.create_session("Taylor")
    
    questions = [
        ("What is 2 + 2?", SubjectType.MATH, "arithmetic"),
        ("How does gravity work?", SubjectType.SCIENCE, "physics"),
        ("What is a C major chord?", SubjectType.MUSIC, "theory"),
        ("Solve for x: 3x + 7 = 16", SubjectType.MATH, "algebra"),
        ("What causes rainbows?", SubjectType.SCIENCE, "optics")
    ]
    
    for question, subject, topic in questions:
        request = StudentRequest(
            session_id=session_id,
            student_id="Taylor",
            subject=subject,
            topic=topic,
            specific_question=question
        )
        await session_manager.process_student_request(request)
    
    # Show session summary
    summary = session_manager.get_session_summary(session_id)
    print("📊 SESSION SUMMARY:")
    print(f"Student: {summary['student_id']}")
    print(f"Total Interactions: {summary['total_interactions']}")
    print(f"Active Subjects: {summary['active_subjects']}")
    print(f"Duration: {summary['duration_minutes']:.1f} minutes")
    print()
    
    # Show agent capabilities
    print("🤖 AVAILABLE AGENTS:")
    agents_info = session_manager.get_available_agents()
    for agent_name, info in agents_info.items():
        print(f"• {agent_name} ({info['subject']}) - Specializations: {info['specializations'][:3]}")


async def demonstrate_practice_generation():
    """Demonstrate automatic practice problem generation."""
    print("💪 === Practice Generation Examples ===\n")
    
    math_agent = MathAgent()
    science_agent = ScienceAgent()
    
    print("📐 MATH PRACTICE PROBLEM:")
    math_practice = await math_agent.generate_practice_problem("algebra", DifficultyLevel.MIDDLE)
    print(math_practice[:300] + "..." if len(math_practice) > 300 else math_practice)
    print()
    
    print("🔬 SCIENCE EXPERIMENT SUGGESTION:")
    science_experiment = await science_agent.suggest_experiment("density", DifficultyLevel.ELEMENTARY)
    print(science_experiment[:300] + "..." if len(science_experiment) > 300 else science_experiment)
    print()


async def main():
    """Run all demonstration examples."""
    print("🎓 ADK Multi-Agent Educator - Interactive Examples\n")
    print("=" * 60)
    
    try:
        await demonstrate_single_subject_questions()
        await demonstrate_interdisciplinary_questions()
        await demonstrate_adaptive_difficulty()
        await demonstrate_session_tracking()
        await demonstrate_practice_generation()
        
        print("✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running examples: {str(e)}")
        print("💡 Make sure your Google API key is set in the .env file")


if __name__ == "__main__":
    asyncio.run(main())
