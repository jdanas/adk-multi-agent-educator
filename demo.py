#!/usr/bin/env python3
"""
Simple demonstration of the ADK Multi-Agent Educator system.

This script shows how to use each specialized agent and demonstrates
collaboration between agents for interdisciplinary questions.
"""

import asyncio
import os
from src.adk_educator.agents.math_agent import MathAgent
from src.adk_educator.agents.science_agent import ScienceAgent
from src.adk_educator.agents.music_agent import MusicAgent
from src.adk_educator.config import StudentRequest, SubjectType, DifficultyLevel


async def demo_individual_agents():
    """Demonstrate each agent working on their specialty."""
    print("🎓 === Individual Agent Demonstrations ===\n")
    
    # Math Agent Demo
    print("📐 MATH AGENT - Solving a quadratic equation:")
    math_agent = MathAgent()
    math_request = StudentRequest(
        session_id="demo_session",
        student_id="Alex",
        subject=SubjectType.MATH,
        topic="quadratic equations",
        difficulty=DifficultyLevel.HIGH,
        specific_question="How do I solve x² + 5x + 6 = 0?"
    )
    
    try:
        math_response = await math_agent.process_request(math_request)
        print(f"Response: {math_response.response_text[:300]}...")
        print(f"Confidence: {math_response.confidence_score:.1%}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Science Agent Demo
    print("🔬 SCIENCE AGENT - Explaining photosynthesis:")
    science_agent = ScienceAgent()
    science_request = StudentRequest(
        session_id="demo_session",
        student_id="Alex",
        subject=SubjectType.SCIENCE,
        topic="biology",
        difficulty=DifficultyLevel.MIDDLE,
        specific_question="How does photosynthesis work in plants?"
    )
    
    try:
        science_response = await science_agent.process_request(science_request)
        print(f"Response: {science_response.response_text[:300]}...")
        print(f"Confidence: {science_response.confidence_score:.1%}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Music Agent Demo
    print("🎵 MUSIC AGENT - Music theory basics:")
    music_agent = MusicAgent()
    music_request = StudentRequest(
        session_id="demo_session",
        student_id="Alex",
        subject=SubjectType.MUSIC,
        topic="music theory",
        difficulty=DifficultyLevel.ELEMENTARY,
        specific_question="What's the difference between major and minor scales?"
    )
    
    try:
        music_response = await music_agent.process_request(music_request)
        print(f"Response: {music_response.response_text[:300]}...")
        print(f"Confidence: {music_response.confidence_score:.1%}\n")
    except Exception as e:
        print(f"Error: {e}\n")


async def main():
    """Main demonstration function."""
    print("🚀 Welcome to the ADK Multi-Agent Educator Demo!\n")
    
    # Check if API key is set
    if not os.getenv('GOOGLE_API_KEY'):
        print("⚠️  Please set your GOOGLE_API_KEY in the .env file first!")
        print("   Get your API key from: https://makersuite.google.com/app/apikey")
        return
    
    await demo_individual_agents()
    
    print("✨ Demo completed! Each agent specializes in their subject area:")
    print("   📐 Math Agent: Problem-solving, step-by-step explanations")
    print("   🔬 Science Agent: Scientific concepts, experiments, research")
    print("   🎵 Music Agent: Theory, history, cultural context")
    print("\n💡 Try the full CLI interface: python -m src.adk_educator.cli")


if __name__ == "__main__":
    asyncio.run(main())
