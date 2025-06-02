#!/usr/bin/env python3
"""
Simple demonstration of the multi-agent educational system.
This example shows how each specialized agent works and how they collaborate.
"""

import asyncio
import os
from pathlib import Path
import sys

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adk_educator.config import Config
from adk_educator.core.coordinator import AgentCoordinator
from adk_educator.agents.math_agent import MathAgent
from adk_educator.agents.science_agent import ScienceAgent
from adk_educator.agents.music_agent import MusicAgent


async def demo_individual_agents():
    """Demonstrate each agent working individually."""
    print("🎓 Multi-Agent Educational System Demo")
    print("=" * 50)
    
    # Initialize configuration
    config = Config()
    
    # Create individual agents
    math_agent = MathAgent(config)
    science_agent = ScienceAgent(config)
    music_agent = MusicAgent(config)
    
    print("\n📐 Math Agent Demo:")
    print("-" * 30)
    math_question = "How do I solve quadratic equations?"
    print(f"Question: {math_question}")
    try:
        math_response = await math_agent.process_query(math_question, "demo_student")
        print(f"Math Agent: {math_response[:200]}...")
    except Exception as e:
        print(f"Math Agent (Demo): Here's how to solve quadratic equations step by step:\n"
              f"1. Standard form: ax² + bx + c = 0\n"
              f"2. Use the quadratic formula: x = (-b ± √(b²-4ac)) / 2a\n"
              f"3. Example: x² + 5x + 6 = 0 → x = (-5 ± √(25-24)) / 2 = (-5 ± 1) / 2\n"
              f"4. Solutions: x = -2 or x = -3")
    
    print("\n🔬 Science Agent Demo:")
    print("-" * 30)
    science_question = "Why do plants need sunlight?"
    print(f"Question: {science_question}")
    try:
        science_response = await science_agent.process_query(science_question, "demo_student")
        print(f"Science Agent: {science_response[:200]}...")
    except Exception as e:
        print(f"Science Agent (Demo): Plants need sunlight for photosynthesis!\n"
              f"🌱 Process: 6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂\n"
              f"🔋 Sunlight provides energy to convert CO₂ and water into glucose\n"
              f"🧪 Experiment: Cover a plant leaf with foil and see what happens!")
    
    print("\n🎵 Music Agent Demo:")
    print("-" * 30)
    music_question = "What are the basic elements of music?"
    print(f"Question: {music_question}")
    try:
        music_response = await music_agent.process_query(music_question, "demo_student")
        print(f"Music Agent: {music_response[:200]}...")
    except Exception as e:
        print(f"Music Agent (Demo): The basic elements of music are:\n"
              f"🎼 Melody - the main tune (horizontal aspect)\n"
              f"🎵 Rhythm - patterns of time and beats\n"
              f"🎶 Harmony - chords and vertical combinations\n"
              f"🔊 Dynamics - loudness and softness\n"
              f"🎨 Timbre - the unique sound quality of instruments")


async def demo_collaboration():
    """Demonstrate agents working together on interdisciplinary questions."""
    print("\n\n🤝 Agent Collaboration Demo:")
    print("=" * 50)
    
    config = Config()
    coordinator = AgentCoordinator(config)
    
    # Example of a question that requires multiple subjects
    interdisciplinary_question = "How does sound frequency relate to musical notes and mathematical patterns?"
    print(f"Complex Question: {interdisciplinary_question}")
    
    try:
        collaborative_response = await coordinator.process_query(
            interdisciplinary_question, 
            "demo_student"
        )
        print(f"\nCollaborative Response:")
        print(collaborative_response)
    except Exception as e:
        print(f"\nCollaborative Response (Demo):")
        print(f"🎵 Music Agent: Musical notes have specific frequencies (A4 = 440 Hz)")
        print(f"📐 Math Agent: Frequencies follow mathematical ratios - octaves double (220 → 440 Hz)")
        print(f"🔬 Science Agent: Sound waves vibrate at these frequencies, creating pitch perception")
        print(f"🤝 Combined: Music theory + physics + mathematics = understanding sound!")


async def demo_adaptive_learning():
    """Demonstrate how the system adapts to student level."""
    print("\n\n🎯 Adaptive Learning Demo:")
    print("=" * 50)
    
    config = Config()
    math_agent = MathAgent(config)
    
    # Simulate beginner level
    print("👶 Beginner Level Question:")
    beginner_question = "What is 2 + 2?"
    print(f"Question: {beginner_question}")
    try:
        beginner_response = await math_agent.process_query(beginner_question, "beginner_student")
        print(f"Response: {beginner_response[:150]}...")
    except Exception as e:
        print(f"Response (Demo): Great question! 2 + 2 = 4. Let's use visual aids: 🍎🍎 + 🍎🍎 = 🍎🍎🍎🍎")
    
    # Simulate advanced level
    print("\n🎓 Advanced Level Question:")
    advanced_question = "Explain the relationship between derivatives and integrals"
    print(f"Question: {advanced_question}")
    try:
        advanced_response = await math_agent.process_query(advanced_question, "advanced_student")
        print(f"Response: {advanced_response[:150]}...")
    except Exception as e:
        print(f"Response (Demo): The Fundamental Theorem of Calculus shows that differentiation and integration are inverse operations. If F'(x) = f(x), then ∫f(x)dx = F(x) + C...")


def main():
    """Run the complete demonstration."""
    print("Setting up Multi-Agent Educational System...")
    
    # Check if API key is set
    if not os.getenv("GOOGLE_API_KEY"):
        print("\n⚠️  Warning: GOOGLE_API_KEY not found in environment.")
        print("Please add your Google AI API key to the .env file:")
        print("GOOGLE_API_KEY=your_api_key_here")
        print("\nRunning demo with simulated responses...\n")
    
    # Run the demonstration
    asyncio.run(demo_individual_agents())
    asyncio.run(demo_collaboration())
    asyncio.run(demo_adaptive_learning())
    
    print("\n\n✨ Demo Complete!")
    print("=" * 50)
    print("🎓 Your multi-agent educational system includes:")
    print("   📐 Math Agent - Specialized in mathematics education")
    print("   🔬 Science Agent - Expert in scientific concepts")
    print("   🎵 Music Agent - Focused on musical theory and practice")
    print("   🤝 Coordinator - Orchestrates collaboration between agents")
    print("   💾 Session Manager - Tracks learning progress")
    print("   🖥️  CLI Interface - Interactive command-line tool")
    print("   🌐 Web API - REST endpoints for web integration")
    
    print("\n🚀 To run the interactive CLI:")
    print("   python -m adk_educator.cli")
    
    print("\n🌐 To start the web API:")
    print("   uvicorn adk_educator.api:app --reload")


if __name__ == "__main__":
    main()
