#!/usr/bin/env python3

"""
Test script to verify the ADK integration provides clean responses without JSON wrappers.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from root_agent import create_adk_agent, MultiAgentEducatorADK

async def test_clean_responses():
    """Test that our ADK agent provides clean responses without JSON wrappers."""
    
    print("🧪 Testing Clean Response Format")
    print("=" * 50)
    
    # Create the ADK agent
    agent = create_adk_agent()
    print(f"✅ Created agent: {type(agent).__name__}")
    
    # Test different types of questions
    test_questions = [
        "What is 2 + 2?",
        "How does photosynthesis work?", 
        "What is a C major scale?",
        "What's the relationship between math and music?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Test {i}: {question}")
        print("-" * 30)
        
        try:
            # Test async interface
            if hasattr(agent, 'run_async'):
                print("🔄 Testing async interface...")
                async for response in agent.run_async(question):
                    print(f"📤 Response Type: {type(response)}")
                    print(f"📤 Response: {response[:100]}...")
                    
                    # Verify it's a clean string, not JSON
                    if isinstance(response, str) and not response.startswith('{"'):
                        print("✅ Clean string response - GOOD")
                    else:
                        print("❌ JSON wrapper detected - NEEDS FIX")
                    break
            
            # Test sync interface
            if hasattr(agent, 'run_live'):
                print("🔄 Testing sync interface...")
                responses = agent.run_live(question)
                if responses and len(responses) > 0:
                    response = responses[0]
                    print(f"📤 Response Type: {type(response)}")
                    print(f"📤 Response: {response[:100]}...")
                    
                    # Verify it's a clean string, not JSON
                    if isinstance(response, str) and not response.startswith('{"'):
                        print("✅ Clean string response - GOOD")
                    else:
                        print("❌ JSON wrapper detected - NEEDS FIX")
                        
        except Exception as e:
            print(f"❌ Error testing question: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Test Summary:")
    print("The agent should provide clean string responses without JSON wrappers.")
    print("This ensures the ADK web interface displays content properly to users.")

if __name__ == "__main__":
    asyncio.run(test_clean_responses())
