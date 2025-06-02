#!/usr/bin/env python3
"""
Multi-Agent Routing Demonstration

This script shows exactly how the ADK integration routes questions 
to specialized agents and demonstrates the multi-agent coordination.
"""

import sys
import os
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_multi_agent_routing():
    """Demonstrate how different types of questions get routed to different agents"""
    print("🎓 Multi-Agent Educational System Routing Demo")
    print("=" * 60)
    
    from src.root_agent import process_educational_query
    
    test_questions = [
        ("Math Question", "What is the derivative of x^2 + 3x?"),
        ("Science Question", "How does photosynthesis work?"),
        ("Music Question", "What are the notes in a C major scale?"),
        ("Interdisciplinary", "How is math used in music composition?"),
        ("General Greeting", "Hello, what can you help me with?")
    ]
    
    for category, question in test_questions:
        print(f"\n🔍 **{category}**")
        print(f"Question: '{question}'")
        print("-" * 40)
        
        try:
            # This simulates what happens when ADK calls our tool
            response = process_educational_query(question)
            print(response)
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "=" * 60)

def test_adk_agent_setup():
    """Test that the ADK agent is properly configured as a router"""
    print("\n🔧 ADK Agent Configuration Test")
    print("=" * 40)
    
    try:
        from src.root_agent import create_adk_agent
        
        agent = create_adk_agent()
        print(f"✅ Agent Name: {agent.name}")
        print(f"✅ Model: {agent.model}")
        print(f"✅ Number of Tools: {len(agent.tools)}")
        print(f"✅ Tool Name: {agent.tools[0].__name__}")
        
        print("\n📋 Agent Instructions (showing routing behavior):")
        print("-" * 40)
        instruction_preview = agent.instruction[:300] + "..." if len(agent.instruction) > 300 else agent.instruction
        print(instruction_preview)
        
        # Check if instruction emphasizes routing
        if "DO NOT attempt to answer" in agent.instruction and "ALWAYS use the process_educational_query tool" in agent.instruction:
            print("\n✅ Agent is properly configured as a ROUTER (will not answer directly)")
        else:
            print("\n⚠️  Agent might try to answer directly instead of routing")
            
    except Exception as e:
        print(f"❌ Error setting up ADK agent: {e}")

def show_architecture():
    """Show the system architecture"""
    print("\n🏗️  MULTI-AGENT SYSTEM ARCHITECTURE")
    print("=" * 50)
    print("""
📱 ADK Web Interface
    ↓ (user question)
🤖 ADK Agent (Gemini Router)
    ↓ (calls tool)
🔧 process_educational_query()
    ↓ (routes based on content)
🎓 RootAgent.process_message()
    ↓ (subject detection & routing)
    
    ├─ 🔢 Professor Mathematics (Math questions)
    ├─ 🔬 Dr. Science Explorer (Science questions)  
    ├─ 🎵 Maestro Harmony (Music questions)
    └─ 🔄 Multi-Agent Coordinator (Complex/interdisciplinary)
    
    ↑ (specialized response)
🔙 Response flows back through tool to ADK to user
""")
    
    print("\n🎯 ROUTING LOGIC:")
    print("• Question analysis → Subject detection")
    print("• Math keywords → Professor Mathematics")
    print("• Science keywords → Dr. Science Explorer") 
    print("• Music keywords → Maestro Harmony")
    print("• Mixed/unclear → All agents collaborate")
    print("• No keywords → Multi-agent coordinator")

if __name__ == "__main__":
    show_architecture()
    test_adk_agent_setup() 
    test_multi_agent_routing()
    
    print("\n🚀 **Next Steps:**")
    print("1. Run: adk web --port 8001")
    print("2. Go to: http://localhost:8001")
    print("3. Select 'src' as the agent")
    print("4. Ask educational questions and watch the routing!")
    print("\n💡 **Try these test questions:**")
    print("• 'What is 2+2?' (should route to Math)")
    print("• 'How do plants grow?' (should route to Science)")
    print("• 'What is a C chord?' (should route to Music)")
    print("• 'How is math used in music?' (should use multi-agent coordination)")
