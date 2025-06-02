#!/usr/bin/env python3
"""
Test script to verify ADK root agent setup
"""

import sys
import os
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_root_agent():
    """Test the root agent setup."""
    print("🧪 Testing ADK root agent setup...")
    
    try:
        # Test different import paths that ADK might use
        print("\n📦 Testing import paths...")
        
        # Direct import
        from src.root_agent import root_agent
        print("✅ Direct import: src.root_agent")
        
        # Via agent module
        from src.agent.root_agent import root_agent as agent_root
        print("✅ Agent module import: src.agent.root_agent")
        
        # Via main src module
        from src import root_agent as src_root
        print("✅ Main src import: src.root_agent")
        
        print("\n🤖 Testing root agent functionality...")
        
        # Test basic message processing
        response = await root_agent.process_message(
            "What is 2 + 2?",
            {"subject": "math", "difficulty": "elementary"}
        )
        print(f"✅ Math question processed: {len(response)} characters")
        
        # Test capabilities
        capabilities = await root_agent.get_capabilities()
        print(f"✅ Capabilities retrieved: {len(capabilities['agents'])} agents")
        
        print("\n🎉 Root agent setup successful!")
        print("   - All import paths work")
        print("   - Message processing functional")
        print("   - Capabilities accessible")
        print("\n💡 You can now use 'adk web' with 'src' as the agent path")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure you're in the project root directory")
        print("   2. Ensure dependencies are installed: uv pip install -r requirements.txt")
        print("   3. Set up your .env file with GOOGLE_API_KEY")

if __name__ == "__main__":
    asyncio.run(test_root_agent())
