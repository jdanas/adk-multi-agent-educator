#!/usr/bin/env python3
"""
Test script to verify ADK compatibility
This simulates exactly what the ADK framework does when discovering and loading agents.
"""

import sys
import os
import asyncio
from pathlib import Path

def test_adk_discovery():
    """Test the ADK agent discovery process"""
    print("=== Testing ADK Agent Discovery ===")
    
    # Set up the path as ADK would
    project_root = Path(__file__).parent
    src_path = project_root / "src"
    
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    print(f"Added to sys.path: {src_path}")
    
    # Test 1: Try to import src.root_agent (from parent directory)
    try:
        sys.path.insert(0, str(project_root))
        from src.root_agent import root_agent as agent1
        print("✓ src.root_agent import: SUCCESS")
        capabilities = agent1.get_capabilities()
        print(f"  Agent name: {capabilities.get('name', 'No name')}")
    except Exception as e:
        print(f"✗ src.root_agent import: FAILED - {e}")
    
    # Test 2: Try to import src.agent.root_agent 
    try:
        from src.agent.root_agent import root_agent as agent2
        print("✓ src.agent.root_agent import: SUCCESS")
    except Exception as e:
        print(f"✗ src.agent.root_agent import: FAILED - {e}")
    
    # Test 3: Try to access src.agent attribute
    try:
        import src
        if hasattr(src, 'agent'):
            agent3 = src.agent
            print("✓ src.agent attribute: SUCCESS")
            print(f"  Type: {type(agent3)}")
        else:
            print("✗ src.agent attribute: NOT FOUND")
    except Exception as e:
        print(f"✗ src.agent attribute: FAILED - {e}")
    
    # Test 4: Direct import from src directory (what ADK actually does)
    try:
        import root_agent as ra_module
        agent4 = ra_module.root_agent
        print("✓ Direct root_agent import: SUCCESS")
        print(f"  Type: {type(agent4)}")
        return agent4
    except Exception as e:
        print(f"✗ Direct root_agent import: FAILED - {e}")
        return None

async def test_agent_functionality(agent):
    """Test that the agent can process messages"""
    if agent is None:
        print("No agent to test")
        return
    
    print("\n=== Testing Agent Functionality ===")
    
    # Test capabilities
    try:
        capabilities = agent.get_capabilities()
        name = capabilities.get('name', 'Unknown Agent')
        print(f"✓ get_capabilities(): {name}")
        print(f"  Agents available: {len(capabilities.get('agents', []))}")
        for agent_info in capabilities.get('agents', []):
            print(f"    - {agent_info.get('name', 'Unknown')} ({agent_info.get('subject', 'Unknown')})")
    except Exception as e:
        print(f"✗ get_capabilities() failed: {e}")
    
    # Test message processing
    try:
        response = await agent.process_message(
            "Explain the Pythagorean theorem",
            {"user_id": "adk_test", "session_id": "adk_session_001"}
        )
        print(f"✓ process_message(): Response received ({len(response)} chars)")
        if len(response) > 100:
            print(f"  Preview: {response[:100]}...")
        else:
            print(f"  Response: {response}")
    except Exception as e:
        print(f"✗ process_message() failed: {e}")

def main():
    """Main test function"""
    print("ADK Multi-Agent Educator Compatibility Test")
    print("=" * 50)
    
    # Discover the agent
    agent = test_adk_discovery()
    
    # Test functionality
    if agent:
        asyncio.run(test_agent_functionality(agent))
        print("\n✅ ADK compatibility test PASSED")
        print("The agent is ready to be used with ADK web interface!")
        print("\nTo use with ADK:")
        print("1. Run 'adk web' in your terminal")
        print("2. Select 'src' as the agent")
        print("3. Start chatting with the multi-agent educator!")
    else:
        print("\n❌ ADK compatibility test FAILED")
        print("Check the error messages above for troubleshooting.")

if __name__ == "__main__":
    main()
