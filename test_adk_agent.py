#!/usr/bin/env python3
"""
Test script to verify ADK Agent compatibility
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_adk_agent_creation():
    """Test creating an ADK-compatible agent"""
    print("Testing ADK Agent creation...")
    
    try:
        from google.adk import Agent
        print("✅ google-adk successfully imported")
        
        # Test our agent creation
        from src.root_agent import create_adk_agent, process_educational_query
        
        print("✅ Agent creation function imported")
        
        # Create the agent
        agent = create_adk_agent()
        print(f"✅ Agent created: {type(agent)}")
        print(f"   Name: {agent.name}")
        print(f"   Description: {agent.description[:100]}...")
        print(f"   Has tools: {len(agent.tools) > 0}")
        
        # Test the tool function directly
        print("\nTesting tool function...")
        response = process_educational_query("What is 2 + 2?")
        print(f"✅ Tool function works: {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_adk_discovery():
    """Test different ways ADK might discover our agent"""
    print("\nTesting ADK discovery patterns...")
    
    discovery_tests = [
        ("src.root_agent", "root_agent"),
        ("src.agent.root_agent", "root_agent"), 
        ("src.agent", "agent"),
        ("src", "root_agent"),
        ("src", "agent")
    ]
    
    for module_path, attr_name in discovery_tests:
        try:
            module = __import__(module_path, fromlist=[attr_name])
            agent = getattr(module, attr_name)
            print(f"✅ {module_path}.{attr_name} -> {type(agent)}")
            
            # If it's an ADK Agent, test basic properties
            if hasattr(agent, 'name') and hasattr(agent, 'tools'):
                print(f"   Agent name: {agent.name}")
                print(f"   Has {len(agent.tools)} tools")
            
        except Exception as e:
            print(f"❌ {module_path}.{attr_name} -> {e}")

def test_validation():
    """Test that our agent passes ADK validation"""
    print("\nTesting ADK validation...")
    
    try:
        from google.adk import Agent
        from src.root_agent import process_educational_query
        
        # Test minimal agent creation
        minimal_agent = Agent(name="test_agent")
        print("✅ Minimal agent creation works")
        
        # Test our agent with required fields
        our_agent = Agent(
            name="multi_agent_educator",
            description="Test educational agent",
            tools=[process_educational_query]
        )
        print("✅ Our agent creation works")
        print(f"   Name: {our_agent.name}")
        print(f"   Valid identifier: {our_agent.name.isidentifier()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("ADK Agent Compatibility Test")
    print("=" * 50)
    
    success = True
    success &= test_adk_agent_creation()
    success &= test_validation()
    test_adk_discovery()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All critical tests passed!")
    else:
        print("❌ Some tests failed")
    
    print("\nTo test with ADK web interface:")
    print("1. Run: adk dev")
    print("2. Select 'src' as the agent")
    print("3. Test with educational queries")
