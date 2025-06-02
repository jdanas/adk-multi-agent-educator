"""
Agent module for ADK Multi-Agent Educator

Alternative entry point for the ADK framework.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from root_agent import root_agent, RootAgent

# Expose the root agent as 'root_agent' for ADK discovery
# Also expose as 'agent' attribute for additional discovery pattern
agent = root_agent

__all__ = ["root_agent", "RootAgent", "agent"]
