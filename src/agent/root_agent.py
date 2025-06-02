"""
Root agent for ADK Multi-Agent Educator - Direct Entry Point
"""

import sys
import os

# Add the parent directory to sys.path to access root_agent
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from root_agent import root_agent

# Direct access to root_agent
__all__ = ["root_agent"]
