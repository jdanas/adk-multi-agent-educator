"""
ADK Multi-Agent Educator - Main Entry Point

This module provides the main entry point for the ADK framework.
"""

from .root_agent import root_agent

# Expose root_agent for ADK discovery
__all__ = ["root_agent"]

# Alternative attribute access
agent = root_agent
