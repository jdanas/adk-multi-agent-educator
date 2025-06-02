"""
ADK Multi-Agent Educator - Main Entry Point

This module provides the main entry point for the ADK framework.
"""

from .root_agent import root_agent, RootAgent

# Expose root_agent for ADK discovery
__all__ = ["root_agent", "RootAgent"]

# Alternative attribute access
agent = root_agent
