"""
ADK Alpha - Google Agent Development Kit Python SDK Implementation

This package provides a comprehensive implementation of Google ADK patterns
for building, evaluating, and deploying AI agents.
"""

__version__ = "0.1.0"
__description__ = "Google ADK Python SDK Alpha Implementation"

from .agents import (
    create_basic_agent,
    create_multi_agent_system,
    BasicAgent,
    ResearchAgent,
    AnalystAgent,
    WriterAgent,
)

from .tools import (
    CustomDatabaseTool,
    WebScrapingTool,
    create_custom_tool,
)

from .config import (
    load_config,
    get_model_config,
    setup_logging,
)

__all__ = [
    "create_basic_agent",
    "create_multi_agent_system",
    "BasicAgent",
    "ResearchAgent", 
    "AnalystAgent",
    "WriterAgent",
    "CustomDatabaseTool",
    "WebScrapingTool",
    "create_custom_tool",
    "load_config",
    "get_model_config",
    "setup_logging",
]
