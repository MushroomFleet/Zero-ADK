"""
Agent implementations for ADK Alpha.

This module provides agent implementations following ADK best practices,
including basic agents, multi-agent orchestration, and specialized agent types.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Union

import structlog
from google.adk.agents import Agent, SequentialAgent, ParallelAgent
from google.adk.tools import google_search, FunctionTool
from google.adk.runners import Runner

from .config import get_model_config, load_config
from .tools import (
    CustomDatabaseTool,
    WebScrapingTool,
    DataAnalysisTool,
    calculator_tool,
    text_processor_tool,
    create_custom_tool,
)

logger = structlog.get_logger(__name__)


class BasicAgent:
  """
  Basic agent implementation with optimized configuration.
  
  Provides a simplified interface for creating agents with
  common tools and best practice configurations.
  """
  
  def __init__(
    self,
    agent_name: str,
    instructions: str,
    model: str = "gemini-2.0-flash-exp",
    tools: Optional[List] = None,
    enable_search: bool = True,
    enable_calculator: bool = True,
    enable_text_processing: bool = True,
    stream: bool = False,
    **model_kwargs
  ):
    """
    Initialize basic agent.
    
    Args:
      agent_name: Name for the agent.
      instructions: Instructions for the agent's behavior.
      model: Model name to use.
      tools: Additional tools to include.
      enable_search: Whether to include Google search tool.
      enable_calculator: Whether to include calculator tool.
      enable_text_processing: Whether to include text processing tool.
      stream: Whether to enable streaming responses.
      **model_kwargs: Additional model configuration parameters.
    """
    self.agent_name = agent_name
    self.instructions = instructions
    self.model = model
    self.stream = stream
    
    # Get optimized model configuration
    model_config = get_model_config(model)
    model_config.update(model_kwargs)
    
    # Build tool list
    agent_tools = []
    
    if enable_search:
      agent_tools.append(google_search)
    
    if enable_calculator:
      agent_tools.append(calculator_tool)
    
    if enable_text_processing:
      agent_tools.append(text_processor_tool)
    
    if tools:
      agent_tools.extend(tools)
    
    # Create the underlying ADK agent
    self.agent = Agent(
      agent_name=agent_name,
      instructions=instructions,
      model=model,
      model_kwargs=model_config,
      tools=agent_tools,
      parallel_tool_calls=True,
      stream=stream
    )
    
    self.runner = Runner(self.agent)
    
    logger.info(
      "basic_agent_created",
      agent_name=agent_name,
      model=model,
      tool_count=len(agent_tools),
      stream=stream
    )
  
  async def execute(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute a query with the agent.
    
    Args:
      query: Query to execute.
      session_id: Optional session ID for conversation continuity.
      
    Returns:
      Dictionary containing execution results.
    """
    try:
      start_time = time.time()
      
      logger.info(
        "agent_execution_started",
        agent_name=self.agent_name,
        query_length=len(query),
        session_id=session_id
      )
      
      # Execute the agent
      if self.stream:
        # For streaming, collect all chunks
        chunks = []
        async for chunk in self.runner.stream(query, session_id=session_id):
          chunks.append(chunk)
        
        # Combine chunks into final response
        content = ""
        tool_calls = []
        for chunk in chunks:
          if hasattr(chunk, 'content') and chunk.content:
            content += chunk.content
          if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
            tool_calls.extend(chunk.tool_calls)
        
        result = {
          "content": content,
          "tool_calls": tool_calls,
          "chunks": len(chunks)
        }
      else:
        # Non-streaming execution
        response = await self.runner.run(query, session_id=session_id)
        result = {
          "content": response.content if hasattr(response, 'content') else str(response),
          "tool_calls": response.tool_calls if hasattr(response, 'tool_calls') else [],
        }
      
      execution_time = time.time() - start_time
      
      logger.info(
        "agent_execution_completed",
        agent_name=self.agent_name,
        execution_time=execution_time,
        success=True,
        session_id=session_id
      )
      
      return {
        "status": "success",
        "agent_name": self.agent_name,
        "query": query,
        "result": result,
        "execution_time": execution_time,
        "session_id": session_id
      }
      
    except Exception as e:
      logger.exception(
        "agent_execution_error",
        agent_name=self.agent_name,
        query_length=len(query),
        error=str(e),
        session_id=session_id
      )
      return {
        "status": "error",
        "agent_name": self.agent_name,
        "query": query,
        "error": str(e),
        "error_type": type(e).__name__,
        "session_id": session_id
      }


class ResearchAgent(BasicAgent):
  """
  Specialized agent for research tasks.
  
  Configured with tools and instructions optimized for
  information gathering and research activities.
  """
  
  def __init__(self, **kwargs):
    # Add web scraping capabilities
    web_scraper = WebScrapingTool(rate_limit=2.0)
    additional_tools = kwargs.pop('tools', [])
    additional_tools.append(web_scraper)
    
    super().__init__(
      agent_name=kwargs.pop('agent_name', 'researcher'),
      instructions=kwargs.pop('instructions', 
        "You are a research specialist. Your role is to gather comprehensive "
        "information from various sources, analyze the credibility of sources, "
        "and provide well-researched, factual information. Use web scraping "
        "and search tools to find relevant information. Always cite your sources."
      ),
      tools=additional_tools,
      **kwargs
    )


class AnalystAgent(BasicAgent):
  """
  Specialized agent for data analysis tasks.
  
  Configured with tools and instructions optimized for
  data processing and analytical insights.
  """
  
  def __init__(self, **kwargs):
    # Add data analysis capabilities
    data_analyzer = DataAnalysisTool()
    additional_tools = kwargs.pop('tools', [])
    additional_tools.append(data_analyzer)
    
    super().__init__(
      agent_name=kwargs.pop('agent_name', 'analyst'),
      instructions=kwargs.pop('instructions',
        "You are a data analysis specialist. Your role is to process structured "
        "data, identify patterns and trends, generate insights, and create "
        "summaries. Use statistical analysis and data visualization concepts "
        "to provide meaningful interpretations of data."
      ),
      tools=additional_tools,
      **kwargs
    )


class WriterAgent(BasicAgent):
  """
  Specialized agent for content writing and documentation.
  
  Configured with tools and instructions optimized for
  writing, editing, and document creation tasks.
  """
  
  def __init__(self, **kwargs):
    super().__init__(
      agent_name=kwargs.pop('agent_name', 'writer'),
      instructions=kwargs.pop('instructions',
        "You are a professional writer and editor. Your role is to create "
        "well-structured, clear, and engaging content. Focus on proper grammar, "
        "style, and organization. Adapt your writing style to the target audience "
        "and purpose. Provide constructive feedback on existing content."
      ),
      enable_search=False,  # Writers typically don't need search
      enable_calculator=False,  # Or calculator
      **kwargs
    )


class MultiAgentOrchestrator:
  """
  Orchestrator for managing multiple agents in workflows.
  
  Provides sequential and parallel execution patterns for
  complex multi-agent tasks.
  """
  
  def __init__(self, agents: Dict[str, BasicAgent]):
    """
    Initialize multi-agent orchestrator.
    
    Args:
      agents: Dictionary mapping agent names to agent instances.
    """
    self.agents = agents
    logger.info(
      "multi_agent_orchestrator_created",
      agent_count=len(agents),
      agent_names=list(agents.keys())
    )
  
  async def execute_sequential(
    self,
    workflow: List[Dict[str, Any]],
    session_id: Optional[str] = None
  ) -> Dict[str, Any]:
    """
    Execute agents sequentially in a workflow.
    
    Args:
      workflow: List of workflow steps, each containing agent_name and query.
      session_id: Optional session ID for conversation continuity.
      
    Returns:
      Dictionary containing workflow execution results.
    """
    try:
      start_time = time.time()
      results = []
      context = ""
      
      logger.info(
        "sequential_workflow_started",
        step_count=len(workflow),
        session_id=session_id
      )
      
      for i, step in enumerate(workflow):
        agent_name = step['agent_name']
        query = step['query']
        
        if agent_name not in self.agents:
          raise ValueError(f"Agent '{agent_name}' not found in orchestrator")
        
        # Add context from previous steps
        if context and step.get('use_context', True):
          contextual_query = f"Previous context: {context}\n\nCurrent task: {query}"
        else:
          contextual_query = query
        
        # Execute step
        agent = self.agents[agent_name]
        step_result = await agent.execute(contextual_query, session_id)
        
        # Update context for next step
        if step_result['status'] == 'success':
          context += f"\nStep {i+1} ({agent_name}): {step_result['result']['content']}"
        
        results.append({
          "step": i + 1,
          "agent_name": agent_name,
          "original_query": query,
          "contextual_query": contextual_query,
          "result": step_result
        })
        
        logger.info(
          "sequential_workflow_step_completed",
          step=i + 1,
          agent_name=agent_name,
          success=step_result['status'] == 'success'
        )
      
      execution_time = time.time() - start_time
      
      logger.info(
        "sequential_workflow_completed",
        total_steps=len(workflow),
        execution_time=execution_time,
        success=True
      )
      
      return {
        "status": "success",
        "workflow_type": "sequential",
        "results": results,
        "execution_time": execution_time,
        "session_id": session_id
      }
      
    except Exception as e:
      logger.exception(
        "sequential_workflow_error",
        workflow_steps=len(workflow),
        error=str(e)
      )
      return {
        "status": "error",
        "workflow_type": "sequential",
        "error": str(e),
        "error_type": type(e).__name__,
        "partial_results": results if 'results' in locals() else []
      }
  
  async def execute_parallel(
    self,
    tasks: List[Dict[str, Any]],
    session_id: Optional[str] = None
  ) -> Dict[str, Any]:
    """
    Execute agents in parallel for independent tasks.
    
    Args:
      tasks: List of tasks, each containing agent_name and query.
      session_id: Optional session ID for conversation continuity.
      
    Returns:
      Dictionary containing parallel execution results.
    """
    try:
      start_time = time.time()
      
      logger.info(
        "parallel_workflow_started",
        task_count=len(tasks),
        session_id=session_id
      )
      
      # Create coroutines for all tasks
      coroutines = []
      for i, task in enumerate(tasks):
        agent_name = task['agent_name']
        query = task['query']
        
        if agent_name not in self.agents:
          raise ValueError(f"Agent '{agent_name}' not found in orchestrator")
        
        agent = self.agents[agent_name]
        coroutines.append(agent.execute(query, session_id))
      
      # Execute all tasks in parallel
      results = await asyncio.gather(*coroutines, return_exceptions=True)
      
      # Process results
      processed_results = []
      for i, (task, result) in enumerate(zip(tasks, results)):
        if isinstance(result, Exception):
          processed_results.append({
            "task": i + 1,
            "agent_name": task['agent_name'],
            "query": task['query'],
            "result": {
              "status": "error",
              "error": str(result),
              "error_type": type(result).__name__
            }
          })
        else:
          processed_results.append({
            "task": i + 1,
            "agent_name": task['agent_name'],
            "query": task['query'],
            "result": result
          })
      
      execution_time = time.time() - start_time
      
      success_count = sum(1 for r in processed_results if r['result']['status'] == 'success')
      
      logger.info(
        "parallel_workflow_completed",
        total_tasks=len(tasks),
        successful_tasks=success_count,
        execution_time=execution_time
      )
      
      return {
        "status": "success",
        "workflow_type": "parallel",
        "results": processed_results,
        "successful_tasks": success_count,
        "total_tasks": len(tasks),
        "execution_time": execution_time,
        "session_id": session_id
      }
      
    except Exception as e:
      logger.exception(
        "parallel_workflow_error",
        task_count=len(tasks),
        error=str(e)
      )
      return {
        "status": "error",
        "workflow_type": "parallel",
        "error": str(e),
        "error_type": type(e).__name__
      }


def create_basic_agent(
  agent_name: str,
  instructions: str,
  **kwargs
) -> BasicAgent:
  """
  Factory function to create a basic agent with default configuration.
  
  Args:
    agent_name: Name for the agent.
    instructions: Instructions for the agent's behavior.
    **kwargs: Additional configuration parameters.
    
  Returns:
    Configured BasicAgent instance.
  """
  return BasicAgent(
    agent_name=agent_name,
    instructions=instructions,
    **kwargs
  )


def create_multi_agent_system(
  agent_configs: List[Dict[str, Any]]
) -> MultiAgentOrchestrator:
  """
  Factory function to create a multi-agent system.
  
  Args:
    agent_configs: List of agent configuration dictionaries.
    
  Returns:
    Configured MultiAgentOrchestrator instance.
  """
  agents = {}
  
  for config in agent_configs:
    agent_type = config.pop('type', 'basic')
    agent_name = config['agent_name']
    
    if agent_type == 'basic':
      agent = BasicAgent(**config)
    elif agent_type == 'research':
      agent = ResearchAgent(**config)
    elif agent_type == 'analyst':
      agent = AnalystAgent(**config)
    elif agent_type == 'writer':
      agent = WriterAgent(**config)
    else:
      raise ValueError(f"Unknown agent type: {agent_type}")
    
    agents[agent_name] = agent
  
  return MultiAgentOrchestrator(agents)


# Example configurations for common use cases
DEFAULT_RESEARCH_WORKFLOW = [
  {
    "agent_name": "researcher",
    "query": "Research the topic and gather relevant information from reliable sources."
  },
  {
    "agent_name": "analyst", 
    "query": "Analyze the research data and identify key insights and patterns."
  },
  {
    "agent_name": "writer",
    "query": "Create a comprehensive report based on the research and analysis."
  }
]

PARALLEL_ANALYSIS_TASKS = [
  {
    "agent_name": "researcher",
    "query": "Gather background information on the topic."
  },
  {
    "agent_name": "analyst",
    "query": "Perform quantitative analysis of available data."
  }
]
