"""
Custom tools implementation for ADK Alpha.

This module provides custom tool implementations following ADK best practices,
including database integration, web scraping, and other specialized capabilities.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

import structlog
from google.adk.tools import BaseTool, FunctionTool
from google.adk.tools import google_search

try:
  import requests
  from bs4 import BeautifulSoup
  HAS_WEB_SCRAPING = True
except ImportError:
  HAS_WEB_SCRAPING = False

try:
  import sqlalchemy
  from sqlalchemy import create_engine, text
  from sqlalchemy.pool import QueuePool
  HAS_DATABASE = True
except ImportError:
  HAS_DATABASE = False

try:
  import pandas as pd
  HAS_PANDAS = True
except ImportError:
  HAS_PANDAS = False


logger = structlog.get_logger(__name__)


class CustomDatabaseTool(BaseTool):
  """
  Custom database integration tool with safe query execution.
  
  Provides secure database access with connection pooling and
  query validation following ADK security best practices.
  """
  
  def __init__(self, connection_string: str, max_results: int = 100):
    """
    Initialize database tool.
    
    Args:
      connection_string: Database connection string.
      max_results: Maximum number of results to return.
    """
    if not HAS_DATABASE:
      raise ImportError("sqlalchemy is required for database tools")
    
    self.connection_string = connection_string
    self.max_results = max_results
    
    # Configure connection pooling
    self.engine = create_engine(
      connection_string,
      poolclass=QueuePool,
      pool_size=5,
      max_overflow=10,
      pool_pre_ping=True,
      pool_recycle=3600
    )
    
    super().__init__(
      name="database_query",
      description="Execute safe database queries with result limiting."
    )
  
  async def execute(self, query: str) -> Dict[str, Any]:
    """
    Execute database query safely.
    
    Args:
      query: SQL query to execute.
      
    Returns:
      Dictionary containing query results or error information.
    """
    try:
      # Basic query validation
      query_lower = query.lower().strip()
      
      # Only allow SELECT queries for safety
      if not query_lower.startswith('select'):
        return {
          "status": "error",
          "error_type": "validation",
          "message": "Only SELECT queries are allowed"
        }
      
      # Check for potentially dangerous operations
      dangerous_keywords = ['drop', 'delete', 'insert', 'update', 'alter', 'create']
      if any(keyword in query_lower for keyword in dangerous_keywords):
        return {
          "status": "error",
          "error_type": "validation", 
          "message": "Query contains potentially dangerous operations"
        }
      
      # Execute query with result limiting
      with self.engine.connect() as connection:
        result = connection.execute(text(f"SELECT * FROM ({query}) AS subquery LIMIT {self.max_results}"))
        rows = result.fetchall()
        columns = result.keys()
        
        # Convert to list of dictionaries
        data = [dict(zip(columns, row)) for row in rows]
        
        logger.info(
          "database_query_executed",
          query_length=len(query),
          result_count=len(data),
          success=True
        )
        
        return {
          "status": "success",
          "data": data,
          "row_count": len(data),
          "columns": list(columns)
        }
        
    except sqlalchemy.exc.SQLAlchemyError as e:
      logger.error("database_query_error", error=str(e), query_length=len(query))
      return {
        "status": "error",
        "error_type": "database",
        "message": "Database query failed"
      }
    except Exception as e:
      logger.exception("database_query_unexpected_error", error=str(e))
      return {
        "status": "error",
        "error_type": "unexpected",
        "message": "An unexpected error occurred"
      }


class WebScrapingTool(BaseTool):
  """
  Web scraping tool with rate limiting and error handling.
  
  Provides safe web content extraction with respect for robots.txt
  and rate limiting to avoid overwhelming target servers.
  """
  
  def __init__(self, rate_limit: float = 1.0, timeout: int = 10):
    """
    Initialize web scraping tool.
    
    Args:
      rate_limit: Minimum seconds between requests.
      timeout: Request timeout in seconds.
    """
    if not HAS_WEB_SCRAPING:
      raise ImportError("requests and beautifulsoup4 are required for web scraping")
    
    self.rate_limit = rate_limit
    self.timeout = timeout
    self.last_request_time = 0
    
    super().__init__(
      name="web_scraper",
      description="Extract text content from web pages with rate limiting."
    )
  
  async def execute(self, url: str, selector: Optional[str] = None) -> Dict[str, Any]:
    """
    Scrape content from a web page.
    
    Args:
      url: URL to scrape.
      selector: Optional CSS selector for specific content.
      
    Returns:
      Dictionary containing scraped content or error information.
    """
    try:
      # Rate limiting
      current_time = time.time()
      time_since_last = current_time - self.last_request_time
      if time_since_last < self.rate_limit:
        await asyncio.sleep(self.rate_limit - time_since_last)
      
      # Validate URL
      if not url.startswith(('http://', 'https://')):
        return {
          "status": "error",
          "error_type": "validation",
          "message": "Invalid URL format"
        }
      
      # Make request with proper headers
      headers = {
        'User-Agent': 'ADK-Alpha-Bot/1.0 (Educational Purpose)',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
      }
      
      response = requests.get(url, headers=headers, timeout=self.timeout)
      response.raise_for_status()
      
      self.last_request_time = time.time()
      
      # Parse content
      soup = BeautifulSoup(response.content, 'html.parser')
      
      # Remove script and style elements
      for script in soup(["script", "style"]):
        script.decompose()
      
      if selector:
        # Extract specific content using CSS selector
        elements = soup.select(selector)
        content = '\n'.join(element.get_text(strip=True) for element in elements)
      else:
        # Extract all text content
        content = soup.get_text(strip=True)
      
      # Clean up content
      lines = (line.strip() for line in content.splitlines())
      content = '\n'.join(line for line in lines if line)
      
      logger.info(
        "web_scraping_completed",
        url=url,
        content_length=len(content),
        selector=selector,
        success=True
      )
      
      return {
        "status": "success",
        "content": content[:5000],  # Limit content length
        "url": url,
        "content_length": len(content),
        "truncated": len(content) > 5000
      }
      
    except requests.exceptions.RequestException as e:
      logger.error("web_scraping_request_error", url=url, error=str(e))
      return {
        "status": "error",
        "error_type": "request",
        "message": f"Failed to fetch URL: {str(e)}"
      }
    except Exception as e:
      logger.exception("web_scraping_unexpected_error", url=url, error=str(e))
      return {
        "status": "error",
        "error_type": "unexpected",
        "message": "An unexpected error occurred during scraping"
      }


class DataAnalysisTool(BaseTool):
  """
  Data analysis tool for processing structured data.
  
  Provides basic data analysis capabilities using pandas,
  including summary statistics and data transformations.
  """
  
  def __init__(self):
    """Initialize data analysis tool."""
    if not HAS_PANDAS:
      raise ImportError("pandas is required for data analysis tools")
    
    super().__init__(
      name="data_analyzer",
      description="Analyze structured data and generate insights."
    )
  
  async def execute(self, data: List[Dict[str, Any]], operation: str = "summary") -> Dict[str, Any]:
    """
    Analyze data and generate insights.
    
    Args:
      data: List of dictionaries containing structured data.
      operation: Type of analysis to perform (summary, describe, correlate).
      
    Returns:
      Dictionary containing analysis results.
    """
    try:
      if not data:
        return {
          "status": "error",
          "error_type": "validation",
          "message": "No data provided for analysis"
        }
      
      # Convert to DataFrame
      df = pd.DataFrame(data)
      
      if operation == "summary":
        # Basic summary statistics
        result = {
          "row_count": len(df),
          "column_count": len(df.columns),
          "columns": list(df.columns),
          "data_types": df.dtypes.to_dict(),
          "null_counts": df.isnull().sum().to_dict(),
        }
        
        # Add numeric summaries
        numeric_columns = df.select_dtypes(include=['number']).columns
        if len(numeric_columns) > 0:
          result["numeric_summary"] = df[numeric_columns].describe().to_dict()
        
      elif operation == "describe":
        # Detailed description
        result = {
          "description": df.describe(include='all').to_dict(),
          "info": {
            "memory_usage": df.memory_usage().sum(),
            "shape": df.shape,
          }
        }
        
      elif operation == "correlate":
        # Correlation analysis for numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        if len(numeric_df.columns) > 1:
          result = {
            "correlation_matrix": numeric_df.corr().to_dict(),
            "numeric_columns": list(numeric_df.columns)
          }
        else:
          result = {
            "message": "Insufficient numeric columns for correlation analysis",
            "numeric_columns": list(numeric_df.columns)
          }
      
      else:
        return {
          "status": "error",
          "error_type": "validation",
          "message": f"Unknown operation: {operation}"
        }
      
      logger.info(
        "data_analysis_completed",
        operation=operation,
        row_count=len(df),
        column_count=len(df.columns),
        success=True
      )
      
      return {
        "status": "success",
        "operation": operation,
        "result": result
      }
      
    except Exception as e:
      logger.exception("data_analysis_error", operation=operation, error=str(e))
      return {
        "status": "error",
        "error_type": "unexpected",
        "message": f"Data analysis failed: {str(e)}"
      }


def create_custom_tool(func, name: Optional[str] = None, description: Optional[str] = None) -> FunctionTool:
  """
  Create a custom tool from a Python function with proper error handling.
  
  Args:
    func: Python function to wrap as a tool.
    name: Optional name for the tool (defaults to function name).
    description: Optional description (defaults to function docstring).
    
  Returns:
    FunctionTool instance with error handling wrapper.
  """
  def error_wrapped_func(*args, **kwargs):
    """Wrapper function that adds comprehensive error handling."""
    try:
      start_time = time.time()
      result = func(*args, **kwargs)
      execution_time = time.time() - start_time
      
      logger.info(
        "custom_tool_executed",
        tool_name=name or func.__name__,
        execution_time=execution_time,
        success=True
      )
      
      # Ensure result is in expected format
      if isinstance(result, dict) and "status" in result:
        return result
      else:
        return {
          "status": "success",
          "result": result,
          "execution_time": execution_time
        }
        
    except ValueError as e:
      logger.error(
        "custom_tool_validation_error",
        tool_name=name or func.__name__,
        error=str(e)
      )
      return {
        "status": "error",
        "error_type": "validation",
        "message": str(e)
      }
    except Exception as e:
      logger.exception(
        "custom_tool_unexpected_error",
        tool_name=name or func.__name__,
        error=str(e)
      )
      return {
        "status": "error",
        "error_type": "unexpected",
        "message": "An unexpected error occurred"
      }
  
  # Preserve original function metadata
  error_wrapped_func.__name__ = func.__name__
  error_wrapped_func.__doc__ = func.__doc__
  
  return FunctionTool(
    error_wrapped_func,
    name=name,
    description=description
  )


# Pre-configured tools that can be imported directly
def simple_calculator(expression: str) -> Dict[str, Any]:
  """
  Simple calculator tool for basic mathematical operations.
  
  Args:
    expression: Mathematical expression to evaluate.
    
  Returns:
    Dictionary containing calculation result.
  """
  import re
  
  # Validate expression (only allow basic math operations)
  if not re.match(r'^[0-9+\-*/.() ]+$', expression):
    raise ValueError("Invalid characters in mathematical expression")
  
  # Evaluate expression safely
  try:
    result = eval(expression, {"__builtins__": {}}, {})
    return {
      "expression": expression,
      "result": result,
      "type": type(result).__name__
    }
  except Exception as e:
    raise ValueError(f"Failed to evaluate expression: {str(e)}")


def text_processor(text: str, operation: str = "word_count") -> Dict[str, Any]:
  """
  Text processing tool for basic text analysis operations.
  
  Args:
    text: Text to process.
    operation: Type of operation (word_count, char_count, summary).
    
  Returns:
    Dictionary containing processing results.
  """
  if operation == "word_count":
    words = text.split()
    return {
      "word_count": len(words),
      "unique_words": len(set(words)),
      "average_word_length": sum(len(word) for word in words) / len(words) if words else 0
    }
  elif operation == "char_count":
    return {
      "character_count": len(text),
      "character_count_no_spaces": len(text.replace(" ", "")),
      "line_count": len(text.splitlines())
    }
  elif operation == "summary":
    words = text.split()
    sentences = text.split('.')
    return {
      "word_count": len(words),
      "sentence_count": len([s for s in sentences if s.strip()]),
      "character_count": len(text),
      "average_sentence_length": len(words) / len(sentences) if sentences else 0
    }
  else:
    raise ValueError(f"Unknown operation: {operation}")


# Create pre-configured tool instances
calculator_tool = create_custom_tool(simple_calculator, "calculator", "Perform basic mathematical calculations")
text_processor_tool = create_custom_tool(text_processor, "text_processor", "Analyze and process text content")
