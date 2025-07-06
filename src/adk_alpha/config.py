"""
Configuration management for ADK Alpha implementation.

This module provides utilities for loading configuration, managing credentials,
and setting up logging based on the ADK handoff specifications.
"""

import os
import logging
import structlog
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from google.cloud import secretmanager
from google.auth import default


def load_config() -> Dict[str, Any]:
  """
  Load configuration from environment variables and .env file.
  
  Returns:
    Dictionary containing all configuration settings.
  """
  # Load .env file if it exists
  load_dotenv()
  
  config = {
    # Core Configuration
    "google_cloud_project": os.getenv("GOOGLE_CLOUD_PROJECT"),
    "google_cloud_location": os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
    "google_genai_use_vertexai": os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "True").lower() == "true",
    
    # Authentication
    "google_application_credentials": os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    "google_api_key": os.getenv("GOOGLE_API_KEY"),
    
    # Session Management
    "session_db_url": os.getenv("SESSION_DB_URL"),
    "session_service_uri": os.getenv("SESSION_SERVICE_URI"),
    
    # Storage
    "google_cloud_bucket": os.getenv("GOOGLE_CLOUD_BUCKET"),
    "storage_bucket": os.getenv("STORAGE_BUCKET"),
    
    # Development
    "serve_web_interface": os.getenv("SERVE_WEB_INTERFACE", "True").lower() == "true",
    "disable_web_driver": os.getenv("DISABLE_WEB_DRIVER", "0") == "0",
    
    # Optional API Keys
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
    
    # Logging
    "log_level": os.getenv("LOG_LEVEL", "INFO"),
  }
  
  return {k: v for k, v in config.items() if v is not None}


def get_model_config(model_name: str = "gemini-2.0-flash-exp") -> Dict[str, Any]:
  """
  Get optimized model configuration based on ADK best practices.
  
  Args:
    model_name: Name of the model to configure.
    
  Returns:
    Dictionary containing model configuration parameters.
  """
  configs = {
    "gemini-2.0-flash-exp": {
      "temperature": 0.1,
      "max_tokens": 1000,
      "top_p": 0.9,
      "top_k": 40,
    },
    "gemini-1.5-pro": {
      "temperature": 0.2,
      "max_tokens": 2000,
      "top_p": 0.8,
      "top_k": 40,
    },
    "claude-3-sonnet": {
      "temperature": 0.1,
      "max_tokens": 1000,
    },
  }
  
  return configs.get(model_name, configs["gemini-2.0-flash-exp"])


def get_secret(secret_id: str, project_id: Optional[str] = None) -> str:
  """
  Retrieve secret from Google Secret Manager.
  
  Args:
    secret_id: ID of the secret to retrieve.
    project_id: Google Cloud project ID. If None, uses default project.
    
  Returns:
    Secret value as string.
    
  Raises:
    Exception: If secret cannot be retrieved.
  """
  if not project_id:
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
  
  if not project_id:
    raise ValueError("Google Cloud project ID not configured")
  
  try:
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")
  except Exception as e:
    raise Exception(f"Failed to retrieve secret {secret_id}: {str(e)}")


def setup_logging(log_level: str = "INFO") -> None:
  """
  Set up structured logging with Google Cloud integration.
  
  Args:
    log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
  """
  # Configure standard logging
  logging.basicConfig(
    level=getattr(logging, log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  )
  
  # Configure structured logging
  structlog.configure(
    processors=[
      structlog.stdlib.filter_by_level,
      structlog.stdlib.add_logger_name,
      structlog.stdlib.add_log_level,
      structlog.stdlib.PositionalArgumentsFormatter(),
      structlog.processors.TimeStamper(fmt="iso"),
      structlog.processors.StackInfoRenderer(),
      structlog.processors.format_exc_info,
      structlog.processors.UnicodeDecoder(),
      structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
  )


def validate_config(config: Dict[str, Any]) -> bool:
  """
  Validate that required configuration is present.
  
  Args:
    config: Configuration dictionary to validate.
    
  Returns:
    True if configuration is valid.
    
  Raises:
    ValueError: If required configuration is missing.
  """
  required_fields = [
    "google_cloud_project",
    "google_cloud_location",
  ]
  
  missing_fields = [field for field in required_fields if not config.get(field)]
  
  if missing_fields:
    raise ValueError(f"Missing required configuration: {', '.join(missing_fields)}")
  
  # Validate authentication
  if not config.get("google_application_credentials") and not config.get("google_api_key"):
    try:
      # Try to get default credentials
      credentials, project = default()
      if not project:
        raise ValueError("No authentication method configured")
    except Exception:
      raise ValueError("No valid authentication method found")
  
  return True


def get_database_config() -> Dict[str, Any]:
  """
  Get database configuration for session management.
  
  Returns:
    Dictionary containing database connection parameters.
  """
  config = load_config()
  
  if config.get("session_db_url"):
    return {
      "url": config["session_db_url"],
      "pool_size": 10,
      "max_overflow": 20,
      "pool_pre_ping": True,
      "pool_recycle": 3600,
    }
  else:
    # Default to in-memory for development
    return {
      "type": "in_memory",
      "persist": False,
    }
