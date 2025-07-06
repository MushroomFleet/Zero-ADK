# ADK-PYTHON-DEV-HANDOFF.md

# Google ADK Python SDK Development Handoff

## Executive Summary

The Google Agent Development Kit (ADK) Python SDK is a **production-ready framework** for building, evaluating, and deploying AI agents. This comprehensive handoff document provides everything needed to develop, deploy, and maintain ADK-based agent systems.

**Key Capabilities**: Multi-agent orchestration, extensive tool ecosystem, streaming support, built-in evaluation framework, and flexible deployment options across Google Cloud services.

**Architecture**: Modular design with clean separation between reasoning (agents), capabilities (tools), execution (runners), and state management (sessions). The framework supports everything from simple single-agent systems to complex multi-agent hierarchies.

## Repository Structure and Codebase

### Core Architecture Overview

```
adk-python/
├── src/google/adk/           # Main package directory
│   ├── agents/              # Agent definitions and types
│   ├── tools/               # Tool implementations and ecosystem
│   ├── runners/             # Agent execution runtime
│   ├── sessions/            # Session management
│   ├── evaluation/          # Agent evaluation framework
│   ├── events/              # Event handling system
│   ├── models/              # LLM abstractions and interfaces
│   ├── memory/              # Memory management services
│   ├── artifacts/           # Artifact handling
│   ├── code_executors/      # Code execution capabilities
│   ├── cli/                 # Command-line interface
│   ├── auth/                # Authentication services
│   ├── flows/               # Workflow orchestration
│   ├── platform/            # Platform-specific utilities
│   └── telemetry/           # Monitoring and tracing
├── tests/                   # Test suites
├── contributing/            # Contribution guidelines and samples
├── pyproject.toml           # Project configuration
└── README.md               # Project documentation
```

### Critical Components

**Agent System**: The `agents/` module provides the core agent classes:
- `BaseAgent`: Abstract foundation for all agents
- `LlmAgent`: Primary LLM-powered agent implementation
- `Agent`: Main entry point (alias for LlmAgent)
- Workflow agents: `Sequential`, `Parallel`, `Loop` for structured orchestration

**Tool Ecosystem**: The `tools/` module offers extensive capabilities:
- `BaseTool`: Abstract base for all tools
- `FunctionTool`: Wraps Python functions as tools
- Built-in tools: Google Search, Code Execution, Vertex AI Search
- Third-party integrations: LangChain, CrewAI, MCP protocol support

**Runtime Engine**: The `runners/` module handles execution:
- `Runner`: Main execution engine
- `InMemoryRunner`: Development and testing execution
- Event streaming and async support

## Dependencies and Requirements

### System Prerequisites

**Python Version**: 3.9+ (3.10+ recommended)
**Operating System**: Cross-platform (Windows, macOS, Linux)
**Build System**: Flit backend (`flit_core >=3.8,<4`)

### Core Runtime Dependencies

```bash
# Essential dependencies
pip install google-adk  # Version 1.4.1+
```

**Critical Dependencies**:
- `google-genai` - Google Generative AI SDK
- `google-cloud-aiplatform` - Google Cloud AI Platform
- `fastapi` - Web framework for APIs
- `pydantic` - Data validation and settings
- `opentelemetry-api` - Observability and tracing
- `uvicorn` - ASGI server
- `click` - Command-line interface framework

**Optional Dependencies**:
- `anthropic` - Claude model support
- `crewai` - CrewAI integration
- `langchain-community` - LangChain tools
- `docker` - Container support
- `beautifulsoup4` - HTML parsing
- `pandas` - Data manipulation

### Installation Methods

```bash
# Production installation
pip install google-adk

# Development installation
pip install git+https://github.com/google/adk-python.git@main

# With evaluation tools
pip install google-adk[eval]
```

## Build Process and Configuration

### Project Configuration

**pyproject.toml**:
```toml
[build-system]
requires = ["flit_core >=3.8,<4"]
build-backend = "flit_core.buildapi"

[project]
name = "google-adk"
dynamic = ["version", "description"]
authors = [{name = "Google", email = "adk-team@google.com"}]
readme = "README.md"
requires-python = ">=3.9"

[tool.flit.module]
name = "google.adk"
```

### Code Quality Configuration

**Formatting and Linting**:
```toml
[tool.pyink]
line-length = 80
unstable = true
pyink-indentation = 2
pyink-use-majority-quotes = true

[tool.isort]
profile = "google"
```

**Build Commands**:
```bash
# Format code
python -m pyink --check src/
python -m isort --check-only src/

# Build package
python -m flit build

# Install in development mode
python -m pip install -e .
```

## Testing Framework and Test Execution

### Test Structure

```
tests/
├── unittests/
│   ├── agents/          # Agent testing
│   ├── tools/           # Tool testing
│   ├── cli/             # CLI testing
│   └── evaluation/      # Evaluation testing
└── integration/         # Integration tests
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test categories
python -m pytest tests/unittests/agents/
python -m pytest tests/unittests/tools/

# Run with coverage
python -m pytest --cov=google.adk

# Run async tests
python -m pytest -v tests/unittests/agents/test_async_agent.py
```

### Test Configuration

**pytest.ini options**:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_default_fixture_loop_scope = "function"
```

**Example Test Pattern**:
```python
import pytest
from google.adk.agents import Agent
from google.adk.tools import FunctionTool

class TestAgent:
    @pytest.fixture(autouse=True)
    def setup_agent(self):
        def sample_tool(query: str) -> str:
            return f"Result for: {query}"
        
        self.agent = Agent(
            agent_name="test_agent",
            tools=[FunctionTool(sample_tool)]
        )

    async def test_agent_execution(self):
        response = await self.agent.execute("test query")
        assert response.status == "success"
```

## Development Setup Instructions

### Local Development Environment

**Step 1: Environment Setup**
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install ADK
pip install google-adk

# Install development dependencies
pip install pytest pytest-asyncio pyink isort mypy
```

**Step 2: Project Structure**
```bash
mkdir my_agent_project
cd my_agent_project

# Create basic structure
touch __init__.py
touch agent.py
touch requirements.txt
touch .env
```

**Step 3: Basic Agent Implementation**
```python
# agent.py
from google.adk.agents import Agent
from google.adk.tools import google_search

def my_custom_tool(query: str) -> str:
    """Custom tool for specific domain logic."""
    # Implementation here
    return f"Processed: {query}"

# Define the root agent
root_agent = Agent(
    agent_name="my_agent",
    instructions="You are a helpful assistant.",
    tools=[
        google_search,
        my_custom_tool,
    ],
    model="gemini-2.0-flash-exp"
)

# Export for deployment
__all__ = ["root_agent"]
```

**Step 4: Environment Configuration**
```bash
# .env
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_GENAI_USE_VERTEXAI=True
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
```

### Development Commands

```bash
# Launch development web interface
adk web

# Run agent in terminal
adk run my_agent

# Start API server
adk api_server

# Run evaluations
adk eval path/to/agent path/to/evalset.json
```

## Configuration Requirements

### Essential Environment Variables

```bash
# Core Configuration
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_GENAI_USE_VERTEXAI=True

# Authentication
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
GOOGLE_API_KEY=your-api-key

# Session Management
SESSION_DB_URL=postgresql://user:pass@host:port/db
SESSION_SERVICE_URI=your-session-service-uri

# Storage
GOOGLE_CLOUD_BUCKET=your-bucket-name
STORAGE_BUCKET=your-cloud-storage-bucket

# Development
SERVE_WEB_INTERFACE=True
DISABLE_WEB_DRIVER=0
```

### Authentication Configuration

**Service Account Setup**:
```json
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "key-id",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
  "client_email": "service-account@your-project-id.iam.gserviceaccount.com",
  "client_id": "client-id",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token"
}
```

**Required IAM Roles**:
- AI Platform User
- Cloud Run Source Developer
- Service Account User
- Storage Object Viewer/Creator (if using Cloud Storage)

## Production Deployment Considerations

### Cloud Run Deployment (Recommended)

**Automated Deployment**:
```bash
# Using ADK CLI (recommended)
adk deploy cloud_run \
  --project=$GOOGLE_CLOUD_PROJECT \
  --region=$GOOGLE_CLOUD_LOCATION \
  --service_name=my-agent-service \
  --app_name=my-agent-app \
  path/to/agent
```

**Manual Deployment**:
```bash
# Using gcloud CLI
gcloud run deploy my-agent-service \
  --source . \
  --region $GOOGLE_CLOUD_LOCATION \
  --project $GOOGLE_CLOUD_PROJECT \
  --allow-unauthenticated \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT,GOOGLE_CLOUD_LOCATION=$GOOGLE_CLOUD_LOCATION"
```

### Deployment Files

**Generated Dockerfile**:
```dockerfile
FROM python:3.12-slim
RUN pip install --no-cache-dir uv==0.7.13
WORKDIR /app
COPY . .
RUN uv pip install --system -r requirements.txt
CMD ["python", "main.py"]
```

**FastAPI Entry Point (main.py)**:
```python
from google.adk.web import get_fast_api_app
from google.adk.sessions import DatabaseSessionService
import uvicorn

# Configure services
session_service = DatabaseSessionService(
    db_url=os.getenv("SESSION_DB_URL")
)

# Create FastAPI app
app = get_fast_api_app(
    session_service=session_service,
    artifact_service=artifact_service,
    memory_service=memory_service
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

**Requirements File**:
```txt
google-adk>=1.4.1
google-cloud-aiplatform
google-cloud-secret-manager
postgresql-adapter  # If using PostgreSQL sessions
```

### Alternative Deployment Options

**Vertex AI Agent Engine**:
```bash
# Deploy to fully managed service
gcloud ai agents deploy \
  --region=$GOOGLE_CLOUD_LOCATION \
  --source=. \
  --app-name=my-agent
```

**Container Deployment**:
```yaml
# docker-compose.yml
version: '3.8'
services:
  agent:
    build: .
    ports:
      - "8080:8080"
    environment:
      - GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT}
      - GOOGLE_CLOUD_LOCATION=${GOOGLE_CLOUD_LOCATION}
      - GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json
    volumes:
      - ./credentials.json:/app/credentials.json
```

## Security Considerations

### Authentication and Authorization

**Service Account Security**:
```python
# Secure credential management
from google.cloud import secretmanager

def get_secret(secret_id: str) -> str:
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# Use in agent configuration
api_key = get_secret("openai-api-key")
```

**Session Security**:
```python
# Implement proper session isolation
from google.adk.sessions import DatabaseSessionService

session_service = DatabaseSessionService(
    db_url=os.getenv("SESSION_DB_URL"),
    user_isolation=True,  # Enforce user-based session isolation
    encryption_key=os.getenv("SESSION_ENCRYPTION_KEY")
)
```

### Input Validation and Sanitization

```python
from pydantic import BaseModel, validator
from typing import Dict, Any

class SecureToolInput(BaseModel):
    query: str
    max_length: int = 1000
    
    @validator('query')
    def validate_query(cls, v):
        if len(v) > cls.max_length:
            raise ValueError('Query too long')
        # Add sanitization logic
        return v.strip()

def secure_tool(input_data: SecureToolInput) -> Dict[str, Any]:
    # Tool implementation with validated input
    return {"result": f"Processed: {input_data.query}"}
```

## Performance Optimization Guidelines

### Model Configuration Optimization

```python
# Optimize model selection and parameters
from google.adk.agents import Agent

agent = Agent(
    agent_name="optimized_agent",
    model="gemini-2.0-flash-exp",  # Fastest model for development
    model_kwargs={
        "temperature": 0.1,  # Lower temperature for consistency
        "max_tokens": 1000,  # Limit token usage
        "top_p": 0.9,
        "top_k": 40
    },
    parallel_tool_calls=True,  # Enable parallel execution
    tools=optimized_tools
)
```

### Session Management Optimization

```python
# Use persistent database sessions
from google.adk.sessions import DatabaseSessionService
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

# Configure connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600
)

session_service = DatabaseSessionService(
    db_url=DATABASE_URL,
    engine=engine
)
```

### Caching Strategies

```python
# Implement caching for expensive operations
from functools import lru_cache
import asyncio

@lru_cache(maxsize=128)
def cached_expensive_operation(param: str) -> str:
    # Expensive computation
    return result

# Async caching
_cache = {}

async def cached_async_operation(param: str) -> str:
    if param not in _cache:
        _cache[param] = await expensive_async_operation(param)
    return _cache[param]
```

## Integration Patterns and Examples

### Multi-Agent System Pattern

```python
from google.adk.agents import Agent, SequentialAgent, ParallelAgent
from google.adk.tools import FunctionTool

# Specialized agents
researcher = Agent(
    agent_name="researcher",
    instructions="You are a research specialist.",
    tools=[google_search, web_scraper_tool]
)

analyst = Agent(
    agent_name="analyst",
    instructions="You analyze and summarize research data.",
    tools=[data_analysis_tool, chart_generator]
)

writer = Agent(
    agent_name="writer",
    instructions="You write professional reports.",
    tools=[document_formatter, grammar_checker]
)

# Orchestrated workflow
report_generator = SequentialAgent(
    agent_name="report_generator",
    agents=[researcher, analyst, writer],
    instructions="Generate comprehensive reports through research, analysis, and writing."
)
```

### Tool Integration Pattern

```python
from google.adk.tools import BaseTool, FunctionTool
from typing import Dict, Any

class DatabaseTool(BaseTool):
    """Custom database integration tool."""
    
    def __init__(self, connection_string: str):
        self.connection = create_connection(connection_string)
        super().__init__(
            name="database_query",
            description="Execute database queries safely."
        )
    
    async def execute(self, query: str) -> Dict[str, Any]:
        try:
            # Implement safe query execution
            result = await self.connection.execute(query)
            return {"status": "success", "data": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}

# Integration with agent
database_tool = DatabaseTool(DATABASE_URL)
agent = Agent(
    agent_name="data_agent",
    tools=[database_tool],
    instructions="You can query the database safely."
)
```

### Streaming Response Pattern

```python
from google.adk.agents import Agent
from google.adk.runners import Runner
import asyncio

async def streaming_agent_example():
    agent = Agent(
        agent_name="streaming_agent",
        model="gemini-2.0-flash-exp",
        stream=True
    )
    
    runner = Runner(agent)
    
    async for chunk in runner.stream("Tell me about AI"):
        if chunk.type == "text":
            print(chunk.content, end="", flush=True)
        elif chunk.type == "tool_call":
            print(f"\n[Tool: {chunk.tool_name}]")
```

## Error Handling and Logging

### Comprehensive Error Handling

```python
import logging
from google.adk.agents import Agent
from google.adk.tools import FunctionTool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def robust_tool(param: str) -> Dict[str, Any]:
    """Tool with comprehensive error handling."""
    try:
        # Tool logic
        result = perform_operation(param)
        logger.info(f"Tool executed successfully: {param}")
        return {"status": "success", "result": result}
    
    except ValueError as e:
        logger.error(f"Validation error in tool: {e}")
        return {"status": "error", "error_type": "validation", "message": str(e)}
    
    except ConnectionError as e:
        logger.error(f"Connection error in tool: {e}")
        return {"status": "error", "error_type": "connection", "message": "Service temporarily unavailable"}
    
    except Exception as e:
        logger.exception(f"Unexpected error in tool: {e}")
        return {"status": "error", "error_type": "unexpected", "message": "An unexpected error occurred"}

# Agent with error callbacks
def error_callback(context):
    logger.error(f"Agent error: {context.error}")
    # Implement recovery logic
    return {"recovered": True, "message": "Error handled gracefully"}

agent = Agent(
    agent_name="robust_agent",
    tools=[FunctionTool(robust_tool)],
    error_callback=error_callback
)
```

### Structured Logging

```python
import structlog
from google.cloud import logging as cloud_logging

# Configure structured logging
cloud_logging.Client().setup_logging()

logger = structlog.get_logger()

def log_agent_execution(agent_name: str, query: str, result: dict):
    logger.info(
        "agent_execution",
        agent_name=agent_name,
        query=query,
        success=result.get("status") == "success",
        execution_time=result.get("execution_time"),
        token_usage=result.get("token_usage")
    )
```

## Monitoring and Maintenance Requirements

### OpenTelemetry Integration

```python
from opentelemetry import trace
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Add Cloud Trace exporter
cloud_trace_exporter = CloudTraceSpanExporter()
span_processor = BatchSpanProcessor(cloud_trace_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Instrument agent execution
@tracer.start_as_current_span("agent_execution")
def execute_agent_with_tracing(query: str):
    with tracer.start_as_current_span("agent_processing"):
        result = agent.execute(query)
        span = trace.get_current_span()
        span.set_attribute("query_length", len(query))
        span.set_attribute("success", result.success)
        return result
```

### Health Checks and Monitoring

```python
from fastapi import FastAPI, HTTPException
from google.adk.web import get_fast_api_app
import asyncio

app = get_fast_api_app()

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        # Test critical components
        session_test = await session_service.test_connection()
        model_test = await test_model_availability()
        
        return {
            "status": "healthy",
            "services": {
                "session_service": session_test,
                "model_service": model_test
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.get("/metrics")
async def metrics():
    """Prometheus-style metrics endpoint."""
    return {
        "active_sessions": await session_service.count_active_sessions(),
        "total_requests": request_counter.get_value(),
        "avg_response_time": response_time_histogram.get_average(),
        "error_rate": error_rate_gauge.get_value()
    }
```

### Performance Monitoring

```python
from google.cloud import monitoring_v3
import time

class PerformanceMonitor:
    def __init__(self, project_id: str):
        self.client = monitoring_v3.MetricServiceClient()
        self.project_name = f"projects/{project_id}"
    
    def record_execution_time(self, agent_name: str, duration: float):
        """Record agent execution time."""
        series = monitoring_v3.TimeSeries()
        series.metric.type = "custom.googleapis.com/agent/execution_time"
        series.metric.labels['agent_name'] = agent_name
        
        now = time.time()
        seconds = int(now)
        nanos = int((now - seconds) * 10**9)
        interval = monitoring_v3.TimeInterval(
            {"end_time": {"seconds": seconds, "nanos": nanos}}
        )
        
        point = monitoring_v3.Point({
            "interval": interval,
            "value": {"double_value": duration}
        })
        series.points = [point]
        
        self.client.create_time_series(
            name=self.project_name,
            time_series=[series]
        )

# Usage
monitor = PerformanceMonitor(PROJECT_ID)

def timed_execution(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        monitor.record_execution_time(func.__name__, duration)
        return result
    return wrapper
```

## Known Issues and Limitations

### Current Limitations

1. **Session Storage**: InMemorySessionService doesn't persist across restarts
2. **Model Constraints**: Some advanced features require specific model versions
3. **Streaming**: Limited streaming support for certain tool combinations
4. **Concurrency**: Tool execution may have concurrency limitations
5. **Memory Usage**: Large conversation histories can impact performance

### Troubleshooting Common Issues

**Authentication Problems**:
```bash
# Verify credentials
gcloud auth application-default login
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Test authentication
python -c "from google.cloud import aiplatform; print('Auth successful')"
```

**Session Issues**:
```python
# Debug session problems
from google.adk.sessions import DatabaseSessionService
import logging

logging.basicConfig(level=logging.DEBUG)
session_service = DatabaseSessionService(db_url=DATABASE_URL)

# Test connection
try:
    result = session_service.test_connection()
    print(f"Session service status: {result}")
except Exception as e:
    print(f"Session service error: {e}")
```

**Performance Issues**:
```python
# Monitor token usage
def monitor_token_usage(agent_response):
    if hasattr(agent_response, 'token_usage'):
        usage = agent_response.token_usage
        print(f"Tokens used: {usage.total_tokens}")
        print(f"Input tokens: {usage.prompt_tokens}")
        print(f"Output tokens: {usage.completion_tokens}")
```

## Conclusion

This comprehensive handoff provides all necessary information for developing, deploying, and maintaining Google ADK Python SDK applications. The framework offers **enterprise-grade capabilities** with extensive customization options, robust error handling, and flexible deployment strategies.

**Key Success Factors**:
- Follow the modular architecture patterns
- Implement comprehensive error handling and logging
- Use proper authentication and security practices
- Monitor performance and optimize based on usage patterns
- Leverage the built-in evaluation framework for quality assurance

The ADK Python SDK is **production-ready** and provides all the tools needed for building sophisticated AI agent systems at scale.