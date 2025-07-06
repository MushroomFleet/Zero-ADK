# ADK Alpha - Google Agent Development Kit Python SDK Implementation

A comprehensive implementation of Google ADK patterns for building, evaluating, and deploying AI agents, based on the ADK-PYTHON-DEV-HANDOFF specifications.

## 🚀 Features

- **Multi-Agent Orchestration**: Sequential and parallel agent workflows
- **Extensive Tool Ecosystem**: Built-in and custom tools for various capabilities  
- **Streaming Support**: Real-time streaming responses
- **Comprehensive Testing**: Full test suite with async support
- **Production Ready**: Security, logging, and monitoring built-in
- **Flexible Deployment**: Ready for Google Cloud services

## 📋 Prerequisites

- Python 3.9+ (3.10+ recommended)
- Google Cloud Project with AI Platform API enabled
- Service Account with appropriate IAM roles

## 🛠️ Installation

### 1. Clone and Install

```bash
git clone <repository-url>
cd ADK-alpha
pip install -e .
```

### 2. Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Development dependencies  
pip install -e ".[dev]"

# Advanced features (optional)
pip install -e ".[advanced]"
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
```

Required environment variables:
```bash
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
```

## 🎯 Quick Start

### Basic Agent Example

```python
import asyncio
from src.adk_alpha.agents import create_basic_agent
from src.adk_alpha.config import setup_logging

async def main():
    setup_logging("INFO")
    
    # Create a basic agent
    agent = create_basic_agent(
        agent_name="assistant",
        instructions="You are a helpful assistant.",
        enable_calculator=True,
        enable_text_processing=True
    )
    
    # Execute a query
    result = await agent.execute("Calculate 15 * 23 + 7")
    print(result['result']['content'])

asyncio.run(main())
```

### Multi-Agent System Example

```python
from src.adk_alpha.agents import create_multi_agent_system

# Define agent configurations
agent_configs = [
    {
        "type": "research",
        "agent_name": "researcher", 
        "instructions": "Gather comprehensive information."
    },
    {
        "type": "analyst",
        "agent_name": "analyst",
        "instructions": "Analyze data and identify patterns."
    },
    {
        "type": "writer", 
        "agent_name": "writer",
        "instructions": "Create well-structured reports."
    }
]

# Create orchestrator
orchestrator = create_multi_agent_system(agent_configs)

# Define workflow
workflow = [
    {"agent_name": "researcher", "query": "Research AI trends"},
    {"agent_name": "analyst", "query": "Analyze the research"},
    {"agent_name": "writer", "query": "Write a summary report"}
]

# Execute sequential workflow
result = await orchestrator.execute_sequential(workflow)
```

## 🏗️ Architecture

### Core Components

```
src/adk_alpha/
├── agents.py          # Agent implementations and orchestration
├── tools.py           # Custom tools and integrations  
├── config.py          # Configuration and credential management
└── __init__.py        # Package exports

tests/                 # Comprehensive test suite
examples/              # Usage examples and patterns
```

### Agent Types

- **BasicAgent**: General purpose agent with common tools
- **ResearchAgent**: Specialized for information gathering
- **AnalystAgent**: Optimized for data analysis  
- **WriterAgent**: Focused on content creation
- **MultiAgentOrchestrator**: Coordinates multiple agents

### Built-in Tools

- **Calculator**: Safe mathematical computations
- **Text Processor**: Text analysis and metrics
- **Web Scraper**: Content extraction with rate limiting
- **Database Tool**: Secure query execution  
- **Data Analyzer**: Statistical analysis with pandas

## 🧪 Testing

### Run Test Suite

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest tests/test_agents.py
pytest tests/test_tools.py
pytest tests/test_config.py
```

### Test Structure

```
tests/
├── test_agents.py     # Agent functionality tests
├── test_tools.py      # Tool implementation tests  
├── test_config.py     # Configuration tests
└── __init__.py
```

## 📖 Examples

### Running Examples

```bash
# Basic agent example
python examples/basic_agent_example.py

# Interactive mode
python examples/basic_agent_example.py interactive

# Multi-agent workflows
python examples/multi_agent_example.py
```

### Example Patterns

1. **Basic Agent Usage**: Single agent with built-in tools
2. **Multi-Agent Workflows**: Sequential and parallel orchestration
3. **Custom Tools**: Creating domain-specific capabilities
4. **Streaming Responses**: Real-time response handling

## 🔧 Configuration

### Model Configuration

```python
from src.adk_alpha.config import get_model_config

# Get optimized settings for specific models
config = get_model_config("gemini-2.0-flash-exp")
# Returns: {"temperature": 0.1, "max_tokens": 1000, ...}
```

### Custom Tools

```python
from src.adk_alpha.tools import create_custom_tool

def my_tool(param: str) -> str:
    return f"Processed: {param}"

custom_tool = create_custom_tool(my_tool, 
    name="my_tool", 
    description="Custom processing tool"
)
```

### Database Integration

```python
from src.adk_alpha.tools import CustomDatabaseTool

db_tool = CustomDatabaseTool("postgresql://user:pass@host/db")
agent = BasicAgent("db_agent", "Database assistant", tools=[db_tool])
```

## 🔒 Security

### Authentication

- Service Account credentials via `GOOGLE_APPLICATION_CREDENTIALS`
- API key authentication via `GOOGLE_API_KEY`
- Secret Manager integration for sensitive data

### Input Validation

- Query sanitization for database tools
- URL validation for web scraping
- Safe evaluation for calculator functions

### Session Management

- User isolation in database sessions
- Optional encryption for session data
- Configurable session persistence

## 📊 Monitoring

### Structured Logging

```python
from src.adk_alpha.config import setup_logging

# Configure structured logging
setup_logging("INFO")

# Logs include execution metrics, tool usage, errors
```

### Performance Tracking

- Execution time monitoring
- Token usage tracking  
- Tool call statistics
- Error rate metrics

## 🚀 Deployment

### Development Setup

```bash
# Install in development mode
pip install -e .

# Run with development settings
export SERVE_WEB_INTERFACE=True
python examples/basic_agent_example.py
```

### Production Considerations

- Configure proper authentication
- Set up database for session persistence
- Enable monitoring and logging
- Use connection pooling for databases
- Implement rate limiting for external APIs

## 🤝 Contributing

### Development Workflow

1. Create feature branch
2. Implement changes with tests
3. Run test suite and linting
4. Submit pull request

### Code Quality

```bash
# Format code
python -m pyink src/

# Sort imports  
python -m isort src/

# Type checking
python -m mypy src/
```

## 📝 API Reference

### Core Classes

#### BasicAgent

```python
BasicAgent(
    agent_name: str,
    instructions: str, 
    model: str = "gemini-2.0-flash-exp",
    tools: Optional[List] = None,
    enable_search: bool = True,
    enable_calculator: bool = True,
    enable_text_processing: bool = True,
    stream: bool = False
)
```

#### MultiAgentOrchestrator

```python
orchestrator = MultiAgentOrchestrator(agents: Dict[str, BasicAgent])

# Sequential execution
await orchestrator.execute_sequential(workflow: List[Dict])

# Parallel execution  
await orchestrator.execute_parallel(tasks: List[Dict])
```

### Configuration Functions

```python
# Load configuration
config = load_config()

# Get model settings
model_config = get_model_config(model_name: str)

# Setup logging
setup_logging(log_level: str = "INFO")

# Validate configuration
validate_config(config: Dict[str, Any]) -> bool
```

## 🐛 Troubleshooting

### Common Issues

**Authentication Errors**
```bash
# Verify credentials
gcloud auth application-default login
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

**Missing Dependencies**
```bash
# Install all optional dependencies
pip install -e ".[dev,advanced]"
```

**Configuration Issues**
```bash
# Check environment variables
python -c "from src.adk_alpha.config import load_config; print(load_config())"
```

### Debug Mode

```python
from src.adk_alpha.config import setup_logging
setup_logging("DEBUG")  # Enable detailed logging
```

## 📚 Additional Resources

- [Google ADK Documentation](https://cloud.google.com/adk)
- [Agent Development Best Practices](docs/best_practices.md)
- [Deployment Guide](docs/deployment.md)
- [API Reference](docs/api_reference.md)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:

1. Check the troubleshooting section
2. Review existing GitHub issues
3. Create a new issue with detailed description
4. Include logs and configuration (sanitized)

---

Built with ❤️ using Google Agent Development Kit specifications.
