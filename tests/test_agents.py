"""
Unit tests for agent implementations.

Tests cover basic agent functionality, specialized agents,
multi-agent orchestration, and error handling.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from src.adk_alpha.agents import (
    BasicAgent,
    ResearchAgent,
    AnalystAgent,
    WriterAgent,
    MultiAgentOrchestrator,
    create_basic_agent,
    create_multi_agent_system,
)
from src.adk_alpha.tools import calculator_tool, text_processor_tool


class TestBasicAgent:
    """Test basic agent functionality."""
    
    @pytest.fixture
    def basic_agent(self):
        """Create a basic agent for testing."""
        return BasicAgent(
            agent_name="test_agent",
            instructions="You are a test agent.",
            model="gemini-2.0-flash-exp",
            enable_search=False,  # Disable external dependencies for testing
        )
    
    def test_agent_initialization(self, basic_agent):
        """Test basic agent initialization."""
        assert basic_agent.agent_name == "test_agent"
        assert basic_agent.instructions == "You are a test agent."
        assert basic_agent.model == "gemini-2.0-flash-exp"
        assert not basic_agent.stream
        assert basic_agent.agent is not None
        assert basic_agent.runner is not None
    
    def test_agent_initialization_with_tools(self):
        """Test agent initialization with custom tools."""
        custom_tools = [calculator_tool, text_processor_tool]
        agent = BasicAgent(
            agent_name="tool_agent",
            instructions="Test agent with tools.",
            tools=custom_tools,
            enable_search=False,
        )
        
        assert agent.agent_name == "tool_agent"
        # The agent should have custom tools plus enabled built-in tools
        # We can't easily test the exact tool count due to internal ADK behavior
    
    def test_agent_initialization_streaming(self):
        """Test agent initialization with streaming enabled."""
        agent = BasicAgent(
            agent_name="stream_agent",
            instructions="Streaming test agent.",
            stream=True,
            enable_search=False,
        )
        
        assert agent.stream
    
    @pytest.mark.asyncio
    async def test_agent_execute_mock(self, basic_agent):
        """Test agent execution with mocked runner."""
        mock_response = Mock()
        mock_response.content = "Test response content"
        mock_response.tool_calls = []
        
        with patch.object(basic_agent.runner, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_response
            
            result = await basic_agent.execute("Test query")
            
            assert result["status"] == "success"
            assert result["agent_name"] == "test_agent"
            assert result["query"] == "Test query"
            assert result["result"]["content"] == "Test response content"
            assert "execution_time" in result
            
            mock_run.assert_called_once_with("Test query", session_id=None)
    
    @pytest.mark.asyncio
    async def test_agent_execute_with_session(self, basic_agent):
        """Test agent execution with session ID."""
        mock_response = Mock()
        mock_response.content = "Session response"
        mock_response.tool_calls = []
        
        with patch.object(basic_agent.runner, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_response
            
            result = await basic_agent.execute("Query with session", session_id="test-session")
            
            assert result["status"] == "success"
            assert result["session_id"] == "test-session"
            mock_run.assert_called_once_with("Query with session", session_id="test-session")
    
    @pytest.mark.asyncio
    async def test_agent_execute_error_handling(self, basic_agent):
        """Test agent error handling."""
        with patch.object(basic_agent.runner, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = Exception("Test error")
            
            result = await basic_agent.execute("Failing query")
            
            assert result["status"] == "error"
            assert result["agent_name"] == "test_agent"
            assert result["query"] == "Failing query"
            assert result["error"] == "Test error"
            assert result["error_type"] == "Exception"
    
    @pytest.mark.asyncio
    async def test_agent_execute_streaming_mock(self):
        """Test streaming agent execution with mocked runner."""
        agent = BasicAgent(
            agent_name="stream_test",
            instructions="Streaming test.",
            stream=True,
            enable_search=False,
        )
        
        # Mock streaming chunks
        mock_chunks = [
            Mock(content="Part 1", tool_calls=[]),
            Mock(content=" Part 2", tool_calls=[]),
            Mock(content=" Part 3", tool_calls=[])
        ]
        
        async def mock_stream_generator(*args, **kwargs):
            for chunk in mock_chunks:
                yield chunk
        
        with patch.object(agent.runner, 'stream', side_effect=mock_stream_generator):
            result = await agent.execute("Streaming query")
            
            assert result["status"] == "success"
            assert result["result"]["content"] == "Part 1 Part 2 Part 3"
            assert result["result"]["chunks"] == 3


class TestSpecializedAgents:
    """Test specialized agent implementations."""
    
    def test_research_agent_initialization(self):
        """Test research agent initialization."""
        agent = ResearchAgent()
        
        assert agent.agent_name == "researcher"
        assert "research specialist" in agent.instructions.lower()
        # Should have web scraping capabilities
    
    def test_analyst_agent_initialization(self):
        """Test analyst agent initialization."""
        agent = AnalystAgent()
        
        assert agent.agent_name == "analyst"
        assert "data analysis" in agent.instructions.lower()
        # Should have data analysis capabilities
    
    def test_writer_agent_initialization(self):
        """Test writer agent initialization."""
        agent = WriterAgent()
        
        assert agent.agent_name == "writer"
        assert "writer" in agent.instructions.lower()
        # Writer agents have search and calculator disabled by default
    
    def test_custom_specialized_agent(self):
        """Test creating specialized agent with custom configuration."""
        agent = ResearchAgent(
            agent_name="custom_researcher",
            instructions="Custom research instructions.",
            model="gemini-1.5-pro"
        )
        
        assert agent.agent_name == "custom_researcher"
        assert agent.instructions == "Custom research instructions."
        assert agent.model == "gemini-1.5-pro"


class TestMultiAgentOrchestrator:
    """Test multi-agent orchestration functionality."""
    
    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing."""
        agents = {}
        for name in ["researcher", "analyst", "writer"]:
            mock_agent = Mock()
            mock_agent.execute = AsyncMock()
            agents[name] = mock_agent
        return agents
    
    @pytest.fixture
    def orchestrator(self, mock_agents):
        """Create orchestrator with mock agents."""
        return MultiAgentOrchestrator(mock_agents)
    
    def test_orchestrator_initialization(self, mock_agents):
        """Test orchestrator initialization."""
        orchestrator = MultiAgentOrchestrator(mock_agents)
        
        assert len(orchestrator.agents) == 3
        assert "researcher" in orchestrator.agents
        assert "analyst" in orchestrator.agents
        assert "writer" in orchestrator.agents
    
    @pytest.mark.asyncio
    async def test_sequential_execution(self, orchestrator, mock_agents):
        """Test sequential workflow execution."""
        # Configure mock responses
        mock_agents["researcher"].execute.return_value = {
            "status": "success",
            "result": {"content": "Research findings"}
        }
        mock_agents["analyst"].execute.return_value = {
            "status": "success", 
            "result": {"content": "Analysis results"}
        }
        mock_agents["writer"].execute.return_value = {
            "status": "success",
            "result": {"content": "Final report"}
        }
        
        workflow = [
            {"agent_name": "researcher", "query": "Research topic X"},
            {"agent_name": "analyst", "query": "Analyze the data"},
            {"agent_name": "writer", "query": "Write a report"}
        ]
        
        result = await orchestrator.execute_sequential(workflow)
        
        assert result["status"] == "success"
        assert result["workflow_type"] == "sequential"
        assert len(result["results"]) == 3
        assert "execution_time" in result
        
        # Verify all agents were called
        mock_agents["researcher"].execute.assert_called_once()
        mock_agents["analyst"].execute.assert_called_once()
        mock_agents["writer"].execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_parallel_execution(self, orchestrator, mock_agents):
        """Test parallel workflow execution."""
        # Configure mock responses
        mock_agents["researcher"].execute.return_value = {
            "status": "success",
            "result": {"content": "Research findings"}
        }
        mock_agents["analyst"].execute.return_value = {
            "status": "success",
            "result": {"content": "Analysis results"}
        }
        
        tasks = [
            {"agent_name": "researcher", "query": "Research background"},
            {"agent_name": "analyst", "query": "Analyze data"}
        ]
        
        result = await orchestrator.execute_parallel(tasks)
        
        assert result["status"] == "success"
        assert result["workflow_type"] == "parallel"
        assert len(result["results"]) == 2
        assert result["successful_tasks"] == 2
        assert result["total_tasks"] == 2
        
        # Verify both agents were called
        mock_agents["researcher"].execute.assert_called_once()
        mock_agents["analyst"].execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_sequential_execution_with_error(self, orchestrator, mock_agents):
        """Test sequential execution with one agent failing."""
        # First agent succeeds, second fails
        mock_agents["researcher"].execute.return_value = {
            "status": "success",
            "result": {"content": "Research findings"}
        }
        mock_agents["analyst"].execute.side_effect = Exception("Analysis failed")
        
        workflow = [
            {"agent_name": "researcher", "query": "Research topic"},
            {"agent_name": "analyst", "query": "Analyze data"}
        ]
        
        result = await orchestrator.execute_sequential(workflow)
        
        assert result["status"] == "error"
        assert "partial_results" in result
        assert len(result["partial_results"]) == 1  # Only first agent completed
    
    @pytest.mark.asyncio
    async def test_parallel_execution_with_mixed_results(self, orchestrator, mock_agents):
        """Test parallel execution with mixed success/failure."""
        # One succeeds, one fails
        mock_agents["researcher"].execute.return_value = {
            "status": "success",
            "result": {"content": "Research findings"}
        }
        mock_agents["analyst"].execute.side_effect = Exception("Analysis failed")
        
        tasks = [
            {"agent_name": "researcher", "query": "Research background"},
            {"agent_name": "analyst", "query": "Analyze data"}
        ]
        
        result = await orchestrator.execute_parallel(tasks)
        
        assert result["status"] == "success"  # Overall success even with partial failures
        assert result["successful_tasks"] == 1
        assert result["total_tasks"] == 2
        
        # Check individual results
        results = result["results"]
        research_result = next(r for r in results if r["agent_name"] == "researcher")
        analyst_result = next(r for r in results if r["agent_name"] == "analyst")
        
        assert research_result["result"]["status"] == "success"
        assert analyst_result["result"]["status"] == "error"
    
    @pytest.mark.asyncio
    async def test_unknown_agent_error(self, orchestrator):
        """Test error handling for unknown agent."""
        workflow = [
            {"agent_name": "unknown_agent", "query": "Do something"}
        ]
        
        result = await orchestrator.execute_sequential(workflow)
        
        assert result["status"] == "error"
        assert "not found" in result["error"]


class TestFactoryFunctions:
    """Test factory functions for creating agents and systems."""
    
    def test_create_basic_agent(self):
        """Test basic agent factory function."""
        agent = create_basic_agent(
            agent_name="factory_agent",
            instructions="Factory-created agent.",
            enable_search=False
        )
        
        assert isinstance(agent, BasicAgent)
        assert agent.agent_name == "factory_agent"
        assert agent.instructions == "Factory-created agent."
    
    def test_create_multi_agent_system(self):
        """Test multi-agent system factory function."""
        agent_configs = [
            {
                "type": "basic",
                "agent_name": "basic_agent",
                "instructions": "Basic agent.",
                "enable_search": False
            },
            {
                "type": "research", 
                "agent_name": "research_agent"
            },
            {
                "type": "analyst",
                "agent_name": "analyst_agent"
            },
            {
                "type": "writer",
                "agent_name": "writer_agent"
            }
        ]
        
        orchestrator = create_multi_agent_system(agent_configs)
        
        assert isinstance(orchestrator, MultiAgentOrchestrator)
        assert len(orchestrator.agents) == 4
        assert isinstance(orchestrator.agents["basic_agent"], BasicAgent)
        assert isinstance(orchestrator.agents["research_agent"], ResearchAgent)
        assert isinstance(orchestrator.agents["analyst_agent"], AnalystAgent)
        assert isinstance(orchestrator.agents["writer_agent"], WriterAgent)
    
    def test_create_multi_agent_system_unknown_type(self):
        """Test error handling for unknown agent type."""
        agent_configs = [
            {
                "type": "unknown_type",
                "agent_name": "unknown_agent",
                "instructions": "Unknown agent type."
            }
        ]
        
        with pytest.raises(ValueError, match="Unknown agent type"):
            create_multi_agent_system(agent_configs)


class TestIntegration:
    """Integration tests for agent components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_basic_flow(self):
        """Test end-to-end basic agent flow with minimal external dependencies."""
        agent = BasicAgent(
            agent_name="integration_test",
            instructions="You are a test agent for integration testing.",
            enable_search=False,  # Disable external dependencies
            enable_calculator=True,
            enable_text_processing=True
        )
        
        # This test would require actual ADK implementation to work
        # For now, we'll mock the runner behavior
        mock_response = Mock()
        mock_response.content = "Integration test response"
        mock_response.tool_calls = []
        
        with patch.object(agent.runner, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_response
            
            result = await agent.execute("Calculate 2 + 2")
            
            assert result["status"] == "success"
            assert "integration test response" in result["result"]["content"].lower()
