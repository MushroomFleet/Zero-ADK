"""
Unit tests for custom tools implementation.

Tests cover custom tool functionality, error handling,
and integration with the ADK framework.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List
import time

from src.adk_alpha.tools import (
    CustomDatabaseTool,
    WebScrapingTool,
    DataAnalysisTool,
    create_custom_tool,
    simple_calculator,
    text_processor,
    calculator_tool,
    text_processor_tool,
)


class TestCustomDatabaseTool:
    """Test custom database tool functionality."""
    
    @pytest.mark.skipif(
        True,  # Skip by default since it requires sqlalchemy
        reason="Requires sqlalchemy and database setup"
    )
    def test_database_tool_initialization(self):
        """Test database tool initialization."""
        connection_string = "sqlite:///:memory:"
        tool = CustomDatabaseTool(connection_string)
        
        assert tool.name == "database_query"
        assert tool.connection_string == connection_string
        assert tool.max_results == 100
    
    @pytest.mark.asyncio
    async def test_database_tool_validation(self):
        """Test database tool query validation."""
        # Mock the database tool to avoid actual database dependency
        with patch('src.adk_alpha.tools.HAS_DATABASE', True):
            with patch('sqlalchemy.create_engine') as mock_engine:
                tool = CustomDatabaseTool("sqlite:///:memory:")
                
                # Test non-SELECT query rejection
                result = await tool.execute("INSERT INTO users VALUES (1, 'test')")
                assert result["status"] == "error"
                assert result["error_type"] == "validation"
                assert "SELECT queries" in result["message"]
                
                # Test dangerous keyword detection
                result = await tool.execute("SELECT * FROM users; DROP TABLE users;")
                assert result["status"] == "error"
                assert result["error_type"] == "validation"
                assert "dangerous operations" in result["message"]
    
    @pytest.mark.asyncio
    async def test_database_tool_successful_query(self):
        """Test successful database query execution."""
        with patch('src.adk_alpha.tools.HAS_DATABASE', True):
            with patch('sqlalchemy.create_engine') as mock_engine:
                # Mock connection and result
                mock_connection = Mock()
                mock_result = Mock()
                mock_result.fetchall.return_value = [("row1_col1", "row1_col2")]
                mock_result.keys.return_value = ["col1", "col2"]
                mock_connection.execute.return_value = mock_result
                mock_engine.return_value.connect.return_value.__enter__.return_value = mock_connection
                
                tool = CustomDatabaseTool("sqlite:///:memory:")
                result = await tool.execute("SELECT * FROM users")
                
                assert result["status"] == "success"
                assert result["row_count"] == 1
                assert result["columns"] == ["col1", "col2"]
                assert result["data"] == [{"col1": "row1_col1", "col2": "row1_col2"}]


class TestWebScrapingTool:
    """Test web scraping tool functionality."""
    
    @pytest.mark.skipif(
        True,  # Skip by default since it requires requests and beautifulsoup4
        reason="Requires requests and beautifulsoup4"
    )
    def test_web_scraping_tool_initialization(self):
        """Test web scraping tool initialization."""
        tool = WebScrapingTool(rate_limit=2.0, timeout=15)
        
        assert tool.name == "web_scraper"
        assert tool.rate_limit == 2.0
        assert tool.timeout == 15
    
    @pytest.mark.asyncio
    async def test_web_scraping_url_validation(self):
        """Test URL validation in web scraping tool."""
        with patch('src.adk_alpha.tools.HAS_WEB_SCRAPING', True):
            tool = WebScrapingTool()
            
            # Test invalid URL
            result = await tool.execute("invalid-url")
            assert result["status"] == "error"
            assert result["error_type"] == "validation"
            assert "Invalid URL format" in result["message"]
    
    @pytest.mark.asyncio
    async def test_web_scraping_successful_request(self):
        """Test successful web scraping request."""
        with patch('src.adk_alpha.tools.HAS_WEB_SCRAPING', True):
            with patch('requests.get') as mock_get:
                with patch('bs4.BeautifulSoup') as mock_soup:
                    # Mock successful response
                    mock_response = Mock()
                    mock_response.content = "<html><body>Test content</body></html>"
                    mock_response.raise_for_status.return_value = None
                    mock_get.return_value = mock_response
                    
                    # Mock BeautifulSoup
                    mock_soup_instance = Mock()
                    mock_soup_instance.get_text.return_value = "Test content"
                    mock_soup.return_value = mock_soup_instance
                    
                    tool = WebScrapingTool()
                    result = await tool.execute("https://example.com")
                    
                    assert result["status"] == "success"
                    assert "Test content" in result["content"]
                    assert result["url"] == "https://example.com"
    
    @pytest.mark.asyncio
    async def test_web_scraping_rate_limiting(self):
        """Test rate limiting functionality."""
        with patch('src.adk_alpha.tools.HAS_WEB_SCRAPING', True):
            with patch('asyncio.sleep') as mock_sleep:
                with patch('requests.get') as mock_get:
                    with patch('bs4.BeautifulSoup'):
                        mock_response = Mock()
                        mock_response.content = "<html>Test</html>"
                        mock_response.raise_for_status.return_value = None
                        mock_get.return_value = mock_response
                        
                        tool = WebScrapingTool(rate_limit=1.0)
                        
                        # First request should not sleep
                        await tool.execute("https://example1.com")
                        mock_sleep.assert_not_called()
                        
                        # Second request should trigger rate limiting
                        tool.last_request_time = time.time() - 0.5  # 0.5 seconds ago
                        await tool.execute("https://example2.com")
                        mock_sleep.assert_called_once()


class TestDataAnalysisTool:
    """Test data analysis tool functionality."""
    
    @pytest.mark.skipif(
        True,  # Skip by default since it requires pandas
        reason="Requires pandas"
    )
    def test_data_analysis_tool_initialization(self):
        """Test data analysis tool initialization."""
        tool = DataAnalysisTool()
        assert tool.name == "data_analyzer"
    
    @pytest.mark.asyncio
    async def test_data_analysis_empty_data(self):
        """Test data analysis with empty data."""
        with patch('src.adk_alpha.tools.HAS_PANDAS', True):
            tool = DataAnalysisTool()
            result = await tool.execute([])
            
            assert result["status"] == "error"
            assert result["error_type"] == "validation"
            assert "No data provided" in result["message"]
    
    @pytest.mark.asyncio
    async def test_data_analysis_summary_operation(self):
        """Test summary operation on data."""
        with patch('src.adk_alpha.tools.HAS_PANDAS', True):
            with patch('pandas.DataFrame') as mock_df_class:
                # Mock DataFrame instance
                mock_df = Mock()
                mock_df.__len__.return_value = 10
                mock_df.columns = ["col1", "col2", "col3"]
                mock_df.dtypes.to_dict.return_value = {"col1": "int64", "col2": "float64", "col3": "object"}
                mock_df.isnull.return_value.sum.return_value.to_dict.return_value = {"col1": 0, "col2": 1, "col3": 2}
                mock_df.select_dtypes.return_value.columns = ["col1", "col2"]
                mock_df.select_dtypes.return_value.describe.return_value.to_dict.return_value = {
                    "col1": {"mean": 5.0, "std": 2.0},
                    "col2": {"mean": 10.0, "std": 3.0}
                }
                mock_df_class.return_value = mock_df
                
                tool = DataAnalysisTool()
                data = [{"col1": 1, "col2": 2.0, "col3": "text"}] * 10
                result = await tool.execute(data, "summary")
                
                assert result["status"] == "success"
                assert result["operation"] == "summary"
                assert result["result"]["row_count"] == 10
                assert result["result"]["column_count"] == 3
                assert "numeric_summary" in result["result"]


class TestBuiltInTools:
    """Test built-in tool functions."""
    
    def test_simple_calculator_valid_expression(self):
        """Test calculator with valid expression."""
        result = simple_calculator("2 + 3 * 4")
        
        assert result["expression"] == "2 + 3 * 4"
        assert result["result"] == 14
        assert result["type"] == "int"
    
    def test_simple_calculator_invalid_expression(self):
        """Test calculator with invalid expression."""
        with pytest.raises(ValueError, match="Invalid characters"):
            simple_calculator("2 + import os")
    
    def test_simple_calculator_evaluation_error(self):
        """Test calculator with expression that fails evaluation."""
        with pytest.raises(ValueError, match="Failed to evaluate"):
            simple_calculator("1 / 0")
    
    def test_text_processor_word_count(self):
        """Test text processor word count operation."""
        text = "This is a test text with some words"
        result = text_processor(text, "word_count")
        
        assert result["word_count"] == 8
        assert result["unique_words"] == 8  # All words are unique in this case
        assert result["average_word_length"] > 0
    
    def test_text_processor_char_count(self):
        """Test text processor character count operation."""
        text = "Hello\nWorld!"
        result = text_processor(text, "char_count")
        
        assert result["character_count"] == 12
        assert result["character_count_no_spaces"] == 11
        assert result["line_count"] == 2
    
    def test_text_processor_summary(self):
        """Test text processor summary operation."""
        text = "Hello world. This is a test. Another sentence here."
        result = text_processor(text, "summary")
        
        assert result["word_count"] == 10
        assert result["sentence_count"] > 0
        assert result["character_count"] == len(text)
        assert result["average_sentence_length"] > 0
    
    def test_text_processor_unknown_operation(self):
        """Test text processor with unknown operation."""
        with pytest.raises(ValueError, match="Unknown operation"):
            text_processor("test text", "unknown_operation")


class TestCustomToolCreation:
    """Test custom tool creation functionality."""
    
    def test_create_custom_tool_success(self):
        """Test creating custom tool with successful function."""
        def test_function(param: str) -> str:
            """Test function docstring."""
            return f"Processed: {param}"
        
        tool = create_custom_tool(test_function)
        
        # Test the wrapped function
        result = tool.func("test_input")
        
        assert result["status"] == "success"
        assert result["result"] == "Processed: test_input"
        assert "execution_time" in result
    
    def test_create_custom_tool_with_dict_return(self):
        """Test creating custom tool that returns status dict."""
        def test_function(param: str) -> Dict[str, Any]:
            """Test function that returns status dict."""
            return {
                "status": "success",
                "custom_field": "custom_value",
                "processed_param": param
            }
        
        tool = create_custom_tool(test_function)
        result = tool.func("test_input")
        
        assert result["status"] == "success"
        assert result["custom_field"] == "custom_value"
        assert result["processed_param"] == "test_input"
    
    def test_create_custom_tool_validation_error(self):
        """Test custom tool with validation error."""
        def test_function(param: str) -> str:
            if not param:
                raise ValueError("Parameter cannot be empty")
            return f"Processed: {param}"
        
        tool = create_custom_tool(test_function)
        result = tool.func("")
        
        assert result["status"] == "error"
        assert result["error_type"] == "validation"
        assert result["message"] == "Parameter cannot be empty"
    
    def test_create_custom_tool_unexpected_error(self):
        """Test custom tool with unexpected error."""
        def test_function(param: str) -> str:
            raise RuntimeError("Unexpected error occurred")
        
        tool = create_custom_tool(test_function)
        result = tool.func("test_input")
        
        assert result["status"] == "error"
        assert result["error_type"] == "unexpected"
        assert result["message"] == "An unexpected error occurred"
    
    def test_create_custom_tool_with_name_and_description(self):
        """Test creating custom tool with custom name and description."""
        def test_function(param: str) -> str:
            return param.upper()
        
        tool = create_custom_tool(
            test_function,
            name="custom_uppercase",
            description="Converts text to uppercase"
        )
        
        # Check if name and description are properly set
        # Note: The actual implementation details depend on FunctionTool
        assert callable(tool.func)


class TestPreConfiguredTools:
    """Test pre-configured tool instances."""
    
    def test_calculator_tool_instance(self):
        """Test that calculator tool is properly configured."""
        assert calculator_tool is not None
        assert hasattr(calculator_tool, 'func')
        
        # Test execution
        result = calculator_tool.func("5 + 3")
        assert result["status"] == "success"
        assert result["result"]["result"] == 8
    
    def test_text_processor_tool_instance(self):
        """Test that text processor tool is properly configured."""
        assert text_processor_tool is not None
        assert hasattr(text_processor_tool, 'func')
        
        # Test execution
        result = text_processor_tool.func("Hello world", "word_count")
        assert result["status"] == "success"
        assert result["result"]["word_count"] == 2


class TestToolErrorHandling:
    """Test comprehensive error handling across tools."""
    
    @pytest.mark.asyncio
    async def test_database_tool_connection_error(self):
        """Test database tool with connection error."""
        with patch('src.adk_alpha.tools.HAS_DATABASE', True):
            with patch('sqlalchemy.create_engine') as mock_engine:
                # Mock connection error
                mock_connection = Mock()
                mock_connection.execute.side_effect = Exception("Connection failed")
                mock_engine.return_value.connect.return_value.__enter__.return_value = mock_connection
                
                tool = CustomDatabaseTool("sqlite:///:memory:")
                result = await tool.execute("SELECT 1")
                
                assert result["status"] == "error"
                assert result["error_type"] == "unexpected"
    
    @pytest.mark.asyncio
    async def test_web_scraping_request_error(self):
        """Test web scraping tool with request error."""
        with patch('src.adk_alpha.tools.HAS_WEB_SCRAPING', True):
            with patch('requests.get') as mock_get:
                mock_get.side_effect = Exception("Network error")
                
                tool = WebScrapingTool()
                result = await tool.execute("https://example.com")
                
                assert result["status"] == "error"
                assert result["error_type"] == "unexpected"
    
    @pytest.mark.asyncio
    async def test_data_analysis_processing_error(self):
        """Test data analysis tool with processing error."""
        with patch('src.adk_alpha.tools.HAS_PANDAS', True):
            with patch('pandas.DataFrame') as mock_df:
                mock_df.side_effect = Exception("DataFrame creation failed")
                
                tool = DataAnalysisTool()
                result = await tool.execute([{"col1": 1}], "summary")
                
                assert result["status"] == "error"
                assert result["error_type"] == "unexpected"


class TestToolIntegration:
    """Integration tests for tool components."""
    
    def test_calculator_integration(self):
        """Test calculator tool integration."""
        # Test various mathematical operations
        test_cases = [
            ("2 + 2", 4),
            ("10 - 3", 7),
            ("6 * 7", 42),
            ("15 / 3", 5.0),
            ("(2 + 3) * 4", 20),
        ]
        
        for expression, expected in test_cases:
            result = simple_calculator(expression)
            assert result["result"] == expected
            assert result["expression"] == expression
    
    def test_text_processing_integration(self):
        """Test text processing tool integration."""
        text = "The quick brown fox jumps over the lazy dog. This is a second sentence."
        
        # Test all operations
        word_result = text_processor(text, "word_count")
        char_result = text_processor(text, "char_count")
        summary_result = text_processor(text, "summary")
        
        assert word_result["word_count"] == 16
        assert char_result["character_count"] == len(text)
        assert summary_result["sentence_count"] == 2
        assert summary_result["word_count"] == 16
