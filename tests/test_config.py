"""
Unit tests for configuration management.

Tests cover configuration loading, validation, 
and integration with Google Cloud services.
"""

import pytest
import os
from unittest.mock import patch, Mock
from typing import Dict, Any

from src.adk_alpha.config import (
    load_config,
    get_model_config,
    get_secret,
    setup_logging,
    validate_config,
    get_database_config,
)


class TestConfigLoading:
    """Test configuration loading functionality."""
    
    @patch.dict(os.environ, {
        'GOOGLE_CLOUD_PROJECT': 'test-project',
        'GOOGLE_CLOUD_LOCATION': 'us-west1',
        'GOOGLE_GENAI_USE_VERTEXAI': 'True',
        'LOG_LEVEL': 'DEBUG'
    })
    def test_load_config_from_environment(self):
        """Test loading configuration from environment variables."""
        config = load_config()
        
        assert config['google_cloud_project'] == 'test-project'
        assert config['google_cloud_location'] == 'us-west1'
        assert config['google_genai_use_vertexai'] is True
        assert config['log_level'] == 'DEBUG'
    
    @patch.dict(os.environ, {}, clear=True)
    def test_load_config_defaults(self):
        """Test loading configuration with default values."""
        config = load_config()
        
        # Should have defaults for some values
        assert config.get('google_cloud_location', 'us-central1') == 'us-central1'
        assert config.get('google_genai_use_vertexai', True) is True
    
    @patch.dict(os.environ, {
        'GOOGLE_GENAI_USE_VERTEXAI': 'false',
        'SERVE_WEB_INTERFACE': 'False',
        'DISABLE_WEB_DRIVER': '1'
    })
    def test_load_config_boolean_parsing(self):
        """Test boolean value parsing in configuration."""
        config = load_config()
        
        assert config['google_genai_use_vertexai'] is False
        assert config['serve_web_interface'] is False
        assert config['disable_web_driver'] is False  # 1 means disabled, so False
    
    @patch('src.adk_alpha.config.load_dotenv')
    def test_load_config_dotenv_integration(self, mock_load_dotenv):
        """Test that load_dotenv is called to load .env file."""
        load_config()
        mock_load_dotenv.assert_called_once()


class TestModelConfiguration:
    """Test model configuration functionality."""
    
    def test_get_model_config_gemini_flash(self):
        """Test getting configuration for Gemini 2.0 Flash."""
        config = get_model_config("gemini-2.0-flash-exp")
        
        assert config['temperature'] == 0.1
        assert config['max_tokens'] == 1000
        assert config['top_p'] == 0.9
        assert config['top_k'] == 40
    
    def test_get_model_config_gemini_pro(self):
        """Test getting configuration for Gemini 1.5 Pro."""
        config = get_model_config("gemini-1.5-pro")
        
        assert config['temperature'] == 0.2
        assert config['max_tokens'] == 2000
        assert config['top_p'] == 0.8
        assert config['top_k'] == 40
    
    def test_get_model_config_claude(self):
        """Test getting configuration for Claude."""
        config = get_model_config("claude-3-sonnet")
        
        assert config['temperature'] == 0.1
        assert config['max_tokens'] == 1000
        # Claude doesn't use top_p and top_k in our config
    
    def test_get_model_config_unknown_model(self):
        """Test getting configuration for unknown model returns default."""
        config = get_model_config("unknown-model")
        
        # Should return the default (gemini-2.0-flash-exp) configuration
        assert config['temperature'] == 0.1
        assert config['max_tokens'] == 1000
        assert config['top_p'] == 0.9
        assert config['top_k'] == 40
    
    def test_get_model_config_default(self):
        """Test getting default model configuration."""
        config = get_model_config()
        
        # Should return gemini-2.0-flash-exp config by default
        assert config['temperature'] == 0.1
        assert config['max_tokens'] == 1000


class TestSecretManagement:
    """Test secret management functionality."""
    
    @patch('src.adk_alpha.config.secretmanager.SecretManagerServiceClient')
    @patch.dict(os.environ, {'GOOGLE_CLOUD_PROJECT': 'test-project'})
    def test_get_secret_success(self, mock_client_class):
        """Test successful secret retrieval."""
        # Mock the secret manager client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.payload.data.decode.return_value = "secret-value"
        mock_client.access_secret_version.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        secret_value = get_secret("test-secret")
        
        assert secret_value == "secret-value"
        mock_client.access_secret_version.assert_called_once()
    
    @patch('src.adk_alpha.config.secretmanager.SecretManagerServiceClient')
    def test_get_secret_with_project_id(self, mock_client_class):
        """Test secret retrieval with explicit project ID."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.payload.data.decode.return_value = "secret-value"
        mock_client.access_secret_version.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        secret_value = get_secret("test-secret", "explicit-project")
        
        assert secret_value == "secret-value"
        # Verify the correct project path was used
        call_args = mock_client.access_secret_version.call_args
        assert "explicit-project" in call_args[1]["request"]["name"]
    
    @patch.dict(os.environ, {}, clear=True)
    def test_get_secret_no_project_id(self):
        """Test secret retrieval fails when no project ID is available."""
        with pytest.raises(ValueError, match="Google Cloud project ID not configured"):
            get_secret("test-secret")
    
    @patch('src.adk_alpha.config.secretmanager.SecretManagerServiceClient')
    @patch.dict(os.environ, {'GOOGLE_CLOUD_PROJECT': 'test-project'})
    def test_get_secret_failure(self, mock_client_class):
        """Test secret retrieval failure handling."""
        mock_client = Mock()
        mock_client.access_secret_version.side_effect = Exception("Access denied")
        mock_client_class.return_value = mock_client
        
        with pytest.raises(Exception, match="Failed to retrieve secret test-secret"):
            get_secret("test-secret")


class TestLoggingSetup:
    """Test logging setup functionality."""
    
    @patch('src.adk_alpha.config.logging.basicConfig')
    @patch('src.adk_alpha.config.structlog.configure')
    def test_setup_logging_default(self, mock_structlog_config, mock_basic_config):
        """Test logging setup with default level."""
        setup_logging()
        
        mock_basic_config.assert_called_once()
        mock_structlog_config.assert_called_once()
        
        # Check that INFO level was used (default)
        call_args = mock_basic_config.call_args
        assert call_args[1]['level'] == 20  # logging.INFO
    
    @patch('src.adk_alpha.config.logging.basicConfig')
    @patch('src.adk_alpha.config.structlog.configure')
    def test_setup_logging_debug(self, mock_structlog_config, mock_basic_config):
        """Test logging setup with DEBUG level."""
        setup_logging("DEBUG")
        
        # Check that DEBUG level was used
        call_args = mock_basic_config.call_args
        assert call_args[1]['level'] == 10  # logging.DEBUG
    
    @patch('src.adk_alpha.config.logging.basicConfig')
    @patch('src.adk_alpha.config.structlog.configure')
    def test_setup_logging_custom_level(self, mock_structlog_config, mock_basic_config):
        """Test logging setup with custom level."""
        setup_logging("WARNING")
        
        # Check that WARNING level was used
        call_args = mock_basic_config.call_args
        assert call_args[1]['level'] == 30  # logging.WARNING


class TestConfigValidation:
    """Test configuration validation functionality."""
    
    def test_validate_config_success(self):
        """Test successful configuration validation."""
        config = {
            'google_cloud_project': 'test-project',
            'google_cloud_location': 'us-central1',
            'google_application_credentials': '/path/to/credentials.json'
        }
        
        result = validate_config(config)
        assert result is True
    
    def test_validate_config_missing_required_fields(self):
        """Test validation failure with missing required fields."""
        config = {
            'google_cloud_location': 'us-central1'
            # Missing google_cloud_project
        }
        
        with pytest.raises(ValueError, match="Missing required configuration"):
            validate_config(config)
    
    def test_validate_config_no_auth_method(self):
        """Test validation failure with no authentication method."""
        config = {
            'google_cloud_project': 'test-project',
            'google_cloud_location': 'us-central1'
            # No authentication credentials
        }
        
        with patch('src.adk_alpha.config.default') as mock_default:
            mock_default.side_effect = Exception("No credentials found")
            
            with pytest.raises(ValueError, match="No valid authentication method found"):
                validate_config(config)
    
    @patch('src.adk_alpha.config.default')
    def test_validate_config_default_credentials_success(self, mock_default):
        """Test validation success with default credentials."""
        config = {
            'google_cloud_project': 'test-project',
            'google_cloud_location': 'us-central1'
            # No explicit credentials, but default should work
        }
        
        mock_default.return_value = (Mock(), 'test-project')
        
        result = validate_config(config)
        assert result is True
    
    @patch('src.adk_alpha.config.default')
    def test_validate_config_default_credentials_no_project(self, mock_default):
        """Test validation failure when default credentials have no project."""
        config = {
            'google_cloud_project': 'test-project',
            'google_cloud_location': 'us-central1'
        }
        
        mock_default.return_value = (Mock(), None)  # No project in credentials
        
        with pytest.raises(ValueError, match="No authentication method configured"):
            validate_config(config)


class TestDatabaseConfiguration:
    """Test database configuration functionality."""
    
    @patch('src.adk_alpha.config.load_config')
    def test_get_database_config_with_url(self, mock_load_config):
        """Test database configuration with explicit URL."""
        mock_load_config.return_value = {
            'session_db_url': 'postgresql://user:pass@host:5432/db'
        }
        
        config = get_database_config()
        
        assert config['url'] == 'postgresql://user:pass@host:5432/db'
        assert config['pool_size'] == 10
        assert config['max_overflow'] == 20
        assert config['pool_pre_ping'] is True
        assert config['pool_recycle'] == 3600
    
    @patch('src.adk_alpha.config.load_config')
    def test_get_database_config_in_memory_default(self, mock_load_config):
        """Test database configuration defaults to in-memory."""
        mock_load_config.return_value = {}  # No database URL configured
        
        config = get_database_config()
        
        assert config['type'] == 'in_memory'
        assert config['persist'] is False


class TestConfigIntegration:
    """Integration tests for configuration components."""
    
    @patch.dict(os.environ, {
        'GOOGLE_CLOUD_PROJECT': 'integration-test-project',
        'GOOGLE_CLOUD_LOCATION': 'us-west2',
        'GOOGLE_GENAI_USE_VERTEXAI': 'True',
        'LOG_LEVEL': 'INFO',
        'SESSION_DB_URL': 'sqlite:///test.db'
    })
    def test_full_configuration_flow(self):
        """Test complete configuration loading and validation flow."""
        # Load configuration
        config = load_config()
        
        # Validate configuration
        # We'll mock the authentication check to avoid actual credential requirements
        with patch('src.adk_alpha.config.default') as mock_default:
            mock_default.return_value = (Mock(), 'integration-test-project')
            is_valid = validate_config(config)
            assert is_valid
        
        # Get model configuration
        model_config = get_model_config()
        assert 'temperature' in model_config
        
        # Get database configuration
        db_config = get_database_config()
        assert 'url' in db_config or 'type' in db_config
        
        # Verify expected values
        assert config['google_cloud_project'] == 'integration-test-project'
        assert config['google_cloud_location'] == 'us-west2'
        assert config['google_genai_use_vertexai'] is True
    
    def test_minimal_configuration(self):
        """Test system works with minimal configuration."""
        minimal_config = {
            'google_cloud_project': 'minimal-project',
            'google_cloud_location': 'us-central1',
            'google_application_credentials': '/path/to/creds.json'
        }
        
        # Should validate successfully
        result = validate_config(minimal_config)
        assert result is True
        
        # Should get model config
        model_config = get_model_config()
        assert isinstance(model_config, dict)
        assert 'temperature' in model_config
