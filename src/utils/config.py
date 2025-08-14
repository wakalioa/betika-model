"""
Configuration Management for Betika Virtual Games Prediction Model

Handles loading and managing application configuration from YAML files and environment variables.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataCollectionConfig:
    """Data collection configuration."""
    base_url: str
    virtual_games: list
    collection_interval: int
    max_retries: int
    timeout: int


@dataclass
class ModelConfig:
    """Model configuration."""
    algorithms: list
    training: dict
    features: list


@dataclass
class PredictionConfig:
    """Prediction configuration."""
    confidence_threshold: float
    max_predictions_per_day: int
    min_data_points: int


@dataclass
class DatabaseConfig:
    """Database configuration."""
    type: str
    path: str
    backup_interval: int


@dataclass
class APIConfig:
    """API configuration."""
    host: str
    port: int
    debug: bool
    cors_origins: list


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str
    format: str
    file: str
    max_file_size: str
    backup_count: int


class Config:
    """Main configuration class that loads and manages all settings."""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or self._find_config_file()
        self.config_data = self._load_config()
        
        # Initialize configuration sections
        self.data_collection = self._load_data_collection_config()
        self.models = self._load_model_config()
        self.prediction = self._load_prediction_config()
        self.database = self._load_database_config()
        self.api = self._load_api_config()
        self.logging = self._load_logging_config()
        
        # Setup logging
        self._setup_logging()
    
    def _find_config_file(self) -> str:
        """Find configuration file in common locations."""
        possible_paths = [
            'config/config.yaml',
            'config.yaml',
            '../config/config.yaml',
            os.path.join(os.path.dirname(__file__), '../../config/config.yaml')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError("Configuration file not found. Please ensure config.yaml exists.")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Override with environment variables
            config = self._override_with_env_vars(config)
            
            return config
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def _override_with_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Override configuration with environment variables."""
        env_mappings = {
            'BETIKA_BASE_URL': ['data_collection', 'base_url'],
            'BETIKA_COLLECTION_INTERVAL': ['data_collection', 'collection_interval'],
            'BETIKA_DB_PATH': ['database', 'path'],
            'BETIKA_API_HOST': ['api', 'host'],
            'BETIKA_API_PORT': ['api', 'port'],
            'BETIKA_LOG_LEVEL': ['logging', 'level'],
            'BETIKA_LOG_FILE': ['logging', 'file'],
        }
        
        for env_var, config_path in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Convert to appropriate type
                if config_path[-1] in ['collection_interval', 'port', 'backup_interval']:
                    value = int(value)
                elif config_path[-1] in ['debug']:
                    value = value.lower() in ('true', '1', 'yes')
                
                # Set nested configuration value
                current = config
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[config_path[-1]] = value
        
        return config
    
    def _load_data_collection_config(self) -> Dict[str, Any]:
        """Load data collection configuration."""
        config = self.config_data.get('data_collection', {})
        return {
            'base_url': config.get('base_url', 'https://www.betika.com'),
            'virtual_games': config.get('virtual_games', ['virtual-football']),
            'collection_interval': config.get('collection_interval', 300),
            'max_retries': config.get('max_retries', 3),
            'timeout': config.get('timeout', 30)
        }
    
    def _load_model_config(self) -> Dict[str, Any]:
        """Load model configuration."""
        config = self.config_data.get('models', {})
        return {
            'algorithms': config.get('algorithms', ['random_forest', 'xgboost']),
            'training': config.get('training', {
                'test_size': 0.2,
                'validation_size': 0.1,
                'cross_validation_folds': 5
            }),
            'features': config.get('features', [
                'team_stats', 'historical_performance', 'time_patterns', 'odds_movements'
            ])
        }
    
    def _load_prediction_config(self) -> Dict[str, Any]:
        """Load prediction configuration."""
        config = self.config_data.get('prediction', {})
        return {
            'confidence_threshold': config.get('confidence_threshold', 0.7),
            'max_predictions_per_day': config.get('max_predictions_per_day', 50),
            'min_data_points': config.get('min_data_points', 100)
        }
    
    def _load_database_config(self) -> Dict[str, Any]:
        """Load database configuration."""
        config = self.config_data.get('database', {})
        db_path = config.get('path', 'data/betika_games.db')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        return {
            'type': config.get('type', 'sqlite'),
            'path': db_path,
            'backup_interval': config.get('backup_interval', 86400)
        }
    
    def _load_api_config(self) -> Dict[str, Any]:
        """Load API configuration."""
        config = self.config_data.get('api', {})
        return {
            'host': config.get('host', '0.0.0.0'),
            'port': config.get('port', 8000),
            'debug': config.get('debug', False),
            'cors_origins': config.get('cors_origins', ['http://localhost:3000'])
        }
    
    def _load_logging_config(self) -> Dict[str, Any]:
        """Load logging configuration."""
        config = self.config_data.get('logging', {})
        log_file = config.get('file', 'logs/betika_model.log')
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        return {
            'level': config.get('level', 'INFO'),
            'format': config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            'file': log_file,
            'max_file_size': config.get('max_file_size', '10MB'),
            'backup_count': config.get('backup_count', 5)
        }
    
    def _setup_logging(self):
        """Setup application logging."""
        import logging.handlers
        import colorlog
        
        # Create logger
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, self.logging['level'].upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler with colors
        console_handler = colorlog.StreamHandler()
        console_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            self.logging['file'],
            maxBytes=self._parse_file_size(self.logging['max_file_size']),
            backupCount=self.logging['backup_count']
        )
        file_formatter = logging.Formatter(self.logging['format'])
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info("Logging configured successfully")
    
    def _parse_file_size(self, size_str: str) -> int:
        """Parse file size string to bytes."""
        size_str = size_str.upper()
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self.config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def reload(self):
        """Reload configuration from file."""
        self.config_data = self._load_config()
        self.data_collection = self._load_data_collection_config()
        self.models = self._load_model_config()
        self.prediction = self._load_prediction_config()
        self.database = self._load_database_config()
        self.api = self._load_api_config()
        self.logging = self._load_logging_config()
        
        logging.getLogger(__name__).info("Configuration reloaded")
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        errors = []
        
        # Validate data collection settings
        if not self.data_collection['base_url']:
            errors.append("Base URL is required")
        
        if self.data_collection['collection_interval'] < 60:
            errors.append("Collection interval should be at least 60 seconds")
        
        # Validate model settings
        if not self.models['algorithms']:
            errors.append("At least one algorithm must be specified")
        
        # Validate prediction settings
        if not 0 < self.prediction['confidence_threshold'] <= 1:
            errors.append("Confidence threshold must be between 0 and 1")
        
        if self.prediction['min_data_points'] < 10:
            errors.append("Minimum data points should be at least 10")
        
        # Validate API settings
        if not 1 <= self.api['port'] <= 65535:
            errors.append("API port must be between 1 and 65535")
        
        if errors:
            logging.getLogger(__name__).error(f"Configuration validation failed: {'; '.join(errors)}")
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'data_collection': self.data_collection,
            'models': self.models,
            'prediction': self.prediction,
            'database': self.database,
            'api': self.api,
            'logging': self.logging
        }