"""
Configuration module for OCR API
Handles environment-based configuration for security and production features
"""
import os
import secrets

class Config:
    """Base configuration"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', secrets.token_hex(32))
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # API Security
    API_KEYS = set(os.environ.get('API_KEYS', 'dev-key-123').split(','))
    
    # CORS Configuration
    ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', '*').split(',')
    
    # Rate Limiting
    RATE_LIMIT_ENABLED = os.environ.get('RATE_LIMIT_ENABLED', 'true').lower() == 'true'
    RATE_LIMIT_STORAGE_URL = os.environ.get('RATE_LIMIT_STORAGE_URL', 'memory://')
    
    # Rate limit values
    GLOBAL_RATE_LIMIT = os.environ.get('GLOBAL_RATE_LIMIT', '100 per hour')
    UPLOAD_RATE_LIMIT = os.environ.get('UPLOAD_RATE_LIMIT', '10 per minute')
    
    # File Upload Configuration
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff', 'jfif', 'pdf'}
    
    # API Metadata
    API_VERSION = '1.0.0'
    API_TITLE = 'Indian Document OCR API'
    SUPPORTED_DOCUMENTS = [
        'Aadhaar Card',
        'PAN Card',
        'Driving License',
        'Voter ID',
        'Passport',
        'E-Aadhaar'
    ]
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    
    @classmethod
    def get_api_keys(cls):
        """Get API keys as a set"""
        return cls.API_KEYS
    
    @classmethod
    def is_valid_api_key(cls, key):
        """Check if API key is valid"""
        return key in cls.API_KEYS
    
    @classmethod
    def get_allowed_origins(cls):
        """Get allowed CORS origins"""
        return cls.ALLOWED_ORIGINS


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    RATE_LIMIT_ENABLED = False  # Disable rate limiting in development


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    RATE_LIMIT_ENABLED = True


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}


def get_config():
    """Get configuration based on environment"""
    env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])
