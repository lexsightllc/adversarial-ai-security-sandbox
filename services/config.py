# SPDX-License-Identifier: MPL-2.0

"""Configuration management for services with environment validation."""

from __future__ import annotations

import os
import logging
from typing import Optional, Any
from functools import lru_cache

from pydantic import BaseSettings, Field, validator


logger = logging.getLogger(__name__)


class APIGatewaySettings(BaseSettings):
    """Configuration for API Gateway service."""

    # Service
    service_name: str = "api-gateway"
    debug: bool = Field(False, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_format: str = Field("json", env="LOG_FORMAT")

    # Security
    secret_key: str = Field("your-super-secret-jwt-key", env="SECRET_KEY")
    algorithm: str = Field("HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    cors_origins: list[str] = Field(["*"], env="CORS_ORIGINS")

    # Database
    database_url: str = Field(
        "postgresql://user:password@postgres:5432/adversarial_sandbox_db",
        env="DATABASE_URL"
    )
    db_pool_size: int = Field(20, env="DB_POOL_SIZE")
    db_max_overflow: int = Field(10, env="DB_MAX_OVERFLOW")
    db_pool_recycle: int = Field(3600, env="DB_POOL_RECYCLE")

    # Rate limiting
    rate_limit_enabled: bool = Field(True, env="RATE_LIMIT_ENABLED")
    rate_limit_requests_per_minute: int = Field(100, env="RATE_LIMIT_RPM")

    # Timeouts
    model_service_timeout: int = Field(60, env="MODEL_SERVICE_TIMEOUT")
    attack_service_timeout: int = Field(300, env="ATTACK_SERVICE_TIMEOUT")

    @validator("secret_key")
    def validate_secret_key(cls, v):
        """Validate that secret key is not default."""
        if v == "your-super-secret-jwt-key":
            logger.warning("Using default secret key! Set SECRET_KEY environment variable.")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class ModelServiceSettings(BaseSettings):
    """Configuration for Model Service."""

    # Service
    service_name: str = "model-service"
    debug: bool = Field(False, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    # Database
    database_url: str = Field(
        "postgresql://user:password@postgres:5432/adversarial_sandbox_db",
        env="DATABASE_URL"
    )
    db_pool_size: int = Field(15, env="DB_POOL_SIZE")

    # Model configuration
    model_cache_capacity: int = Field(5, env="MODEL_CACHE_CAPACITY")
    model_cache_ttl: int = Field(3600, env="MODEL_CACHE_TTL")
    huggingface_cache_dir: str = Field("/tmp/hf_cache", env="HUGGINGFACE_CACHE_DIR")

    # Timeouts
    model_loading_timeout: int = Field(120, env="MODEL_LOADING_TIMEOUT")
    inference_timeout: int = Field(30, env="INFERENCE_TIMEOUT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class AttackServiceSettings(BaseSettings):
    """Configuration for Attack Service."""

    # Service
    service_name: str = "attack-service"
    debug: bool = Field(False, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    # Database
    database_url: str = Field(
        "postgresql://user:password@postgres:5432/adversarial_sandbox_db",
        env="DATABASE_URL"
    )
    db_pool_size: int = Field(15, env="DB_POOL_SIZE")

    # RabbitMQ
    rabbitmq_url: str = Field(
        "amqp://guest:guest@rabbitmq:5672/",
        env="RABBITMQ_URL"
    )

    # Attack configuration
    max_concurrent_attacks: int = Field(10, env="MAX_CONCURRENT_ATTACKS")
    attack_timeout: int = Field(300, env="ATTACK_TIMEOUT")

    # Webhook configuration
    webhook_listener_url: str = Field(
        "http://webhook-listener:8003/webhook",
        env="WEBHOOK_LISTENER_URL"
    )
    webhook_retry_attempts: int = Field(3, env="WEBHOOK_RETRY_ATTEMPTS")
    webhook_retry_delay: int = Field(5, env="WEBHOOK_RETRY_DELAY")

    # Timeouts
    model_service_timeout: int = Field(60, env="MODEL_SERVICE_TIMEOUT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class DatabaseSettings(BaseSettings):
    """Shared database configuration."""

    database_url: str = Field(
        "postgresql://user:password@postgres:5432/adversarial_sandbox_db",
        env="DATABASE_URL"
    )
    pool_size: int = Field(20, env="DB_POOL_SIZE")
    max_overflow: int = Field(10, env="DB_MAX_OVERFLOW")
    pool_recycle: int = Field(3600, env="DB_POOL_RECYCLE")
    echo: bool = Field(False, env="DB_ECHO")
    connect_timeout: int = Field(10, env="DB_CONNECT_TIMEOUT")
    statement_timeout: int = Field(30000, env="DB_STATEMENT_TIMEOUT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_api_gateway_settings() -> APIGatewaySettings:
    """Get cached API Gateway settings."""
    return APIGatewaySettings()


@lru_cache()
def get_model_service_settings() -> ModelServiceSettings:
    """Get cached Model Service settings."""
    return ModelServiceSettings()


@lru_cache()
def get_attack_service_settings() -> AttackServiceSettings:
    """Get cached Attack Service settings."""
    return AttackServiceSettings()


@lru_cache()
def get_database_settings() -> DatabaseSettings:
    """Get cached Database settings."""
    return DatabaseSettings()


def validate_environment() -> bool:
    """Validate that all required environment variables are set.

    Returns:
        True if all required variables are present, False otherwise
    """
    required_vars = [
        "DATABASE_URL",
        "SECRET_KEY",
        "RABBITMQ_URL",
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        return False

    logger.info("Environment validation successful")
    return True
