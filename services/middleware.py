# SPDX-License-Identifier: MPL-2.0

"""Middleware for rate limiting, request validation, and security enhancements."""

from __future__ import annotations

import time
import logging
from typing import Callable, Optional
from functools import wraps
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import FastAPI, Request, HTTPException, status
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, ValidationError


logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self, requests_per_minute: int = 60):
        """Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute per IP
        """
        self.requests_per_minute = requests_per_minute
        self.requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        """Check if client is allowed to make a request.

        Args:
            client_id: Client identifier (IP address or user ID)

        Returns:
            True if request is allowed, False otherwise
        """
        now = time.time()
        minute_ago = now - 60

        # Remove old requests outside the window
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > minute_ago
        ]

        # Check if limit exceeded
        if len(self.requests[client_id]) >= self.requests_per_minute:
            return False

        # Add current request
        self.requests[client_id].append(now)
        return True

    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for this minute.

        Args:
            client_id: Client identifier

        Returns:
            Number of remaining requests
        """
        now = time.time()
        minute_ago = now - 60

        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > minute_ago
        ]

        return max(0, self.requests_per_minute - len(self.requests[client_id]))


# Default rate limiter instance
limiter = Limiter(key_func=get_remote_address)


def add_rate_limiting(app: FastAPI, requests_per_minute: int = 100) -> FastAPI:
    """Add rate limiting to FastAPI app.

    Args:
        app: FastAPI application instance
        requests_per_minute: Maximum requests per minute

    Returns:
        FastAPI app with rate limiting configured
    """
    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
        logger.warning(
            f"Rate limit exceeded for {request.client.host}: {exc.detail}"
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please try again later.",
            headers={"Retry-After": "60"},
        )

    return app


def add_security_headers(app: FastAPI) -> FastAPI:
    """Add security headers to all responses.

    Args:
        app: FastAPI application instance

    Returns:
        FastAPI app with security headers middleware
    """
    @app.middleware("http")
    async def add_security_headers_middleware(request: Request, call_next):
        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"

        return response

    return app


def add_request_validation(app: FastAPI) -> FastAPI:
    """Add request validation and sanitization middleware.

    Args:
        app: FastAPI application instance

    Returns:
        FastAPI app with request validation middleware
    """
    @app.middleware("http")
    async def validate_request(request: Request, call_next):
        # Check content-type for POST/PUT requests
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            if content_type and "application/json" not in content_type:
                logger.warning(
                    f"Invalid Content-Type for {request.method} request: {content_type}"
                )
                # Allow the request to continue but log it

        # Validate request size (max 10MB)
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 10 * 1024 * 1024:
            logger.error(f"Request too large: {content_length} bytes")
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Request body too large"
            )

        response = await call_next(request)
        return response

    return app


def add_cors_headers(app: FastAPI) -> FastAPI:
    """Add CORS headers for API access.

    Args:
        app: FastAPI application instance

    Returns:
        FastAPI app with CORS middleware
    """
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Change to specific origins in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


def add_logging_middleware(app: FastAPI) -> FastAPI:
    """Add detailed request/response logging middleware.

    Args:
        app: FastAPI application instance

    Returns:
        FastAPI app with logging middleware
    """
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(time.time()))
        start_time = time.time()

        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"from {request.client.host}"
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            logger.error(
                f"[{request_id}] Request failed: {exc}",
                exc_info=True
            )
            raise

        process_time = time.time() - start_time
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"completed in {process_time:.3f}s with status {response.status_code}"
        )

        return response

    return app


def setup_middleware(app: FastAPI, config: Optional[dict] = None) -> FastAPI:
    """Setup all middleware for a FastAPI application.

    Args:
        app: FastAPI application instance
        config: Configuration dictionary for middleware

    Returns:
        Configured FastAPI app
    """
    if config is None:
        config = {}

    # Add all middleware layers
    app = add_cors_headers(app)
    app = add_security_headers(app)
    app = add_request_validation(app)
    app = add_logging_middleware(app)

    return app
