"""
API Package
FastAPI-based REST API for Healthcare IDP System
"""

from .main import app
from .schemas import (
    DocumentInput,
    ProcessingResult,
    HealthResponse
)

__all__ = [
    'app',
    'DocumentInput',
    'ProcessingResult',
    'HealthResponse'
]
