"""
Healthcare IDP System
Intelligent Document Processing for insurance claims, enrollment, and policies.

Showcases:
- Document Intelligence (classification, extraction, normalization)
- Claims & Policy Automation (rule-based adjudication, model-driven decisioning)
- Policy & Enrollment Parsing (eligibility matching, complex plan interpretation)
- High-Precision Modeling (97-99% accuracy target)

Tech Stack: AWS Bedrock, LangChain, spaCy, Tesseract, FastAPI
"""

__version__ = "2.0.0"
__author__ = "Healthcare IDP Team"

# Core pipeline
from .pipeline import IDPPipeline
from .enhanced_pipeline import EnhancedIDPPipeline

# Document processing
from .document_classifier import DocumentClassifier
from .document_processor import DocumentProcessor

# Entity extraction
from .entity_extractor import EntityExtractor
from .enhanced_extractor import EnhancedEntityExtractor

# Data processing
from .data_normalizer import DataNormalizer, ValidationEngine

# Business logic
from .claim_adjudicator import ClaimAdjudicator
from .enhanced_adjudicator import EnhancedClaimAdjudicator
from .eligibility_engine import EligibilityMatchingEngine
from .policy_interpreter import PolicyInterpreter

# LLM integration
from .llm_integration import BedrockProvider, OpenAIProvider, LangChainIntegration, DocumentAIOrchestrator

# Metrics
from .metrics_dashboard import MetricsDashboard, AccuracyTracker, get_dashboard

__all__ = [
    # Pipelines
    'IDPPipeline',
    'EnhancedIDPPipeline',
    
    # Document Processing
    'DocumentClassifier',
    'DocumentProcessor',
    
    # Extraction
    'EntityExtractor',
    'EnhancedEntityExtractor',
    
    # Data Processing
    'DataNormalizer',
    'ValidationEngine',
    
    # Business Logic
    'ClaimAdjudicator',
    'EnhancedClaimAdjudicator',
    'EligibilityMatchingEngine',
    'PolicyInterpreter',
    
    # LLM
    'BedrockProvider',
    'OpenAIProvider',
    'LangChainIntegration',
    'DocumentAIOrchestrator',
    
    # Metrics
    'MetricsDashboard',
    'AccuracyTracker',
    'get_dashboard',
]
