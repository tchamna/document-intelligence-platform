"""
API Schemas
Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime


class DocumentInput(BaseModel):
    """Input model for document processing"""
    
    text: str = Field(..., min_length=50, description="Document text content")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata about the document"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "DISABILITY CLAIM FORM\n\nClaim Number: CLM-2024-789456...",
                "metadata": {"source": "email", "filename": "claim_form.pdf"}
            }
        }


class AdjudicationResult(BaseModel):
    """Adjudication result model"""
    
    eligibility_status: str
    confidence: float
    coverage_verified: bool
    waiting_period_met: bool
    estimated_benefit_start: Optional[str]
    required_documents: List[str]
    denial_reasons: List[str]


class PipelineStage(BaseModel):
    """Individual pipeline stage result"""
    
    status: str
    confidence: Optional[float] = None
    entities_found: Optional[int] = None
    decision: Optional[str] = None
    quality_score: Optional[float] = None
    duration_seconds: Optional[float] = None


class ProcessingMetrics(BaseModel):
    """Processing metrics model"""
    
    total_stages: int
    overall_confidence: float
    production_ready: bool


class ProcessingResult(BaseModel):
    """Complete processing result model"""
    
    status: str
    document_type: Optional[str] = None
    classification_confidence: Optional[float] = None
    extracted_entities: Optional[Dict[str, Any]] = None
    adjudication: Optional[AdjudicationResult] = None
    quality_score: Optional[float] = None
    pipeline_stages: Dict[str, PipelineStage] = {}
    processing_metrics: Optional[ProcessingMetrics] = None
    processing_time_seconds: float = 0.0
    error: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "document_type": "disability_claim",
                "classification_confidence": 0.97,
                "extracted_entities": {
                    "claim_number": "CLM-2024-789456",
                    "policy_number": "POL-456-7890"
                },
                "quality_score": 0.95,
                "processing_time_seconds": 1.23
            }
        }


class HealthResponse(BaseModel):
    """Health check response model"""
    
    status: str = "healthy"
    version: str
    timestamp: datetime
    components: Dict[str, str] = {}
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2024-12-08T10:00:00Z",
                "components": {
                    "classifier": "ready",
                    "extractor": "ready",
                    "adjudicator": "ready"
                }
            }
        }


class BatchDocumentInput(BaseModel):
    """Input for batch document processing"""
    
    documents: List[DocumentInput] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of documents to process"
    )


class BatchProcessingResult(BaseModel):
    """Result of batch document processing"""
    
    total_documents: int
    successful: int
    failed: int
    results: List[ProcessingResult]
    total_processing_time_seconds: float


class ClassificationRequest(BaseModel):
    """Request for document classification only"""
    
    text: str = Field(..., min_length=50)


class ClassificationResult(BaseModel):
    """Classification result model"""
    
    document_type: str
    confidence: float
    reasoning: Optional[str] = None


class ExtractionRequest(BaseModel):
    """Request for entity extraction"""
    
    text: str = Field(..., min_length=50)
    document_type: str = Field(..., description="Type of document")


class ExtractionResult(BaseModel):
    """Entity extraction result model"""
    
    entities: Dict[str, Any]
    entity_count: int
