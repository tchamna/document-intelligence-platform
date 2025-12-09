"""
FastAPI Application
REST API for Healthcare IDP System
"""

import os
import sys
from datetime import datetime
from typing import List

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline import IDPPipeline
from src.document_classifier import DocumentClassifier
from src.entity_extractor import EntityExtractor
from src.document_processor import DocumentProcessor
from api.schemas import (
    DocumentInput,
    ProcessingResult,
    HealthResponse,
    BatchDocumentInput,
    BatchProcessingResult,
    ClassificationRequest,
    ClassificationResult,
    ExtractionRequest,
    ExtractionResult
)

# Initialize FastAPI app
app = FastAPI(
    title="Healthcare IDP API",
    description="Intelligent Document Processing API for healthcare benefits administration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
static_dir = os.path.join(os.path.dirname(__file__), '..', 'static')
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize pipeline and document processor (singletons)
pipeline = None
doc_processor = None


def get_pipeline() -> IDPPipeline:
    """Get or create pipeline instance"""
    global pipeline
    if pipeline is None:
        pipeline = IDPPipeline()
    return pipeline


def get_document_processor() -> DocumentProcessor:
    """Get or create document processor instance"""
    global doc_processor
    if doc_processor is None:
        doc_processor = DocumentProcessor()
    return doc_processor


# Allowed file extensions
ALLOWED_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.txt'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


def validate_file(filename: str, content_length: int = 0) -> None:
    """Validate uploaded file"""
    import os
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"File type not supported. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    if content_length > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)} MB"
        )


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check system health and component status"""
    processor = get_document_processor()
    deps = processor.check_dependencies()
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.utcnow(),
        components={
            "classifier": "ready",
            "extractor": "ready",
            "adjudicator": "ready",
            "document_processor": "ready",
            "ocr_tesseract": "ready" if deps['tesseract'] else "not available",
            "pdf_support": "ready" if deps['pdfplumber'] else "limited",
            "api": "ready"
        }
    )


@app.get("/", tags=["System"])
async def root():
    """API root endpoint - redirects to web UI"""
    return {
        "message": "Healthcare IDP API",
        "version": "1.0.0",
        "web_ui": "/ui",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/ui", tags=["System"], include_in_schema=False)
async def web_ui():
    """Serve the batch processing web UI for document upload"""
    # Serve the batch UI by default
    ui_path = os.path.join(os.path.dirname(__file__), '..', 'static', 'batch_ui.html')
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    # Fallback to enhanced UI
    ui_path_fallback = os.path.join(os.path.dirname(__file__), '..', 'static', 'enhanced_ui.html')
    if os.path.exists(ui_path_fallback):
        return FileResponse(ui_path_fallback)
    raise HTTPException(status_code=404, detail="Web UI not found")


@app.get("/ui/classic", tags=["System"], include_in_schema=False)
async def classic_ui():
    """Serve the classic web UI"""
    ui_path = os.path.join(os.path.dirname(__file__), '..', 'static', 'index.html')
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    raise HTTPException(status_code=404, detail="Classic UI not found")


@app.get("/ui/enhanced", tags=["System"], include_in_schema=False)
async def enhanced_ui():
    """Serve the enhanced web UI with all features"""
    ui_path = os.path.join(os.path.dirname(__file__), '..', 'static', 'enhanced_ui.html')
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    raise HTTPException(status_code=404, detail="Enhanced UI not found")


@app.get("/batch", tags=["System"], include_in_schema=False)
async def batch_ui_shortcut():
    """Serve the batch processing UI (shortcut)"""
    ui_path = os.path.join(os.path.dirname(__file__), '..', 'static', 'batch_ui.html')
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    raise HTTPException(status_code=404, detail="Batch UI not found")


@app.get("/ui/batch", tags=["System"], include_in_schema=False)
async def batch_ui():
    """Serve the simplified batch processing UI"""
    ui_path = os.path.join(os.path.dirname(__file__), '..', 'static', 'batch_ui.html')
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    raise HTTPException(status_code=404, detail="Batch UI not found")


@app.get("/ui/samples", tags=["System"], include_in_schema=False)
async def samples_page():
    """Serve the sample documents download page"""
    ui_path = os.path.join(os.path.dirname(__file__), '..', 'static', 'samples', 'index.html')
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    raise HTTPException(status_code=404, detail="Samples page not found")


@app.get("/ui/docs", tags=["System"], include_in_schema=False)
async def documentation_page():
    """Serve the documentation page with README, workflow, and tech stack"""
    ui_path = os.path.join(os.path.dirname(__file__), '..', 'static', 'docs.html')
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    raise HTTPException(status_code=404, detail="Documentation page not found")


# Document processing endpoints
@app.post("/process", response_model=ProcessingResult, tags=["Processing"])
async def process_document(document: DocumentInput):
    """
    Process a document through the complete IDP pipeline.
    
    This endpoint:
    1. Classifies the document type
    2. Extracts relevant entities
    3. Performs claim adjudication (if applicable)
    4. Returns comprehensive results with quality metrics
    """
    try:
        idp_pipeline = get_pipeline()
        result = idp_pipeline.process_document(
            document_text=document.text,
            metadata=document.metadata
        )
        return ProcessingResult(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", tags=["File Upload"])
async def upload_and_process(file: UploadFile = File(...)):
    """
    Upload a document file (PDF, image, or text) and process it through the IDP pipeline.
    
    Supported formats:
    - PDF files (.pdf)
    - Images (.png, .jpg, .jpeg, .tiff, .tif, .bmp)
    - Text files (.txt)
    
    The system will:
    1. Extract text using OCR (for PDFs and images) or direct reading (for text)
    2. Classify the document type
    3. Extract relevant entities
    4. Perform claim adjudication (if applicable)
    
    Returns comprehensive processing results including extracted text and quality metrics.
    """
    import time
    start_time = time.time()
    
    # Validate file
    validate_file(file.filename)
    
    try:
        # Read file content
        content = await file.read()
        
        # Process the file (OCR/text extraction)
        processor = get_document_processor()
        processed = processor.process_file(content, file.filename)
        
        # Check if text was extracted
        if not processed.text or len(processed.text.strip()) < 50:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "error": "Could not extract sufficient text from document",
                    "ocr_confidence": processed.confidence,
                    "warnings": processed.warnings
                }
            )
        
        # Run through IDP pipeline
        idp_pipeline = get_pipeline()
        result = idp_pipeline.process_document(
            document_text=processed.text,
            metadata={
                "filename": file.filename,
                "file_format": processed.format.value,
                "page_count": processed.page_count,
                "ocr_confidence": processed.confidence,
                "extraction_method": processed.metadata.get('method', 'unknown')
            }
        )
        
        # Add file processing info to result
        result['file_info'] = {
            'filename': file.filename,
            'format': processed.format.value,
            'page_count': processed.page_count,
            'ocr_confidence': processed.confidence,
            'extraction_method': processed.metadata.get('method', 'unknown'),
            'text_length': len(processed.text),
            'warnings': processed.warnings
        }
        result['extracted_text_preview'] = processed.text[:500] + '...' if len(processed.text) > 500 else processed.text
        result['total_processing_time'] = time.time() - start_time
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/upload/batch", tags=["File Upload"])
async def upload_batch(files: List[UploadFile] = File(...)):
    """
    Upload multiple document files for batch processing.
    
    Maximum 10 files per batch.
    Supported formats: PDF, images (PNG, JPG, TIFF, BMP), and text files.
    """
    import time
    start_time = time.time()
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
    
    processor = get_document_processor()
    idp_pipeline = get_pipeline()
    
    results = []
    successful = 0
    failed = 0
    
    for file in files:
        try:
            # Validate
            validate_file(file.filename)
            
            # Read and process
            content = await file.read()
            processed = processor.process_file(content, file.filename)
            
            if processed.text and len(processed.text.strip()) >= 50:
                # Run pipeline
                result = idp_pipeline.process_document(
                    document_text=processed.text,
                    metadata={'filename': file.filename}
                )
                result['filename'] = file.filename
                result['ocr_confidence'] = processed.confidence
                results.append(result)
                successful += 1
            else:
                results.append({
                    'filename': file.filename,
                    'status': 'error',
                    'error': 'Insufficient text extracted'
                })
                failed += 1
                
        except Exception as e:
            results.append({
                'filename': file.filename,
                'status': 'error',
                'error': str(e)
            })
            failed += 1
    
    return {
        'total_files': len(files),
        'successful': successful,
        'failed': failed,
        'results': results,
        'total_processing_time': time.time() - start_time
    }


@app.post("/upload/extract-text", tags=["File Upload"])
async def extract_text_only(file: UploadFile = File(...)):
    """
    Upload a document and extract text only (no IDP processing).
    
    Useful for previewing OCR results before full processing.
    """
    validate_file(file.filename)
    
    try:
        content = await file.read()
        processor = get_document_processor()
        processed = processor.process_file(content, file.filename)
        
        return {
            'filename': file.filename,
            'format': processed.format.value,
            'page_count': processed.page_count,
            'confidence': processed.confidence,
            'extraction_method': processed.metadata.get('method', 'unknown'),
            'text_length': len(processed.text),
            'text': processed.text,
            'warnings': processed.warnings
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")


@app.get("/upload/supported-formats", tags=["File Upload"])
async def get_supported_formats():
    """Get list of supported file formats and OCR status"""
    processor = get_document_processor()
    deps = processor.check_dependencies()
    
    return {
        'supported_extensions': list(ALLOWED_EXTENSIONS),
        'max_file_size_mb': MAX_FILE_SIZE // (1024 * 1024),
        'ocr_available': deps['tesseract'],
        'pdf_ocr_available': deps['tesseract'] and deps['poppler'],
        'dependencies': deps,
        'notes': [
            'PDF files: Text extraction attempted first, OCR used as fallback',
            'Image files: Processed with Tesseract OCR',
            'Text files: Direct reading',
            'For best OCR results, use high-resolution scans (300+ DPI)'
        ]
    }


@app.post("/process/batch", response_model=BatchProcessingResult, tags=["Processing"])
async def process_batch(batch: BatchDocumentInput, background_tasks: BackgroundTasks):
    """
    Process multiple documents in batch.
    
    Maximum 100 documents per batch.
    """
    import time
    start_time = time.time()
    
    idp_pipeline = get_pipeline()
    results = []
    successful = 0
    failed = 0
    
    for doc in batch.documents:
        try:
            result = idp_pipeline.process_document(
                document_text=doc.text,
                metadata=doc.metadata
            )
            results.append(ProcessingResult(**result))
            if result['status'] == 'success':
                successful += 1
            else:
                failed += 1
        except Exception as e:
            results.append(ProcessingResult(
                status='error',
                error=str(e),
                processing_time_seconds=0.0
            ))
            failed += 1
    
    return BatchProcessingResult(
        total_documents=len(batch.documents),
        successful=successful,
        failed=failed,
        results=results,
        total_processing_time_seconds=time.time() - start_time
    )


@app.post("/classify", response_model=ClassificationResult, tags=["Classification"])
async def classify_document(request: ClassificationRequest):
    """
    Classify a document without full processing.
    
    Returns document type and confidence score.
    """
    try:
        classifier = DocumentClassifier()
        doc_type, confidence = classifier.classify(request.text)
        return ClassificationResult(
            document_type=doc_type,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract", response_model=ExtractionResult, tags=["Extraction"])
async def extract_entities(request: ExtractionRequest):
    """
    Extract entities from a document.
    
    Requires document type to be specified for optimal extraction.
    """
    try:
        extractor = EntityExtractor()
        entities = extractor.extract(request.text, request.document_type)
        return ExtractionResult(
            entities=entities,
            entity_count=len(entities)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/document-types", tags=["Reference"])
async def get_document_types():
    """Get list of supported document types"""
    return {
        "document_types": [
            {
                "type": "disability_claim",
                "description": "Disability benefit claim forms",
                "key_entities": ["claim_number", "policy_number", "date_of_disability", "diagnosis"]
            },
            {
                "type": "enrollment",
                "description": "Employee enrollment and election forms",
                "key_entities": ["employee_name", "plan_type", "coverage_level", "effective_date"]
            },
            {
                "type": "policy",
                "description": "Insurance policy documents and certificates",
                "key_entities": ["policy_number", "elimination_period", "max_benefit", "exclusions"]
            },
            {
                "type": "rfp",
                "description": "Requests for proposals and bids",
                "key_entities": ["requirements", "pricing", "scope_of_work"]
            }
        ]
    }


@app.get("/metrics", tags=["Metrics"])
async def get_metrics():
    """Get system performance metrics"""
    from src.metrics_dashboard import get_dashboard
    
    dashboard = get_dashboard()
    return dashboard.get_dashboard_summary()


@app.get("/metrics/export", tags=["Metrics"])
async def export_metrics():
    """Export metrics report to file"""
    from src.metrics_dashboard import get_dashboard
    
    dashboard = get_dashboard()
    filepath = dashboard.export_report()
    return {"status": "success", "filepath": filepath}


# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED PIPELINE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/v2/process", tags=["Enhanced Processing"])
async def process_document_v2(document: DocumentInput):
    """
    Process document with enhanced pipeline (v2).
    
    Features:
    - High-precision entity extraction (97-99% target)
    - Data normalization and validation
    - Automated adjudication with business rules
    - Eligibility matching for enrollments
    - Policy interpretation
    - Quality scoring and metrics
    """
    from src.enhanced_pipeline import EnhancedIDPPipeline
    
    try:
        pipeline = EnhancedIDPPipeline()
        result = pipeline.process_document(
            document_text=document.text,
            metadata=document.metadata,
            policy_context=document.metadata.get('policy_context') if document.metadata else None
        )
        return pipeline.to_dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v2/upload", tags=["Enhanced Processing"])
async def upload_and_process_v2(file: UploadFile = File(...)):
    """
    Upload and process with enhanced pipeline (v2).
    
    All features of the enhanced pipeline with file upload support.
    """
    import time
    from src.enhanced_pipeline import EnhancedIDPPipeline
    
    start_time = time.time()
    validate_file(file.filename)
    
    try:
        content = await file.read()
        processor = get_document_processor()
        processed = processor.process_file(content, file.filename)
        
        if not processed.text or len(processed.text.strip()) < 50:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "error": "Could not extract sufficient text from document",
                    "ocr_confidence": processed.confidence,
                    "warnings": processed.warnings
                }
            )
        
        # Use enhanced pipeline
        pipeline = EnhancedIDPPipeline()
        result = pipeline.process_document(
            document_text=processed.text,
            metadata={
                "filename": file.filename,
                "file_format": processed.format.value,
                "page_count": processed.page_count,
                "ocr_confidence": processed.confidence
            }
        )
        
        response = pipeline.to_dict(result)
        response['file_info'] = {
            'filename': file.filename,
            'format': processed.format.value,
            'page_count': processed.page_count,
            'ocr_confidence': processed.confidence,
            'text_length': len(processed.text)
        }
        response['total_processing_time'] = time.time() - start_time
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════════
# ADJUDICATION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/adjudicate", tags=["Adjudication"])
async def adjudicate_claim(claim_data: dict, policy_data: dict = None):
    """
    Adjudicate a disability claim against policy terms.
    
    Features:
    - Coverage verification
    - Elimination period check
    - Pre-existing condition screening
    - Documentation verification
    - Benefit calculation
    - Exclusion checking
    """
    from src.enhanced_adjudicator import EnhancedClaimAdjudicator
    
    try:
        adjudicator = EnhancedClaimAdjudicator()
        
        # Use default policy if not provided
        if policy_data is None:
            policy_data = {
                'policy_number': claim_data.get('policy_number', 'DEFAULT'),
                'effective_date': '2024-01-01',
                'elimination_period': '90 days',
                'benefit_percentage': '60%',
                'max_benefit': '10000'
            }
        
        decision = adjudicator.adjudicate(claim_data, policy_data)
        return adjudicator.to_dict(decision)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# ELIGIBILITY ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/eligibility/check", tags=["Eligibility"])
async def check_eligibility(enrollment_data: dict, employee_data: dict = None):
    """
    Check employee eligibility for benefit enrollment.
    
    Features:
    - Plan matching based on employee profile
    - Coverage level determination
    - Dependent validation
    - Premium calculation
    - Effective date calculation
    """
    from src.eligibility_engine import EligibilityMatchingEngine
    
    try:
        engine = EligibilityMatchingEngine()
        result = engine.check_eligibility(enrollment_data, employee_data)
        return engine.to_dict(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/eligibility/plans", tags=["Eligibility"])
async def get_available_plans():
    """Get list of available benefit plans"""
    from src.eligibility_engine import PlanMatcher
    
    matcher = PlanMatcher()
    plans = []
    
    for plan_id, plan in matcher.plan_catalog.items():
        plans.append({
            'plan_id': plan_id,
            'name': plan['name'],
            'type': plan['type'],
            'coverage_levels': plan['coverage_levels'],
            'waiting_period_days': plan['waiting_period_days']
        })
    
    return {'plans': plans}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PROCESSING ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/normalize", tags=["Data Processing"])
async def normalize_data(entities: dict):
    """
    Normalize extracted entities.
    
    Standardizes:
    - Dates to ISO format (YYYY-MM-DD)
    - Money values to float
    - Phone numbers to (XXX) XXX-XXXX
    - SSN to masked format
    - Names to title case
    """
    from src.data_normalizer import DataNormalizer
    
    try:
        normalizer = DataNormalizer()
        normalized = normalizer.normalize_all(entities)
        
        # Extract report
        report = normalized.pop('_normalization_report', [])
        
        return {
            'normalized_entities': normalized,
            'normalization_report': report
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate", tags=["Data Processing"])
async def validate_data(entities: dict, document_type: str = "general"):
    """
    Validate extracted and normalized data.
    
    Checks:
    - Required fields present
    - Field formats valid
    - Business rule compliance
    """
    from src.data_normalizer import ValidationEngine
    
    try:
        validator = ValidationEngine()
        result = validator.validate(entities, document_type)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACTION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/v2/extract", tags=["Enhanced Extraction"])
async def extract_entities_v2(text: str, document_type: str = "general"):
    """
    Extract entities with enhanced extractor (v2).
    
    Uses ensemble of:
    - Regex patterns (high precision)
    - spaCy NER (names, organizations)
    - LLM enhancement (complex entities)
    
    Returns confidence scores per entity.
    """
    from src.enhanced_extractor import EnhancedEntityExtractor
    
    try:
        extractor = EnhancedEntityExtractor()
        result = extractor.extract(text, document_type)
        
        return {
            'entities': result.entities,
            'extraction_confidence': result.extraction_confidence,
            'coverage_score': result.coverage_score,
            'entity_count': result.entity_count,
            'extraction_time_ms': result.extraction_time_ms,
            'method_breakdown': result.method_breakdown
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# LLM ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/llm/status", tags=["LLM"])
async def llm_status():
    """
    Check LLM provider status.
    
    Returns whether LLM is connected (AWS Bedrock or OpenAI) or running in mock mode.
    """
    import os
    from src.llm_integration import BedrockProvider, OpenAIProvider
    
    providers_status = []
    active_provider = None
    
    # Check Bedrock
    try:
        bedrock = BedrockProvider()
        bedrock_connected = bedrock.client is not None
        if bedrock_connected:
            try:
                response = bedrock.invoke("Test", max_tokens=5)
                if not response.metadata.get("mock"):
                    active_provider = {
                        'name': 'AWS Bedrock',
                        'model': bedrock.model_id,
                        'status': 'connected'
                    }
            except:
                bedrock_connected = False
        providers_status.append({
            'provider': 'AWS Bedrock',
            'status': 'connected' if active_provider and active_provider['name'] == 'AWS Bedrock' else 'not configured',
            'models': list(bedrock.MODEL_IDS.keys())
        })
    except:
        providers_status.append({'provider': 'AWS Bedrock', 'status': 'error'})
    
    # Check OpenAI
    if not active_provider:
        try:
            openai_key = os.getenv('OPENAI_API_KEY')
            if openai_key:
                openai = OpenAIProvider()
                if openai.client:
                    try:
                        response = openai.invoke("Test", max_tokens=5)
                        if not response.metadata.get("mock"):
                            active_provider = {
                                'name': 'OpenAI',
                                'model': openai.model_id,
                                'status': 'connected'
                            }
                    except:
                        pass
            providers_status.append({
                'provider': 'OpenAI',
                'status': 'connected' if active_provider and active_provider['name'] == 'OpenAI' else 'not configured',
                'models': list(OpenAIProvider.MODEL_IDS.keys()),
                'hint': 'Set OPENAI_API_KEY environment variable'
            })
        except:
            providers_status.append({'provider': 'OpenAI', 'status': 'not available'})
    
    if active_provider:
        return {
            'status': 'connected',
            'provider': active_provider['name'],
            'model': active_provider['model'],
            'message': f"LLM fully operational via {active_provider['name']}",
            'mock_mode': False,
            'available_providers': providers_status
        }
    else:
        return {
            'status': 'mock',
            'provider': 'Mock Provider',
            'model': 'mock',
            'message': 'Running in mock mode. Configure AWS Bedrock or set OPENAI_API_KEY for full LLM capabilities.',
            'mock_mode': True,
            'available_providers': providers_status,
            'setup_hints': [
                'AWS Bedrock: Configure AWS credentials and enable Claude 3 model access',
                'OpenAI: Set OPENAI_API_KEY environment variable and pip install openai'
            ]
        }


@app.post("/llm/classify", tags=["LLM"])
async def llm_classify(text: str):
    """
    Classify document using LLM (AWS Bedrock/Claude).
    
    Provides reasoning for classification decision.
    """
    from src.llm_integration import LangChainIntegration
    
    try:
        langchain = LangChainIntegration()
        result = langchain.classify_document(text[:4000])  # Limit for API
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/llm/interpret-policy", tags=["LLM"])
async def llm_interpret_policy(policy_text: str):
    """
    Interpret policy document using LLM.
    
    Extracts:
    - Coverage terms and conditions
    - Elimination/waiting periods
    - Benefit calculations
    - Exclusions and limitations
    - Key definitions
    """
    from src.llm_integration import LangChainIntegration
    
    try:
        langchain = LangChainIntegration()
        result = langchain.interpret_policy(policy_text[:5000])
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE INFO ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/pipeline/info", tags=["System"])
async def get_pipeline_info():
    """Get comprehensive pipeline capabilities and tech stack info"""
    from src.enhanced_pipeline import EnhancedIDPPipeline
    
    pipeline = EnhancedIDPPipeline()
    return pipeline.get_pipeline_info()


@app.get("/pipeline/components", tags=["System"])
async def get_pipeline_components():
    """Get status of all pipeline components"""
    processor = get_document_processor()
    deps = processor.check_dependencies()
    
    return {
        'components': {
            'document_classifier': {
                'status': 'ready',
                'description': 'Multi-stage classification (rule-based + ML + LLM)'
            },
            'entity_extractor': {
                'status': 'ready',
                'description': 'Ensemble extraction (regex + spaCy + LLM)',
                'spacy_model': 'en_core_web_lg'
            },
            'data_normalizer': {
                'status': 'ready',
                'description': 'Standardize dates, money, names, identifiers'
            },
            'claim_adjudicator': {
                'status': 'ready',
                'description': 'Rule-based business logic engine',
                'rules_count': 6
            },
            'eligibility_engine': {
                'status': 'ready',
                'description': 'Plan matching and eligibility determination',
                'plans_count': 6
            },
            'policy_interpreter': {
                'status': 'ready',
                'description': 'NLP and LLM-based policy interpretation'
            },
            'llm_provider': {
                'status': 'ready (mock mode)' if not deps.get('bedrock') else 'ready',
                'description': 'AWS Bedrock (Claude 3)',
                'models': ['claude-3-sonnet', 'claude-3-haiku', 'titan-text']
            },
            'ocr_engine': {
                'status': 'ready' if deps['tesseract'] else 'not available',
                'description': 'Tesseract OCR',
                'pdf_support': deps['poppler']
            },
            'metrics_dashboard': {
                'status': 'ready',
                'description': 'Accuracy tracking and performance monitoring'
            }
        },
        'tech_stack': {
            'nlp': 'spaCy 3.7 + en_core_web_lg',
            'llm': 'AWS Bedrock (Claude 3)',
            'ocr': 'Tesseract 5.x',
            'pdf': 'pdfplumber + pdf2image + Poppler',
            'api': 'FastAPI + uvicorn',
            'ml': 'scikit-learn, transformers'
        },
        'accuracy_targets': {
            'extraction_precision': '97-99%',
            'classification_accuracy': '99%',
            'adjudication_accuracy': '95%'
        }
    }


# Run server if executed directly
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )
