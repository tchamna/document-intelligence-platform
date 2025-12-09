"""
Main IDP Pipeline
End-to-end orchestration of all components
"""

import json
from typing import Dict, Any, Optional
import logging
import time

from .document_classifier import DocumentClassifier
from .entity_extractor import EntityExtractor
from .claim_adjudicator import ClaimAdjudicator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IDPPipeline:
    """
    End-to-end IDP pipeline orchestrating all components.
    Handles document processing from upload to final output.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the IDP pipeline.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.classifier = DocumentClassifier(config=self.config.get('classifier'))
        self.extractor = EntityExtractor()
        self.adjudicator = ClaimAdjudicator()
        
        logger.info("IDPPipeline initialized successfully")
    
    def process_document(self, document_text: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process a document through the complete pipeline.
        
        Args:
            document_text: Raw text from OCR or document
            metadata: Optional metadata about the document
            
        Returns:
            Complete processing results with metrics
        """
        start_time = time.time()
        logger.info("=" * 80)
        logger.info("Starting document processing pipeline")
        logger.info("=" * 80)
        
        results = {
            'status': 'success',
            'pipeline_stages': {},
            'processing_time_seconds': 0
        }
        
        try:
            # Stage 1: Classification
            logger.info("Stage 1: Document classification")
            stage_start = time.time()
            
            doc_type, confidence = self.classifier.classify(document_text, metadata)
            
            results['document_type'] = doc_type
            results['classification_confidence'] = confidence
            results['pipeline_stages']['classification'] = {
                'status': 'completed',
                'confidence': confidence,
                'duration_seconds': time.time() - stage_start
            }
            
            logger.info(f"  Document type: {doc_type} (confidence: {confidence:.2%})")
            
            # Stage 2: Entity Extraction
            logger.info("Stage 2: Entity extraction")
            stage_start = time.time()
            
            entities = self.extractor.extract(document_text, doc_type)
            
            results['extracted_entities'] = entities
            results['pipeline_stages']['extraction'] = {
                'status': 'completed',
                'entities_found': len(entities),
                'duration_seconds': time.time() - stage_start
            }
            
            logger.info(f"  Extracted {len(entities)} entities")
            
            # Stage 3: Business Logic (if applicable)
            if doc_type == 'disability_claim':
                logger.info("Stage 3: Claim adjudication")
                stage_start = time.time()
                
                # Get policy data (would query database in production)
                policy_data = self._get_policy_data(entities.get('policy_number'))
                
                adjudication = self.adjudicator.adjudicate(entities, policy_data)
                
                results['adjudication'] = {
                    'eligibility_status': adjudication.eligibility_status,
                    'confidence': adjudication.confidence,
                    'coverage_verified': adjudication.coverage_verified,
                    'waiting_period_met': adjudication.waiting_period_met,
                    'estimated_benefit_start': adjudication.estimated_benefit_start,
                    'required_documents': adjudication.required_documents,
                    'denial_reasons': adjudication.denial_reasons
                }
                
                results['pipeline_stages']['adjudication'] = {
                    'status': 'completed',
                    'decision': adjudication.eligibility_status,
                    'duration_seconds': time.time() - stage_start
                }
                
                logger.info(f"  Adjudication: {adjudication.eligibility_status}")
            
            # Stage 4: Quality Checks
            logger.info("Stage 4: Quality validation")
            stage_start = time.time()
            
            quality_score = self._calculate_quality_score(results)
            results['quality_score'] = quality_score
            results['pipeline_stages']['validation'] = {
                'status': 'completed',
                'quality_score': quality_score,
                'duration_seconds': time.time() - stage_start
            }
            
            logger.info(f"  Quality score: {quality_score:.2%}")
            
            # Final metrics
            results['processing_time_seconds'] = time.time() - start_time
            results['processing_metrics'] = {
                'total_stages': len(results['pipeline_stages']),
                'overall_confidence': self._calculate_overall_confidence(results),
                'production_ready': quality_score >= 0.97
            }
            
            logger.info("=" * 80)
            logger.info(f"Pipeline completed successfully in {results['processing_time_seconds']:.2f}s")
            logger.info(f"Quality score: {quality_score:.2%}")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            results['status'] = 'error'
            results['error'] = str(e)
            results['processing_time_seconds'] = time.time() - start_time
        
        return results
    
    def _get_policy_data(self, policy_number: str) -> Dict:
        """Retrieve policy data (would query database in production)"""
        # Mock policy data for demo
        return {
            'policy_number': policy_number or 'POL-456-7890',
            'effective_date': '2024-01-01',
            'elimination_period': '90 days',
            'max_benefit': '$10,000',
            'preexisting_condition_clause': 'Excluded for 12 months if treated in prior 90 days'
        }
    
    def _calculate_quality_score(self, results: Dict) -> float:
        """Calculate overall quality score for the processing"""
        
        scores = []
        
        # Classification confidence
        if 'classification_confidence' in results:
            scores.append(results['classification_confidence'])
        
        # Extraction completeness
        entities = results.get('extracted_entities', {})
        if entities:
            expected_fields = 8  # Varies by doc type
            found_fields = len(entities)
            extraction_score = min(found_fields / expected_fields, 1.0)
            scores.append(extraction_score)
        
        # Adjudication confidence (if applicable)
        if 'adjudication' in results:
            scores.append(results['adjudication']['confidence'])
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _calculate_overall_confidence(self, results: Dict) -> float:
        """Calculate weighted overall confidence"""
        
        classification_weight = 0.3
        extraction_weight = 0.5
        adjudication_weight = 0.2
        
        confidence = (
            results.get('classification_confidence', 0) * classification_weight +
            self._calculate_quality_score(results) * extraction_weight
        )
        
        if 'adjudication' in results:
            confidence += results['adjudication']['confidence'] * adjudication_weight
        
        return confidence


# Main entry point for standalone execution
if __name__ == "__main__":
    # Sample document text (simulating OCR output)
    sample_claim = """
    DISABILITY CLAIM FORM
    
    Claim Number: CLM-2024-789456
    Policy Number: POL-456-7890
    
    Claimant Information:
    Name: Sarah Johnson
    Date of Birth: 05/15/1985
    Social Security Number: 123-45-6789
    
    Employer: Acme Corporation
    Employment Start Date: 01/15/2020
    
    Disability Information:
    Date of Disability: 11/15/2024
    Nature of Disability: Acute lumbar strain with radiculopathy
    
    Physician Information:
    Dr. Michael Roberts, MD
    Orthopedic Spine Specialist
    
    Monthly Earnings: $5,000
    Requested Benefit Amount: $2,500/month (60% of earnings)
    """
    
    # Initialize pipeline
    pipeline = IDPPipeline()
    
    # Process document
    results = pipeline.process_document(sample_claim)
    
    # Display results
    print("=" * 80)
    print("DOCUMENT PROCESSING RESULTS")
    print("=" * 80)
    print(json.dumps(results, indent=2))
    
    print("\n" + "=" * 80)
    print("KEY METRICS")
    print("=" * 80)
    print(f"Document Type: {results['document_type']}")
    print(f"Classification Confidence: {results['classification_confidence']:.2%}")
    print(f"Quality Score: {results['quality_score']:.2%}")
    print(f"Overall Confidence: {results['processing_metrics']['overall_confidence']:.2%}")
    print(f"Production Ready: {results['processing_metrics']['production_ready']}")
    
    if 'adjudication' in results:
        print("\n" + "=" * 80)
        print("ADJUDICATION DECISION")
        print("=" * 80)
        adj = results['adjudication']
        print(f"Status: {adj['eligibility_status']}")
        print(f"Confidence: {adj['confidence']:.2%}")
        print(f"Coverage Verified: {adj['coverage_verified']}")
        print(f"Waiting Period Met: {adj['waiting_period_met']}")
        if adj['estimated_benefit_start']:
            print(f"Benefit Start Date: {adj['estimated_benefit_start']}")
