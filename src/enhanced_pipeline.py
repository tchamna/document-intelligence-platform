"""
Enhanced IDP Pipeline
Production-grade document intelligence pipeline
Integrates all components for 97-99% accuracy
"""

import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

from .document_classifier import DocumentClassifier
from .enhanced_extractor import EnhancedEntityExtractor
from .enhanced_adjudicator import EnhancedClaimAdjudicator, AdjudicationStatus
from .data_normalizer import DataNormalizer, ValidationEngine
from .eligibility_engine import EligibilityMatchingEngine
from .policy_interpreter import PolicyInterpreter
from .metrics_dashboard import get_dashboard

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete pipeline processing result"""
    status: str
    document_type: str
    classification_confidence: float
    extracted_entities: Dict[str, Any]
    normalized_entities: Dict[str, Any]
    validation_result: Dict[str, Any]
    adjudication_result: Optional[Dict[str, Any]]
    eligibility_result: Optional[Dict[str, Any]]
    policy_interpretation: Optional[Dict[str, Any]]
    processing_metrics: Dict[str, Any]
    quality_score: float
    recommendations: List[str]


class EnhancedIDPPipeline:
    """
    Production-grade IDP pipeline for healthcare documents.
    
    Capabilities:
    - Document classification (RFP, claims, enrollment, policy)
    - High-precision entity extraction (97-99% target)
    - Data normalization and validation
    - Automated claim adjudication
    - Eligibility matching
    - Policy interpretation
    - Metrics tracking and accuracy monitoring
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the enhanced IDP pipeline.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        logger.info("Initializing Enhanced IDP Pipeline components...")
        
        self.classifier = DocumentClassifier(config=self.config.get('classifier'))
        self.extractor = EnhancedEntityExtractor(
            use_llm=self.config.get('use_llm', True),
            confidence_threshold=self.config.get('confidence_threshold', 0.7)
        )
        self.normalizer = DataNormalizer()
        self.validator = ValidationEngine()
        self.adjudicator = EnhancedClaimAdjudicator()
        self.eligibility_engine = EligibilityMatchingEngine()
        self.policy_interpreter = PolicyInterpreter()
        self.metrics = get_dashboard()
        
        logger.info("Enhanced IDP Pipeline initialized successfully")
    
    def process_document(self, document_text: str, 
                        metadata: Optional[Dict] = None,
                        policy_context: Optional[Dict] = None) -> PipelineResult:
        """
        Process a document through the complete pipeline.
        
        Args:
            document_text: Raw text from OCR or document
            metadata: Optional document metadata
            policy_context: Optional policy data for adjudication
            
        Returns:
            Complete processing result with all stages
        """
        start_time = time.time()
        metadata = metadata or {}
        
        logger.info("=" * 80)
        logger.info("ENHANCED IDP PIPELINE - Starting Document Processing")
        logger.info("=" * 80)
        logger.info(f"Document length: {len(document_text)} characters")
        
        processing_metrics = {
            'stages': {},
            'total_time_ms': 0
        }
        
        recommendations = []
        
        try:
            # ═══════════════════════════════════════════════════════════════
            # Stage 1: Document Classification
            # ═══════════════════════════════════════════════════════════════
            stage_start = time.time()
            logger.info("\n📋 Stage 1: Document Classification")
            
            doc_type, classification_confidence = self.classifier.classify(
                document_text, metadata
            )
            
            processing_metrics['stages']['classification'] = {
                'duration_ms': (time.time() - stage_start) * 1000,
                'confidence': classification_confidence
            }
            
            logger.info(f"   → Document Type: {doc_type}")
            logger.info(f"   → Confidence: {classification_confidence:.2%}")
            
            # Record for metrics
            self.metrics.record_classification(doc_type, None, classification_confidence)
            
            # ═══════════════════════════════════════════════════════════════
            # Stage 2: Entity Extraction
            # ═══════════════════════════════════════════════════════════════
            stage_start = time.time()
            logger.info("\n🔍 Stage 2: Entity Extraction")
            
            extraction_result = self.extractor.extract(document_text, doc_type)
            extracted_entities = extraction_result.entities
            
            processing_metrics['stages']['extraction'] = {
                'duration_ms': extraction_result.extraction_time_ms,
                'entities_found': extraction_result.entity_count,
                'confidence': extraction_result.extraction_confidence,
                'coverage': extraction_result.coverage_score,
                'method_breakdown': extraction_result.method_breakdown
            }
            
            logger.info(f"   → Entities Found: {extraction_result.entity_count}")
            logger.info(f"   → Extraction Confidence: {extraction_result.extraction_confidence:.2%}")
            logger.info(f"   → Coverage Score: {extraction_result.coverage_score:.2%}")
            
            # Record for metrics
            self.metrics.record_extraction(
                extracted_entities, None, 
                extraction_result.extraction_confidence,
                extraction_result.extraction_time_ms
            )
            
            # ═══════════════════════════════════════════════════════════════
            # Stage 3: Data Normalization
            # ═══════════════════════════════════════════════════════════════
            stage_start = time.time()
            logger.info("\n📐 Stage 3: Data Normalization")
            
            normalized_entities = self.normalizer.normalize_all(extracted_entities)
            
            # Get normalization report
            norm_report = normalized_entities.pop('_normalization_report', [])
            
            processing_metrics['stages']['normalization'] = {
                'duration_ms': (time.time() - stage_start) * 1000,
                'fields_normalized': len(norm_report),
                'normalization_types': list(set(r['type'] for r in norm_report))
            }
            
            logger.info(f"   → Fields Normalized: {len(norm_report)}")
            
            # ═══════════════════════════════════════════════════════════════
            # Stage 4: Data Validation
            # ═══════════════════════════════════════════════════════════════
            stage_start = time.time()
            logger.info("\n✅ Stage 4: Data Validation")
            
            validation_result = self.validator.validate(normalized_entities, doc_type)
            
            processing_metrics['stages']['validation'] = {
                'duration_ms': (time.time() - stage_start) * 1000,
                'is_valid': validation_result['is_valid'],
                'validation_score': validation_result['validation_score'],
                'issues_count': len(validation_result['issues'])
            }
            
            logger.info(f"   → Valid: {validation_result['is_valid']}")
            logger.info(f"   → Validation Score: {validation_result['validation_score']:.2%}")
            
            if validation_result['issues']:
                for issue in validation_result['issues'][:3]:
                    logger.info(f"   ⚠ Issue: {issue}")
            
            # ═══════════════════════════════════════════════════════════════
            # Stage 5: Document-Type Specific Processing
            # ═══════════════════════════════════════════════════════════════
            adjudication_result = None
            eligibility_result = None
            policy_interpretation = None
            
            # Get default policy context if not provided
            if policy_context is None:
                policy_context = self._get_default_policy_context()
            
            if doc_type == 'disability_claim':
                # ═══════════════════════════════════════════════════════════
                # Stage 5a: Claim Adjudication
                # ═══════════════════════════════════════════════════════════
                stage_start = time.time()
                logger.info("\n⚖️ Stage 5: Claim Adjudication")
                
                decision = self.adjudicator.adjudicate(
                    normalized_entities, policy_context
                )
                adjudication_result = self.adjudicator.to_dict(decision)
                
                processing_metrics['stages']['adjudication'] = {
                    'duration_ms': (time.time() - stage_start) * 1000,
                    'status': decision.status.value,
                    'confidence': decision.confidence,
                    'rules_passed': sum(1 for r in decision.rule_results if r.passed)
                }
                
                logger.info(f"   → Decision: {decision.status.value}")
                logger.info(f"   → Confidence: {decision.confidence:.2%}")
                
                if decision.benefit_calculation:
                    logger.info(f"   → Monthly Benefit: ${decision.benefit_calculation.get('final_monthly_benefit', 0):,.2f}")
                
                # Add recommendations
                recommendations.extend(decision.required_actions)
                
                self.metrics.record_adjudication(decision.status.value)
                
            elif doc_type == 'enrollment':
                # ═══════════════════════════════════════════════════════════
                # Stage 5b: Eligibility Matching
                # ═══════════════════════════════════════════════════════════
                stage_start = time.time()
                logger.info("\n🎯 Stage 5: Eligibility Matching")
                
                eligibility = self.eligibility_engine.check_eligibility(
                    normalized_entities
                )
                eligibility_result = self.eligibility_engine.to_dict(eligibility)
                
                processing_metrics['stages']['eligibility'] = {
                    'duration_ms': (time.time() - stage_start) * 1000,
                    'status': eligibility.status.value,
                    'confidence': eligibility.confidence,
                    'matched_plan': eligibility.matched_plan
                }
                
                logger.info(f"   → Status: {eligibility.status.value}")
                logger.info(f"   → Matched Plan: {eligibility.matched_plan}")
                
                if eligibility.premium_amount:
                    logger.info(f"   → Premium: ${eligibility.premium_amount:.2f}/month")
                
                recommendations.extend(eligibility.recommendations)
                
            elif doc_type == 'policy':
                # ═══════════════════════════════════════════════════════════
                # Stage 5c: Policy Interpretation
                # ═══════════════════════════════════════════════════════════
                stage_start = time.time()
                logger.info("\n📜 Stage 5: Policy Interpretation")
                
                policy_interpretation = self.policy_interpreter.interpret(document_text)
                
                processing_metrics['stages']['policy_interpretation'] = {
                    'duration_ms': (time.time() - stage_start) * 1000,
                    'clauses_found': len(policy_interpretation.get('clauses', {})),
                    'exclusions_found': len(policy_interpretation.get('exclusions', []))
                }
                
                logger.info(f"   → Clauses Identified: {len(policy_interpretation.get('clauses', {}))}")
                logger.info(f"   → Exclusions Found: {len(policy_interpretation.get('exclusions', []))}")
            
            # ═══════════════════════════════════════════════════════════════
            # Stage 6: Quality Scoring
            # ═══════════════════════════════════════════════════════════════
            logger.info("\n📊 Stage 6: Quality Scoring")
            
            quality_score = self._calculate_quality_score(
                classification_confidence,
                extraction_result.extraction_confidence,
                validation_result['validation_score'],
                extraction_result.coverage_score
            )
            
            processing_metrics['quality_score'] = quality_score
            
            logger.info(f"   → Overall Quality Score: {quality_score:.2%}")
            
            # Add quality-based recommendations
            if quality_score < 0.9:
                recommendations.append("Manual review recommended due to quality score below 90%")
            if extraction_result.coverage_score < 0.8:
                recommendations.append("Some expected fields may be missing - verify document completeness")
            
            # Calculate total time
            total_time = (time.time() - start_time) * 1000
            processing_metrics['total_time_ms'] = total_time
            
            logger.info("\n" + "=" * 80)
            logger.info(f"PIPELINE COMPLETE - Total Time: {total_time:.1f}ms")
            logger.info(f"Quality Score: {quality_score:.2%}")
            logger.info("=" * 80)
            
            return PipelineResult(
                status='success',
                document_type=doc_type,
                classification_confidence=classification_confidence,
                extracted_entities=extracted_entities,
                normalized_entities=normalized_entities,
                validation_result=validation_result,
                adjudication_result=adjudication_result,
                eligibility_result=eligibility_result,
                policy_interpretation=policy_interpretation,
                processing_metrics=processing_metrics,
                quality_score=quality_score,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}", exc_info=True)
            
            return PipelineResult(
                status='error',
                document_type='unknown',
                classification_confidence=0,
                extracted_entities={},
                normalized_entities={},
                validation_result={'is_valid': False, 'issues': [str(e)]},
                adjudication_result=None,
                eligibility_result=None,
                policy_interpretation=None,
                processing_metrics={'error': str(e)},
                quality_score=0,
                recommendations=['Manual processing required due to pipeline error']
            )
    
    def _calculate_quality_score(self, classification_conf: float,
                                 extraction_conf: float,
                                 validation_score: float,
                                 coverage_score: float) -> float:
        """Calculate overall quality score"""
        # Weighted average
        weights = {
            'classification': 0.2,
            'extraction': 0.35,
            'validation': 0.25,
            'coverage': 0.2
        }
        
        score = (
            classification_conf * weights['classification'] +
            extraction_conf * weights['extraction'] +
            validation_score * weights['validation'] +
            coverage_score * weights['coverage']
        )
        
        return min(1.0, max(0.0, score))
    
    def _get_default_policy_context(self) -> Dict:
        """Get default policy context for adjudication"""
        return {
            'policy_number': 'DEFAULT',
            'effective_date': '2024-01-01',
            'elimination_period': '90 days',
            'benefit_percentage': '60%',
            'max_benefit': '10000',
            'benefit_period': '24 months',
            'own_occupation_period': '24 months',
            'exclusions': [
                'pre-existing conditions within 12 months',
                'self-inflicted injuries',
                'war or act of war'
            ]
        }
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about pipeline capabilities"""
        return {
            'name': 'Healthcare IDP Pipeline',
            'version': '2.0.0',
            'capabilities': [
                'Document Classification (RFP, Claims, Enrollment, Policy)',
                'High-Precision Entity Extraction (97-99% target)',
                'Data Normalization and Validation',
                'Automated Claim Adjudication',
                'Eligibility Matching',
                'Policy Interpretation',
                'Metrics Tracking'
            ],
            'supported_document_types': [
                'disability_claim',
                'enrollment',
                'policy',
                'rfp'
            ],
            'extraction_methods': [
                'regex',
                'spacy_ner',
                'llm_enhanced',
                'ensemble'
            ],
            'tech_stack': {
                'nlp': 'spaCy 3.7',
                'llm': 'AWS Bedrock (Claude)',
                'ocr': 'Tesseract',
                'framework': 'FastAPI'
            },
            'accuracy_targets': {
                'extraction': '97-99%',
                'classification': '99%',
                'adjudication': '95%'
            }
        }
    
    def to_dict(self, result: PipelineResult) -> Dict[str, Any]:
        """Convert pipeline result to dictionary for API response"""
        return {
            'status': result.status,
            'document_type': result.document_type,
            'classification_confidence': result.classification_confidence,
            'extracted_entities': result.extracted_entities,
            'normalized_entities': result.normalized_entities,
            'validation': result.validation_result,
            'adjudication': result.adjudication_result,
            'eligibility': result.eligibility_result,
            'policy_interpretation': result.policy_interpretation,
            'processing_metrics': result.processing_metrics,
            'quality_score': result.quality_score,
            'recommendations': result.recommendations
        }


# Backward compatibility
IDPPipeline = EnhancedIDPPipeline
