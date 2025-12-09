"""
Document Classification Module
Multi-stage classifier using rule-based + LLM ensemble approach
"""

import json
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DocumentClassifier:
    """
    Multi-stage document classification using ensemble approach:
    1. Rule-based pre-classification (fast path)
    2. LLM-based semantic classification for ambiguous cases
    """
    
    def __init__(self, bedrock_client=None, config: Optional[Dict] = None):
        """
        Initialize the document classifier.
        
        Args:
            bedrock_client: Optional pre-configured boto3 bedrock client
            config: Optional configuration dictionary
        """
        self.bedrock_client = bedrock_client
        self.config = config or {}
        
        # Document type keywords for rule-based classification
        self.doc_types = {
            'disability_claim': [
                'claim form', 'disability', 'claimant', 'diagnosis',
                'date of disability', 'attending physician'
            ],
            'enrollment': [
                'enroll', 'employee', 'dependent', 'coverage election',
                'beneficiary', 'plan selection'
            ],
            'policy': [
                'policy', 'certificate', 'terms', 'conditions',
                'exclusions', 'benefit schedule'
            ],
            'rfp': [
                'request for proposal', 'bid', 'pricing',
                'requirements', 'scope of work'
            ]
        }
        
        self.confidence_threshold = self.config.get('confidence_threshold', 0.85)
        logger.info("DocumentClassifier initialized")
    
    def classify(self, text: str, metadata: Optional[Dict] = None) -> Tuple[str, float]:
        """
        Classify document with confidence score.
        
        Args:
            text: Document text content
            metadata: Optional metadata about the document
            
        Returns:
            Tuple of (document_type, confidence_score)
        """
        logger.info("Starting document classification")
        
        # Stage 1: Rule-based quick classification
        rule_result = self._rule_based_classify(text)
        logger.debug(f"Rule-based result: {rule_result}")
        
        if rule_result[1] >= self.confidence_threshold:
            logger.info(f"High confidence classification: {rule_result}")
            return rule_result
        
        # Stage 2: LLM-based classification for ambiguous cases
        if self.bedrock_client:
            try:
                llm_result = self._llm_classify(text)
                logger.info(f"LLM classification result: {llm_result}")
                return llm_result
            except Exception as e:
                logger.warning(f"LLM classification failed: {e}, using rule-based result")
                return rule_result
        
        return rule_result
    
    def _rule_based_classify(self, text: str) -> Tuple[str, float]:
        """Fast keyword-based classification"""
        text_lower = text.lower()
        scores = {}
        
        for doc_type, keywords in self.doc_types.items():
            # Count keyword matches
            matches = sum(1 for kw in keywords if kw in text_lower)
            # Calculate score as percentage of keywords found
            scores[doc_type] = matches / len(keywords)
        
        # Get best match
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        
        return (best_type, confidence)
    
    def _llm_classify(self, text: str) -> Tuple[str, float]:
        """LLM-based classification using AWS Bedrock"""
        
        prompt = f"""Classify this insurance document into one of these categories:
- disability_claim: Claim forms for disability benefits
- enrollment: Employee enrollment or election forms
- policy: Policy documents, certificates of insurance
- rfp: Requests for proposals or bids

Document excerpt (first 2000 characters):
{text[:2000]}

Respond in JSON format:
{{"document_type": "category", "confidence": 0.95, "reasoning": "brief explanation"}}"""

        try:
            response = self.bedrock_client.invoke_model(
                modelId='anthropic.claude-3-sonnet-20240229-v1:0',
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                    "temperature": 0.1
                })
            )
            
            result = json.loads(response['body'].read())
            content = result['content'][0]['text']
            
            # Parse JSON response
            classification = json.loads(content)
            
            return (classification['document_type'], classification['confidence'])
        
        except Exception as e:
            logger.error(f"LLM classification error: {e}")
            raise
