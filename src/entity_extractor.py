"""
Entity Extraction Module
High-precision entity extraction using hybrid approach
Target: 97-99% precision
"""

import re
from typing import Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    High-precision entity extraction using hybrid approach:
    - Regex patterns for structured fields (dates, IDs, amounts)
    - NER models for person/organization names
    - Validation and normalization
    """
    
    def __init__(self):
        """Initialize the entity extractor"""
        
        # Regex patterns for high-precision extraction
        self.patterns = {
            'claim_number': r'(?:claim|clm)[\s#:-]*([A-Z0-9-]{8,})',
            'policy_number': r'(?:policy|pol)[\s#:-]*([A-Z0-9-]{8,})',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'date': r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            'money': r'\$[\d,]+(?:\.\d{2})?',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'percentage': r'\b\d{1,3}%\b'
        }
        
        # Try to load spaCy model
        self.nlp = None
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_lg")
            logger.info("spaCy model loaded successfully")
        except Exception as e:
            logger.warning(f"spaCy model not available: {e}")
        
        logger.info("EntityExtractor initialized")
    
    def extract(self, text: str, doc_type: str) -> Dict[str, Any]:
        """
        Extract entities based on document type.
        
        Args:
            text: Document text content
            doc_type: Type of document (disability_claim, enrollment, etc.)
            
        Returns:
            Dictionary of extracted entities
        """
        logger.info(f"Extracting entities for document type: {doc_type}")
        
        entities = {}
        
        # Pattern-based extraction (high precision)
        for entity_type, pattern in self.patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[entity_type] = matches[0] if len(matches) == 1 else matches
                logger.debug(f"Found {entity_type}: {entities[entity_type]}")
        
        # NER-based extraction if available
        if self.nlp:
            doc = self.nlp(text[:10000])  # Limit for performance
            entities['persons'] = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
            entities['organizations'] = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
        
        # Document-specific extraction
        if doc_type == 'disability_claim':
            entities.update(self._extract_claim_fields(text))
        elif doc_type == 'enrollment':
            entities.update(self._extract_enrollment_fields(text))
        elif doc_type == 'policy':
            entities.update(self._extract_policy_fields(text))
        
        # Normalize and validate
        entities = self._normalize_entities(entities)
        
        logger.info(f"Extracted {len(entities)} entities")
        return entities
    
    def _extract_claim_fields(self, text: str) -> Dict:
        """Extract disability claim specific fields"""
        fields = {}
        
        # Date of disability
        dod_pattern = r'date\s+of\s+disability[:\s]+(\d{1,2}/\d{1,2}/\d{4})'
        match = re.search(dod_pattern, text, re.IGNORECASE)
        if match:
            fields['date_of_disability'] = match.group(1)
        
        # Diagnosis
        diagnosis_pattern = r'(?:diagnosis|condition|nature\s+of\s+disability)[:\s]+([^\n]{10,100})'
        match = re.search(diagnosis_pattern, text, re.IGNORECASE)
        if match:
            fields['diagnosis'] = match.group(1).strip()
        
        # Benefit amount
        benefit_pattern = r'benefit\s+amount[:\s]+\$?([\d,]+(?:\.\d{2})?)'
        match = re.search(benefit_pattern, text, re.IGNORECASE)
        if match:
            fields['benefit_amount'] = match.group(1)
        
        # Employer
        employer_pattern = r'employer[:\s]+([^\n]{5,50})'
        match = re.search(employer_pattern, text, re.IGNORECASE)
        if match:
            fields['employer'] = match.group(1).strip()
        
        return fields
    
    def _extract_enrollment_fields(self, text: str) -> Dict:
        """Extract enrollment specific fields"""
        fields = {}
        
        # Plan selection
        plan_pattern = r'plan\s+(?:selection|type|name)[:\s]+([^\n]{5,50})'
        match = re.search(plan_pattern, text, re.IGNORECASE)
        if match:
            fields['plan_type'] = match.group(1).strip()
        
        # Coverage level
        coverage_pattern = r'coverage\s+level[:\s]+(employee\s+only|employee\s*\+\s*spouse|employee\s*\+\s*family|family)'
        match = re.search(coverage_pattern, text, re.IGNORECASE)
        if match:
            fields['coverage_level'] = match.group(1).strip()
        
        # Effective date
        effective_pattern = r'effective\s+date[:\s]+(\d{1,2}/\d{1,2}/\d{4})'
        match = re.search(effective_pattern, text, re.IGNORECASE)
        if match:
            fields['effective_date'] = match.group(1)
        
        return fields
    
    def _extract_policy_fields(self, text: str) -> Dict:
        """Extract policy document specific fields"""
        fields = {}
        
        # Elimination period
        elim_pattern = r'elimination\s+(?:period|waiting\s+period)[:\s]+(\d+)\s+days?'
        match = re.search(elim_pattern, text, re.IGNORECASE)
        if match:
            fields['elimination_period'] = f"{match.group(1)} days"
        
        # Maximum benefit
        max_benefit_pattern = r'maximum\s+(?:monthly\s+)?benefit[:\s]+\$?([\d,]+(?:\.\d{2})?)'
        match = re.search(max_benefit_pattern, text, re.IGNORECASE)
        if match:
            fields['max_benefit'] = match.group(1)
        
        # Benefit percentage
        benefit_pct_pattern = r'benefit\s+(?:percentage|amount)[:\s]+(\d{1,3})%'
        match = re.search(benefit_pct_pattern, text, re.IGNORECASE)
        if match:
            fields['benefit_percentage'] = f"{match.group(1)}%"
        
        return fields
    
    def _normalize_entities(self, entities: Dict) -> Dict:
        """Normalize and validate extracted entities"""
        normalized = {}
        
        for key, value in entities.items():
            if 'date' in key.lower() and isinstance(value, str):
                # Normalize dates to YYYY-MM-DD
                try:
                    dt = datetime.strptime(value, '%m/%d/%Y')
                    normalized[key] = dt.strftime('%Y-%m-%d')
                except:
                    normalized[key] = value
            elif 'amount' in key.lower() or 'benefit' in key.lower():
                # Normalize money amounts
                if isinstance(value, str):
                    cleaned = value.replace(',', '').replace('$', '')
                    normalized[key] = f"${cleaned}"
                else:
                    normalized[key] = value
            else:
                normalized[key] = value
        
        return normalized
