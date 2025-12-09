"""
Enhanced Entity Extractor with Confidence Scoring
High-precision NER using ensemble of regex, spaCy, and LLM
Target: 97-99% precision
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """Represents a single extracted entity with metadata"""
    value: Any
    entity_type: str
    confidence: float
    source: str  # 'regex', 'spacy', 'llm', 'ensemble'
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None
    context: Optional[str] = None
    validation_status: str = 'pending'
    normalized_value: Optional[Any] = None


@dataclass
class ExtractionResult:
    """Complete extraction result with metrics"""
    entities: Dict[str, Any]
    raw_entities: List[ExtractedEntity]
    extraction_confidence: float
    coverage_score: float
    entity_count: int
    extraction_time_ms: float
    method_breakdown: Dict[str, int] = field(default_factory=dict)


class EnhancedEntityExtractor:
    """
    Production-grade entity extraction with ensemble methods.
    Combines regex, spaCy NER, and LLM for maximum accuracy.
    """
    
    def __init__(self, use_llm: bool = True, confidence_threshold: float = 0.7):
        """
        Initialize the enhanced entity extractor.
        
        Args:
            use_llm: Whether to use LLM for extraction enhancement
            confidence_threshold: Minimum confidence to accept an entity
        """
        self.confidence_threshold = confidence_threshold
        self.use_llm = use_llm
        
        # Initialize components
        self.nlp = self._load_spacy()
        self.llm_provider = self._load_llm() if use_llm else None
        
        # Enhanced regex patterns with named groups
        self.patterns = self._build_patterns()
        
        # Entity type configurations
        self.entity_config = self._build_entity_config()
        
        logger.info(f"EnhancedEntityExtractor initialized (LLM: {use_llm})")
    
    def _load_spacy(self):
        """Load spaCy model with custom extensions"""
        try:
            import spacy
            nlp = spacy.load("en_core_web_lg")
            
            # Add custom entity ruler for domain-specific entities
            if 'entity_ruler' not in nlp.pipe_names:
                ruler = nlp.add_pipe("entity_ruler", before="ner")
                
                # Add healthcare-specific patterns
                patterns = [
                    {"label": "DIAGNOSIS", "pattern": [{"LOWER": {"IN": ["diagnosis", "condition"]}}]},
                    {"label": "POLICY_NUM", "pattern": [{"TEXT": {"REGEX": r"POL-\d{3}-\d{4}"}}]},
                    {"label": "CLAIM_NUM", "pattern": [{"TEXT": {"REGEX": r"CLM-\d{4}-\d{3}"}}]},
                ]
                ruler.add_patterns(patterns)
            
            logger.info("spaCy model loaded with custom extensions")
            return nlp
        except Exception as e:
            logger.warning(f"spaCy not available: {e}")
            return None
    
    def _load_llm(self):
        """Load LLM provider for extraction enhancement"""
        try:
            from .llm_integration import BedrockProvider
            return BedrockProvider()
        except Exception as e:
            logger.warning(f"LLM provider not available: {e}")
            return None
    
    def _build_patterns(self) -> Dict[str, Dict]:
        """Build comprehensive regex patterns"""
        return {
            'claim_number': {
                'patterns': [
                    r'(?:claim|clm)[\s#:.-]*([A-Z]{2,4}[\-_]?\d{4,}[\-_]?[A-Z0-9]*)',
                    r'\b(CLM[\-_]\d{4}[\-_]\d{3,})\b',
                    r'claim\s*(?:number|#|no\.?)[:\s]*(\S+)',
                ],
                'confidence': 0.95
            },
            'policy_number': {
                'patterns': [
                    r'(?:policy|pol)[\s#:.-]*([A-Z]{2,4}[\-_]?\d{3,}[\-_]?[A-Z0-9]*)',
                    r'\b(POL[\-_]\d{3}[\-_]\d{4})\b',
                    r'policy\s*(?:number|#|no\.?)[:\s]*(\S+)',
                ],
                'confidence': 0.95
            },
            'ssn': {
                'patterns': [
                    r'\b(\d{3}[-\s]?\d{2}[-\s]?\d{4})\b',
                ],
                'confidence': 0.98,
                'sensitive': True
            },
            'date': {
                'patterns': [
                    r'\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b',
                    r'\b(\d{4}[/\-]\d{1,2}[/\-]\d{1,2})\b',
                    r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b',
                ],
                'confidence': 0.92
            },
            'money': {
                'patterns': [
                    r'\$\s*([\d,]+(?:\.\d{2})?)',
                    r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:dollars?|usd)',
                    r'(?:amount|benefit|premium)[:\s]*\$?\s*([\d,]+(?:\.\d{2})?)',
                ],
                'confidence': 0.94
            },
            'email': {
                'patterns': [
                    r'\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b',
                ],
                'confidence': 0.99
            },
            'phone': {
                'patterns': [
                    r'\b(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b',
                    r'\b(1[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4})\b',
                ],
                'confidence': 0.96
            },
            'percentage': {
                'patterns': [
                    r'(\d{1,3}(?:\.\d+)?)\s*%',
                    r'(\d{1,3}(?:\.\d+)?)\s*percent',
                ],
                'confidence': 0.95
            },
            'date_of_disability': {
                'patterns': [
                    r'date\s+of\s+disability[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',
                    r'disability\s+(?:began|started|onset)[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',
                    r'disabled\s+(?:on|since)[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',
                ],
                'confidence': 0.96
            },
            'diagnosis': {
                'patterns': [
                    r'(?:diagnosis|condition|nature\s+of\s+(?:illness|disability))[:\s]+([^\n]{10,100})',
                    r'(?:medical\s+)?diagnosis[:\s]+([^\n]{10,100})',
                ],
                'confidence': 0.85
            },
            'benefit_amount': {
                'patterns': [
                    r'benefit\s+amount[:\s]*\$?\s*([\d,]+(?:\.\d{2})?)',
                    r'monthly\s+benefit[:\s]*\$?\s*([\d,]+(?:\.\d{2})?)',
                    r'weekly\s+benefit[:\s]*\$?\s*([\d,]+(?:\.\d{2})?)',
                ],
                'confidence': 0.94
            },
            'elimination_period': {
                'patterns': [
                    r'elimination\s+(?:period)?[:\s]*(\d+)\s*(?:days?|weeks?|months?)',
                    r'waiting\s+period[:\s]*(\d+)\s*(?:days?|weeks?|months?)',
                ],
                'confidence': 0.95
            },
            'employer': {
                'patterns': [
                    r'employer[:\s]+([^\n]{5,50})',
                    r'company[:\s]+([^\n]{5,50})',
                    r'works?\s+(?:for|at)[:\s]+([^\n]{5,50})',
                ],
                'confidence': 0.85
            }
        }
    
    def _build_entity_config(self) -> Dict:
        """Build entity type configuration"""
        return {
            'disability_claim': {
                'required': ['claim_number', 'policy_number', 'date_of_disability'],
                'optional': ['diagnosis', 'benefit_amount', 'employer', 'ssn', 'phone'],
                'extracted_from_text': ['persons', 'organizations', 'dates']
            },
            'enrollment': {
                'required': ['policy_number', 'effective_date'],
                'optional': ['plan_type', 'coverage_level', 'premium_amount'],
                'extracted_from_text': ['persons', 'dates', 'money']
            },
            'policy': {
                'required': ['policy_number', 'effective_date'],
                'optional': ['elimination_period', 'benefit_period', 'max_benefit'],
                'extracted_from_text': ['dates', 'money', 'percentage']
            },
            'rfp': {
                'required': [],
                'optional': ['due_date', 'contact_email', 'requirements'],
                'extracted_from_text': ['organizations', 'dates', 'money']
            }
        }
    
    def extract(self, text: str, doc_type: str = 'general') -> ExtractionResult:
        """
        Extract entities using ensemble of methods.
        
        Args:
            text: Document text
            doc_type: Type of document for context-aware extraction
            
        Returns:
            ExtractionResult with all extracted entities
        """
        import time
        start_time = time.time()
        
        logger.info(f"Starting extraction for {doc_type} document ({len(text)} chars)")
        
        all_entities: List[ExtractedEntity] = []
        method_counts = defaultdict(int)
        
        # Method 1: Regex extraction
        regex_entities = self._extract_with_regex(text)
        all_entities.extend(regex_entities)
        method_counts['regex'] = len(regex_entities)
        logger.debug(f"Regex extracted {len(regex_entities)} entities")
        
        # Method 2: spaCy NER
        if self.nlp:
            spacy_entities = self._extract_with_spacy(text)
            all_entities.extend(spacy_entities)
            method_counts['spacy'] = len(spacy_entities)
            logger.debug(f"spaCy extracted {len(spacy_entities)} entities")
        
        # Method 3: Document-type specific extraction
        specific_entities = self._extract_document_specific(text, doc_type)
        all_entities.extend(specific_entities)
        method_counts['specific'] = len(specific_entities)
        
        # Method 4: LLM enhancement (for high-value documents)
        if self.use_llm and self.llm_provider and len(text) > 100:
            llm_entities = self._extract_with_llm(text, doc_type)
            all_entities.extend(llm_entities)
            method_counts['llm'] = len(llm_entities)
            logger.debug(f"LLM extracted {len(llm_entities)} entities")
        
        # Ensemble: merge and deduplicate
        merged_entities = self._ensemble_merge(all_entities)
        
        # Calculate confidence and coverage
        entities_dict = self._entities_to_dict(merged_entities)
        confidence = self._calculate_confidence(merged_entities, doc_type)
        coverage = self._calculate_coverage(entities_dict, doc_type)
        
        extraction_time = (time.time() - start_time) * 1000
        
        result = ExtractionResult(
            entities=entities_dict,
            raw_entities=merged_entities,
            extraction_confidence=confidence,
            coverage_score=coverage,
            entity_count=len(merged_entities),
            extraction_time_ms=extraction_time,
            method_breakdown=dict(method_counts)
        )
        
        logger.info(f"Extraction complete: {len(merged_entities)} entities, "
                   f"{confidence:.2%} confidence, {extraction_time:.1f}ms")
        
        return result
    
    def _extract_with_regex(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using regex patterns"""
        entities = []
        
        for entity_type, config in self.patterns.items():
            for pattern in config['patterns']:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    # Get the captured group or full match
                    value = match.group(1) if match.groups() else match.group(0)
                    
                    # Get context (50 chars before and after)
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end]
                    
                    entities.append(ExtractedEntity(
                        value=value.strip(),
                        entity_type=entity_type,
                        confidence=config['confidence'],
                        source='regex',
                        start_pos=match.start(),
                        end_pos=match.end(),
                        context=context
                    ))
        
        return entities
    
    def _extract_with_spacy(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using spaCy NER"""
        entities = []
        
        # Process in chunks for long documents
        chunk_size = 100000
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            doc = self.nlp(chunk)
            
            for ent in doc.ents:
                entity_type = self._map_spacy_label(ent.label_)
                
                entities.append(ExtractedEntity(
                    value=ent.text,
                    entity_type=entity_type,
                    confidence=0.85,  # Default spaCy confidence
                    source='spacy',
                    start_pos=ent.start_char + i,
                    end_pos=ent.end_char + i,
                    context=chunk[max(0, ent.start_char-30):min(len(chunk), ent.end_char+30)]
                ))
        
        return entities
    
    def _map_spacy_label(self, label: str) -> str:
        """Map spaCy entity labels to our types"""
        mapping = {
            'PERSON': 'persons',
            'ORG': 'organizations',
            'GPE': 'locations',
            'DATE': 'date',
            'MONEY': 'money',
            'PERCENT': 'percentage',
            'CARDINAL': 'number',
            'TIME': 'time'
        }
        return mapping.get(label, label.lower())
    
    def _extract_document_specific(self, text: str, doc_type: str) -> List[ExtractedEntity]:
        """Extract document-type specific entities"""
        entities = []
        
        if doc_type == 'disability_claim':
            entities.extend(self._extract_claim_entities(text))
        elif doc_type == 'enrollment':
            entities.extend(self._extract_enrollment_entities(text))
        elif doc_type == 'policy':
            entities.extend(self._extract_policy_entities(text))
        
        return entities
    
    def _extract_claim_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract disability claim specific entities"""
        entities = []
        
        # Attending physician
        physician_pattern = r'(?:attending|treating)\s+physician[:\s]+(?:Dr\.?\s+)?([A-Za-z\s]+)'
        match = re.search(physician_pattern, text, re.IGNORECASE)
        if match:
            entities.append(ExtractedEntity(
                value=match.group(1).strip(),
                entity_type='attending_physician',
                confidence=0.88,
                source='specific',
                start_pos=match.start(),
                end_pos=match.end()
            ))
        
        # Occupation
        occupation_pattern = r'occupation[:\s]+([^\n]{5,40})'
        match = re.search(occupation_pattern, text, re.IGNORECASE)
        if match:
            entities.append(ExtractedEntity(
                value=match.group(1).strip(),
                entity_type='occupation',
                confidence=0.85,
                source='specific',
                start_pos=match.start(),
                end_pos=match.end()
            ))
        
        return entities
    
    def _extract_enrollment_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract enrollment form specific entities"""
        entities = []
        
        # Plan selection
        plan_pattern = r'plan\s+(?:selection|type|name)[:\s]+([^\n]{5,50})'
        match = re.search(plan_pattern, text, re.IGNORECASE)
        if match:
            entities.append(ExtractedEntity(
                value=match.group(1).strip(),
                entity_type='plan_type',
                confidence=0.90,
                source='specific'
            ))
        
        # Coverage level
        coverage_pattern = r'coverage\s+(?:level|type)[:\s]+(employee\s+only|employee\s*\+\s*spouse|employee\s*\+\s*family|family)'
        match = re.search(coverage_pattern, text, re.IGNORECASE)
        if match:
            entities.append(ExtractedEntity(
                value=match.group(1).strip(),
                entity_type='coverage_level',
                confidence=0.92,
                source='specific'
            ))
        
        # Beneficiary
        beneficiary_pattern = r'beneficiary[:\s]+([^\n]{5,50})'
        match = re.search(beneficiary_pattern, text, re.IGNORECASE)
        if match:
            entities.append(ExtractedEntity(
                value=match.group(1).strip(),
                entity_type='beneficiary',
                confidence=0.87,
                source='specific'
            ))
        
        return entities
    
    def _extract_policy_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract policy document specific entities"""
        entities = []
        
        # Own occupation period
        own_occ_pattern = r'own\s+occupation\s+(?:period)?[:\s]*(\d+)\s*(months?|years?)'
        match = re.search(own_occ_pattern, text, re.IGNORECASE)
        if match:
            entities.append(ExtractedEntity(
                value=f"{match.group(1)} {match.group(2)}",
                entity_type='own_occupation_period',
                confidence=0.93,
                source='specific'
            ))
        
        # Benefit period
        benefit_period_pattern = r'benefit\s+period[:\s]*(?:to\s+)?(age\s+\d+|\d+\s*(?:months?|years?))'
        match = re.search(benefit_period_pattern, text, re.IGNORECASE)
        if match:
            entities.append(ExtractedEntity(
                value=match.group(1).strip(),
                entity_type='benefit_period',
                confidence=0.91,
                source='specific'
            ))
        
        # Maximum benefit
        max_benefit_pattern = r'maximum\s+(?:monthly\s+)?benefit[:\s]*\$?\s*([\d,]+)'
        match = re.search(max_benefit_pattern, text, re.IGNORECASE)
        if match:
            entities.append(ExtractedEntity(
                value=match.group(1).strip(),
                entity_type='max_benefit',
                confidence=0.94,
                source='specific'
            ))
        
        return entities
    
    def _extract_with_llm(self, text: str, doc_type: str) -> List[ExtractedEntity]:
        """Extract entities using LLM for enhanced accuracy"""
        entities = []
        
        try:
            prompt = f"""Extract all relevant entities from this {doc_type} document.

Document text:
{text[:4000]}

Extract:
- Names (claimant, physician, employer)
- IDs (claim number, policy number, SSN)
- Dates (disability date, effective date)
- Amounts (benefit, premium, salary)
- Medical info (diagnosis, condition)

Return JSON: {{"entities": [{{"type": "...", "value": "...", "confidence": 0.0}}]}}"""

            response = self.llm_provider.invoke(prompt, max_tokens=1000)
            
            import json
            try:
                content = response.content
                if '```' in content:
                    content = content.split('```')[1]
                    if content.startswith('json'):
                        content = content[4:]
                
                data = json.loads(content.strip())
                
                for entity in data.get('entities', []):
                    entities.append(ExtractedEntity(
                        value=entity.get('value', ''),
                        entity_type=entity.get('type', 'unknown'),
                        confidence=entity.get('confidence', 0.85),
                        source='llm'
                    ))
            except json.JSONDecodeError:
                logger.warning("Could not parse LLM extraction response")
                
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
        
        return entities
    
    def _ensemble_merge(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Merge entities from different sources, boost confidence for agreement"""
        
        # Group by type and approximate value
        groups = defaultdict(list)
        
        for entity in entities:
            key = (entity.entity_type, self._normalize_for_matching(str(entity.value)))
            groups[key].append(entity)
        
        merged = []
        for key, group in groups.items():
            if len(group) == 1:
                merged.append(group[0])
            else:
                # Multiple sources agree - boost confidence
                best = max(group, key=lambda e: e.confidence)
                
                # Boost confidence if multiple methods agree
                source_count = len(set(e.source for e in group))
                confidence_boost = 0.05 * (source_count - 1)
                
                merged.append(ExtractedEntity(
                    value=best.value,
                    entity_type=best.entity_type,
                    confidence=min(0.99, best.confidence + confidence_boost),
                    source='ensemble',
                    start_pos=best.start_pos,
                    end_pos=best.end_pos,
                    context=best.context
                ))
        
        return merged
    
    def _normalize_for_matching(self, value: str) -> str:
        """Normalize value for matching/deduplication"""
        return re.sub(r'\s+', ' ', value.lower().strip())
    
    def _entities_to_dict(self, entities: List[ExtractedEntity]) -> Dict[str, Any]:
        """Convert entity list to dictionary"""
        result = {}
        
        for entity in entities:
            if entity.confidence >= self.confidence_threshold:
                if entity.entity_type in result:
                    # Handle multiple values
                    existing = result[entity.entity_type]
                    if isinstance(existing, list):
                        existing.append(entity.value)
                    else:
                        result[entity.entity_type] = [existing, entity.value]
                else:
                    result[entity.entity_type] = entity.value
        
        return result
    
    def _calculate_confidence(self, entities: List[ExtractedEntity], doc_type: str) -> float:
        """Calculate overall extraction confidence"""
        if not entities:
            return 0.0
        
        # Weight by entity importance
        config = self.entity_config.get(doc_type, {})
        required = set(config.get('required', []))
        
        weighted_sum = 0
        weight_total = 0
        
        for entity in entities:
            weight = 2.0 if entity.entity_type in required else 1.0
            weighted_sum += entity.confidence * weight
            weight_total += weight
        
        return weighted_sum / weight_total if weight_total > 0 else 0.0
    
    def _calculate_coverage(self, entities: Dict, doc_type: str) -> float:
        """Calculate how many expected fields were extracted"""
        config = self.entity_config.get(doc_type, {})
        required = config.get('required', [])
        optional = config.get('optional', [])
        
        all_expected = required + optional
        if not all_expected:
            return 1.0
        
        found = sum(1 for field in all_expected if field in entities)
        return found / len(all_expected)


# Backward compatibility
EntityExtractor = EnhancedEntityExtractor
