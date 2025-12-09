"""
Data Normalization Module
Standardize and normalize extracted data across document types
Ensures consistency for downstream processing and 97-99% accuracy targets
"""

import re
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class NormalizationResult:
    """Result of data normalization"""
    original_value: Any
    normalized_value: Any
    confidence: float
    normalization_type: str
    warnings: List[str]


class DataNormalizer:
    """
    Normalize and standardize extracted data.
    Handles dates, amounts, names, identifiers, and more.
    """
    
    def __init__(self):
        """Initialize the data normalizer"""
        self.date_formats = [
            '%m/%d/%Y', '%m-%d-%Y', '%Y-%m-%d', '%Y/%m/%d',
            '%d/%m/%Y', '%d-%m-%Y', '%m/%d/%y', '%B %d, %Y',
            '%b %d, %Y', '%d %B %Y', '%d %b %Y'
        ]
        
        self.state_abbreviations = {
            'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR',
            'california': 'CA', 'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE',
            'florida': 'FL', 'georgia': 'GA', 'hawaii': 'HI', 'idaho': 'ID',
            'illinois': 'IL', 'indiana': 'IN', 'iowa': 'IA', 'kansas': 'KS',
            'kentucky': 'KY', 'louisiana': 'LA', 'maine': 'ME', 'maryland': 'MD',
            'massachusetts': 'MA', 'michigan': 'MI', 'minnesota': 'MN', 'mississippi': 'MS',
            'missouri': 'MO', 'montana': 'MT', 'nebraska': 'NE', 'nevada': 'NV',
            'new hampshire': 'NH', 'new jersey': 'NJ', 'new mexico': 'NM', 'new york': 'NY',
            'north carolina': 'NC', 'north dakota': 'ND', 'ohio': 'OH', 'oklahoma': 'OK',
            'oregon': 'OR', 'pennsylvania': 'PA', 'rhode island': 'RI', 'south carolina': 'SC',
            'south dakota': 'SD', 'tennessee': 'TN', 'texas': 'TX', 'utah': 'UT',
            'vermont': 'VT', 'virginia': 'VA', 'washington': 'WA', 'west virginia': 'WV',
            'wisconsin': 'WI', 'wyoming': 'WY'
        }
        
        logger.info("DataNormalizer initialized")
    
    def normalize_all(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize all entities in a dictionary.
        
        Args:
            entities: Dictionary of extracted entities
            
        Returns:
            Dictionary with normalized values
        """
        normalized = {}
        normalization_report = []
        
        for key, value in entities.items():
            if value is None:
                normalized[key] = None
                continue
            
            # Determine normalization type based on key name
            if any(d in key.lower() for d in ['date', 'dob', 'effective', 'disability']):
                result = self.normalize_date(value)
            elif any(m in key.lower() for m in ['amount', 'benefit', 'premium', 'salary', 'earnings']):
                result = self.normalize_money(value)
            elif 'ssn' in key.lower() or 'social' in key.lower():
                result = self.normalize_ssn(value)
            elif 'phone' in key.lower() or 'tel' in key.lower():
                result = self.normalize_phone(value)
            elif any(n in key.lower() for n in ['name', 'person', 'claimant', 'employee']):
                result = self.normalize_name(value)
            elif 'email' in key.lower():
                result = self.normalize_email(value)
            elif any(p in key.lower() for p in ['percent', 'pct', 'rate']):
                result = self.normalize_percentage(value)
            elif any(i in key.lower() for i in ['id', 'number', 'policy', 'claim']):
                result = self.normalize_identifier(value)
            else:
                result = NormalizationResult(
                    original_value=value,
                    normalized_value=value,
                    confidence=1.0,
                    normalization_type='passthrough',
                    warnings=[]
                )
            
            normalized[key] = result.normalized_value
            normalization_report.append({
                'field': key,
                'original': result.original_value,
                'normalized': result.normalized_value,
                'confidence': result.confidence,
                'type': result.normalization_type,
                'warnings': result.warnings
            })
        
        normalized['_normalization_report'] = normalization_report
        return normalized
    
    def normalize_date(self, value: Union[str, List[str]]) -> NormalizationResult:
        """
        Normalize date values to ISO format (YYYY-MM-DD).
        
        Args:
            value: Date string or list of date strings
            
        Returns:
            NormalizationResult with standardized date
        """
        if isinstance(value, list):
            return NormalizationResult(
                original_value=value,
                normalized_value=[self.normalize_date(v).normalized_value for v in value],
                confidence=0.9,
                normalization_type='date_list',
                warnings=[]
            )
        
        value = str(value).strip()
        warnings = []
        
        for fmt in self.date_formats:
            try:
                dt = datetime.strptime(value, fmt)
                
                # Validate reasonable date range
                if dt.year < 1900 or dt.year > 2100:
                    warnings.append(f"Date year {dt.year} outside expected range")
                
                return NormalizationResult(
                    original_value=value,
                    normalized_value=dt.strftime('%Y-%m-%d'),
                    confidence=0.98,
                    normalization_type='date',
                    warnings=warnings
                )
            except ValueError:
                continue
        
        # Try to extract date components
        date_pattern = r'(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})'
        match = re.search(date_pattern, value)
        if match:
            try:
                m, d, y = match.groups()
                y = int(y)
                if y < 100:
                    y = 2000 + y if y < 50 else 1900 + y
                dt = datetime(y, int(m), int(d))
                return NormalizationResult(
                    original_value=value,
                    normalized_value=dt.strftime('%Y-%m-%d'),
                    confidence=0.85,
                    normalization_type='date_extracted',
                    warnings=['Date extracted with pattern matching']
                )
            except:
                pass
        
        return NormalizationResult(
            original_value=value,
            normalized_value=value,
            confidence=0.5,
            normalization_type='date_failed',
            warnings=['Could not parse date format']
        )
    
    def normalize_money(self, value: Union[str, float, int]) -> NormalizationResult:
        """
        Normalize monetary values to standard format.
        
        Args:
            value: Money value
            
        Returns:
            NormalizationResult with standardized money format
        """
        original = value
        warnings = []
        
        if isinstance(value, (int, float)):
            return NormalizationResult(
                original_value=original,
                normalized_value=round(float(value), 2),
                confidence=0.99,
                normalization_type='money',
                warnings=[]
            )
        
        value = str(value).strip()
        
        # Extract numeric value
        cleaned = re.sub(r'[^\d.,]', '', value)
        
        # Handle comma as thousands separator vs decimal
        if ',' in cleaned and '.' in cleaned:
            # Assume comma is thousands separator
            cleaned = cleaned.replace(',', '')
        elif ',' in cleaned:
            # Check if comma is decimal separator (European format)
            parts = cleaned.split(',')
            if len(parts) == 2 and len(parts[1]) == 2:
                cleaned = cleaned.replace(',', '.')
            else:
                cleaned = cleaned.replace(',', '')
        
        try:
            amount = round(float(cleaned), 2)
            return NormalizationResult(
                original_value=original,
                normalized_value=amount,
                confidence=0.95,
                normalization_type='money',
                warnings=warnings
            )
        except ValueError:
            return NormalizationResult(
                original_value=original,
                normalized_value=value,
                confidence=0.5,
                normalization_type='money_failed',
                warnings=['Could not parse monetary value']
            )
    
    def normalize_ssn(self, value: str) -> NormalizationResult:
        """
        Normalize and validate SSN format.
        
        Args:
            value: SSN string
            
        Returns:
            NormalizationResult with standardized SSN (masked for security)
        """
        value = str(value).strip()
        digits = re.sub(r'\D', '', value)
        warnings = []
        
        if len(digits) != 9:
            return NormalizationResult(
                original_value=value,
                normalized_value=value,
                confidence=0.5,
                normalization_type='ssn_invalid',
                warnings=['SSN must be 9 digits']
            )
        
        # Format as XXX-XX-XXXX (masked)
        masked = f"XXX-XX-{digits[-4:]}"
        full_formatted = f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"
        
        return NormalizationResult(
            original_value=value,
            normalized_value={
                'masked': masked,
                'last_four': digits[-4:],
                '_full': full_formatted  # Store securely
            },
            confidence=0.98,
            normalization_type='ssn',
            warnings=warnings
        )
    
    def normalize_phone(self, value: str) -> NormalizationResult:
        """
        Normalize phone number to standard format.
        
        Args:
            value: Phone number string
            
        Returns:
            NormalizationResult with standardized phone format
        """
        value = str(value).strip()
        digits = re.sub(r'\D', '', value)
        
        if len(digits) == 10:
            formatted = f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
            return NormalizationResult(
                original_value=value,
                normalized_value=formatted,
                confidence=0.98,
                normalization_type='phone',
                warnings=[]
            )
        elif len(digits) == 11 and digits[0] == '1':
            formatted = f"({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
            return NormalizationResult(
                original_value=value,
                normalized_value=formatted,
                confidence=0.96,
                normalization_type='phone',
                warnings=['Country code stripped']
            )
        else:
            return NormalizationResult(
                original_value=value,
                normalized_value=value,
                confidence=0.6,
                normalization_type='phone_invalid',
                warnings=['Unexpected phone number length']
            )
    
    def normalize_name(self, value: Union[str, List[str]]) -> NormalizationResult:
        """
        Normalize person names to standard format.
        
        Args:
            value: Name string or list
            
        Returns:
            NormalizationResult with standardized name
        """
        if isinstance(value, list):
            return NormalizationResult(
                original_value=value,
                normalized_value=[self.normalize_name(v).normalized_value for v in value],
                confidence=0.9,
                normalization_type='name_list',
                warnings=[]
            )
        
        value = str(value).strip()
        warnings = []
        
        # Remove extra whitespace
        cleaned = ' '.join(value.split())
        
        # Title case (respecting common exceptions)
        exceptions = {'jr', 'sr', 'ii', 'iii', 'iv', 'md', 'phd', 'esq'}
        suffixes = {'jr.', 'sr.', 'ii', 'iii', 'iv', 'm.d.', 'ph.d.', 'esq.'}
        
        parts = cleaned.split()
        normalized_parts = []
        
        for part in parts:
            lower = part.lower().rstrip('.,')
            if lower in exceptions:
                normalized_parts.append(part.upper())
            elif "'" in part or "-" in part:
                # Handle names like O'Brien or Smith-Jones
                subparts = re.split(r"(['-])", part)
                normalized_parts.append(''.join(
                    p.capitalize() if p not in "'-" else p for p in subparts
                ))
            else:
                normalized_parts.append(part.capitalize())
        
        normalized = ' '.join(normalized_parts)
        
        return NormalizationResult(
            original_value=value,
            normalized_value=normalized,
            confidence=0.95,
            normalization_type='name',
            warnings=warnings
        )
    
    def normalize_email(self, value: str) -> NormalizationResult:
        """
        Normalize and validate email address.
        
        Args:
            value: Email string
            
        Returns:
            NormalizationResult with standardized email
        """
        value = str(value).strip().lower()
        
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if re.match(email_pattern, value):
            return NormalizationResult(
                original_value=value,
                normalized_value=value,
                confidence=0.98,
                normalization_type='email',
                warnings=[]
            )
        else:
            return NormalizationResult(
                original_value=value,
                normalized_value=value,
                confidence=0.5,
                normalization_type='email_invalid',
                warnings=['Invalid email format']
            )
    
    def normalize_percentage(self, value: Union[str, float, int]) -> NormalizationResult:
        """
        Normalize percentage values.
        
        Args:
            value: Percentage value
            
        Returns:
            NormalizationResult with standardized percentage
        """
        original = value
        
        if isinstance(value, (int, float)):
            # If <= 1, assume decimal format
            if value <= 1:
                return NormalizationResult(
                    original_value=original,
                    normalized_value=value * 100,
                    confidence=0.95,
                    normalization_type='percentage',
                    warnings=['Converted from decimal format']
                )
            return NormalizationResult(
                original_value=original,
                normalized_value=float(value),
                confidence=0.98,
                normalization_type='percentage',
                warnings=[]
            )
        
        value = str(value).strip()
        cleaned = re.sub(r'[^\d.]', '', value)
        
        try:
            pct = float(cleaned)
            return NormalizationResult(
                original_value=original,
                normalized_value=pct,
                confidence=0.95,
                normalization_type='percentage',
                warnings=[]
            )
        except ValueError:
            return NormalizationResult(
                original_value=original,
                normalized_value=value,
                confidence=0.5,
                normalization_type='percentage_failed',
                warnings=['Could not parse percentage']
            )
    
    def normalize_identifier(self, value: str) -> NormalizationResult:
        """
        Normalize identifier values (policy numbers, claim numbers, etc.).
        
        Args:
            value: Identifier string
            
        Returns:
            NormalizationResult with standardized identifier
        """
        value = str(value).strip().upper()
        
        # Remove extra whitespace and standardize separators
        cleaned = re.sub(r'\s+', '', value)
        cleaned = re.sub(r'[-_]+', '-', cleaned)
        
        return NormalizationResult(
            original_value=value,
            normalized_value=cleaned,
            confidence=0.97,
            normalization_type='identifier',
            warnings=[]
        )


class ValidationEngine:
    """
    Validate extracted and normalized data against business rules.
    """
    
    def __init__(self):
        """Initialize validation engine"""
        self.rules = self._load_validation_rules()
        logger.info("ValidationEngine initialized")
    
    def _load_validation_rules(self) -> Dict:
        """Load validation rules"""
        return {
            'ssn': {
                'pattern': r'^\d{3}-\d{2}-\d{4}$',
                'length': 11
            },
            'policy_number': {
                'pattern': r'^[A-Z]{2,4}-?\d{3,}',
                'min_length': 6
            },
            'claim_number': {
                'pattern': r'^[A-Z]{2,4}-?\d{4}',
                'min_length': 6
            },
            'date': {
                'pattern': r'^\d{4}-\d{2}-\d{2}$',
                'min_year': 1900,
                'max_year': 2100
            },
            'email': {
                'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            }
        }
    
    def validate(self, data: Dict, doc_type: str) -> Dict[str, Any]:
        """
        Validate extracted data.
        
        Args:
            data: Normalized data dictionary
            doc_type: Document type for context-specific validation
            
        Returns:
            Validation results with issues and scores
        """
        results = {
            'is_valid': True,
            'validation_score': 1.0,
            'field_validations': {},
            'issues': [],
            'warnings': []
        }
        
        total_fields = 0
        valid_fields = 0
        
        for field, value in data.items():
            if field.startswith('_'):
                continue
            
            total_fields += 1
            field_result = self._validate_field(field, value)
            results['field_validations'][field] = field_result
            
            if field_result['is_valid']:
                valid_fields += 1
            else:
                results['issues'].append({
                    'field': field,
                    'issue': field_result['issue']
                })
                results['is_valid'] = False
            
            if field_result.get('warnings'):
                results['warnings'].extend(field_result['warnings'])
        
        if total_fields > 0:
            results['validation_score'] = valid_fields / total_fields
        
        # Document type specific validation
        doc_validation = self._validate_document_type(data, doc_type)
        results['document_validation'] = doc_validation
        
        if not doc_validation['is_valid']:
            results['is_valid'] = False
            results['issues'].extend(doc_validation['issues'])
        
        return results
    
    def _validate_field(self, field: str, value: Any) -> Dict:
        """Validate a single field"""
        result = {
            'is_valid': True,
            'issue': None,
            'warnings': []
        }
        
        if value is None:
            result['warnings'].append(f"Field '{field}' is empty")
            return result
        
        # Apply type-specific validation
        for rule_type, rule in self.rules.items():
            if rule_type in field.lower():
                if 'pattern' in rule and isinstance(value, str):
                    if not re.match(rule['pattern'], str(value)):
                        result['is_valid'] = False
                        result['issue'] = f"Value doesn't match expected pattern for {rule_type}"
                
                if 'min_length' in rule:
                    if len(str(value)) < rule['min_length']:
                        result['is_valid'] = False
                        result['issue'] = f"Value too short for {rule_type}"
                
                break
        
        return result
    
    def _validate_document_type(self, data: Dict, doc_type: str) -> Dict:
        """Validate data completeness for document type"""
        required_fields = {
            'disability_claim': ['claim_number', 'policy_number', 'date_of_disability'],
            'enrollment': ['effective_date', 'plan_type'],
            'policy': ['policy_number', 'effective_date'],
            'rfp': []
        }
        
        result = {
            'is_valid': True,
            'issues': [],
            'missing_fields': []
        }
        
        required = required_fields.get(doc_type, [])
        for field in required:
            if field not in data or data[field] is None:
                result['missing_fields'].append(field)
                result['issues'].append({
                    'type': 'missing_required',
                    'field': field,
                    'message': f"Required field '{field}' is missing"
                })
                result['is_valid'] = False
        
        return result
