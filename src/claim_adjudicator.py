"""
Claim Adjudication Module
Automated claim adjudication using business rules engine
"""

from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class AdjudicationResult:
    """Result of claim adjudication"""
    eligibility_status: str  # 'Approved', 'Denied', 'Pending'
    confidence: float
    coverage_verified: bool
    waiting_period_met: bool
    estimated_benefit_start: Optional[str]
    required_documents: List[str]
    denial_reasons: List[str]


class ClaimAdjudicator:
    """
    Automate claim adjudication using rule engine.
    Implements business logic for disability claims processing.
    """
    
    def __init__(self):
        """Initialize the claim adjudicator"""
        self.business_rules = self._load_business_rules()
        logger.info("ClaimAdjudicator initialized")
    
    def adjudicate(self, claim_data: Dict, policy_data: Dict) -> AdjudicationResult:
        """
        Adjudicate a claim based on extracted data and policy terms.
        
        Args:
            claim_data: Dictionary of extracted claim information
            policy_data: Dictionary of policy terms and conditions
            
        Returns:
            AdjudicationResult with decision and reasoning
        """
        logger.info("Starting claim adjudication")
        
        # Initialize result
        result = AdjudicationResult(
            eligibility_status='Pending',
            confidence=0.0,
            coverage_verified=False,
            waiting_period_met=False,
            estimated_benefit_start=None,
            required_documents=[],
            denial_reasons=[]
        )
        
        # Step 1: Verify coverage
        coverage_check = self._verify_coverage(claim_data, policy_data)
        result.coverage_verified = coverage_check['verified']
        
        if not coverage_check['verified']:
            result.eligibility_status = 'Denied'
            result.denial_reasons.append(coverage_check['reason'])
            result.confidence = 0.95
            logger.info(f"Coverage verification failed: {coverage_check['reason']}")
            return result
        
        # Step 2: Check waiting period
        waiting_check = self._check_waiting_period(claim_data, policy_data)
        result.waiting_period_met = waiting_check['met']
        logger.debug(f"Waiting period check: {waiting_check}")
        
        # Step 3: Check pre-existing conditions
        preex_check = self._check_preexisting_conditions(claim_data, policy_data)
        
        if not preex_check['passed']:
            result.eligibility_status = 'Denied'
            result.denial_reasons.append(preex_check['reason'])
            result.confidence = 0.92
            logger.info(f"Pre-existing condition check failed: {preex_check['reason']}")
            return result
        
        # Step 4: Verify required documentation
        doc_check = self._verify_documentation(claim_data)
        result.required_documents = doc_check['missing_documents']
        
        # Step 5: Calculate benefit start date
        if result.waiting_period_met and not result.required_documents:
            result.eligibility_status = 'Approved'
            result.confidence = 0.97
            result.estimated_benefit_start = self._calculate_benefit_start(
                claim_data, policy_data
            )
            logger.info("Claim approved")
        elif not result.required_documents:
            result.eligibility_status = 'Approved - Waiting Period'
            result.confidence = 0.95
            result.estimated_benefit_start = self._calculate_benefit_start(
                claim_data, policy_data
            )
            logger.info("Claim approved pending waiting period")
        else:
            result.eligibility_status = 'Pending - Documentation Required'
            result.confidence = 0.85
            logger.info(f"Pending documentation: {result.required_documents}")
        
        return result
    
    def _verify_coverage(self, claim_data: Dict, policy_data: Dict) -> Dict:
        """Verify active coverage at time of disability"""
        
        # Check policy number match
        claim_policy = claim_data.get('policy_number', '')
        policy_number = policy_data.get('policy_number', '')
        
        if claim_policy and policy_number and claim_policy != policy_number:
            return {
                'verified': False,
                'reason': 'Policy number mismatch'
            }
        
        # Check effective dates
        dod = claim_data.get('date_of_disability')
        policy_effective = policy_data.get('effective_date')
        
        if dod and policy_effective:
            try:
                dod_date = datetime.strptime(dod, '%Y-%m-%d')
                effective_date = datetime.strptime(policy_effective, '%Y-%m-%d')
                
                if dod_date < effective_date:
                    return {
                        'verified': False,
                        'reason': 'Disability occurred before coverage effective date'
                    }
            except Exception as e:
                logger.warning(f"Date parsing error: {e}")
        
        return {'verified': True, 'reason': None}
    
    def _check_waiting_period(self, claim_data: Dict, policy_data: Dict) -> Dict:
        """Check if elimination period has been satisfied"""
        
        dod = claim_data.get('date_of_disability')
        elimination_period = policy_data.get('elimination_period', '90 days')
        
        if not dod:
            return {'met': False, 'reason': 'Date of disability not provided'}
        
        try:
            # Extract days from elimination period
            days_match = re.search(r'\d+', elimination_period)
            if not days_match:
                return {'met': False, 'reason': 'Invalid elimination period format'}
            
            days = int(days_match.group())
            dod_date = datetime.strptime(dod, '%Y-%m-%d')
            required_date = dod_date + timedelta(days=days)
            
            if datetime.now() >= required_date:
                return {'met': True, 'days_remaining': 0}
            else:
                days_remaining = (required_date - datetime.now()).days
                return {'met': False, 'days_remaining': days_remaining}
        
        except Exception as e:
            logger.error(f"Error checking waiting period: {e}")
            return {'met': False, 'reason': 'Unable to calculate waiting period'}
    
    def _check_preexisting_conditions(self, claim_data: Dict, policy_data: Dict) -> Dict:
        """Check for pre-existing condition exclusions"""
        
        preex_clause = policy_data.get('preexisting_condition_clause', '')
        
        # Simplified logic - in production would check medical history
        if 'excluded' in preex_clause.lower():
            # Would need medical records to properly evaluate
            return {
                'passed': True,
                'reason': None,
                'requires_review': True
            }
        
        return {'passed': True, 'reason': None}
    
    def _verify_documentation(self, claim_data: Dict) -> Dict:
        """Check for required supporting documentation"""
        
        required_docs = [
            'Medical certification',
            'Employer verification',
            'Proof of earnings'
        ]
        
        # In production, would check against actual document uploads
        missing = []
        
        if not claim_data.get('diagnosis'):
            missing.append('Medical certification')
        if not claim_data.get('employer'):
            missing.append('Employer verification')
        
        return {'missing_documents': missing}
    
    def _calculate_benefit_start(self, claim_data: Dict, policy_data: Dict) -> Optional[str]:
        """Calculate when benefits should begin"""
        
        dod = claim_data.get('date_of_disability')
        elimination_period = policy_data.get('elimination_period', '90 days')
        
        if not dod:
            return None
        
        try:
            days_match = re.search(r'\d+', elimination_period)
            if not days_match:
                return None
            
            days = int(days_match.group())
            dod_date = datetime.strptime(dod, '%Y-%m-%d')
            benefit_start = dod_date + timedelta(days=days)
            return benefit_start.strftime('%Y-%m-%d')
        
        except Exception as e:
            logger.error(f"Error calculating benefit start: {e}")
            return None
    
    def _load_business_rules(self) -> Dict:
        """Load business rules configuration"""
        return {
            'max_benefit_period': '24 months',
            'own_occupation_period': '24 months',
            'required_documentation': [
                'Medical certification',
                'Employer verification',
                'Proof of earnings'
            ]
        }
