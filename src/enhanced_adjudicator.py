"""
Enhanced Claim Adjudicator with Comprehensive Business Rules
Production-grade rule engine for automated claim processing
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AdjudicationStatus(Enum):
    """Claim adjudication status codes"""
    APPROVED = "approved"
    DENIED = "denied"
    PENDING = "pending"
    PENDING_DOCUMENTATION = "pending_documentation"
    PENDING_REVIEW = "pending_review"
    PENDING_WAITING_PERIOD = "pending_waiting_period"
    REFERRED_TO_MEDICAL = "referred_to_medical"


@dataclass
class RuleResult:
    """Result of a single rule evaluation"""
    rule_id: str
    rule_name: str
    passed: bool
    confidence: float
    message: str
    severity: str  # 'critical', 'major', 'minor', 'info'
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdjudicationDecision:
    """Complete adjudication decision"""
    claim_id: str
    status: AdjudicationStatus
    confidence: float
    rule_results: List[RuleResult]
    required_actions: List[str]
    missing_documents: List[str]
    denial_reasons: List[str]
    benefit_calculation: Optional[Dict[str, Any]]
    estimated_benefit_start: Optional[str]
    processing_notes: List[str]
    audit_trail: List[Dict[str, Any]]
    sla_met: bool
    decision_timestamp: str


class BusinessRule:
    """Base class for business rules"""
    
    def __init__(self, rule_id: str, rule_name: str, severity: str = 'major'):
        self.rule_id = rule_id
        self.rule_name = rule_name
        self.severity = severity
    
    def evaluate(self, claim_data: Dict, policy_data: Dict) -> RuleResult:
        raise NotImplementedError


class CoverageVerificationRule(BusinessRule):
    """Verify active coverage at time of disability"""
    
    def __init__(self):
        super().__init__('CVR001', 'Coverage Verification', 'critical')
    
    def evaluate(self, claim_data: Dict, policy_data: Dict) -> RuleResult:
        # Check policy numbers match
        claim_policy = claim_data.get('policy_number', '').upper().replace('-', '')
        policy_number = policy_data.get('policy_number', '').upper().replace('-', '')
        
        if claim_policy and policy_number:
            if claim_policy != policy_number:
                return RuleResult(
                    rule_id=self.rule_id,
                    rule_name=self.rule_name,
                    passed=False,
                    confidence=0.98,
                    message=f"Policy number mismatch: claim shows {claim_data.get('policy_number')}, "
                            f"but policy is {policy_data.get('policy_number')}",
                    severity='critical',
                    data={'claim_policy': claim_policy, 'actual_policy': policy_number}
                )
        
        # Check coverage dates
        dod = claim_data.get('date_of_disability') or claim_data.get('disability_date')
        policy_effective = policy_data.get('effective_date')
        policy_termination = policy_data.get('termination_date')
        
        if dod and policy_effective:
            try:
                dod_date = self._parse_date(dod)
                effective_date = self._parse_date(policy_effective)
                
                if dod_date < effective_date:
                    return RuleResult(
                        rule_id=self.rule_id,
                        rule_name=self.rule_name,
                        passed=False,
                        confidence=0.97,
                        message=f"Disability date ({dod}) is before policy effective date ({policy_effective})",
                        severity='critical'
                    )
                
                if policy_termination:
                    term_date = self._parse_date(policy_termination)
                    if dod_date > term_date:
                        return RuleResult(
                            rule_id=self.rule_id,
                            rule_name=self.rule_name,
                            passed=False,
                            confidence=0.97,
                            message=f"Disability date ({dod}) is after policy termination ({policy_termination})",
                            severity='critical'
                        )
            except Exception as e:
                logger.warning(f"Date parsing error in coverage verification: {e}")
        
        return RuleResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            passed=True,
            confidence=0.95,
            message="Coverage verified for date of disability",
            severity='info'
        )
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime"""
        formats = ['%Y-%m-%d', '%m/%d/%Y', '%m-%d-%Y', '%d/%m/%Y']
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"Could not parse date: {date_str}")


class EliminationPeriodRule(BusinessRule):
    """Check if elimination/waiting period has been satisfied"""
    
    def __init__(self):
        super().__init__('EPR001', 'Elimination Period Check', 'major')
    
    def evaluate(self, claim_data: Dict, policy_data: Dict) -> RuleResult:
        dod = claim_data.get('date_of_disability') or claim_data.get('disability_date')
        elimination_period = policy_data.get('elimination_period', '90 days')
        
        if not dod:
            return RuleResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                passed=False,
                confidence=0.99,
                message="Date of disability not provided",
                severity='critical'
            )
        
        try:
            # Extract days from elimination period
            days_match = re.search(r'(\d+)', str(elimination_period))
            if not days_match:
                days = 90  # Default
            else:
                days = int(days_match.group(1))
                # Handle weeks/months
                if 'week' in str(elimination_period).lower():
                    days *= 7
                elif 'month' in str(elimination_period).lower():
                    days *= 30
            
            dod_date = datetime.strptime(dod, '%Y-%m-%d') if '-' in dod else \
                       datetime.strptime(dod, '%m/%d/%Y')
            
            required_date = dod_date + timedelta(days=days)
            today = datetime.now()
            
            if today >= required_date:
                return RuleResult(
                    rule_id=self.rule_id,
                    rule_name=self.rule_name,
                    passed=True,
                    confidence=0.98,
                    message=f"Elimination period of {days} days satisfied",
                    severity='info',
                    data={
                        'elimination_days': days,
                        'satisfied_date': required_date.strftime('%Y-%m-%d')
                    }
                )
            else:
                days_remaining = (required_date - today).days
                return RuleResult(
                    rule_id=self.rule_id,
                    rule_name=self.rule_name,
                    passed=False,
                    confidence=0.97,
                    message=f"Elimination period not met. {days_remaining} days remaining",
                    severity='major',
                    data={
                        'elimination_days': days,
                        'days_remaining': days_remaining,
                        'satisfied_date': required_date.strftime('%Y-%m-%d')
                    }
                )
        except Exception as e:
            logger.error(f"Error in elimination period calculation: {e}")
            return RuleResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                passed=False,
                confidence=0.5,
                message=f"Could not calculate elimination period: {str(e)}",
                severity='major'
            )


class PreExistingConditionRule(BusinessRule):
    """Check pre-existing condition exclusions"""
    
    COMMON_PREEX_CONDITIONS = [
        'diabetes', 'heart disease', 'cancer', 'arthritis', 'hypertension',
        'depression', 'anxiety', 'back pain', 'chronic pain', 'asthma'
    ]
    
    def __init__(self):
        super().__init__('PEC001', 'Pre-Existing Condition Check', 'major')
    
    def evaluate(self, claim_data: Dict, policy_data: Dict) -> RuleResult:
        diagnosis = claim_data.get('diagnosis', '')
        # Handle if diagnosis is a list
        if isinstance(diagnosis, list):
            diagnosis = ' '.join(str(d) for d in diagnosis)
        diagnosis = str(diagnosis).lower()
        
        preex_clause = policy_data.get('preexisting_condition_clause', '')
        preex_period = policy_data.get('preexisting_lookback_period', '12 months')
        
        # Check if diagnosis is a common pre-existing condition
        flagged_conditions = []
        for condition in self.COMMON_PREEX_CONDITIONS:
            if condition in diagnosis:
                flagged_conditions.append(condition)
        
        if flagged_conditions:
            return RuleResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                passed=False,
                confidence=0.75,
                message=f"Potential pre-existing condition detected: {', '.join(flagged_conditions)}. "
                        f"Manual review required.",
                severity='major',
                data={
                    'flagged_conditions': flagged_conditions,
                    'lookback_period': preex_period,
                    'requires_medical_review': True
                }
            )
        
        return RuleResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            passed=True,
            confidence=0.85,
            message="No obvious pre-existing conditions detected",
            severity='info',
            data={'requires_medical_review': False}
        )


class DocumentationRule(BusinessRule):
    """Verify required documentation is present"""
    
    REQUIRED_DOCS = {
        'disability_claim': [
            ('attending_physician_statement', 'Attending Physician Statement (APS)'),
            ('employer_statement', 'Employer Statement'),
            ('earnings_documentation', 'Proof of Earnings (W2/Pay Stubs)'),
            ('claimant_statement', 'Claimant Statement'),
        ],
        'short_term_disability': [
            ('attending_physician_statement', 'Attending Physician Statement'),
            ('return_to_work_date', 'Expected Return to Work Date'),
        ],
        'long_term_disability': [
            ('attending_physician_statement', 'Attending Physician Statement'),
            ('independent_medical_exam', 'Independent Medical Examination'),
            ('functional_capacity_eval', 'Functional Capacity Evaluation'),
        ]
    }
    
    def __init__(self):
        super().__init__('DOC001', 'Documentation Verification', 'major')
    
    def evaluate(self, claim_data: Dict, policy_data: Dict) -> RuleResult:
        claim_type = claim_data.get('claim_type', 'disability_claim')
        required = self.REQUIRED_DOCS.get(claim_type, self.REQUIRED_DOCS['disability_claim'])
        
        missing = []
        present = []
        
        # Check for evidence of each required document
        for doc_key, doc_name in required:
            # Check if document data exists
            if claim_data.get(doc_key) or \
               claim_data.get('diagnosis') and 'physician' in doc_key.lower() or \
               claim_data.get('employer') and 'employer' in doc_key.lower():
                present.append(doc_name)
            else:
                missing.append(doc_name)
        
        if missing:
            return RuleResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                passed=False,
                confidence=0.92,
                message=f"Missing required documentation: {', '.join(missing)}",
                severity='major',
                data={
                    'missing_documents': missing,
                    'present_documents': present,
                    'completeness_pct': len(present) / len(required) * 100
                }
            )
        
        return RuleResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            passed=True,
            confidence=0.95,
            message="All required documentation present",
            severity='info',
            data={'present_documents': present}
        )


class BenefitCalculationRule(BusinessRule):
    """Calculate benefit amount based on policy terms"""
    
    def __init__(self):
        super().__init__('BEN001', 'Benefit Calculation', 'major')
    
    def evaluate(self, claim_data: Dict, policy_data: Dict) -> RuleResult:
        # Get earnings
        earnings = claim_data.get('monthly_earnings') or \
                   claim_data.get('salary') or \
                   claim_data.get('benefit_amount')
        
        if not earnings:
            return RuleResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                passed=False,
                confidence=0.95,
                message="Cannot calculate benefit - earnings information not provided",
                severity='major',
                data={'requires': ['monthly_earnings', 'salary', 'W2']}
            )
        
        # Parse earnings
        if isinstance(earnings, str):
            earnings = float(re.sub(r'[^\d.]', '', earnings))
        
        # Get benefit percentage
        benefit_pct_str = policy_data.get('benefit_percentage', '60%')
        benefit_pct = float(re.sub(r'[^\d.]', '', str(benefit_pct_str))) / 100
        
        # Get maximum benefit
        max_benefit_str = policy_data.get('max_benefit', '10000')
        max_benefit = float(re.sub(r'[^\d.]', '', str(max_benefit_str)))
        
        # Calculate
        calculated_benefit = earnings * benefit_pct
        final_benefit = min(calculated_benefit, max_benefit)
        
        return RuleResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            passed=True,
            confidence=0.97,
            message=f"Monthly benefit calculated: ${final_benefit:,.2f}",
            severity='info',
            data={
                'monthly_earnings': earnings,
                'benefit_percentage': benefit_pct * 100,
                'calculated_benefit': calculated_benefit,
                'max_benefit': max_benefit,
                'final_monthly_benefit': final_benefit,
                'annual_benefit': final_benefit * 12
            }
        )


class ExclusionRule(BusinessRule):
    """Check for policy exclusions"""
    
    STANDARD_EXCLUSIONS = [
        'self-inflicted',
        'war',
        'illegal act',
        'cosmetic surgery',
        'pre-existing',
        'experimental treatment',
        'workers compensation',
        'intoxication'
    ]
    
    def __init__(self):
        super().__init__('EXC001', 'Exclusion Check', 'critical')
    
    def _safe_to_string(self, value) -> str:
        """Safely convert value to lowercase string"""
        if isinstance(value, list):
            return ' '.join(str(v) for v in value).lower()
        return str(value).lower() if value else ''
    
    def evaluate(self, claim_data: Dict, policy_data: Dict) -> RuleResult:
        diagnosis = self._safe_to_string(claim_data.get('diagnosis', ''))
        cause = self._safe_to_string(claim_data.get('cause_of_disability', ''))
        notes = self._safe_to_string(claim_data.get('notes', ''))
        
        combined_text = f"{diagnosis} {cause} {notes}"
        
        # Get policy-specific exclusions
        policy_exclusions = policy_data.get('exclusions', [])
        if isinstance(policy_exclusions, str):
            policy_exclusions = [policy_exclusions]
        
        all_exclusions = self.STANDARD_EXCLUSIONS + [e.lower() for e in policy_exclusions]
        
        triggered = []
        for exclusion in all_exclusions:
            if exclusion in combined_text:
                triggered.append(exclusion)
        
        if triggered:
            return RuleResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                passed=False,
                confidence=0.88,
                message=f"Potential exclusion triggered: {', '.join(triggered)}",
                severity='critical',
                data={
                    'triggered_exclusions': triggered,
                    'requires_review': True
                }
            )
        
        return RuleResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            passed=True,
            confidence=0.90,
            message="No exclusions triggered",
            severity='info'
        )


class EnhancedClaimAdjudicator:
    """
    Production-grade claim adjudication engine.
    Implements comprehensive business rules for automated decision-making.
    """
    
    def __init__(self):
        """Initialize the adjudicator with all business rules"""
        self.rules: List[BusinessRule] = [
            CoverageVerificationRule(),
            EliminationPeriodRule(),
            PreExistingConditionRule(),
            DocumentationRule(),
            BenefitCalculationRule(),
            ExclusionRule(),
        ]
        
        self.audit_trail = []
        logger.info(f"EnhancedClaimAdjudicator initialized with {len(self.rules)} rules")
    
    def adjudicate(self, claim_data: Dict, policy_data: Dict) -> AdjudicationDecision:
        """
        Execute full adjudication workflow.
        
        Args:
            claim_data: Extracted claim information
            policy_data: Policy terms and conditions
            
        Returns:
            Complete adjudication decision
        """
        logger.info("=" * 60)
        logger.info("Starting claim adjudication")
        logger.info("=" * 60)
        
        claim_id = claim_data.get('claim_number', 'UNKNOWN')
        start_time = datetime.now()
        
        # Execute all rules
        rule_results: List[RuleResult] = []
        for rule in self.rules:
            logger.debug(f"Evaluating rule: {rule.rule_name}")
            result = rule.evaluate(claim_data, policy_data)
            rule_results.append(result)
            
            self._log_rule_result(result)
        
        # Determine overall decision
        status, confidence = self._determine_status(rule_results)
        
        # Collect issues
        denial_reasons = [r.message for r in rule_results 
                        if not r.passed and r.severity == 'critical']
        
        missing_docs = []
        for r in rule_results:
            if r.rule_id == 'DOC001' and not r.passed:
                missing_docs = r.data.get('missing_documents', [])
        
        required_actions = self._determine_required_actions(rule_results, status)
        
        # Get benefit calculation
        benefit_calc = None
        for r in rule_results:
            if r.rule_id == 'BEN001' and r.passed:
                benefit_calc = r.data
        
        # Calculate benefit start date
        benefit_start = None
        for r in rule_results:
            if r.rule_id == 'EPR001':
                benefit_start = r.data.get('satisfied_date')
        
        # Build audit trail
        audit_entry = {
            'timestamp': start_time.isoformat(),
            'claim_id': claim_id,
            'rules_executed': len(rule_results),
            'rules_passed': sum(1 for r in rule_results if r.passed),
            'final_status': status.value,
            'confidence': confidence
        }
        self.audit_trail.append(audit_entry)
        
        # Calculate SLA
        processing_time = (datetime.now() - start_time).total_seconds()
        sla_met = processing_time < 5.0  # 5 second SLA for automated decisions
        
        decision = AdjudicationDecision(
            claim_id=claim_id,
            status=status,
            confidence=confidence,
            rule_results=rule_results,
            required_actions=required_actions,
            missing_documents=missing_docs,
            denial_reasons=denial_reasons,
            benefit_calculation=benefit_calc,
            estimated_benefit_start=benefit_start,
            processing_notes=self._generate_notes(rule_results),
            audit_trail=[audit_entry],
            sla_met=sla_met,
            decision_timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Adjudication complete: {status.value} ({confidence:.1%} confidence)")
        return decision
    
    def _determine_status(self, results: List[RuleResult]) -> Tuple[AdjudicationStatus, float]:
        """Determine overall status from rule results"""
        
        # Check for critical failures
        critical_failures = [r for r in results 
                           if not r.passed and r.severity == 'critical']
        
        if critical_failures:
            avg_confidence = sum(r.confidence for r in critical_failures) / len(critical_failures)
            return AdjudicationStatus.DENIED, avg_confidence
        
        # Check for documentation issues
        doc_rule = next((r for r in results if r.rule_id == 'DOC001'), None)
        if doc_rule and not doc_rule.passed:
            return AdjudicationStatus.PENDING_DOCUMENTATION, doc_rule.confidence
        
        # Check for waiting period
        elim_rule = next((r for r in results if r.rule_id == 'EPR001'), None)
        if elim_rule and not elim_rule.passed:
            return AdjudicationStatus.PENDING_WAITING_PERIOD, elim_rule.confidence
        
        # Check for medical review needs
        preex_rule = next((r for r in results if r.rule_id == 'PEC001'), None)
        if preex_rule and not preex_rule.passed:
            return AdjudicationStatus.REFERRED_TO_MEDICAL, preex_rule.confidence
        
        # Check for other major failures
        major_failures = [r for r in results 
                        if not r.passed and r.severity == 'major']
        
        if major_failures:
            return AdjudicationStatus.PENDING_REVIEW, 0.85
        
        # All passed
        avg_confidence = sum(r.confidence for r in results if r.passed) / max(1, len(results))
        return AdjudicationStatus.APPROVED, min(0.97, avg_confidence)
    
    def _determine_required_actions(self, results: List[RuleResult], 
                                    status: AdjudicationStatus) -> List[str]:
        """Determine required follow-up actions"""
        actions = []
        
        if status == AdjudicationStatus.PENDING_DOCUMENTATION:
            actions.append("Request missing documentation from claimant")
        
        if status == AdjudicationStatus.REFERRED_TO_MEDICAL:
            actions.append("Schedule medical review with clinical team")
        
        if status == AdjudicationStatus.PENDING_WAITING_PERIOD:
            elim_rule = next((r for r in results if r.rule_id == 'EPR001'), None)
            if elim_rule:
                days = elim_rule.data.get('days_remaining', 'unknown')
                actions.append(f"Re-evaluate claim after {days} days")
        
        if status == AdjudicationStatus.APPROVED:
            actions.append("Issue benefit determination letter")
            actions.append("Set up payment schedule")
        
        if status == AdjudicationStatus.DENIED:
            actions.append("Generate denial letter with appeal rights")
        
        return actions
    
    def _generate_notes(self, results: List[RuleResult]) -> List[str]:
        """Generate processing notes"""
        notes = []
        
        for result in results:
            if not result.passed:
                notes.append(f"[{result.rule_id}] {result.message}")
            elif result.data:
                if 'final_monthly_benefit' in result.data:
                    notes.append(f"Calculated monthly benefit: ${result.data['final_monthly_benefit']:,.2f}")
        
        return notes
    
    def _log_rule_result(self, result: RuleResult):
        """Log rule evaluation result"""
        status = "✓" if result.passed else "✗"
        logger.info(f"  {status} [{result.rule_id}] {result.rule_name}: {result.message}")
    
    def to_dict(self, decision: AdjudicationDecision) -> Dict[str, Any]:
        """Convert decision to dictionary for API response"""
        return {
            'claim_id': decision.claim_id,
            'status': decision.status.value,
            'confidence': decision.confidence,
            'denial_reasons': decision.denial_reasons,
            'missing_documents': decision.missing_documents,
            'required_actions': decision.required_actions,
            'benefit_calculation': decision.benefit_calculation,
            'estimated_benefit_start': decision.estimated_benefit_start,
            'processing_notes': decision.processing_notes,
            'sla_met': decision.sla_met,
            'decision_timestamp': decision.decision_timestamp,
            'rules_evaluated': len(decision.rule_results),
            'rules_passed': sum(1 for r in decision.rule_results if r.passed)
        }


# Backward compatibility alias
ClaimAdjudicator = EnhancedClaimAdjudicator
