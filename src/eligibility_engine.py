"""
Eligibility Matching Engine
Match enrollment data to policy structures and verify eligibility
Handles complex benefit structures and customized plans
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


class EligibilityStatus(Enum):
    """Eligibility determination status"""
    ELIGIBLE = "eligible"
    NOT_ELIGIBLE = "not_eligible"
    CONDITIONALLY_ELIGIBLE = "conditionally_eligible"
    PENDING_VERIFICATION = "pending_verification"
    WAIVER_REQUIRED = "waiver_required"


@dataclass
class CoverageLevel:
    """Coverage level details"""
    code: str
    name: str
    covers_employee: bool = True
    covers_spouse: bool = False
    covers_children: bool = False
    covers_domestic_partner: bool = False
    tier: int = 1


@dataclass
class EligibilityRule:
    """Rule for eligibility determination"""
    rule_id: str
    description: str
    field: str
    operator: str  # 'eq', 'ne', 'gt', 'lt', 'gte', 'lte', 'in', 'contains', 'regex'
    value: Any
    is_required: bool = True
    waivable: bool = False


@dataclass
class EligibilityResult:
    """Result of eligibility check"""
    status: EligibilityStatus
    confidence: float
    matched_plan: Optional[str]
    coverage_level: Optional[CoverageLevel]
    effective_date: Optional[str]
    issues: List[str]
    satisfied_rules: List[str]
    failed_rules: List[str]
    dependents_eligible: List[Dict[str, Any]]
    premium_amount: Optional[float]
    recommendations: List[str]


class PlanMatcher:
    """Match employee data to available plans"""
    
    def __init__(self):
        """Initialize plan matcher with common plan structures"""
        self.plan_catalog = self._load_plan_catalog()
        logger.info("PlanMatcher initialized")
    
    def _load_plan_catalog(self) -> Dict[str, Dict]:
        """Load plan definitions"""
        return {
            'gold_health': {
                'name': 'Gold Health Plan',
                'type': 'medical',
                'coverage_levels': ['employee', 'employee_spouse', 'employee_family'],
                'waiting_period_days': 0,
                'eligibility_rules': [
                    {'field': 'employment_status', 'operator': 'in', 'value': ['active', 'full-time']},
                    {'field': 'hours_per_week', 'operator': 'gte', 'value': 30},
                ],
                'premium_tiers': {
                    'employee': 150.00,
                    'employee_spouse': 350.00,
                    'employee_family': 500.00
                }
            },
            'silver_health': {
                'name': 'Silver Health Plan',
                'type': 'medical',
                'coverage_levels': ['employee', 'employee_spouse', 'employee_family'],
                'waiting_period_days': 30,
                'eligibility_rules': [
                    {'field': 'employment_status', 'operator': 'in', 'value': ['active', 'full-time', 'part-time']},
                    {'field': 'hours_per_week', 'operator': 'gte', 'value': 20},
                ],
                'premium_tiers': {
                    'employee': 100.00,
                    'employee_spouse': 250.00,
                    'employee_family': 375.00
                }
            },
            'std_disability': {
                'name': 'Short-Term Disability',
                'type': 'disability',
                'coverage_levels': ['employee'],
                'waiting_period_days': 90,
                'eligibility_rules': [
                    {'field': 'employment_status', 'operator': 'eq', 'value': 'active'},
                    {'field': 'tenure_months', 'operator': 'gte', 'value': 3},
                ],
                'benefit_percentage': 60,
                'max_benefit': 2500
            },
            'ltd_disability': {
                'name': 'Long-Term Disability',
                'type': 'disability',
                'coverage_levels': ['employee'],
                'waiting_period_days': 180,
                'eligibility_rules': [
                    {'field': 'employment_status', 'operator': 'eq', 'value': 'active'},
                    {'field': 'tenure_months', 'operator': 'gte', 'value': 12},
                    {'field': 'job_classification', 'operator': 'in', 'value': ['exempt', 'management']},
                ],
                'benefit_percentage': 60,
                'max_benefit': 10000
            },
            'basic_life': {
                'name': 'Basic Life Insurance',
                'type': 'life',
                'coverage_levels': ['employee'],
                'waiting_period_days': 0,
                'eligibility_rules': [
                    {'field': 'employment_status', 'operator': 'in', 'value': ['active']},
                ],
                'coverage_amount': '1x salary',
                'max_coverage': 500000
            },
            'dental_plan': {
                'name': 'Dental Plan',
                'type': 'dental',
                'coverage_levels': ['employee', 'employee_spouse', 'employee_family'],
                'waiting_period_days': 0,
                'eligibility_rules': [
                    {'field': 'employment_status', 'operator': 'in', 'value': ['active', 'full-time']},
                ],
                'premium_tiers': {
                    'employee': 25.00,
                    'employee_spouse': 50.00,
                    'employee_family': 75.00
                }
            },
            'vision_plan': {
                'name': 'Vision Plan',
                'type': 'vision',
                'coverage_levels': ['employee', 'employee_spouse', 'employee_family'],
                'waiting_period_days': 0,
                'eligibility_rules': [],
                'premium_tiers': {
                    'employee': 10.00,
                    'employee_spouse': 20.00,
                    'employee_family': 30.00
                }
            }
        }
    
    def find_matching_plans(self, employee_data: Dict, requested_plan: Optional[str] = None) -> List[Dict]:
        """
        Find plans that employee is eligible for.
        
        Args:
            employee_data: Employee information
            requested_plan: Specific plan requested (optional)
            
        Returns:
            List of matching plans with eligibility details
        """
        matches = []
        
        for plan_id, plan in self.plan_catalog.items():
            if requested_plan and plan_id != requested_plan.lower().replace(' ', '_'):
                continue
            
            eligibility = self._check_plan_eligibility(employee_data, plan)
            
            if eligibility['eligible'] or eligibility['conditionally_eligible']:
                matches.append({
                    'plan_id': plan_id,
                    'plan_name': plan['name'],
                    'plan_type': plan['type'],
                    'eligibility': eligibility,
                    'coverage_levels': plan['coverage_levels'],
                    'waiting_period': plan['waiting_period_days']
                })
        
        return matches
    
    def _check_plan_eligibility(self, employee_data: Dict, plan: Dict) -> Dict:
        """Check if employee meets plan eligibility rules"""
        result = {
            'eligible': True,
            'conditionally_eligible': False,
            'failed_rules': [],
            'satisfied_rules': [],
            'waiver_needed': False
        }
        
        for rule in plan.get('eligibility_rules', []):
            field = rule['field']
            operator = rule['operator']
            expected = rule['value']
            actual = employee_data.get(field)
            
            passed = self._evaluate_rule(actual, operator, expected)
            
            if passed:
                result['satisfied_rules'].append(f"{field} {operator} {expected}")
            else:
                result['failed_rules'].append(f"{field}: expected {operator} {expected}, got {actual}")
                result['eligible'] = False
                
                if rule.get('waivable', False):
                    result['conditionally_eligible'] = True
                    result['waiver_needed'] = True
        
        return result
    
    def _evaluate_rule(self, actual: Any, operator: str, expected: Any) -> bool:
        """Evaluate a single eligibility rule"""
        if actual is None:
            return False
        
        try:
            if operator == 'eq':
                return str(actual).lower() == str(expected).lower()
            elif operator == 'ne':
                return str(actual).lower() != str(expected).lower()
            elif operator == 'gt':
                return float(actual) > float(expected)
            elif operator == 'lt':
                return float(actual) < float(expected)
            elif operator == 'gte':
                return float(actual) >= float(expected)
            elif operator == 'lte':
                return float(actual) <= float(expected)
            elif operator == 'in':
                return str(actual).lower() in [str(v).lower() for v in expected]
            elif operator == 'contains':
                return str(expected).lower() in str(actual).lower()
            elif operator == 'regex':
                return bool(re.match(expected, str(actual), re.IGNORECASE))
            else:
                logger.warning(f"Unknown operator: {operator}")
                return False
        except (ValueError, TypeError) as e:
            logger.warning(f"Rule evaluation error: {e}")
            return False


class DependentValidator:
    """Validate dependent eligibility"""
    
    MAX_CHILD_AGE = 26
    
    def __init__(self):
        """Initialize dependent validator"""
        logger.info("DependentValidator initialized")
    
    def validate_dependents(self, dependents: List[Dict], 
                           coverage_level: str) -> List[Dict]:
        """
        Validate dependent eligibility.
        
        Args:
            dependents: List of dependent information
            coverage_level: Selected coverage level
            
        Returns:
            List of validated dependents with eligibility status
        """
        validated = []
        
        for dep in dependents:
            validation = {
                'name': dep.get('name'),
                'relationship': dep.get('relationship'),
                'eligible': True,
                'issues': []
            }
            
            # Check relationship type
            relationship = dep.get('relationship', '').lower()
            
            if relationship in ['spouse', 'husband', 'wife']:
                if 'spouse' not in coverage_level and 'family' not in coverage_level:
                    validation['eligible'] = False
                    validation['issues'].append("Coverage level does not include spouse")
            
            elif relationship in ['child', 'son', 'daughter', 'dependent child']:
                if 'family' not in coverage_level and 'children' not in coverage_level:
                    validation['eligible'] = False
                    validation['issues'].append("Coverage level does not include children")
                
                # Check age
                dob = dep.get('date_of_birth')
                if dob:
                    age = self._calculate_age(dob)
                    if age and age > self.MAX_CHILD_AGE:
                        validation['eligible'] = False
                        validation['issues'].append(f"Child exceeds maximum age ({self.MAX_CHILD_AGE})")
                    validation['age'] = age
            
            elif relationship in ['domestic partner']:
                # May require special verification
                validation['requires_verification'] = True
                validation['issues'].append("Domestic partner status requires verification")
            
            validated.append(validation)
        
        return validated
    
    def _calculate_age(self, dob: str) -> Optional[int]:
        """Calculate age from date of birth"""
        try:
            if isinstance(dob, str):
                for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%m-%d-%Y']:
                    try:
                        birth_date = datetime.strptime(dob, fmt).date()
                        break
                    except ValueError:
                        continue
                else:
                    return None
            else:
                birth_date = dob
            
            today = date.today()
            age = today.year - birth_date.year
            if (today.month, today.day) < (birth_date.month, birth_date.day):
                age -= 1
            
            return age
        except Exception as e:
            logger.warning(f"Could not calculate age: {e}")
            return None


class EligibilityMatchingEngine:
    """
    Main eligibility matching engine.
    Combines plan matching, dependent validation, and eligibility determination.
    """
    
    def __init__(self):
        """Initialize the eligibility matching engine"""
        self.plan_matcher = PlanMatcher()
        self.dependent_validator = DependentValidator()
        logger.info("EligibilityMatchingEngine initialized")
    
    def check_eligibility(self, enrollment_data: Dict, 
                         employee_data: Optional[Dict] = None) -> EligibilityResult:
        """
        Check eligibility for enrollment request.
        
        Args:
            enrollment_data: Enrollment form data
            employee_data: Employee HR data (optional, can be derived)
            
        Returns:
            Complete eligibility determination
        """
        logger.info("Starting eligibility check")
        
        # Merge enrollment data with employee data
        if employee_data:
            combined_data = {**employee_data, **enrollment_data}
        else:
            combined_data = self._infer_employee_data(enrollment_data)
        
        # Get requested plan
        requested_plan = enrollment_data.get('plan_type') or \
                        enrollment_data.get('plan_selection') or \
                        enrollment_data.get('plan_name')
        
        # Find matching plans
        matching_plans = self.plan_matcher.find_matching_plans(
            combined_data, requested_plan
        )
        
        if not matching_plans:
            return EligibilityResult(
                status=EligibilityStatus.NOT_ELIGIBLE,
                confidence=0.92,
                matched_plan=None,
                coverage_level=None,
                effective_date=None,
                issues=["No eligible plans found for this employee profile"],
                satisfied_rules=[],
                failed_rules=["Plan eligibility requirements not met"],
                dependents_eligible=[],
                premium_amount=None,
                recommendations=["Review eligibility requirements", 
                               "Contact HR for alternative options"]
            )
        
        # Use first matching plan (best match)
        best_match = matching_plans[0]
        
        # Determine coverage level
        coverage_level = self._determine_coverage_level(enrollment_data)
        
        # Validate dependents if applicable
        dependents = enrollment_data.get('dependents', [])
        if not dependents and enrollment_data.get('dependent_name'):
            dependents = [{
                'name': enrollment_data.get('dependent_name'),
                'relationship': enrollment_data.get('dependent_relationship', 'dependent')
            }]
        
        validated_dependents = self.dependent_validator.validate_dependents(
            dependents, coverage_level.code if coverage_level else 'employee'
        )
        
        # Calculate effective date
        effective_date = self._calculate_effective_date(
            enrollment_data.get('requested_effective_date') or 
            enrollment_data.get('effective_date'),
            best_match['waiting_period']
        )
        
        # Calculate premium
        premium = self._calculate_premium(best_match, coverage_level)
        
        # Determine final status
        if best_match['eligibility']['eligible']:
            status = EligibilityStatus.ELIGIBLE
            confidence = 0.97
        elif best_match['eligibility']['conditionally_eligible']:
            status = EligibilityStatus.CONDITIONALLY_ELIGIBLE
            confidence = 0.85
        else:
            status = EligibilityStatus.PENDING_VERIFICATION
            confidence = 0.75
        
        # Check for dependent issues
        dep_issues = [d for d in validated_dependents if not d.get('eligible', True)]
        if dep_issues:
            status = EligibilityStatus.CONDITIONALLY_ELIGIBLE
        
        issues = best_match['eligibility']['failed_rules']
        for dep in validated_dependents:
            issues.extend(dep.get('issues', []))
        
        return EligibilityResult(
            status=status,
            confidence=confidence,
            matched_plan=best_match['plan_name'],
            coverage_level=coverage_level,
            effective_date=effective_date,
            issues=issues,
            satisfied_rules=best_match['eligibility']['satisfied_rules'],
            failed_rules=best_match['eligibility']['failed_rules'],
            dependents_eligible=validated_dependents,
            premium_amount=premium,
            recommendations=self._generate_recommendations(
                status, issues, best_match
            )
        )
    
    def _infer_employee_data(self, enrollment_data: Dict) -> Dict:
        """Infer employee data from enrollment form"""
        inferred = dict(enrollment_data)
        
        # Infer employment status
        if 'employment_status' not in inferred:
            inferred['employment_status'] = 'active'
        
        # Infer hours from coverage type
        if 'hours_per_week' not in inferred:
            if 'full-time' in str(enrollment_data.get('employment_type', '')).lower():
                inferred['hours_per_week'] = 40
            elif 'part-time' in str(enrollment_data.get('employment_type', '')).lower():
                inferred['hours_per_week'] = 25
            else:
                inferred['hours_per_week'] = 40  # Assume full-time
        
        return inferred
    
    def _determine_coverage_level(self, enrollment_data: Dict) -> Optional[CoverageLevel]:
        """Determine coverage level from enrollment data"""
        coverage = enrollment_data.get('coverage_level', '').lower()
        
        # Normalize variations
        if 'family' in coverage:
            return CoverageLevel(
                code='employee_family',
                name='Employee + Family',
                covers_employee=True,
                covers_spouse=True,
                covers_children=True,
                tier=3
            )
        elif 'spouse' in coverage or 'partner' in coverage:
            return CoverageLevel(
                code='employee_spouse',
                name='Employee + Spouse',
                covers_employee=True,
                covers_spouse=True,
                tier=2
            )
        elif 'children' in coverage:
            return CoverageLevel(
                code='employee_children',
                name='Employee + Children',
                covers_employee=True,
                covers_children=True,
                tier=2
            )
        else:
            return CoverageLevel(
                code='employee',
                name='Employee Only',
                tier=1
            )
    
    def _calculate_effective_date(self, requested: Optional[str], 
                                  waiting_days: int) -> str:
        """Calculate effective date considering waiting period"""
        today = date.today()
        
        # First of next month is common default
        if today.day > 15:
            # Use first of month after next
            if today.month == 12:
                default_date = date(today.year + 1, 2, 1)
            elif today.month == 11:
                default_date = date(today.year + 1, 1, 1)
            else:
                default_date = date(today.year, today.month + 2, 1)
        else:
            # Use first of next month
            if today.month == 12:
                default_date = date(today.year + 1, 1, 1)
            else:
                default_date = date(today.year, today.month + 1, 1)
        
        # Parse requested date if provided
        if requested:
            try:
                for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%m-%d-%Y']:
                    try:
                        requested_date = datetime.strptime(requested, fmt).date()
                        break
                    except ValueError:
                        continue
                else:
                    requested_date = default_date
            except:
                requested_date = default_date
        else:
            requested_date = default_date
        
        # Apply waiting period
        from datetime import timedelta
        min_date = today + timedelta(days=waiting_days)
        
        effective = max(requested_date, min_date)
        return effective.strftime('%Y-%m-%d')
    
    def _calculate_premium(self, plan_match: Dict, 
                          coverage_level: Optional[CoverageLevel]) -> Optional[float]:
        """Calculate premium for coverage"""
        plan = self.plan_matcher.plan_catalog.get(plan_match['plan_id'], {})
        premiums = plan.get('premium_tiers', {})
        
        if not premiums or not coverage_level:
            return None
        
        return premiums.get(coverage_level.code, premiums.get('employee'))
    
    def _generate_recommendations(self, status: EligibilityStatus, 
                                 issues: List[str], 
                                 plan_match: Dict) -> List[str]:
        """Generate recommendations based on eligibility result"""
        recommendations = []
        
        if status == EligibilityStatus.ELIGIBLE:
            recommendations.append(f"Proceed with enrollment in {plan_match['plan_name']}")
            recommendations.append("Review plan documents for coverage details")
        
        elif status == EligibilityStatus.CONDITIONALLY_ELIGIBLE:
            recommendations.append("Submit required documentation for eligibility verification")
            recommendations.append("Contact HR/Benefits for waiver process if applicable")
        
        elif status == EligibilityStatus.NOT_ELIGIBLE:
            recommendations.append("Review other available benefit options")
            recommendations.append("Check eligibility requirements for alternative plans")
        
        if issues:
            recommendations.append("Address eligibility issues listed above")
        
        return recommendations
    
    def to_dict(self, result: EligibilityResult) -> Dict[str, Any]:
        """Convert result to dictionary for API response"""
        return {
            'status': result.status.value,
            'confidence': result.confidence,
            'matched_plan': result.matched_plan,
            'coverage_level': {
                'code': result.coverage_level.code,
                'name': result.coverage_level.name,
                'tier': result.coverage_level.tier
            } if result.coverage_level else None,
            'effective_date': result.effective_date,
            'premium_amount': result.premium_amount,
            'issues': result.issues,
            'satisfied_rules': result.satisfied_rules,
            'failed_rules': result.failed_rules,
            'dependents': result.dependents_eligible,
            'recommendations': result.recommendations
        }
