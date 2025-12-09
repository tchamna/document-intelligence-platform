#!/usr/bin/env python3
"""
Comprehensive Demo Script for Healthcare IDP System
===================================================

This script demonstrates all Document Intelligence capabilities:
1. Document Classification (RFP, Claims, Enrollment, Policy)
2. Entity Extraction with NER (spaCy + Regex + LLM ensemble)
3. Claim Adjudication (Rule-based business logic)
4. Eligibility Matching (Plan/coverage verification)
5. Policy Interpretation (NLP clause analysis)
6. LLM Integration (AWS Bedrock ready)
7. Metrics Dashboard (97-99% precision tracking)

Author: Data Science / Document Intelligence Engineer
Target: Production-grade system with high precision
"""

import os
import sys
import json
import time
import io
from datetime import datetime, timedelta
from typing import Dict, Any

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import IDPPipeline
from src.enhanced_pipeline import EnhancedIDPPipeline
from src.document_classifier import DocumentClassifier
from src.entity_extractor import EntityExtractor
from src.enhanced_extractor import EnhancedEntityExtractor
from src.claim_adjudicator import ClaimAdjudicator
from src.enhanced_adjudicator import EnhancedClaimAdjudicator
from src.eligibility_engine import EligibilityMatchingEngine, EligibilityResult
from src.policy_interpreter import PolicyInterpreter
from src.data_normalizer import DataNormalizer
from src.metrics_dashboard import MetricsDashboard


def print_header(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_subheader(title: str):
    """Print a formatted subsection header"""
    print(f"\n{'─' * 40}")
    print(f" {title}")
    print(f"{'─' * 40}")


def format_json(data: Any) -> str:
    """Format data as pretty JSON"""
    return json.dumps(data, indent=2, default=str)


# =============================================================================
# SAMPLE DOCUMENT DATA
# =============================================================================

SAMPLE_DISABILITY_CLAIM = """
SHORT-TERM DISABILITY CLAIM FORM
================================

CLAIMANT INFORMATION
--------------------
Name: Michael Thompson
Employee ID: EMP-2024-7892
Date of Birth: 03/15/1985
SSN: XXX-XX-4567
Department: Engineering
Hire Date: 06/01/2018
Annual Salary: $85,000

EMPLOYER INFORMATION
--------------------
Company: Tech Solutions Inc.
Policy Number: POL-STD-2024-001
Group Number: GRP-500123

DISABILITY INFORMATION
----------------------
Type of Disability: Short-Term Disability
First Date Unable to Work: 11/15/2024
Last Day Worked: 11/14/2024
Primary Diagnosis: Lumbar spine injury
ICD-10 Code: M54.5
Treating Physician: Dr. Sarah Johnson, MD
Physician Phone: (555) 234-5678
Physician NPI: 1234567890

Expected Return to Work: 12/15/2024
Is this Work-Related: No

CERTIFICATION
-------------
I certify that the above information is true and accurate.
Signature: Michael Thompson
Date: 11/18/2024
"""

SAMPLE_ENROLLMENT_FORM = """
GROUP BENEFITS ENROLLMENT FORM
==============================

EMPLOYEE INFORMATION
--------------------
Full Name: Jennifer Marie Williams
Employee ID: EMP-2024-3456
Date of Birth: 08/22/1990
SSN: XXX-XX-7890
Email: jennifer.williams@company.com
Phone: (555) 987-6543
Address: 456 Oak Avenue, Suite 201
City: Chicago, IL 60601

EMPLOYMENT DETAILS
------------------
Hire Date: 01/15/2024
Job Title: Senior Analyst
Department: Finance
Employment Status: Full-Time
Work Schedule: 40 hours/week

COVERAGE SELECTIONS
-------------------
☑ Medical Insurance - PPO Gold Plan
☑ Dental Insurance - Standard Plan
☑ Vision Insurance - Basic Plan
☑ Short-Term Disability (60% salary)
☑ Long-Term Disability (60% salary)
☑ Life Insurance - 2x Annual Salary

DEPENDENT INFORMATION
---------------------
Spouse: Robert Williams, DOB: 05/10/1988
Child 1: Emma Williams, DOB: 02/14/2018
Child 2: James Williams, DOB: 07/30/2020

Effective Date Requested: 02/01/2024

Signature: Jennifer M. Williams
Date: 01/16/2024
"""

SAMPLE_POLICY_DOCUMENT = """
GROUP SHORT-TERM DISABILITY POLICY
==================================
Policy Number: POL-STD-2024-001
Effective Date: January 1, 2024
Group Name: Tech Solutions Inc.

ARTICLE I - ELIGIBILITY
-----------------------
1.1 Eligible Employees: All active full-time employees working 30+ hours/week
1.2 Waiting Period: 90 days from date of hire
1.3 Minimum Hours: Must work minimum 30 hours per week
1.4 Excluded Classes: Temporary, seasonal, and contract workers

ARTICLE II - BENEFITS
---------------------
2.1 Benefit Percentage: 60% of weekly covered earnings
2.2 Maximum Weekly Benefit: $2,500
2.3 Minimum Weekly Benefit: $100
2.4 Elimination Period: 7 calendar days
2.5 Maximum Benefit Period: 26 weeks per disability

ARTICLE III - COVERED CONDITIONS
--------------------------------
3.1 This policy covers disabilities resulting from:
    - Illness or injury not related to employment
    - Pregnancy and childbirth complications
    - Mental/nervous conditions (subject to 90-day limit)
    
3.2 Pre-existing Conditions:
    - 12/12 lookback/exclusion period applies
    - Condition must be disclosed at enrollment

ARTICLE IV - CLAIM PROCEDURES
-----------------------------
4.1 Notice of Claim: Within 30 days of disability onset
4.2 Proof of Loss: Within 90 days of notice
4.3 Required Documentation:
    - Completed claim form
    - Attending physician statement
    - Employer verification
"""

SAMPLE_RFP = """
REQUEST FOR PROPOSAL: GROUP BENEFITS ADMINISTRATION
====================================================
RFP Number: RFP-2024-GB-001
Issue Date: October 1, 2024
Due Date: November 15, 2024

SECTION 1: ORGANIZATION OVERVIEW
--------------------------------
Company: Healthcare Partners Inc.
Industry: Healthcare Technology
Employees: 2,500 full-time
Locations: 15 offices nationwide
Current Carrier: XYZ Insurance Co.

SECTION 2: SCOPE OF SERVICES
----------------------------
2.1 Required Coverage Types:
    - Medical Insurance (PPO/HMO options)
    - Dental Insurance
    - Vision Insurance
    - Short-Term Disability (STD)
    - Long-Term Disability (LTD)
    - Life Insurance and AD&D
    
2.2 Administrative Services:
    - Online enrollment platform
    - Claims processing and adjudication
    - Customer service support
    - Compliance reporting (ERISA, ACA)

SECTION 3: REQUIREMENTS
-----------------------
3.1 Technology Requirements:
    - API integration capabilities
    - Real-time eligibility verification
    - Automated claims processing
    - Document intelligence/OCR support
    
3.2 Performance Standards:
    - Claims processing: 97% accuracy
    - Turnaround time: 5 business days
    - Customer satisfaction: 95%+ rating

SECTION 4: PRICING
------------------
Please provide pricing for:
- Per employee per month (PEPM) administrative fees
- Fully-insured premium rates by coverage type
- Implementation and setup fees
"""


def demo_document_classification():
    """Demonstrate document classification capabilities"""
    print_header("1. DOCUMENT CLASSIFICATION")
    
    classifier = DocumentClassifier()
    
    documents = [
        ("Disability Claim", SAMPLE_DISABILITY_CLAIM),
        ("Enrollment Form", SAMPLE_ENROLLMENT_FORM),
        ("Policy Document", SAMPLE_POLICY_DOCUMENT),
        ("RFP Document", SAMPLE_RFP),
    ]
    
    print("\nClassifying documents with NLP model...")
    print("-" * 60)
    
    for doc_name, doc_text in documents:
        doc_type, confidence = classifier.classify(doc_text)
        confidence_bar = "█" * int(confidence * 20) + "░" * (20 - int(confidence * 20))
        
        print(f"\n📄 Document: {doc_name}")
        print(f"   Predicted Type: {doc_type.upper()}")
        print(f"   Confidence: {confidence*100:.1f}% [{confidence_bar}]")
    
    print("\n✅ Classification supports: disability_claim, enrollment, policy, rfp")


def demo_entity_extraction():
    """Demonstrate entity extraction with multiple methods"""
    print_header("2. ENTITY EXTRACTION (NER)")
    
    # Standard extraction
    print_subheader("2.1 Standard Extraction (spaCy + Regex)")
    
    extractor = EntityExtractor()
    entities = extractor.extract(SAMPLE_DISABILITY_CLAIM, doc_type="disability_claim")
    
    print("\nExtracted Entities from Disability Claim:")
    print("-" * 50)
    for entity_type, value in entities.items():
        print(f"  • {entity_type:25s}: {value}")
    
    # Enhanced extraction
    print_subheader("2.2 Enhanced Extraction (Ensemble Method)")
    
    enhanced_extractor = EnhancedEntityExtractor()
    enhanced_result = enhanced_extractor.extract(
        SAMPLE_DISABILITY_CLAIM, 
        doc_type="disability_claim"
    )
    
    print("\nExtracted Entities with Confidence Scores:")
    print("-" * 60)
    for entity in enhanced_result.raw_entities[:10]:  # Show first 10 entities
        conf_bar = "█" * int(entity.confidence * 10) + "░" * (10 - int(entity.confidence * 10))
        val_str = str(entity.value)[:25]
        print(f"  • {entity.entity_type:20s}: {val_str:25s} [{conf_bar}] {entity.confidence*100:.0f}%")
        print(f"    Source: {entity.source}, Status: {entity.validation_status}")
    
    print(f"\n📊 Overall Extraction Confidence: {enhanced_result.extraction_confidence*100:.1f}%")
    print(f"📈 Coverage Score: {enhanced_result.coverage_score*100:.1f}%")
    print(f"⏱️ Processing Time: {enhanced_result.extraction_time_ms:.0f}ms")


def demo_data_normalization():
    """Demonstrate data normalization"""
    print_header("3. DATA NORMALIZATION")
    
    normalizer = DataNormalizer()
    
    # Sample extracted data with various formats
    raw_data = {
        "claimant_name": "MICHAEL  THOMPSON",
        "date_of_birth": "03/15/1985",
        "ssn": "123-45-6789",
        "phone": "5552345678",
        "salary": "$85,000",
        "first_day_absent": "Nov 15, 2024",
    }
    
    print("\nBefore Normalization:")
    print("-" * 50)
    for key, value in raw_data.items():
        print(f"  • {key:20s}: '{value}'")
    
    # Normalize using normalize_all
    normalized = normalizer.normalize_all(raw_data)
    
    print("\nAfter Normalization:")
    print("-" * 50)
    for key, value in normalized.items():
        if isinstance(value, dict):
            print(f"  • {key:20s}: '{value.get('normalized', value.get('value', 'N/A'))}' (conf: {value.get('confidence', 0)*100:.0f}%)")
        else:
            print(f"  • {key:20s}: '{value}'")


def demo_claim_adjudication():
    """Demonstrate claim adjudication with business rules"""
    print_header("4. CLAIM ADJUDICATION")
    
    print_subheader("4.1 Standard Adjudication")
    
    adjudicator = ClaimAdjudicator()
    
    # Claim data
    claim_data = {
        "employee_id": "EMP-2024-7892",
        "policy_number": "POL-STD-2024-001",
        "first_day_absent": "2024-11-15",
        "last_day_worked": "2024-11-14",
        "claim_submission_date": "2024-11-18",
        "diagnosis_code": "M54.5",
        "is_work_related": False,
        "annual_salary": 85000,
        "weekly_salary": 85000 / 52,
    }
    
    policy_info = {
        "benefit_percentage": 0.60,
        "elimination_period_days": 7,
        "max_weekly_benefit": 2500,
        "max_benefit_weeks": 26,
        "active": True,
    }
    
    result = adjudicator.adjudicate(claim_data, policy_info)
    
    print("\n📋 Claim Adjudication Result:")
    print("-" * 50)
    print(f"  Status: {result.eligibility_status.upper()}")
    print(f"  Confidence: {result.confidence*100:.1f}%")
    print(f"  Coverage Verified: {'Yes' if result.coverage_verified else 'No'}")
    print(f"  Waiting Period Met: {'Yes' if result.waiting_period_met else 'No'}")
    print(f"  Estimated Benefit Start: {result.estimated_benefit_start or 'N/A'}")
    if result.required_documents:
        print(f"  Required Docs: {', '.join(result.required_documents)}")
    
    # Enhanced adjudication
    print_subheader("4.2 Enhanced Adjudication (Full Rule Engine)")
    
    enhanced_adjudicator = EnhancedClaimAdjudicator()
    enhanced_result = enhanced_adjudicator.adjudicate(claim_data, policy_info)
    
    print("\n📋 Enhanced Adjudication Result:")
    print("-" * 50)
    print(f"  Status: {enhanced_result.status.value.upper()}")
    print(f"  Confidence: {enhanced_result.confidence*100:.1f}%")
    print(f"  Rules Evaluated: {len(enhanced_result.rule_results)}")
    rules_passed = sum(1 for r in enhanced_result.rule_results if r.passed)
    print(f"  Rules Passed: {rules_passed}")
    
    if enhanced_result.benefit_calculation:
        bc = enhanced_result.benefit_calculation
        print(f"\n💰 Benefit Calculation:")
        print(f"    Weekly Earnings: ${bc.get('weekly_earnings', 0):,.2f}")
        print(f"    Benefit Rate: {bc.get('benefit_percentage', 0)*100:.0f}%")
        print(f"    Calculated Weekly: ${bc.get('calculated_weekly_benefit', 0):,.2f}")
        print(f"    Final Weekly Benefit: ${bc.get('final_weekly_benefit', 0):,.2f}")
    
    print(f"\n📝 Recommendations:")
    for rec in enhanced_result.required_actions[:3]:
        print(f"    → {rec}")


def demo_eligibility_engine():
    """Demonstrate eligibility matching"""
    print_header("5. ELIGIBILITY MATCHING ENGINE")
    
    engine = EligibilityMatchingEngine()
    
    # Sample policy data
    policy_data = {
        "policy_id": "POL-STD-2024-001",
        "policy_name": "Short-Term Disability",
        "effective_date": "2024-01-01",
        "min_hours_per_week": 30,
        "waiting_period_days": 90,
        "eligible_classes": ["full-time", "part-time-30+"],
        "coverage_types": ["std", "disability"],
        "benefit_tiers": {
            "standard": {
                "benefit_percentage": 0.60,
                "max_weekly_benefit": 2500,
                "elimination_period": 7,
                "max_benefit_weeks": 26,
            }
        }
    }
    
    # Sample enrollment data
    enrollment_data = {
        "employee_id": "EMP-2024-7892",
        "employee_name": "Michael Thompson",
        "hire_date": "2018-06-01",
        "employment_status": "full-time",
        "weekly_hours": 40,
        "annual_salary": 85000,
        "selected_coverages": ["std", "life"],
        "dependents": []
    }
    
    # Check eligibility
    result = engine.check_eligibility(enrollment_data, policy_data)
    
    print("\n📋 Eligibility Check Result:")
    print("-" * 50)
    print(f"  Status: {result.status.value.upper()}")
    print(f"  Confidence: {result.confidence*100:.1f}%")
    print(f"  Matched Plan: {result.matched_plan or 'N/A'}")
    print(f"  Effective Date: {result.effective_date or 'N/A'}")
    
    if result.satisfied_rules:
        print(f"\n✓ Rules Satisfied ({len(result.satisfied_rules)}):")
        for rule in result.satisfied_rules[:5]:
            print(f"    ✓ {rule}")
    
    if result.failed_rules:
        print(f"\n✗ Rules Failed ({len(result.failed_rules)}):")
        for rule in result.failed_rules:
            print(f"    ✗ {rule}")


def demo_policy_interpretation():
    """Demonstrate policy interpretation"""
    print_header("6. POLICY INTERPRETATION (NLP)")
    
    interpreter = PolicyInterpreter()
    
    result = interpreter.interpret(SAMPLE_POLICY_DOCUMENT)
    
    print("\n📜 Policy Interpretation Result:")
    print("-" * 50)
    
    print(f"\n📋 Extracted Clauses ({len(result.get('clauses', {}))}):")
    for term, clause in result.get("clauses", {}).items():
        if clause:
            clause_preview = clause[:70] + "..." if len(clause) > 70 else clause
            print(f"    • {term}: {clause_preview}")
    
    print(f"\n📋 Key Terms ({len(result.get('terms', {}))}):")
    for term, value in result.get("terms", {}).items():
        if value:
            print(f"    • {term}: {value}")
    
    print(f"\n⚠️ Exclusions ({len(result.get('exclusions', []))}):")
    for exclusion in result.get("exclusions", [])[:3]:
        if exclusion:
            print(f"    → {exclusion[:70]}...")


def demo_metrics_dashboard():
    """Demonstrate metrics and monitoring"""
    print_header("7. METRICS DASHBOARD")
    
    dashboard = MetricsDashboard()
    
    # Simulate some processing results
    print("\nSimulating 100 document processing results...")
    
    import random
    for i in range(100):
        doc_type = random.choice(["disability_claim", "enrollment", "policy", "rfp"])
        
        # Simulate extraction metrics
        predicted = {"claimant_name": "John Doe", "date": "2024-01-15", "amount": "$1000"}
        actual = {"claimant_name": "John Doe", "date": "2024-01-15", "amount": "$1000"}
        
        if random.random() < 0.03:  # 3% error rate
            predicted["amount"] = "$1001"
        
        confidence = 0.90 + random.random() * 0.09  # 90-99% confidence
        processing_time = random.uniform(100, 500)  # 100-500ms
        
        dashboard.record_extraction(
            extracted=predicted,
            ground_truth=actual,
            confidence=confidence,
            processing_time_ms=processing_time
        )
        
        # Record classification
        dashboard.record_classification(doc_type, doc_type, confidence)
    
    # Get metrics
    metrics = dashboard.get_dashboard_summary()
    
    print("\n📊 System Performance Metrics:")
    print("-" * 50)
    
    perf = metrics.get("performance", {})
    print(f"  Average Processing Time: {perf.get('average_time_ms', 0):.0f}ms")
    print(f"  Throughput: {perf.get('throughput_per_minute', 0):.1f} docs/min")
    
    extraction = metrics.get("extraction", {})
    if extraction:
        print(f"  Average Confidence: {extraction.get('average_confidence', 0)*100:.1f}%")
    
    print(f"\n✅ Metrics tracking active - Ready for 97-99% precision monitoring")


def demo_full_pipeline():
    """Demonstrate the full IDP pipeline"""
    print_header("8. FULL IDP PIPELINE DEMO")
    
    # Initialize enhanced pipeline
    print("\nInitializing Enhanced IDP Pipeline...")
    pipeline = EnhancedIDPPipeline()
    
    print_subheader("Processing Disability Claim")
    
    start_time = time.time()
    result = pipeline.process_document(SAMPLE_DISABILITY_CLAIM)
    processing_time = time.time() - start_time
    
    print(f"\n📄 Pipeline Results:")
    print("-" * 60)
    print(f"  Document Type: {result.document_type.upper()}")
    print(f"  Classification Confidence: {result.classification_confidence*100:.1f}%")
    print(f"  Quality Score: {result.quality_score*100:.1f}%")
    print(f"  Processing Time: {processing_time*1000:.0f}ms")
    
    print(f"\n📝 Extracted Entities:")
    entities = result.normalized_entities or result.extracted_entities or {}
    for key, value in list(entities.items())[:8]:
        print(f"    • {key}: {value}")
    
    if result.adjudication_result:
        adj = result.adjudication_result
        status = adj.get('status', 'N/A') if isinstance(adj, dict) else getattr(adj, 'status', 'N/A')
        conf = adj.get('confidence', 0) if isinstance(adj, dict) else getattr(adj, 'confidence', 0)
        print(f"\n⚖️ Adjudication:")
        print(f"    Status: {str(status).upper()}")
        print(f"    Confidence: {conf*100:.1f}%")
    
    print(f"\n💡 Recommendations:")
    for rec in (result.recommendations or [])[:3]:
        print(f"    → {rec}")


def demo_technology_stack():
    """Display technology stack information"""
    print_header("9. TECHNOLOGY STACK")
    
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │                   Healthcare IDP System                         │
    │            Document Intelligence Platform                       │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  📦 CORE COMPONENTS                                             │
    │  ├── Document Classification (NLP-based)                        │
    │  ├── Entity Extraction (spaCy + Regex + LLM ensemble)           │
    │  ├── Claim Adjudication (Rule-based engine)                     │
    │  ├── Eligibility Matching (Plan verification)                   │
    │  ├── Policy Interpretation (Clause analysis)                    │
    │  └── Data Normalization (Field standardization)                 │
    │                                                                 │
    │  🔧 TECHNOLOGY STACK                                            │
    │  ├── Python 3.11+                                               │
    │  ├── FastAPI (REST API)                                         │
    │  ├── spaCy 3.7 (NLP/NER)                                        │
    │  ├── Tesseract OCR                                              │
    │  ├── PyMuPDF (PDF processing)                                   │
    │  ├── LangChain (Document AI pipeline)                           │
    │  └── AWS Bedrock (LLM integration)                              │
    │                                                                 │
    │  📊 PERFORMANCE TARGETS                                         │
    │  ├── Precision: 97-99%                                          │
    │  ├── Recall: 95%+                                               │
    │  └── Processing Time: <2s per document                          │
    │                                                                 │
    │  ☁️ DEPLOYMENT OPTIONS                                          │
    │  ├── Docker / Docker Compose                                    │
    │  ├── AWS Lambda                                                 │
    │  ├── Kubernetes                                                 │
    │  └── Azure / GCP                                                │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    """)


def main():
    """Run the complete demonstration"""
    print("\n" + "═" * 80)
    print("    HEALTHCARE IDP SYSTEM - DOCUMENT INTELLIGENCE DEMONSTRATION")
    print("    Intelligent Document Processing for Group Benefits")
    print("═" * 80)
    
    print(f"\n📅 Demo Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Target Accuracy: 97-99% precision for production use")
    print(f"📄 Document Types: Disability Claims, Enrollment, Policy, RFP")
    
    # Run demonstrations
    demo_document_classification()
    demo_entity_extraction()
    demo_data_normalization()
    demo_claim_adjudication()
    demo_eligibility_engine()
    demo_policy_interpretation()
    demo_metrics_dashboard()
    demo_full_pipeline()
    demo_technology_stack()
    
    # Summary
    print_header("DEMONSTRATION COMPLETE")
    print("""
    ✅ All Document Intelligence capabilities demonstrated:
    
    1. Document Classification - Multi-class NLP classification
    2. Entity Extraction - Ensemble NER with confidence scoring
    3. Data Normalization - Field standardization and validation
    4. Claim Adjudication - Rule-based business logic engine
    5. Eligibility Matching - Plan and coverage verification
    6. Policy Interpretation - NLP-based clause analysis
    7. Metrics Dashboard - Production monitoring (97-99% precision)
    8. Full Pipeline - End-to-end document processing
    
    📚 API Documentation: http://localhost:8000/docs
    🌐 Web Interface: http://localhost:8000/ui
    📊 Metrics Endpoint: http://localhost:8000/metrics/dashboard
    
    Ready for production deployment with AWS Bedrock LLM integration!
    """)


if __name__ == "__main__":
    main()
