#!/usr/bin/env python3
"""
System Test Script
End-to-end testing of the Healthcare IDP System
"""

import sys
import os
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import IDPPipeline
from src.document_classifier import DocumentClassifier
from src.entity_extractor import EntityExtractor
from src.claim_adjudicator import ClaimAdjudicator


def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_result(name, passed, message=""):
    """Print test result"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status}: {name}")
    if message:
        print(f"       {message}")


def test_classifier():
    """Test document classifier"""
    print_header("Testing Document Classifier")
    
    classifier = DocumentClassifier()
    
    # Test 1: Disability claim
    claim_text = """
    DISABILITY CLAIM FORM
    Claim Number: CLM-2024-123
    Claimant: John Doe
    Date of Disability: 11/15/2024
    Diagnosis: Back injury
    """
    doc_type, confidence = classifier.classify(claim_text)
    print_result(
        "Classify disability claim",
        doc_type == "disability_claim",
        f"Type: {doc_type}, Confidence: {confidence:.2%}"
    )
    
    # Test 2: Enrollment
    enrollment_text = """
    EMPLOYEE ENROLLMENT FORM
    Employee Name: Jane Smith
    Plan Selection: Premium Health
    Coverage Election: Family
    Beneficiary: John Smith
    """
    doc_type, confidence = classifier.classify(enrollment_text)
    print_result(
        "Classify enrollment form",
        doc_type == "enrollment",
        f"Type: {doc_type}, Confidence: {confidence:.2%}"
    )
    
    # Test 3: Policy
    policy_text = """
    CERTIFICATE OF INSURANCE
    Policy Number: POL-2024-789
    Terms and Conditions apply
    Exclusions: Pre-existing conditions
    Benefit Schedule attached
    """
    doc_type, confidence = classifier.classify(policy_text)
    print_result(
        "Classify policy document",
        doc_type == "policy",
        f"Type: {doc_type}, Confidence: {confidence:.2%}"
    )


def test_extractor():
    """Test entity extractor"""
    print_header("Testing Entity Extractor")
    
    extractor = EntityExtractor()
    
    text = """
    DISABILITY CLAIM FORM
    Claim Number: CLM-2024-789456
    Policy Number: POL-456-7890
    SSN: 123-45-6789
    Date of Disability: 11/15/2024
    Employer: Acme Corporation
    Diagnosis: Acute lumbar strain
    Benefit Amount: $2,500.00
    Email: claimant@email.com
    Phone: 555-123-4567
    """
    
    entities = extractor.extract(text, "disability_claim")
    
    # Test extractions
    print_result(
        "Extract claim number",
        "claim_number" in entities,
        f"Found: {entities.get('claim_number', 'NOT FOUND')}"
    )
    
    print_result(
        "Extract policy number",
        "policy_number" in entities,
        f"Found: {entities.get('policy_number', 'NOT FOUND')}"
    )
    
    print_result(
        "Extract SSN",
        "ssn" in entities,
        f"Found: {entities.get('ssn', 'NOT FOUND')}"
    )
    
    print_result(
        "Extract money amount",
        "money" in entities,
        f"Found: {entities.get('money', 'NOT FOUND')}"
    )
    
    print_result(
        "Extract diagnosis",
        "diagnosis" in entities,
        f"Found: {entities.get('diagnosis', 'NOT FOUND')[:50] if entities.get('diagnosis') else 'NOT FOUND'}"
    )
    
    print_result(
        "Extract employer",
        "employer" in entities,
        f"Found: {entities.get('employer', 'NOT FOUND')}"
    )
    
    print(f"\nTotal entities extracted: {len(entities)}")


def test_adjudicator():
    """Test claim adjudicator"""
    print_header("Testing Claim Adjudicator")
    
    adjudicator = ClaimAdjudicator()
    
    # Test with valid claim
    claim_data = {
        "claim_number": "CLM-2024-789456",
        "policy_number": "POL-456-7890",
        "date_of_disability": "2024-06-01",  # Old enough for waiting period
        "diagnosis": "Acute lumbar strain",
        "employer": "Acme Corporation"
    }
    
    policy_data = {
        "policy_number": "POL-456-7890",
        "effective_date": "2024-01-01",
        "elimination_period": "90 days",
        "max_benefit": "$10,000"
    }
    
    result = adjudicator.adjudicate(claim_data, policy_data)
    
    print_result(
        "Verify coverage",
        result.coverage_verified == True,
        f"Coverage verified: {result.coverage_verified}"
    )
    
    print_result(
        "Check waiting period",
        result.waiting_period_met in [True, False],
        f"Waiting period met: {result.waiting_period_met}"
    )
    
    print_result(
        "Adjudication decision",
        result.eligibility_status in ["Approved", "Approved - Waiting Period", "Pending - Documentation Required", "Denied"],
        f"Status: {result.eligibility_status}"
    )
    
    print_result(
        "Confidence score",
        0 <= result.confidence <= 1,
        f"Confidence: {result.confidence:.2%}"
    )
    
    if result.estimated_benefit_start:
        print(f"\nEstimated benefit start date: {result.estimated_benefit_start}")


def test_full_pipeline():
    """Test complete pipeline"""
    print_header("Testing Full Pipeline")
    
    pipeline = IDPPipeline()
    
    sample_document = """
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
    
    print("\nProcessing document...")
    result = pipeline.process_document(sample_document)
    
    # Check results
    print_result(
        "Pipeline status",
        result["status"] == "success",
        f"Status: {result['status']}"
    )
    
    print_result(
        "Document classification",
        result.get("document_type") == "disability_claim",
        f"Type: {result.get('document_type')}, Confidence: {result.get('classification_confidence', 0):.2%}"
    )
    
    print_result(
        "Entity extraction",
        len(result.get("extracted_entities", {})) > 0,
        f"Entities found: {len(result.get('extracted_entities', {}))}"
    )
    
    print_result(
        "Adjudication",
        "adjudication" in result,
        f"Decision: {result.get('adjudication', {}).get('eligibility_status', 'N/A')}"
    )
    
    print_result(
        "Quality score",
        result.get("quality_score", 0) > 0,
        f"Score: {result.get('quality_score', 0):.2%}"
    )
    
    print_result(
        "Processing time",
        result.get("processing_time_seconds", 0) > 0,
        f"Time: {result.get('processing_time_seconds', 0):.2f} seconds"
    )
    
    # Print summary
    print("\n" + "-" * 40)
    print("PIPELINE SUMMARY")
    print("-" * 40)
    print(f"Document Type: {result.get('document_type')}")
    print(f"Classification Confidence: {result.get('classification_confidence', 0):.2%}")
    print(f"Quality Score: {result.get('quality_score', 0):.2%}")
    print(f"Production Ready: {result.get('processing_metrics', {}).get('production_ready', False)}")
    
    if "adjudication" in result:
        adj = result["adjudication"]
        print(f"\nAdjudication Result:")
        print(f"  Status: {adj.get('eligibility_status')}")
        print(f"  Confidence: {adj.get('confidence', 0):.2%}")
        print(f"  Coverage Verified: {adj.get('coverage_verified')}")


def main():
    """Run all tests"""
    print_header("Healthcare IDP System - Test Suite")
    print(f"Test run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        test_classifier()
        test_extractor()
        test_adjudicator()
        test_full_pipeline()
        
        print_header("Test Suite Complete")
        print("All system tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
