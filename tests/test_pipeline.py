"""
Pipeline Integration Tests
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline import IDPPipeline
from src.claim_adjudicator import ClaimAdjudicator, AdjudicationResult


class TestIDPPipeline:
    """Integration tests for IDPPipeline"""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance"""
        return IDPPipeline()
    
    @pytest.fixture
    def sample_claim(self):
        """Sample claim document"""
        return """
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
    
    @pytest.fixture
    def sample_enrollment(self):
        """Sample enrollment document"""
        return """
        GROUP BENEFITS ENROLLMENT FORM
        
        Employee Information:
        Name: Michael Chen
        Employee ID: EMP-12345
        
        Plan Selection: Premium Health Plus
        Coverage Level: Employee + Family
        
        Dependents:
        - Lisa Chen (Spouse)
        - Emma Chen (Child, Age 8)
        
        Effective Date: 01/01/2025
        Premium Amount: $450/month
        """
    
    def test_process_claim_document(self, pipeline, sample_claim):
        """Test full pipeline processing of claim document"""
        result = pipeline.process_document(sample_claim)
        
        assert result['status'] == 'success'
        assert result['document_type'] == 'disability_claim'
        assert result['classification_confidence'] > 0.5
        assert 'extracted_entities' in result
        assert 'adjudication' in result
        assert 'quality_score' in result
    
    def test_process_enrollment_document(self, pipeline, sample_enrollment):
        """Test full pipeline processing of enrollment document"""
        result = pipeline.process_document(sample_enrollment)
        
        assert result['status'] == 'success'
        assert result['document_type'] == 'enrollment'
        assert 'extracted_entities' in result
        assert 'adjudication' not in result  # No adjudication for enrollment
    
    def test_pipeline_stages(self, pipeline, sample_claim):
        """Test that all pipeline stages complete"""
        result = pipeline.process_document(sample_claim)
        
        stages = result['pipeline_stages']
        
        assert 'classification' in stages
        assert stages['classification']['status'] == 'completed'
        
        assert 'extraction' in stages
        assert stages['extraction']['status'] == 'completed'
        
        assert 'validation' in stages
        assert stages['validation']['status'] == 'completed'
    
    def test_processing_metrics(self, pipeline, sample_claim):
        """Test processing metrics are generated"""
        result = pipeline.process_document(sample_claim)
        
        assert 'processing_metrics' in result
        metrics = result['processing_metrics']
        
        assert 'total_stages' in metrics
        assert 'overall_confidence' in metrics
        assert 'production_ready' in metrics
    
    def test_processing_time(self, pipeline, sample_claim):
        """Test that processing time is tracked"""
        result = pipeline.process_document(sample_claim)
        
        assert 'processing_time_seconds' in result
        assert result['processing_time_seconds'] > 0
    
    def test_quality_score(self, pipeline, sample_claim):
        """Test quality score calculation"""
        result = pipeline.process_document(sample_claim)
        
        assert 'quality_score' in result
        assert 0 <= result['quality_score'] <= 1
    
    def test_error_handling(self, pipeline):
        """Test error handling for invalid input"""
        # Empty text should still return a result
        result = pipeline.process_document("")
        
        assert 'status' in result
    
    def test_metadata_handling(self, pipeline, sample_claim):
        """Test that metadata is handled correctly"""
        metadata = {
            'source': 'email',
            'filename': 'claim_form.pdf',
            'received_date': '2024-12-01'
        }
        
        result = pipeline.process_document(sample_claim, metadata=metadata)
        
        assert result['status'] == 'success'


class TestClaimAdjudicator:
    """Test cases for ClaimAdjudicator"""
    
    @pytest.fixture
    def adjudicator(self):
        """Create adjudicator instance"""
        return ClaimAdjudicator()
    
    @pytest.fixture
    def valid_claim_data(self):
        """Valid claim data"""
        return {
            'claim_number': 'CLM-2024-789456',
            'policy_number': 'POL-456-7890',
            'date_of_disability': '2024-06-15',  # Old enough for waiting period
            'diagnosis': 'Acute lumbar strain',
            'employer': 'Acme Corporation'
        }
    
    @pytest.fixture
    def policy_data(self):
        """Policy data"""
        return {
            'policy_number': 'POL-456-7890',
            'effective_date': '2024-01-01',
            'elimination_period': '90 days',
            'max_benefit': '$10,000',
            'preexisting_condition_clause': 'Excluded for 12 months if treated in prior 90 days'
        }
    
    def test_adjudicate_valid_claim(self, adjudicator, valid_claim_data, policy_data):
        """Test adjudication of valid claim"""
        result = adjudicator.adjudicate(valid_claim_data, policy_data)
        
        assert isinstance(result, AdjudicationResult)
        assert result.coverage_verified == True
        assert result.eligibility_status in ['Approved', 'Approved - Waiting Period', 'Pending - Documentation Required']
    
    def test_verify_coverage(self, adjudicator, valid_claim_data, policy_data):
        """Test coverage verification"""
        result = adjudicator._verify_coverage(valid_claim_data, policy_data)
        
        assert result['verified'] == True
    
    def test_coverage_mismatch(self, adjudicator, policy_data):
        """Test coverage verification with mismatched policy"""
        claim_data = {
            'policy_number': 'WRONG-POLICY',
            'date_of_disability': '2024-11-15'
        }
        
        result = adjudicator._verify_coverage(claim_data, policy_data)
        
        assert result['verified'] == False
        assert 'mismatch' in result['reason'].lower()
    
    def test_disability_before_effective_date(self, adjudicator, policy_data):
        """Test denial when disability occurred before coverage"""
        claim_data = {
            'policy_number': 'POL-456-7890',
            'date_of_disability': '2023-06-01'  # Before effective date
        }
        
        result = adjudicator._verify_coverage(claim_data, policy_data)
        
        assert result['verified'] == False
    
    def test_waiting_period_check(self, adjudicator, policy_data):
        """Test waiting period calculation"""
        claim_data = {
            'date_of_disability': '2024-06-01'  # Old enough for waiting period
        }
        
        result = adjudicator._check_waiting_period(claim_data, policy_data)
        
        assert 'met' in result
    
    def test_documentation_check(self, adjudicator):
        """Test documentation verification"""
        # Complete claim data
        claim_data = {
            'diagnosis': 'Back pain',
            'employer': 'Acme Corp'
        }
        
        result = adjudicator._verify_documentation(claim_data)
        
        assert 'missing_documents' in result
        assert len(result['missing_documents']) == 0
    
    def test_missing_documentation(self, adjudicator):
        """Test detection of missing documentation"""
        # Incomplete claim data
        claim_data = {}
        
        result = adjudicator._verify_documentation(claim_data)
        
        assert len(result['missing_documents']) > 0
    
    def test_benefit_start_calculation(self, adjudicator, policy_data):
        """Test benefit start date calculation"""
        claim_data = {
            'date_of_disability': '2024-06-01'
        }
        
        result = adjudicator._calculate_benefit_start(claim_data, policy_data)
        
        assert result is not None
        assert result == '2024-08-30'  # 90 days after 6/1
