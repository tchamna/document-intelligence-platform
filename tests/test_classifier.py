"""
Document Classifier Tests
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.document_classifier import DocumentClassifier


class TestDocumentClassifier:
    """Test cases for DocumentClassifier"""
    
    @pytest.fixture
    def classifier(self):
        """Create classifier instance"""
        return DocumentClassifier()
    
    def test_classify_disability_claim(self, classifier):
        """Test classification of disability claim document"""
        text = """
        DISABILITY CLAIM FORM
        
        Claim Number: CLM-2024-789456
        Claimant: John Doe
        Date of Disability: 11/15/2024
        Diagnosis: Acute back pain
        Attending Physician: Dr. Smith
        """
        
        doc_type, confidence = classifier.classify(text)
        
        assert doc_type == 'disability_claim'
        assert confidence > 0.5
    
    def test_classify_enrollment(self, classifier):
        """Test classification of enrollment document"""
        text = """
        EMPLOYEE ENROLLMENT FORM
        
        Employee Name: Jane Smith
        Plan Selection: Premium Health Plus
        Coverage Election: Employee + Family
        Dependent: John Smith (spouse)
        Beneficiary: Jane Smith Jr.
        """
        
        doc_type, confidence = classifier.classify(text)
        
        assert doc_type == 'enrollment'
        assert confidence > 0.5
    
    def test_classify_policy(self, classifier):
        """Test classification of policy document"""
        text = """
        GROUP DISABILITY POLICY
        CERTIFICATE OF INSURANCE
        
        Policy Number: POL-2024-12345
        Terms and Conditions
        The following exclusions apply:
        - Pre-existing conditions
        
        Benefit Schedule:
        - Maximum benefit: $10,000
        """
        
        doc_type, confidence = classifier.classify(text)
        
        assert doc_type == 'policy'
        assert confidence > 0.5
    
    def test_classify_rfp(self, classifier):
        """Test classification of RFP document"""
        text = """
        REQUEST FOR PROPOSAL
        
        Scope of Work: Insurance Administration Services
        Requirements:
        - Claims processing
        - Customer service
        
        Pricing: Please provide detailed pricing
        Bid deadline: December 31, 2024
        """
        
        doc_type, confidence = classifier.classify(text)
        
        assert doc_type == 'rfp'
        assert confidence > 0.5
    
    def test_rule_based_classify(self, classifier):
        """Test rule-based classification method"""
        text = "disability claim form claimant diagnosis"
        
        doc_type, confidence = classifier._rule_based_classify(text)
        
        assert doc_type == 'disability_claim'
        assert 0 <= confidence <= 1
    
    def test_empty_text(self, classifier):
        """Test handling of empty text"""
        text = ""
        
        doc_type, confidence = classifier.classify(text)
        
        # Should return some result even with empty text
        assert doc_type in ['disability_claim', 'enrollment', 'policy', 'rfp']
        assert confidence == 0.0
    
    def test_ambiguous_text(self, classifier):
        """Test handling of ambiguous text"""
        text = "This is a generic document without specific keywords."
        
        doc_type, confidence = classifier.classify(text)
        
        # Should still return a classification
        assert doc_type is not None
        assert 0 <= confidence <= 1
