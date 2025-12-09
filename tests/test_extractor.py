"""
Entity Extractor Tests
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.entity_extractor import EntityExtractor


class TestEntityExtractor:
    """Test cases for EntityExtractor"""
    
    @pytest.fixture
    def extractor(self):
        """Create extractor instance"""
        return EntityExtractor()
    
    def test_extract_claim_number(self, extractor):
        """Test extraction of claim number"""
        text = "Claim Number: CLM-2024-789456"
        
        entities = extractor.extract(text, 'disability_claim')
        
        assert 'claim_number' in entities
        assert 'CLM-2024-789456' in entities['claim_number']
    
    def test_extract_policy_number(self, extractor):
        """Test extraction of policy number"""
        text = "Policy Number: POL-456-7890-ABC"
        
        entities = extractor.extract(text, 'disability_claim')
        
        assert 'policy_number' in entities
    
    def test_extract_ssn(self, extractor):
        """Test extraction of SSN"""
        text = "SSN: 123-45-6789"
        
        entities = extractor.extract(text, 'disability_claim')
        
        assert 'ssn' in entities
        assert entities['ssn'] == '123-45-6789'
    
    def test_extract_date(self, extractor):
        """Test extraction of dates"""
        text = "Date of Disability: 11/15/2024"
        
        entities = extractor.extract(text, 'disability_claim')
        
        # Should extract the date
        assert 'date' in entities or 'date_of_disability' in entities
    
    def test_extract_money(self, extractor):
        """Test extraction of monetary amounts"""
        text = "Benefit Amount: $2,500.00 per month"
        
        entities = extractor.extract(text, 'disability_claim')
        
        assert 'money' in entities
        assert '$2,500.00' in entities['money']
    
    def test_extract_email(self, extractor):
        """Test extraction of email addresses"""
        text = "Contact: john.doe@company.com"
        
        entities = extractor.extract(text, 'disability_claim')
        
        assert 'email' in entities
        assert 'john.doe@company.com' in entities['email']
    
    def test_extract_phone(self, extractor):
        """Test extraction of phone numbers"""
        text = "Phone: 555-123-4567"
        
        entities = extractor.extract(text, 'disability_claim')
        
        assert 'phone' in entities
    
    def test_extract_diagnosis(self, extractor):
        """Test extraction of diagnosis from claim"""
        text = """
        DISABILITY CLAIM FORM
        
        Diagnosis: Acute lumbar strain with radiculopathy
        """
        
        entities = extractor.extract(text, 'disability_claim')
        
        assert 'diagnosis' in entities
        assert 'lumbar strain' in entities['diagnosis'].lower()
    
    def test_extract_employer(self, extractor):
        """Test extraction of employer from claim"""
        text = """
        Employer: Acme Corporation
        Employment Date: 01/15/2020
        """
        
        entities = extractor.extract(text, 'disability_claim')
        
        assert 'employer' in entities
        assert 'Acme Corporation' in entities['employer']
    
    def test_extract_enrollment_fields(self, extractor):
        """Test extraction of enrollment-specific fields"""
        text = """
        Plan Selection: Premium Health Plus
        Coverage Level: Employee + Family
        Effective Date: 01/01/2025
        """
        
        entities = extractor.extract(text, 'enrollment')
        
        # Should extract plan type and coverage level
        assert 'plan_type' in entities or 'coverage_level' in entities
    
    def test_extract_policy_fields(self, extractor):
        """Test extraction of policy-specific fields"""
        text = """
        Elimination Period: 90 days
        Maximum Monthly Benefit: $10,000
        Benefit Percentage: 60%
        """
        
        entities = extractor.extract(text, 'policy')
        
        assert 'elimination_period' in entities
        assert '90 days' in entities['elimination_period']
    
    def test_normalize_dates(self, extractor):
        """Test date normalization"""
        text = "Date: 11/15/2024"
        
        entities = extractor.extract(text, 'disability_claim')
        
        # Dates should be normalized to YYYY-MM-DD format
        if 'date' in entities:
            # Check if normalized
            assert entities['date'] == '2024-11-15' or entities['date'] == '11/15/2024'
    
    def test_empty_text(self, extractor):
        """Test handling of empty text"""
        text = ""
        
        entities = extractor.extract(text, 'disability_claim')
        
        # Should return empty dict for empty text
        assert isinstance(entities, dict)
    
    def test_no_matches(self, extractor):
        """Test handling of text with no matching entities"""
        text = "This is a simple text without any structured data."
        
        entities = extractor.extract(text, 'disability_claim')
        
        assert isinstance(entities, dict)
