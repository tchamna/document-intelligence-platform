"""
API Integration Tests
Tests for the FastAPI endpoints
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from api.main import app


class TestHealthEndpoints:
    """Test health and system endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'components' in data
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert 'message' in data
        assert 'version' in data
    
    def test_document_types(self, client):
        """Test document types reference endpoint"""
        response = client.get("/document-types")
        
        assert response.status_code == 200
        data = response.json()
        assert 'document_types' in data
        assert len(data['document_types']) >= 4
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert 'target_precision' in data
        assert data['target_precision'] >= 0.95


class TestProcessingEndpoints:
    """Test document processing endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def sample_claim(self):
        """Sample claim document"""
        return {
            "text": """
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
            
            Monthly Earnings: $5,000
            Requested Benefit Amount: $2,500/month (60% of earnings)
            """,
            "metadata": {
                "source": "email",
                "filename": "claim_form.pdf"
            }
        }
    
    def test_process_document(self, client, sample_claim):
        """Test document processing endpoint"""
        response = client.post("/process", json=sample_claim)
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'success'
        assert data['document_type'] == 'disability_claim'
        assert 'extracted_entities' in data
        assert 'processing_time_seconds' in data
    
    def test_process_invalid_document(self, client):
        """Test processing with invalid input"""
        response = client.post("/process", json={"text": "too short"})
        
        assert response.status_code == 422  # Validation error
    
    def test_classify_document(self, client, sample_claim):
        """Test classification endpoint"""
        response = client.post("/classify", json={"text": sample_claim["text"]})
        
        assert response.status_code == 200
        data = response.json()
        assert data['document_type'] == 'disability_claim'
        assert 'confidence' in data
    
    def test_extract_entities(self, client, sample_claim):
        """Test extraction endpoint"""
        response = client.post("/extract", json={
            "text": sample_claim["text"],
            "document_type": "disability_claim"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert 'entities' in data
        assert 'entity_count' in data
        assert data['entity_count'] > 0


class TestBatchProcessing:
    """Test batch processing endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_batch_processing(self, client):
        """Test batch document processing"""
        documents = [
            {
                "text": """
                DISABILITY CLAIM FORM
                Claim Number: CLM-001
                Claimant: Test User 1
                Date of Disability: 11/01/2024
                Diagnosis: Back injury
                Employer: Company A
                """,
            },
            {
                "text": """
                EMPLOYEE ENROLLMENT FORM
                Employee: Test User 2
                Plan Selection: Health Plus
                Coverage Election: Family
                Beneficiary: Spouse
                Effective Date: 01/01/2025
                """
            }
        ]
        
        response = client.post("/process/batch", json={"documents": documents})
        
        assert response.status_code == 200
        data = response.json()
        assert data['total_documents'] == 2
        assert data['successful'] == 2
        assert len(data['results']) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
