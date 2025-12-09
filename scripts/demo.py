#!/usr/bin/env python3
"""
Demo Script
Demonstrates the Healthcare IDP System capabilities
"""

import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import IDPPipeline
from src.document_classifier import DocumentClassifier
from src.entity_extractor import EntityExtractor
from src.policy_interpreter import PolicyInterpreter


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def load_sample_documents():
    """Load sample documents from data/samples"""
    samples_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'samples')
    documents = {}
    
    sample_files = [
        'disability_claim_sample.txt',
        'enrollment_form_sample.txt',
        'policy_document_sample.txt',
        'rfp_sample.txt'
    ]
    
    for filename in sample_files:
        filepath = os.path.join(samples_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                key = filename.replace('_sample.txt', '')
                documents[key] = f.read()
    
    return documents


def demo_classification():
    """Demonstrate document classification"""
    print_section("Document Classification Demo")
    
    classifier = DocumentClassifier()
    documents = load_sample_documents()
    
    print("\nClassifying sample documents...\n")
    
    for doc_name, doc_text in documents.items():
        doc_type, confidence = classifier.classify(doc_text)
        status = "✅" if doc_type.replace('_', '') in doc_name.replace('_', '') else "❓"
        print(f"{status} {doc_name}:")
        print(f"   Classified as: {doc_type}")
        print(f"   Confidence: {confidence:.2%}")
        print()


def demo_entity_extraction():
    """Demonstrate entity extraction"""
    print_section("Entity Extraction Demo")
    
    extractor = EntityExtractor()
    documents = load_sample_documents()
    
    # Demo on disability claim
    if 'disability_claim' in documents:
        print("\nExtracting entities from disability claim...\n")
        entities = extractor.extract(documents['disability_claim'], 'disability_claim')
        
        print("Extracted Entities:")
        print("-" * 40)
        for key, value in entities.items():
            if isinstance(value, list):
                if len(value) <= 3:
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: [{value[0]}, {value[1]}, ...] ({len(value)} items)")
            else:
                print(f"  {key}: {value}")
    
    # Demo on enrollment form
    if 'enrollment_form' in documents:
        print("\n\nExtracting entities from enrollment form...\n")
        entities = extractor.extract(documents['enrollment_form'], 'enrollment')
        
        print("Extracted Entities:")
        print("-" * 40)
        for key, value in entities.items():
            if isinstance(value, list) and len(value) > 3:
                print(f"  {key}: [{value[0]}, {value[1]}, ...] ({len(value)} items)")
            else:
                print(f"  {key}: {value}")


def demo_full_pipeline():
    """Demonstrate full pipeline processing"""
    print_section("Full Pipeline Processing Demo")
    
    pipeline = IDPPipeline()
    documents = load_sample_documents()
    
    if 'disability_claim' in documents:
        print("\nProcessing disability claim through full pipeline...\n")
        result = pipeline.process_document(documents['disability_claim'])
        
        print("Pipeline Results:")
        print("-" * 40)
        print(f"Status: {result['status']}")
        print(f"Document Type: {result['document_type']}")
        print(f"Classification Confidence: {result['classification_confidence']:.2%}")
        print(f"Entities Extracted: {len(result.get('extracted_entities', {}))}")
        print(f"Quality Score: {result['quality_score']:.2%}")
        print(f"Processing Time: {result['processing_time_seconds']:.3f}s")
        
        if 'adjudication' in result:
            adj = result['adjudication']
            print("\nAdjudication Decision:")
            print(f"  Status: {adj['eligibility_status']}")
            print(f"  Confidence: {adj['confidence']:.2%}")
            print(f"  Coverage Verified: {adj['coverage_verified']}")
            print(f"  Waiting Period Met: {adj['waiting_period_met']}")
            if adj.get('estimated_benefit_start'):
                print(f"  Benefit Start Date: {adj['estimated_benefit_start']}")
        
        print("\nProcessing Metrics:")
        metrics = result.get('processing_metrics', {})
        print(f"  Overall Confidence: {metrics.get('overall_confidence', 0):.2%}")
        print(f"  Production Ready: {metrics.get('production_ready', False)}")


def demo_policy_interpretation():
    """Demonstrate policy interpretation"""
    print_section("Policy Interpretation Demo")
    
    interpreter = PolicyInterpreter()
    documents = load_sample_documents()
    
    if 'policy_document' in documents:
        print("\nInterpreting policy document...\n")
        interpretation = interpreter.interpret(documents['policy_document'])
        
        print("Policy Clauses Found:")
        print("-" * 40)
        for clause_name, clause_text in interpretation.get('clauses', {}).items():
            if clause_text:
                print(f"\n{clause_name}:")
                # Truncate long clauses
                display_text = clause_text[:150] + "..." if len(clause_text) > 150 else clause_text
                print(f"  {display_text}")


def demo_batch_processing():
    """Demonstrate batch processing"""
    print_section("Batch Processing Demo")
    
    pipeline = IDPPipeline()
    documents = load_sample_documents()
    
    print("\nProcessing all sample documents in batch...\n")
    
    results = []
    for doc_name, doc_text in documents.items():
        result = pipeline.process_document(doc_text)
        results.append({
            'document': doc_name,
            'type': result['document_type'],
            'confidence': result['classification_confidence'],
            'entities': len(result.get('extracted_entities', {})),
            'quality': result['quality_score'],
            'time': result['processing_time_seconds']
        })
    
    print("Batch Results Summary:")
    print("-" * 70)
    print(f"{'Document':<20} {'Type':<18} {'Conf.':<8} {'Entities':<10} {'Quality':<8} {'Time':<8}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['document']:<20} {r['type']:<18} {r['confidence']:.2%}   {r['entities']:<10} {r['quality']:.2%}   {r['time']:.3f}s")
    
    print("-" * 70)
    avg_quality = sum(r['quality'] for r in results) / len(results)
    total_time = sum(r['time'] for r in results)
    print(f"{'TOTAL':<20} {'':<18} {'':8} {sum(r['entities'] for r in results):<10} {avg_quality:.2%}   {total_time:.3f}s")


def main():
    """Run all demos"""
    print("\n" + "=" * 70)
    print(" HEALTHCARE IDP SYSTEM - DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo showcases the capabilities of the Healthcare")
    print("Intelligent Document Processing (IDP) System.")
    
    try:
        # Run demos
        demo_classification()
        demo_entity_extraction()
        demo_policy_interpretation()
        demo_full_pipeline()
        demo_batch_processing()
        
        print_section("Demo Complete")
        print("\nAll demonstrations completed successfully!")
        print("\nNext steps:")
        print("  1. Start the API server: uvicorn api.main:app --reload")
        print("  2. Open API docs: http://localhost:8000/docs")
        print("  3. Run tests: pytest tests/ -v")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
