#!/usr/bin/env python3
"""
Batch Document Processor
========================
Automated batch processing for healthcare documents.
Supports folder watching, scheduled processing, and CSV export.

Usage:
    # Process all documents in a folder
    python scripts/batch_processor.py --input data/inbox --output data/processed
    
    # Watch folder for new documents
    python scripts/batch_processor.py --watch data/inbox --output data/processed
    
    # Process with CSV export
    python scripts/batch_processor.py --input data/inbox --csv results.csv
"""

import os
import sys
import csv
import time
import json
import logging
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.enhanced_pipeline import EnhancedIDPPipeline
from src.document_processor import DocumentProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result from processing a single document"""
    filename: str
    document_type: str
    classification_confidence: float
    quality_score: float
    status: str  # 'success' or 'failed'
    adjudication_status: Optional[str]
    adjudication_confidence: Optional[float]
    entity_count: int
    processing_time_ms: float
    timestamp: str
    error: Optional[str] = None
    # Store full extracted entities for detailed export
    extracted_entities: Optional[Dict[str, Any]] = None


@dataclass
class EntityExtraction:
    """Flattened entity extraction for CSV export"""
    filename: str
    document_type: str
    timestamp: str
    # Person identifiers
    claimant_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    ssn: Optional[str] = None
    age: Optional[str] = None
    # Contact info
    email: Optional[str] = None
    phone: Optional[str] = None
    # Policy/Claim identifiers
    policy_number: Optional[str] = None
    claim_number: Optional[str] = None
    # Employment
    employer: Optional[str] = None
    occupation: Optional[str] = None
    # Medical
    diagnosis: Optional[str] = None
    date_of_disability: Optional[str] = None
    treating_physician: Optional[str] = None
    # Financial
    benefit_amount: Optional[str] = None
    max_benefit: Optional[str] = None
    benefit_percentage: Optional[str] = None
    # Plan details
    plan_type: Optional[str] = None
    coverage_level: Optional[str] = None
    effective_date: Optional[str] = None
    elimination_period: Optional[str] = None
    # Additional extracted entities
    all_dates: Optional[str] = None
    all_amounts: Optional[str] = None
    persons_found: Optional[str] = None
    organizations_found: Optional[str] = None


class BatchProcessor:
    """
    Batch document processor for automated IDP workflows.
    
    Features:
    - Process multiple documents from a folder
    - Watch folder for new documents
    - Export results to CSV
    - Move processed documents to archive
    - Error handling and retry logic
    """
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.txt'}
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the batch processor"""
        logger.info("Initializing Batch Processor...")
        
        self.pipeline = EnhancedIDPPipeline()
        self.doc_processor = DocumentProcessor()
        self.output_dir = Path(output_dir) if output_dir else None
        self.results: List[ProcessingResult] = []
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'processed').mkdir(exist_ok=True)
            (self.output_dir / 'failed').mkdir(exist_ok=True)
            (self.output_dir / 'results').mkdir(exist_ok=True)
        
        logger.info("Batch Processor initialized successfully")
    
    def process_folder(self, input_dir: str, recursive: bool = False) -> List[ProcessingResult]:
        """
        Process all documents in a folder.
        
        Args:
            input_dir: Path to input folder
            recursive: Whether to process subdirectories
            
        Returns:
            List of processing results
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        # Find all supported files
        if recursive:
            files = [f for f in input_path.rglob('*') if f.suffix.lower() in self.SUPPORTED_EXTENSIONS]
        else:
            files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS]
        
        logger.info(f"Found {len(files)} documents to process in {input_dir}")
        
        results = []
        for i, file_path in enumerate(files, 1):
            logger.info(f"Processing [{i}/{len(files)}]: {file_path.name}")
            result = self.process_document(file_path)
            results.append(result)
            
            # Move file to appropriate folder
            if self.output_dir:
                self._move_processed_file(file_path, result)
        
        self.results.extend(results)
        return results
    
    def process_document(self, file_path: Path) -> ProcessingResult:
        """
        Process a single document.
        
        Args:
            file_path: Path to the document
            
        Returns:
            ProcessingResult with extraction details
        """
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        try:
            # Read file content
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            # Extract text from document
            processed = self.doc_processor.process_file(file_content, file_path.name)
            text = processed.text
            
            if not text or len(text.strip()) < 10:
                return ProcessingResult(
                    filename=file_path.name,
                    document_type='unknown',
                    classification_confidence=0.0,
                    quality_score=0.0,
                    status='failed',
                    adjudication_status=None,
                    adjudication_confidence=None,
                    entity_count=0,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    timestamp=timestamp,
                    error="No text extracted from document"
                )
            
            # Process through pipeline
            result = self.pipeline.process_document(text)
            
            # Extract adjudication info
            adj_status = None
            adj_confidence = None
            if result.adjudication_result:
                adj_status = result.adjudication_result.get('status', 'N/A')
                adj_confidence = result.adjudication_result.get('confidence', 0.0)
            
            processing_time = (time.time() - start_time) * 1000
            
            return ProcessingResult(
                filename=file_path.name,
                document_type=result.document_type,
                classification_confidence=result.classification_confidence,
                quality_score=result.quality_score,
                status='success',
                adjudication_status=adj_status,
                adjudication_confidence=adj_confidence,
                entity_count=len(result.extracted_entities),
                processing_time_ms=processing_time,
                timestamp=timestamp,
                extracted_entities=result.extracted_entities  # Store entities for detailed export
            )
            
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            return ProcessingResult(
                filename=file_path.name,
                document_type='unknown',
                classification_confidence=0.0,
                quality_score=0.0,
                status='failed',
                adjudication_status=None,
                adjudication_confidence=None,
                entity_count=0,
                processing_time_ms=(time.time() - start_time) * 1000,
                timestamp=timestamp,
                error=str(e)
            )
    
    def _move_processed_file(self, file_path: Path, result: ProcessingResult):
        """Move processed file to appropriate output folder"""
        if not self.output_dir:
            return
        
        if result.status == 'success':
            dest_dir = self.output_dir / 'processed' / result.document_type
        else:
            dest_dir = self.output_dir / 'failed'
        
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / file_path.name
        
        # Handle duplicate filenames
        counter = 1
        while dest_path.exists():
            stem = file_path.stem
            suffix = file_path.suffix
            dest_path = dest_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        
        shutil.move(str(file_path), str(dest_path))
        logger.info(f"  Moved to: {dest_path}")
    
    def export_to_csv(self, csv_path: str, results: Optional[List[ProcessingResult]] = None):
        """
        Export processing results to CSV.
        
        Args:
            csv_path: Path to output CSV file
            results: Results to export (uses self.results if not provided)
        """
        results = results or self.results
        if not results:
            logger.warning("No results to export")
            return
        
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        fieldnames = [
            'filename', 'document_type', 'classification_confidence', 
            'quality_score', 'status', 'adjudication_status', 
            'adjudication_confidence', 'entity_count', 'processing_time_ms',
            'timestamp', 'error'
        ]
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                # Exclude extracted_entities from summary CSV
                row = {k: v for k, v in asdict(result).items() if k != 'extracted_entities'}
                writer.writerow(row)
        
        logger.info(f"Exported {len(results)} results to {csv_path}")
    
    def _extract_entity_value(self, entities: Dict, key: str, default: str = None) -> Optional[str]:
        """Safely extract a value from entities dict, handling lists"""
        if not entities or key not in entities:
            return default
        value = entities[key]
        if isinstance(value, list):
            return '; '.join(str(v) for v in value[:5])  # Join first 5 items
        return str(value) if value else default
    
    def _result_to_entity_extraction(self, result: ProcessingResult) -> EntityExtraction:
        """Convert ProcessingResult to flattened EntityExtraction for CSV"""
        entities = result.extracted_entities or {}
        
        # Get first person as claimant name (or from specific field)
        persons = entities.get('persons', [])
        claimant_name = persons[0] if persons else None
        
        return EntityExtraction(
            filename=result.filename,
            document_type=result.document_type,
            timestamp=result.timestamp,
            # Person identifiers
            claimant_name=claimant_name,
            date_of_birth=self._extract_entity_value(entities, 'date_of_birth'),
            ssn=self._extract_entity_value(entities, 'ssn'),
            age=self._extract_entity_value(entities, 'age'),
            # Contact info
            email=self._extract_entity_value(entities, 'email'),
            phone=self._extract_entity_value(entities, 'phone'),
            # Policy/Claim identifiers
            policy_number=self._extract_entity_value(entities, 'policy_number'),
            claim_number=self._extract_entity_value(entities, 'claim_number'),
            # Employment
            employer=self._extract_entity_value(entities, 'employer'),
            occupation=self._extract_entity_value(entities, 'occupation'),
            # Medical
            diagnosis=self._extract_entity_value(entities, 'diagnosis'),
            date_of_disability=self._extract_entity_value(entities, 'date_of_disability'),
            treating_physician=self._extract_entity_value(entities, 'treating_physician'),
            # Financial
            benefit_amount=self._extract_entity_value(entities, 'benefit_amount'),
            max_benefit=self._extract_entity_value(entities, 'max_benefit'),
            benefit_percentage=self._extract_entity_value(entities, 'benefit_percentage'),
            # Plan details
            plan_type=self._extract_entity_value(entities, 'plan_type'),
            coverage_level=self._extract_entity_value(entities, 'coverage_level'),
            effective_date=self._extract_entity_value(entities, 'effective_date'),
            elimination_period=self._extract_entity_value(entities, 'elimination_period'),
            # Additional entities
            all_dates=self._extract_entity_value(entities, 'date'),
            all_amounts=self._extract_entity_value(entities, 'money'),
            persons_found='; '.join(persons[:5]) if persons else None,
            organizations_found='; '.join(entities.get('organizations', [])[:5]) if entities.get('organizations') else None
        )
    
    def export_entities_csv(self, csv_path: str, results: Optional[List[ProcessingResult]] = None):
        """
        Export detailed entity extractions to CSV.
        
        This creates a table with key extracted fields like:
        - Claimant name, DOB, SSN, age
        - Policy number, claim number
        - Diagnosis, employer, benefit amounts
        
        Args:
            csv_path: Path to output CSV file
            results: Results to export (uses self.results if not provided)
        """
        results = results or self.results
        if not results:
            logger.warning("No results to export")
            return
        
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to EntityExtraction objects
        extractions = [self._result_to_entity_extraction(r) for r in results if r.status == 'success']
        
        if not extractions:
            logger.warning("No successful extractions to export")
            return
        
        fieldnames = [
            'filename', 'document_type', 'timestamp',
            'claimant_name', 'date_of_birth', 'ssn', 'age',
            'email', 'phone',
            'policy_number', 'claim_number',
            'employer', 'occupation',
            'diagnosis', 'date_of_disability', 'treating_physician',
            'benefit_amount', 'max_benefit', 'benefit_percentage',
            'plan_type', 'coverage_level', 'effective_date', 'elimination_period',
            'all_dates', 'all_amounts', 'persons_found', 'organizations_found'
        ]
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for extraction in extractions:
                writer.writerow(asdict(extraction))
        
        logger.info(f"Exported {len(extractions)} entity extractions to {csv_path}")
    
    def export_detailed_json(self, json_path: str, results: Optional[List[ProcessingResult]] = None):
        """Export detailed results to JSON"""
        results = results or self.results
        if not results:
            return
        
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        
        logger.info(f"Exported detailed results to {json_path}")
    
    def watch_folder(self, input_dir: str, poll_interval: int = 5):
        """
        Watch a folder for new documents and process them automatically.
        
        Args:
            input_dir: Path to folder to watch
            poll_interval: Seconds between checks for new files
        """
        input_path = Path(input_dir)
        input_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Watching folder: {input_dir}")
        logger.info(f"Poll interval: {poll_interval} seconds")
        logger.info("Press Ctrl+C to stop...")
        
        processed_files = set()
        
        try:
            while True:
                # Find new files
                current_files = {
                    f for f in input_path.iterdir() 
                    if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS
                }
                
                new_files = current_files - processed_files
                
                if new_files:
                    logger.info(f"Found {len(new_files)} new document(s)")
                    for file_path in new_files:
                        logger.info(f"Processing: {file_path.name}")
                        result = self.process_document(file_path)
                        self.results.append(result)
                        
                        if self.output_dir:
                            self._move_processed_file(file_path, result)
                        
                        processed_files.add(file_path)
                        
                        # Auto-export results (summary and entities)
                        if self.output_dir:
                            self.export_to_csv(
                                str(self.output_dir / 'results' / 'batch_results.csv')
                            )
                            self.export_entities_csv(
                                str(self.output_dir / 'results' / 'extracted_entities.csv')
                            )
                
                time.sleep(poll_interval)
                
        except KeyboardInterrupt:
            logger.info("\nStopping folder watch...")
            if self.results:
                logger.info(f"Total documents processed: {len(self.results)}")
    
    def print_summary(self):
        """Print processing summary"""
        if not self.results:
            print("\nNo documents processed")
            return
        
        total = len(self.results)
        successful = sum(1 for r in self.results if r.status == 'success')
        failed = total - successful
        
        avg_time = sum(r.processing_time_ms for r in self.results) / total
        avg_quality = sum(r.quality_score for r in self.results if r.status == 'success') / max(successful, 1)
        
        doc_types = {}
        for r in self.results:
            doc_types[r.document_type] = doc_types.get(r.document_type, 0) + 1
        
        print("\n" + "=" * 60)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 60)
        print(f"  Total Documents:     {total}")
        print(f"  Successful:          {successful} ({successful/total*100:.1f}%)")
        print(f"  Failed:              {failed} ({failed/total*100:.1f}%)")
        print(f"  Avg Processing Time: {avg_time:.0f}ms")
        print(f"  Avg Quality Score:   {avg_quality*100:.1f}%")
        print("\n  Document Types:")
        for doc_type, count in sorted(doc_types.items()):
            print(f"    - {doc_type}: {count}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Batch Document Processor for Healthcare IDP System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all documents in a folder (auto-exports both summary and entities CSVs)
  python batch_processor.py --input data/inbox --output data/processed
  
  # Watch folder for new documents
  python batch_processor.py --watch data/inbox --output data/processed
  
  # Process and export summary CSV only
  python batch_processor.py --input data/inbox --csv results.csv
  
  # Export key extracted entities to CSV (claimant, SSN, DOB, policy#, etc.)
  python batch_processor.py --input data/inbox --entities extracted_entities.csv
  
  # Export both summary and entities CSVs
  python batch_processor.py --input data/inbox --csv results.csv --entities entities.csv
  
  # Process recursively
  python batch_processor.py --input data/inbox --recursive --output data/processed

Output Files:
  - batch_results.csv: Processing summary (document type, confidence, quality, timing)
  - extracted_entities.csv: Key extracted fields (claimant name, SSN, policy#, diagnosis, etc.)
        """
    )
    
    parser.add_argument('--input', '-i', help='Input folder to process')
    parser.add_argument('--watch', '-w', help='Folder to watch for new documents')
    parser.add_argument('--output', '-o', help='Output folder for processed documents')
    parser.add_argument('--csv', help='Export processing summary to CSV file')
    parser.add_argument('--entities', '-e', help='Export extracted entities table to CSV file')
    parser.add_argument('--json', help='Export detailed results to JSON file')
    parser.add_argument('--recursive', '-r', action='store_true', help='Process subdirectories')
    parser.add_argument('--interval', type=int, default=5, help='Watch poll interval (seconds)')
    
    args = parser.parse_args()
    
    if not args.input and not args.watch:
        parser.print_help()
        print("\nError: Either --input or --watch must be specified")
        sys.exit(1)
    
    # Initialize processor
    processor = BatchProcessor(output_dir=args.output)
    
    try:
        if args.watch:
            # Watch mode
            processor.watch_folder(args.watch, poll_interval=args.interval)
        else:
            # Batch mode
            results = processor.process_folder(args.input, recursive=args.recursive)
            
            # Export results
            if args.csv:
                processor.export_to_csv(args.csv, results)
            
            if args.entities:
                processor.export_entities_csv(args.entities, results)
            
            if args.json:
                processor.export_detailed_json(args.json, results)
            
            # Auto-export to output dir if specified
            if args.output and not args.csv:
                processor.export_to_csv(str(Path(args.output) / 'results' / 'batch_results.csv'), results)
            
            if args.output and not args.entities:
                processor.export_entities_csv(str(Path(args.output) / 'results' / 'extracted_entities.csv'), results)
            
            # Print summary
            processor.print_summary()
            
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
