"""
Document Processor Module
Handles file uploads, OCR, and text extraction from various document formats
"""

import os
import io
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DocumentFormat(Enum):
    """Supported document formats"""
    PDF = "pdf"
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    TIFF = "tiff"
    TIF = "tif"
    BMP = "bmp"
    TEXT = "txt"
    UNKNOWN = "unknown"


@dataclass
class ProcessedDocument:
    """Result of document processing"""
    text: str
    format: DocumentFormat
    page_count: int
    confidence: float
    metadata: Dict[str, Any]
    warnings: List[str]


class DocumentProcessor:
    """
    Process various document formats and extract text using OCR.
    Supports PDF, images (PNG, JPG, TIFF, BMP), and text files.
    """
    
    def __init__(self, tesseract_path: Optional[str] = None):
        """
        Initialize document processor.
        
        Args:
            tesseract_path: Path to Tesseract executable (optional)
        """
        self.tesseract_available = False
        self.poppler_available = False
        
        # Check for Tesseract - try common Windows paths if not provided
        try:
            import pytesseract
            
            # Try to find Tesseract in common locations
            if not tesseract_path:
                import os
                common_paths = [
                    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                    os.path.expanduser(r'~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'),
                ]
                for path in common_paths:
                    if os.path.exists(path):
                        tesseract_path = path
                        break
            
            if tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                logger.info(f"Using Tesseract at: {tesseract_path}")
            
            # Test if Tesseract is available
            pytesseract.get_tesseract_version()
            self.tesseract_available = True
            logger.info("Tesseract OCR is available")
        except Exception as e:
            logger.warning(f"Tesseract OCR not available: {e}")
            logger.warning("Install Tesseract for OCR support: https://github.com/tesseract-ocr/tesseract")
        
        # Check for PDF support
        try:
            import pdf2image
            self.poppler_available = True
            logger.info("PDF processing is available")
        except Exception as e:
            logger.warning(f"PDF processing not available: {e}")
        
        logger.info("DocumentProcessor initialized")
    
    def process_file(self, file_content: bytes, filename: str) -> ProcessedDocument:
        """
        Process an uploaded file and extract text.
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            
        Returns:
            ProcessedDocument with extracted text and metadata
        """
        logger.info(f"Processing file: {filename}")
        
        # Determine file format
        file_format = self._detect_format(filename)
        
        warnings = []
        
        if file_format == DocumentFormat.TEXT:
            return self._process_text_file(file_content, filename)
        elif file_format == DocumentFormat.PDF:
            return self._process_pdf(file_content, filename)
        elif file_format in [DocumentFormat.PNG, DocumentFormat.JPG, 
                            DocumentFormat.JPEG, DocumentFormat.TIFF,
                            DocumentFormat.TIF, DocumentFormat.BMP]:
            return self._process_image(file_content, filename, file_format)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
    
    def _detect_format(self, filename: str) -> DocumentFormat:
        """Detect document format from filename"""
        ext = Path(filename).suffix.lower().lstrip('.')
        
        format_map = {
            'pdf': DocumentFormat.PDF,
            'png': DocumentFormat.PNG,
            'jpg': DocumentFormat.JPG,
            'jpeg': DocumentFormat.JPEG,
            'tiff': DocumentFormat.TIFF,
            'tif': DocumentFormat.TIF,
            'bmp': DocumentFormat.BMP,
            'txt': DocumentFormat.TEXT,
        }
        
        return format_map.get(ext, DocumentFormat.UNKNOWN)
    
    def _process_text_file(self, content: bytes, filename: str) -> ProcessedDocument:
        """Process plain text file"""
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            text = content.decode('latin-1')
        
        return ProcessedDocument(
            text=text,
            format=DocumentFormat.TEXT,
            page_count=1,
            confidence=1.0,
            metadata={'filename': filename, 'method': 'direct_read'},
            warnings=[]
        )
    
    def _process_pdf(self, content: bytes, filename: str) -> ProcessedDocument:
        """Process PDF file - try text extraction first, then OCR"""
        warnings = []
        text_parts = []
        page_count = 0
        confidence = 1.0
        method = 'text_extraction'
        
        # First, try direct text extraction with pdfplumber
        try:
            import pdfplumber
            
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                page_count = len(pdf.pages)
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            
            extracted_text = '\n\n'.join(text_parts)
            
            # If we got sufficient text, return it
            if len(extracted_text.strip()) > 100:
                logger.info(f"Extracted text directly from PDF: {len(extracted_text)} chars")
                return ProcessedDocument(
                    text=extracted_text,
                    format=DocumentFormat.PDF,
                    page_count=page_count,
                    confidence=0.98,
                    metadata={'filename': filename, 'method': 'pdfplumber'},
                    warnings=warnings
                )
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
            warnings.append(f"Text extraction failed: {str(e)}")
        
        # Fallback: Try PyPDF2
        try:
            import PyPDF2
            
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            page_count = len(pdf_reader.pages)
            text_parts = []
            
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            
            extracted_text = '\n\n'.join(text_parts)
            
            if len(extracted_text.strip()) > 100:
                logger.info(f"Extracted text with PyPDF2: {len(extracted_text)} chars")
                return ProcessedDocument(
                    text=extracted_text,
                    format=DocumentFormat.PDF,
                    page_count=page_count,
                    confidence=0.95,
                    metadata={'filename': filename, 'method': 'pypdf2'},
                    warnings=warnings
                )
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}")
            warnings.append(f"PyPDF2 failed: {str(e)}")
        
        # Last resort: OCR the PDF
        if self.tesseract_available and self.poppler_available:
            return self._ocr_pdf(content, filename, warnings)
        else:
            warnings.append("OCR not available - install Tesseract and Poppler")
            return ProcessedDocument(
                text="[Unable to extract text from PDF - OCR not available]",
                format=DocumentFormat.PDF,
                page_count=page_count,
                confidence=0.0,
                metadata={'filename': filename, 'method': 'failed'},
                warnings=warnings
            )
    
    def _ocr_pdf(self, content: bytes, filename: str, warnings: List[str]) -> ProcessedDocument:
        """OCR a PDF by converting to images first"""
        import pytesseract
        from pdf2image import convert_from_bytes
        
        logger.info("Using OCR for PDF processing")
        
        try:
            # Convert PDF to images
            images = convert_from_bytes(content, dpi=300)
            page_count = len(images)
            
            text_parts = []
            confidences = []
            
            for i, image in enumerate(images):
                # Get OCR data with confidence scores
                ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                
                # Extract text
                page_text = pytesseract.image_to_string(image)
                text_parts.append(page_text)
                
                # Calculate average confidence for this page
                conf_values = [int(c) for c in ocr_data['conf'] if int(c) > 0]
                if conf_values:
                    confidences.append(sum(conf_values) / len(conf_values))
            
            extracted_text = '\n\n--- Page Break ---\n\n'.join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) / 100 if confidences else 0.5
            
            logger.info(f"OCR extracted {len(extracted_text)} chars with {avg_confidence:.2%} confidence")
            
            return ProcessedDocument(
                text=extracted_text,
                format=DocumentFormat.PDF,
                page_count=page_count,
                confidence=avg_confidence,
                metadata={'filename': filename, 'method': 'ocr_pdf'},
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"PDF OCR failed: {e}")
            warnings.append(f"PDF OCR failed: {str(e)}")
            return ProcessedDocument(
                text="[OCR failed for PDF]",
                format=DocumentFormat.PDF,
                page_count=0,
                confidence=0.0,
                metadata={'filename': filename, 'method': 'failed'},
                warnings=warnings
            )
    
    def _process_image(self, content: bytes, filename: str, 
                       file_format: DocumentFormat) -> ProcessedDocument:
        """Process image file with OCR"""
        from PIL import Image
        
        warnings = []
        
        if not self.tesseract_available:
            warnings.append("Tesseract OCR not available")
            return ProcessedDocument(
                text="[OCR not available - install Tesseract]",
                format=file_format,
                page_count=1,
                confidence=0.0,
                metadata={'filename': filename, 'method': 'failed'},
                warnings=warnings
            )
        
        import pytesseract
        
        try:
            # Open image
            image = Image.open(io.BytesIO(content))
            
            # Preprocess image for better OCR
            image = self._preprocess_image(image)
            
            # Get OCR data with confidence
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Extract text
            text = pytesseract.image_to_string(image)
            
            # Calculate confidence
            conf_values = [int(c) for c in ocr_data['conf'] if int(c) > 0]
            avg_confidence = sum(conf_values) / len(conf_values) / 100 if conf_values else 0.5
            
            logger.info(f"OCR extracted {len(text)} chars with {avg_confidence:.2%} confidence")
            
            return ProcessedDocument(
                text=text,
                format=file_format,
                page_count=1,
                confidence=avg_confidence,
                metadata={
                    'filename': filename,
                    'method': 'ocr_image',
                    'image_size': image.size,
                    'image_mode': image.mode
                },
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Image OCR failed: {e}")
            warnings.append(f"Image OCR failed: {str(e)}")
            return ProcessedDocument(
                text="[OCR failed for image]",
                format=file_format,
                page_count=1,
                confidence=0.0,
                metadata={'filename': filename, 'method': 'failed'},
                warnings=warnings
            )
    
    def _preprocess_image(self, image):
        """Preprocess image for better OCR results"""
        from PIL import Image, ImageEnhance, ImageFilter
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Increase contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        # Sharpen
        image = image.filter(ImageFilter.SHARPEN)
        
        # Resize if too small (OCR works better with larger images)
        min_width = 1000
        if image.width < min_width:
            ratio = min_width / image.width
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file extensions"""
        formats = ['.txt']
        
        # PDF support
        formats.append('.pdf')
        
        # Image formats
        if self.tesseract_available:
            formats.extend(['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'])
        
        return formats
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check status of OCR dependencies"""
        return {
            'tesseract': self.tesseract_available,
            'poppler': self.poppler_available,
            'pdfplumber': self._check_module('pdfplumber'),
            'pypdf2': self._check_module('PyPDF2'),
            'pillow': self._check_module('PIL'),
        }
    
    def _check_module(self, module_name: str) -> bool:
        """Check if a Python module is available"""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False
