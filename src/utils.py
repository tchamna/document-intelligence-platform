"""
Utility Functions for Healthcare IDP System
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default location.
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Default to config/config.yaml relative to project root
        project_root = Path(__file__).parent.parent
        config_path = project_root / 'config' / 'config.yaml'
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return {}
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        # Ensure log directory exists
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    logger.info(f"Logging configured at {log_level} level")


def mask_pii(text: str) -> str:
    """
    Mask personally identifiable information in text.
    
    Args:
        text: Text potentially containing PII
        
    Returns:
        Text with PII masked
    """
    import re
    
    # Mask SSN
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', 'XXX-XX-XXXX', text)
    
    # Mask phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', 'XXX-XXX-XXXX', text)
    
    # Mask email addresses
    text = re.sub(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'email@masked.com',
        text
    )
    
    return text


def calculate_metrics(predictions: list, actuals: list) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        predictions: List of predicted values
        actuals: List of actual values
        
    Returns:
        Dictionary with precision, recall, f1 scores
    """
    from collections import Counter
    
    if len(predictions) != len(actuals):
        raise ValueError("Predictions and actuals must have same length")
    
    if not predictions:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # Calculate true positives, false positives, false negatives
    tp = sum(1 for p, a in zip(predictions, actuals) if p == a and p is not None)
    fp = sum(1 for p, a in zip(predictions, actuals) if p != a and p is not None)
    fn = sum(1 for p, a in zip(predictions, actuals) if p != a and a is not None)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': tp / len(predictions) if predictions else 0.0
    }


def validate_document(text: str, min_length: int = 50) -> Dict[str, Any]:
    """
    Validate document text before processing.
    
    Args:
        text: Document text to validate
        min_length: Minimum required text length
        
    Returns:
        Validation result with status and issues
    """
    issues = []
    
    if not text:
        issues.append("Document text is empty")
    elif len(text) < min_length:
        issues.append(f"Document too short (minimum {min_length} characters)")
    
    # Check for minimum word count
    word_count = len(text.split()) if text else 0
    if word_count < 10:
        issues.append("Document has too few words")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'character_count': len(text) if text else 0,
        'word_count': word_count
    }


def format_currency(amount: float) -> str:
    """Format number as currency string"""
    return f"${amount:,.2f}"


def parse_date(date_str: str) -> Optional[str]:
    """
    Parse various date formats to ISO format.
    
    Args:
        date_str: Date string in various formats
        
    Returns:
        ISO formatted date string or None
    """
    from datetime import datetime
    
    formats = [
        '%m/%d/%Y',
        '%Y-%m-%d',
        '%d/%m/%Y',
        '%B %d, %Y',
        '%b %d, %Y',
        '%m-%d-%Y'
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            continue
    
    return None
