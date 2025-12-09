"""
Accuracy Tracking and Metrics Dashboard
Production-grade metrics for monitoring model performance
Target: 97-99% precision
"""

import json
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import logging
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics tracked"""
    EXTRACTION_ACCURACY = "extraction_accuracy"
    CLASSIFICATION_ACCURACY = "classification_accuracy"
    ADJUDICATION_ACCURACY = "adjudication_accuracy"
    PROCESSING_TIME = "processing_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CONFIDENCE_CALIBRATION = "confidence_calibration"


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: str
    metric_type: MetricType
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccuracyMetrics:
    """Comprehensive accuracy metrics"""
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    confidence_mean: float
    confidence_std: float
    sample_count: int
    by_field: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion_matrix: Optional[Dict[str, Dict[str, int]]] = None


@dataclass
class PerformanceMetrics:
    """Performance and throughput metrics"""
    avg_processing_time_ms: float
    p50_processing_time_ms: float
    p95_processing_time_ms: float
    p99_processing_time_ms: float
    throughput_per_minute: float
    error_rate: float
    success_count: int
    error_count: int


class AccuracyTracker:
    """
    Track and compute accuracy metrics for IDP components.
    Supports ground truth comparison and confidence calibration.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize accuracy tracker.
        
        Args:
            storage_path: Path to store metrics data
        """
        self.storage_path = storage_path or "data/metrics"
        self.predictions: List[Dict] = []
        self.ground_truths: List[Dict] = []
        
        # Ensure storage directory exists
        os.makedirs(self.storage_path, exist_ok=True)
        
        logger.info(f"AccuracyTracker initialized (storage: {self.storage_path})")
    
    def record_prediction(self, prediction: Dict, ground_truth: Optional[Dict] = None,
                         doc_type: str = 'unknown', confidence: float = 0.0):
        """
        Record a prediction for accuracy tracking.
        
        Args:
            prediction: Predicted values
            ground_truth: Actual values (if available)
            doc_type: Type of document
            confidence: Model confidence
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'ground_truth': ground_truth,
            'doc_type': doc_type,
            'confidence': confidence
        }
        
        self.predictions.append(record)
        
        if ground_truth:
            self.ground_truths.append(ground_truth)
    
    def compute_extraction_accuracy(self, predictions: List[Dict], 
                                   ground_truths: List[Dict]) -> AccuracyMetrics:
        """
        Compute extraction accuracy metrics.
        
        Args:
            predictions: List of predicted entities
            ground_truths: List of actual entities
            
        Returns:
            Comprehensive accuracy metrics
        """
        if not predictions or not ground_truths:
            return AccuracyMetrics(
                precision=0.0, recall=0.0, f1_score=0.0, accuracy=0.0,
                confidence_mean=0.0, confidence_std=0.0, sample_count=0
            )
        
        # Field-level metrics
        field_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        confidences = []
        
        for pred, truth in zip(predictions, ground_truths):
            pred_dict = pred if isinstance(pred, dict) else {}
            truth_dict = truth if isinstance(truth, dict) else {}
            
            # Get all fields
            all_fields = set(pred_dict.keys()) | set(truth_dict.keys())
            
            for field in all_fields:
                if field.startswith('_'):
                    continue
                
                pred_val = self._normalize_value(pred_dict.get(field))
                truth_val = self._normalize_value(truth_dict.get(field))
                
                if pred_val and truth_val:
                    if self._values_match(pred_val, truth_val):
                        field_metrics[field]['tp'] += 1
                    else:
                        field_metrics[field]['fp'] += 1
                        field_metrics[field]['fn'] += 1
                elif pred_val and not truth_val:
                    field_metrics[field]['fp'] += 1
                elif truth_val and not pred_val:
                    field_metrics[field]['fn'] += 1
            
            if 'confidence' in pred_dict:
                confidences.append(pred_dict['confidence'])
        
        # Compute overall metrics
        total_tp = sum(m['tp'] for m in field_metrics.values())
        total_fp = sum(m['fp'] for m in field_metrics.values())
        total_fn = sum(m['fn'] for m in field_metrics.values())
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0
        
        # Per-field metrics
        by_field = {}
        for field, counts in field_metrics.items():
            tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
            field_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            field_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            field_f1 = 2 * field_precision * field_recall / (field_precision + field_recall) \
                       if (field_precision + field_recall) > 0 else 0
            
            by_field[field] = {
                'precision': field_precision,
                'recall': field_recall,
                'f1_score': field_f1,
                'support': tp + fn
            }
        
        return AccuracyMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            confidence_mean=statistics.mean(confidences) if confidences else 0,
            confidence_std=statistics.stdev(confidences) if len(confidences) > 1 else 0,
            sample_count=len(predictions),
            by_field=by_field
        )
    
    def compute_classification_accuracy(self, predictions: List[str], 
                                        ground_truths: List[str]) -> AccuracyMetrics:
        """
        Compute classification accuracy metrics.
        
        Args:
            predictions: Predicted document types
            ground_truths: Actual document types
            
        Returns:
            Classification accuracy metrics with confusion matrix
        """
        if not predictions or not ground_truths:
            return AccuracyMetrics(
                precision=0.0, recall=0.0, f1_score=0.0, accuracy=0.0,
                confidence_mean=0.0, confidence_std=0.0, sample_count=0
            )
        
        # Build confusion matrix
        all_classes = set(predictions) | set(ground_truths)
        confusion = {c: {c2: 0 for c2 in all_classes} for c in all_classes}
        
        correct = 0
        for pred, truth in zip(predictions, ground_truths):
            confusion[truth][pred] += 1
            if pred == truth:
                correct += 1
        
        # Per-class metrics
        by_field = {}
        for cls in all_classes:
            tp = confusion[cls][cls]
            fp = sum(confusion[other][cls] for other in all_classes if other != cls)
            fn = sum(confusion[cls][other] for other in all_classes if other != cls)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            by_field[cls] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': tp + fn
            }
        
        accuracy = correct / len(predictions) if predictions else 0
        
        # Macro averages
        macro_precision = statistics.mean(m['precision'] for m in by_field.values())
        macro_recall = statistics.mean(m['recall'] for m in by_field.values())
        macro_f1 = statistics.mean(m['f1_score'] for m in by_field.values())
        
        return AccuracyMetrics(
            precision=macro_precision,
            recall=macro_recall,
            f1_score=macro_f1,
            accuracy=accuracy,
            confidence_mean=0,
            confidence_std=0,
            sample_count=len(predictions),
            by_field=by_field,
            confusion_matrix=confusion
        )
    
    def _normalize_value(self, value: Any) -> Optional[str]:
        """Normalize value for comparison"""
        if value is None:
            return None
        
        if isinstance(value, list):
            return str(sorted([str(v).lower().strip() for v in value]))
        
        return str(value).lower().strip()
    
    def _values_match(self, pred: str, truth: str, threshold: float = 0.9) -> bool:
        """Check if predicted value matches ground truth"""
        if pred == truth:
            return True
        
        # Fuzzy matching for partial matches
        if pred in truth or truth in pred:
            overlap = len(set(pred) & set(truth)) / max(len(pred), len(truth))
            return overlap >= threshold
        
        return False
    
    def save_metrics(self, metrics: Dict, filename: str = None):
        """Save metrics to file"""
        if filename is None:
            filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.storage_path, filename)
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        logger.info(f"Metrics saved to {filepath}")


class PerformanceTracker:
    """Track processing performance and throughput"""
    
    def __init__(self, window_minutes: int = 60):
        """
        Initialize performance tracker.
        
        Args:
            window_minutes: Time window for calculating metrics
        """
        self.window_minutes = window_minutes
        self.processing_times: List[Tuple[datetime, float]] = []
        self.errors: List[Tuple[datetime, str]] = []
        self.successes: List[datetime] = []
        
        logger.info(f"PerformanceTracker initialized (window: {window_minutes} min)")
    
    def record_processing(self, duration_ms: float, success: bool = True, 
                         error_msg: Optional[str] = None):
        """
        Record a processing event.
        
        Args:
            duration_ms: Processing duration in milliseconds
            success: Whether processing was successful
            error_msg: Error message if not successful
        """
        now = datetime.now()
        
        if success:
            self.processing_times.append((now, duration_ms))
            self.successes.append(now)
        else:
            self.errors.append((now, error_msg or "Unknown error"))
        
        # Clean old data
        self._cleanup_old_data()
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        self._cleanup_old_data()
        
        times = [t[1] for t in self.processing_times]
        
        if not times:
            return PerformanceMetrics(
                avg_processing_time_ms=0,
                p50_processing_time_ms=0,
                p95_processing_time_ms=0,
                p99_processing_time_ms=0,
                throughput_per_minute=0,
                error_rate=0,
                success_count=0,
                error_count=0
            )
        
        sorted_times = sorted(times)
        n = len(sorted_times)
        
        # Calculate percentiles
        p50 = sorted_times[int(n * 0.50)] if n > 0 else 0
        p95 = sorted_times[int(n * 0.95)] if n > 0 else 0
        p99 = sorted_times[int(n * 0.99)] if n > 0 else 0
        
        # Throughput
        success_count = len(self.successes)
        error_count = len(self.errors)
        total = success_count + error_count
        
        throughput = success_count / self.window_minutes if success_count > 0 else 0
        error_rate = error_count / total if total > 0 else 0
        
        return PerformanceMetrics(
            avg_processing_time_ms=statistics.mean(times),
            p50_processing_time_ms=p50,
            p95_processing_time_ms=p95,
            p99_processing_time_ms=p99,
            throughput_per_minute=throughput,
            error_rate=error_rate,
            success_count=success_count,
            error_count=error_count
        )
    
    def _cleanup_old_data(self):
        """Remove data outside the tracking window"""
        cutoff = datetime.now() - timedelta(minutes=self.window_minutes)
        
        self.processing_times = [(t, d) for t, d in self.processing_times if t > cutoff]
        self.errors = [(t, e) for t, e in self.errors if t > cutoff]
        self.successes = [t for t in self.successes if t > cutoff]


class MetricsDashboard:
    """
    Central metrics dashboard for IDP system.
    Aggregates accuracy and performance metrics.
    """
    
    def __init__(self, storage_path: str = "data/metrics"):
        """Initialize metrics dashboard"""
        self.accuracy_tracker = AccuracyTracker(storage_path)
        self.performance_tracker = PerformanceTracker()
        
        # Component-specific trackers
        self.component_metrics = {
            'classifier': {'predictions': [], 'truths': [], 'confidences': []},
            'extractor': {'predictions': [], 'truths': [], 'confidences': []},
            'adjudicator': {'decisions': [], 'outcomes': []},
        }
        
        # Targets
        self.targets = {
            'extraction_accuracy': 0.97,
            'classification_accuracy': 0.99,
            'adjudication_accuracy': 0.95,
            'processing_time_p95_ms': 5000,
            'error_rate': 0.01
        }
        
        logger.info("MetricsDashboard initialized")
    
    def record_classification(self, predicted: str, actual: Optional[str], 
                             confidence: float):
        """Record classification result"""
        self.component_metrics['classifier']['predictions'].append(predicted)
        self.component_metrics['classifier']['confidences'].append(confidence)
        
        if actual:
            self.component_metrics['classifier']['truths'].append(actual)
    
    def record_extraction(self, extracted: Dict, ground_truth: Optional[Dict],
                         confidence: float, processing_time_ms: float):
        """Record extraction result"""
        self.component_metrics['extractor']['predictions'].append(extracted)
        self.component_metrics['extractor']['confidences'].append(confidence)
        
        if ground_truth:
            self.component_metrics['extractor']['truths'].append(ground_truth)
        
        self.performance_tracker.record_processing(processing_time_ms)
    
    def record_adjudication(self, decision: str, outcome: Optional[str] = None):
        """Record adjudication result"""
        self.component_metrics['adjudicator']['decisions'].append(decision)
        
        if outcome:
            self.component_metrics['adjudicator']['outcomes'].append(outcome)
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get comprehensive dashboard summary"""
        # Classification metrics
        classifier = self.component_metrics['classifier']
        if classifier['predictions'] and classifier['truths']:
            classification_metrics = self.accuracy_tracker.compute_classification_accuracy(
                classifier['predictions'][-100:],  # Last 100
                classifier['truths'][-100:]
            )
        else:
            classification_metrics = None
        
        # Extraction metrics
        extractor = self.component_metrics['extractor']
        if extractor['predictions'] and extractor['truths']:
            extraction_metrics = self.accuracy_tracker.compute_extraction_accuracy(
                extractor['predictions'][-100:],
                extractor['truths'][-100:]
            )
        else:
            extraction_metrics = None
        
        # Performance metrics
        performance_metrics = self.performance_tracker.get_metrics()
        
        # Target comparison
        target_status = self._check_targets(
            classification_metrics, extraction_metrics, performance_metrics
        )
        
        return {
            'timestamp': datetime.now().isoformat(),
            'classification': {
                'accuracy': classification_metrics.accuracy if classification_metrics else None,
                'precision': classification_metrics.precision if classification_metrics else None,
                'f1_score': classification_metrics.f1_score if classification_metrics else None,
                'sample_count': classification_metrics.sample_count if classification_metrics else 0,
                'by_class': classification_metrics.by_field if classification_metrics else {}
            },
            'extraction': {
                'accuracy': extraction_metrics.accuracy if extraction_metrics else None,
                'precision': extraction_metrics.precision if extraction_metrics else None,
                'recall': extraction_metrics.recall if extraction_metrics else None,
                'f1_score': extraction_metrics.f1_score if extraction_metrics else None,
                'confidence_mean': extraction_metrics.confidence_mean if extraction_metrics else None,
                'sample_count': extraction_metrics.sample_count if extraction_metrics else 0,
                'by_field': extraction_metrics.by_field if extraction_metrics else {}
            },
            'performance': {
                'avg_processing_time_ms': performance_metrics.avg_processing_time_ms,
                'p95_processing_time_ms': performance_metrics.p95_processing_time_ms,
                'throughput_per_minute': performance_metrics.throughput_per_minute,
                'error_rate': performance_metrics.error_rate,
                'success_count': performance_metrics.success_count,
                'error_count': performance_metrics.error_count
            },
            'targets': target_status,
            'health_status': self._calculate_health_status(target_status)
        }
    
    def _check_targets(self, classification: Optional[AccuracyMetrics],
                      extraction: Optional[AccuracyMetrics],
                      performance: PerformanceMetrics) -> Dict[str, Dict]:
        """Check metrics against targets"""
        status = {}
        
        if extraction:
            status['extraction_accuracy'] = {
                'target': self.targets['extraction_accuracy'],
                'actual': extraction.accuracy,
                'met': extraction.accuracy >= self.targets['extraction_accuracy'],
                'gap': extraction.accuracy - self.targets['extraction_accuracy']
            }
        
        if classification:
            status['classification_accuracy'] = {
                'target': self.targets['classification_accuracy'],
                'actual': classification.accuracy,
                'met': classification.accuracy >= self.targets['classification_accuracy'],
                'gap': classification.accuracy - self.targets['classification_accuracy']
            }
        
        status['processing_time'] = {
            'target': self.targets['processing_time_p95_ms'],
            'actual': performance.p95_processing_time_ms,
            'met': performance.p95_processing_time_ms <= self.targets['processing_time_p95_ms'],
            'gap': self.targets['processing_time_p95_ms'] - performance.p95_processing_time_ms
        }
        
        status['error_rate'] = {
            'target': self.targets['error_rate'],
            'actual': performance.error_rate,
            'met': performance.error_rate <= self.targets['error_rate'],
            'gap': self.targets['error_rate'] - performance.error_rate
        }
        
        return status
    
    def _calculate_health_status(self, targets: Dict) -> str:
        """Calculate overall system health"""
        met_count = sum(1 for t in targets.values() if t.get('met', False))
        total = len(targets)
        
        if total == 0:
            return 'unknown'
        
        ratio = met_count / total
        
        if ratio >= 0.9:
            return 'healthy'
        elif ratio >= 0.7:
            return 'degraded'
        else:
            return 'unhealthy'
    
    def export_report(self, filepath: str = None) -> str:
        """Export metrics report"""
        if filepath is None:
            filepath = f"data/metrics/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        report = self.get_dashboard_summary()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Metrics report exported to {filepath}")
        return filepath


# Global dashboard instance
_dashboard = None

def get_dashboard() -> MetricsDashboard:
    """Get global metrics dashboard instance"""
    global _dashboard
    if _dashboard is None:
        _dashboard = MetricsDashboard()
    return _dashboard
