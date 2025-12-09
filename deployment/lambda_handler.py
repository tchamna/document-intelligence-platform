"""
AWS Lambda Handler
Serverless deployment for Healthcare IDP System
"""

import json
import base64
import logging
from typing import Dict, Any

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Import pipeline (lazy loading for cold start optimization)
_pipeline = None


def get_pipeline():
    """Lazy load pipeline for Lambda cold start optimization"""
    global _pipeline
    if _pipeline is None:
        from src.pipeline import IDPPipeline
        _pipeline = IDPPipeline()
    return _pipeline


def lambda_handler(event: Dict, context: Any) -> Dict:
    """
    AWS Lambda handler for document processing.
    
    Supports:
    - API Gateway REST API events
    - Direct invocation
    - S3 trigger events
    
    Args:
        event: Lambda event object
        context: Lambda context object
        
    Returns:
        Response object with processing results
    """
    logger.info(f"Received event: {json.dumps(event)[:500]}")
    
    try:
        # Determine event source and extract document
        document_text = None
        metadata = {}
        
        # API Gateway event
        if 'httpMethod' in event:
            return handle_api_gateway_event(event)
        
        # S3 trigger event
        elif 'Records' in event and event['Records'][0].get('eventSource') == 'aws:s3':
            return handle_s3_event(event)
        
        # Direct invocation
        elif 'document_text' in event:
            document_text = event['document_text']
            metadata = event.get('metadata', {})
        
        else:
            return create_response(400, {'error': 'Unsupported event format'})
        
        # Process document
        if document_text:
            pipeline = get_pipeline()
            result = pipeline.process_document(document_text, metadata)
            return create_response(200, result)
        
        return create_response(400, {'error': 'No document text provided'})
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return create_response(500, {'error': str(e)})


def handle_api_gateway_event(event: Dict) -> Dict:
    """Handle API Gateway REST API events"""
    
    method = event.get('httpMethod', '')
    path = event.get('path', '')
    
    # Health check
    if path == '/health' and method == 'GET':
        return create_response(200, {
            'status': 'healthy',
            'version': '1.0.0'
        })
    
    # Process document
    if path == '/process' and method == 'POST':
        body = event.get('body', '{}')
        
        # Handle base64 encoded body
        if event.get('isBase64Encoded'):
            body = base64.b64decode(body).decode('utf-8')
        
        data = json.loads(body)
        document_text = data.get('text', '')
        metadata = data.get('metadata', {})
        
        if not document_text:
            return create_response(400, {'error': 'No document text provided'})
        
        pipeline = get_pipeline()
        result = pipeline.process_document(document_text, metadata)
        return create_response(200, result)
    
    # Classify only
    if path == '/classify' and method == 'POST':
        body = event.get('body', '{}')
        if event.get('isBase64Encoded'):
            body = base64.b64decode(body).decode('utf-8')
        
        data = json.loads(body)
        text = data.get('text', '')
        
        from src.document_classifier import DocumentClassifier
        classifier = DocumentClassifier()
        doc_type, confidence = classifier.classify(text)
        
        return create_response(200, {
            'document_type': doc_type,
            'confidence': confidence
        })
    
    return create_response(404, {'error': 'Endpoint not found'})


def handle_s3_event(event: Dict) -> Dict:
    """Handle S3 trigger events for automatic document processing"""
    import boto3
    
    s3 = boto3.client('s3')
    results = []
    
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        
        logger.info(f"Processing S3 object: s3://{bucket}/{key}")
        
        try:
            # Get document from S3
            response = s3.get_object(Bucket=bucket, Key=key)
            document_text = response['Body'].read().decode('utf-8')
            
            # Process document
            pipeline = get_pipeline()
            result = pipeline.process_document(
                document_text,
                metadata={'source': 's3', 'bucket': bucket, 'key': key}
            )
            
            # Optionally store results back to S3
            result_key = key.replace('.txt', '_result.json').replace('.pdf', '_result.json')
            s3.put_object(
                Bucket=bucket,
                Key=f"processed/{result_key}",
                Body=json.dumps(result, indent=2),
                ContentType='application/json'
            )
            
            results.append({
                'source': f"s3://{bucket}/{key}",
                'status': result['status']
            })
            
        except Exception as e:
            logger.error(f"Error processing S3 object {key}: {str(e)}")
            results.append({
                'source': f"s3://{bucket}/{key}",
                'status': 'error',
                'error': str(e)
            })
    
    return create_response(200, {'results': results})


def create_response(status_code: int, body: Dict) -> Dict:
    """Create API Gateway compatible response"""
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS'
        },
        'body': json.dumps(body, default=str)
    }
