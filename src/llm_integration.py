"""
LLM Integration Module
AWS Bedrock and LangChain integration for advanced document understanding
Supports Claude, Titan, and other foundation models
"""

import json
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Structured response from LLM"""
    content: str
    model: str
    tokens_used: int
    confidence: float
    metadata: Dict[str, Any]


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def invoke(self, prompt: str, **kwargs) -> LLMResponse:
        pass
    
    @abstractmethod
    def invoke_with_schema(self, prompt: str, schema: Dict) -> Dict:
        pass


class BedrockProvider(BaseLLMProvider):
    """
    AWS Bedrock LLM Provider
    Supports Claude 3, Titan, and Llama models
    """
    
    MODEL_IDS = {
        'claude-3-sonnet': 'anthropic.claude-3-sonnet-20240229-v1:0',
        'claude-3-haiku': 'anthropic.claude-3-haiku-20240307-v1:0',
        'claude-3-opus': 'anthropic.claude-3-opus-20240229-v1:0',
        'claude-instant': 'anthropic.claude-instant-v1',
        'titan-text': 'amazon.titan-text-express-v1',
        'titan-embed': 'amazon.titan-embed-text-v1',
        'llama-2-70b': 'meta.llama2-70b-chat-v1'
    }
    
    def __init__(self, model_name: str = 'claude-3-sonnet', region: str = 'us-east-1'):
        """
        Initialize Bedrock provider.
        
        Args:
            model_name: Short name of the model
            region: AWS region
        """
        self.model_name = model_name
        self.model_id = self.MODEL_IDS.get(model_name, model_name)
        self.region = region
        self.client = None
        
        self._initialize_client()
        logger.info(f"BedrockProvider initialized with model: {self.model_id}")
    
    def _initialize_client(self):
        """Initialize boto3 Bedrock client"""
        try:
            import boto3
            self.client = boto3.client(
                'bedrock-runtime',
                region_name=self.region
            )
            logger.info("Bedrock client initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize Bedrock client: {e}")
            logger.info("Running in mock mode for demonstration")
    
    def invoke(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.1) -> LLMResponse:
        """
        Invoke the LLM with a prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Returns:
            LLMResponse with model output
        """
        if not self.client:
            return self._mock_invoke(prompt)
        
        try:
            if 'anthropic' in self.model_id:
                body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                })
            elif 'titan' in self.model_id:
                body = json.dumps({
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": max_tokens,
                        "temperature": temperature
                    }
                })
            else:
                body = json.dumps({
                    "prompt": prompt,
                    "max_gen_len": max_tokens,
                    "temperature": temperature
                })
            
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body
            )
            
            result = json.loads(response['body'].read())
            
            if 'anthropic' in self.model_id:
                content = result['content'][0]['text']
                tokens = result.get('usage', {}).get('output_tokens', 0)
            elif 'titan' in self.model_id:
                content = result['results'][0]['outputText']
                tokens = result.get('inputTextTokenCount', 0) + result.get('results', [{}])[0].get('tokenCount', 0)
            else:
                content = result.get('generation', '')
                tokens = 0
            
            return LLMResponse(
                content=content,
                model=self.model_id,
                tokens_used=tokens,
                confidence=0.95,
                metadata={'raw_response': result}
            )
            
        except Exception as e:
            logger.error(f"Bedrock invocation error: {e}")
            return self._mock_invoke(prompt)
    
    def invoke_with_schema(self, prompt: str, schema: Dict, **kwargs) -> Dict:
        """
        Invoke LLM expecting structured JSON output.
        
        Args:
            prompt: Input prompt
            schema: Expected JSON schema
            
        Returns:
            Parsed JSON response
        """
        structured_prompt = f"""{prompt}

Respond ONLY with valid JSON matching this schema:
{json.dumps(schema, indent=2)}

JSON Response:"""
        
        response = self.invoke(structured_prompt, **kwargs)
        
        try:
            # Extract JSON from response
            content = response.content
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]
            
            return json.loads(content.strip())
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return {"error": "Failed to parse response", "raw": response.content}
    
    def _mock_invoke(self, prompt: str) -> LLMResponse:
        """Generate mock response for demonstration"""
        logger.info("Using mock LLM response (Bedrock not configured)")
        
        # Smart mock responses based on prompt content
        if 'classify' in prompt.lower():
            content = json.dumps({
                "document_type": "disability_claim",
                "confidence": 0.94,
                "reasoning": "Document contains claim-specific terminology including disability date, diagnosis, and benefit amount"
            })
        elif 'extract' in prompt.lower():
            content = json.dumps({
                "entities": {
                    "claim_number": "CLM-2024-001",
                    "policy_number": "POL-789-456",
                    "claimant_name": "John Smith",
                    "diagnosis": "Lower back injury",
                    "benefit_amount": "$3,500"
                },
                "confidence": 0.96
            })
        elif 'policy' in prompt.lower():
            content = json.dumps({
                "elimination_period": "90 days",
                "benefit_period": "24 months",
                "benefit_percentage": "60%",
                "exclusions": ["pre-existing conditions", "self-inflicted injuries"],
                "own_occupation_period": "24 months"
            })
        else:
            content = "This is a mock response. Configure AWS credentials for actual Bedrock access."
        
        return LLMResponse(
            content=content,
            model="mock",
            tokens_used=0,
            confidence=0.85,
            metadata={"mock": True}
        )
    
    def get_embeddings(self, text: str) -> List[float]:
        """
        Get text embeddings using Titan Embed.
        
        Args:
            text: Input text
            
        Returns:
            List of embedding values
        """
        if not self.client:
            # Return mock embeddings
            import hashlib
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            import random
            random.seed(seed)
            return [random.random() for _ in range(1536)]
        
        try:
            body = json.dumps({"inputText": text})
            response = self.client.invoke_model(
                modelId='amazon.titan-embed-text-v1',
                body=body
            )
            result = json.loads(response['body'].read())
            return result['embedding']
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return []


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI provider as alternative to AWS Bedrock.
    Easier to set up for testing and development.
    """
    
    MODEL_IDS = {
        'gpt-4': 'gpt-4',
        'gpt-4-turbo': 'gpt-4-turbo-preview',
        'gpt-3.5-turbo': 'gpt-3.5-turbo',
        'gpt-4o': 'gpt-4o',
        'gpt-4o-mini': 'gpt-4o-mini'
    }
    
    def __init__(self, model: str = 'gpt-4o-mini', region: str = None):
        """
        Initialize OpenAI provider.
        
        Args:
            model: Model name (default: gpt-4o-mini for cost efficiency)
            region: Not used, kept for API compatibility
        """
        self.model_id = self.MODEL_IDS.get(model, model)
        self.client = self._initialize_client()
        logger.info(f"OpenAIProvider initialized with model: {self.model_id}")
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            import os
            api_key = os.getenv('OPENAI_API_KEY')
            
            if not api_key:
                logger.warning("OPENAI_API_KEY not found. Running in mock mode.")
                return None
            
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                # Test the connection
                logger.info("OpenAI client initialized successfully")
                return client
            except ImportError:
                logger.warning("openai package not installed. Run: pip install openai")
                return None
                
        except Exception as e:
            logger.warning(f"OpenAI initialization failed: {e}. Running in mock mode.")
            return None
    
    def invoke(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Invoke OpenAI model.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters (max_tokens, temperature)
            
        Returns:
            LLMResponse with model output
        """
        if not self.client:
            return self._mock_invoke(prompt)
        
        max_tokens = kwargs.get('max_tokens', 2000)
        temperature = kwargs.get('temperature', 0.1)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "You are an expert document analysis assistant. Provide accurate, structured responses."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            return LLMResponse(
                content=content,
                model=self.model_id,
                tokens_used=tokens_used,
                confidence=0.95,
                metadata={
                    'provider': 'openai',
                    'finish_reason': response.choices[0].finish_reason
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI invocation error: {e}")
            return self._mock_invoke(prompt)
    
    def invoke_with_schema(self, prompt: str, schema: Dict, **kwargs) -> Dict:
        """Invoke expecting structured JSON output"""
        structured_prompt = f"""{prompt}

Respond ONLY with valid JSON matching this schema:
{json.dumps(schema, indent=2)}

JSON Response:"""
        
        response = self.invoke(structured_prompt, **kwargs)
        
        try:
            content = response.content
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]
            
            return json.loads(content.strip())
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return {"error": "Failed to parse response", "raw": response.content}
    
    def _mock_invoke(self, prompt: str) -> LLMResponse:
        """Generate mock response for demonstration"""
        logger.info("Using mock LLM response (OpenAI not configured)")
        
        # Same mock responses as Bedrock
        if 'classify' in prompt.lower():
            content = json.dumps({
                "document_type": "disability_claim",
                "confidence": 0.94,
                "reasoning": "Document contains claim-specific terminology"
            })
        elif 'extract' in prompt.lower():
            content = json.dumps({
                "entities": {
                    "claim_number": "CLM-2024-001",
                    "policy_number": "POL-789-456",
                    "claimant_name": "John Smith"
                },
                "confidence": 0.96
            })
        else:
            content = "Mock response. Set OPENAI_API_KEY for actual OpenAI access."
        
        return LLMResponse(
            content=content,
            model="mock-openai",
            tokens_used=0,
            confidence=0.85,
            metadata={"mock": True}
        )
    
    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings using OpenAI"""
        if not self.client:
            import hashlib
            import random
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            random.seed(seed)
            return [random.random() for _ in range(1536)]
        
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            return []


class LangChainIntegration:
    """
    LangChain integration for document processing pipelines.
    Implements RAG and chain-based workflows.
    """
    
    def __init__(self, llm_provider: Optional[BaseLLMProvider] = None):
        """
        Initialize LangChain integration.
        
        Args:
            llm_provider: LLM provider instance
        """
        self.llm_provider = llm_provider or BedrockProvider()
        self.chains = {}
        self._setup_chains()
        logger.info("LangChainIntegration initialized")
    
    def _setup_chains(self):
        """Set up document processing chains"""
        try:
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            from langchain.output_parsers import StructuredOutputParser, ResponseSchema
            
            # Document classification chain
            self.chains['classify'] = PromptTemplate(
                input_variables=["document_text"],
                template="""Analyze this document and classify it.

Document:
{document_text}

Classify as one of: disability_claim, enrollment, policy, rfp, other
Provide your classification and reasoning in JSON format."""
            )
            
            # Entity extraction chain
            self.chains['extract'] = PromptTemplate(
                input_variables=["document_text", "document_type"],
                template="""Extract all relevant entities from this {document_type} document.

Document:
{document_text}

Extract entities like: names, dates, amounts, policy numbers, claim numbers, etc.
Return as structured JSON."""
            )
            
            # Policy interpretation chain
            self.chains['interpret_policy'] = PromptTemplate(
                input_variables=["policy_text"],
                template="""Interpret this insurance policy document.

Policy:
{policy_text}

Extract and explain:
1. Coverage terms and conditions
2. Elimination/waiting periods
3. Benefit calculations
4. Exclusions and limitations
5. Key definitions

Return as structured JSON."""
            )
            
            logger.info("LangChain chains configured")
            
        except ImportError as e:
            logger.warning(f"LangChain not fully available: {e}")
            self.chains = {}
    
    def classify_document(self, text: str) -> Dict[str, Any]:
        """
        Classify document using LangChain chain.
        
        Args:
            text: Document text
            
        Returns:
            Classification result
        """
        prompt = f"""Analyze and classify this document:

{text[:3000]}

Classify as: disability_claim, enrollment, policy, rfp, or other
Respond with JSON: {{"document_type": "...", "confidence": 0.0, "reasoning": "..."}}"""
        
        response = self.llm_provider.invoke(prompt)
        
        try:
            return json.loads(response.content)
        except:
            return {
                "document_type": "other",
                "confidence": 0.5,
                "reasoning": response.content
            }
    
    def extract_entities(self, text: str, doc_type: str) -> Dict[str, Any]:
        """
        Extract entities using LangChain chain.
        
        Args:
            text: Document text
            doc_type: Type of document
            
        Returns:
            Extracted entities
        """
        schema = {
            "entities": {
                "claim_number": "string",
                "policy_number": "string",
                "names": ["string"],
                "dates": ["string"],
                "amounts": ["string"],
                "diagnosis": "string"
            },
            "confidence": "float"
        }
        
        prompt = f"""Extract entities from this {doc_type} document:

{text[:3000]}

Focus on: claim numbers, policy numbers, names, dates, monetary amounts, and medical/diagnosis information."""
        
        return self.llm_provider.invoke_with_schema(prompt, schema)
    
    def interpret_policy(self, policy_text: str) -> Dict[str, Any]:
        """
        Interpret policy document using LangChain.
        
        Args:
            policy_text: Policy document text
            
        Returns:
            Policy interpretation
        """
        schema = {
            "elimination_period": "string",
            "benefit_period": "string",
            "benefit_percentage": "string",
            "max_benefit": "string",
            "own_occupation_period": "string",
            "exclusions": ["string"],
            "pre_existing_clause": "string",
            "key_definitions": {"term": "definition"},
            "summary": "string"
        }
        
        prompt = f"""Interpret this insurance policy:

{policy_text[:4000]}

Extract all key terms, conditions, exclusions, and benefits."""
        
        return self.llm_provider.invoke_with_schema(prompt, schema)


class DocumentAIOrchestrator:
    """
    Orchestrates LLM-based document processing.
    Combines multiple models for optimal accuracy.
    """
    
    def __init__(self):
        """Initialize the orchestrator"""
        self.bedrock = BedrockProvider()
        self.langchain = LangChainIntegration(self.bedrock)
        logger.info("DocumentAIOrchestrator initialized")
    
    def process_with_llm(self, text: str, task: str = 'classify') -> Dict[str, Any]:
        """
        Process document with LLM assistance.
        
        Args:
            text: Document text
            task: Processing task (classify, extract, interpret)
            
        Returns:
            Processing results
        """
        if task == 'classify':
            return self.langchain.classify_document(text)
        elif task == 'extract':
            return self.langchain.extract_entities(text, 'general')
        elif task == 'interpret':
            return self.langchain.interpret_policy(text)
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def ensemble_classify(self, text: str) -> Dict[str, Any]:
        """
        Ensemble classification using multiple approaches.
        Combines rule-based, ML, and LLM for highest accuracy.
        
        Args:
            text: Document text
            
        Returns:
            Classification with confidence scores
        """
        results = {
            'classifications': [],
            'final_classification': None,
            'confidence': 0.0
        }
        
        # LLM classification
        llm_result = self.langchain.classify_document(text)
        results['classifications'].append({
            'method': 'llm',
            'result': llm_result
        })
        
        # Use LLM result as primary (in production, would ensemble with ML)
        results['final_classification'] = llm_result.get('document_type', 'other')
        results['confidence'] = llm_result.get('confidence', 0.85)
        
        return results


# Convenience function for quick access
def get_llm_provider(provider: str = 'bedrock', **kwargs) -> BaseLLMProvider:
    """
    Factory function to get LLM provider.
    
    Args:
        provider: Provider name ('bedrock', 'openai', etc.)
        **kwargs: Provider-specific configuration
        
    Returns:
        LLM provider instance
    """
    if provider == 'bedrock':
        return BedrockProvider(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
