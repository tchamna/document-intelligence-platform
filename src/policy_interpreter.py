"""
Policy Interpretation Module
NLP and LLM-based policy clause interpretation
"""

import json
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class PolicyInterpreter:
    """
    Interpret policy clauses using NLP and LLM.
    Extract and explain policy terms, conditions, and exclusions.
    """
    
    def __init__(self, bedrock_client=None):
        """
        Initialize the policy interpreter.
        
        Args:
            bedrock_client: Optional pre-configured boto3 bedrock client
        """
        self.bedrock_client = bedrock_client
        
        # Common policy terms to identify
        self.policy_terms = {
            'elimination_period': [
                'elimination period', 'waiting period', 'qualifying period'
            ],
            'benefit_period': [
                'benefit period', 'maximum duration', 'benefit duration'
            ],
            'own_occupation': [
                'own occupation', 'regular occupation', 'own occ'
            ],
            'any_occupation': [
                'any occupation', 'any occ', 'gainful occupation'
            ],
            'pre_existing': [
                'pre-existing', 'preexisting', 'prior condition'
            ],
            'exclusions': [
                'exclusion', 'not covered', 'excluded', 'exception'
            ]
        }
        
        logger.info("PolicyInterpreter initialized")
    
    def interpret(self, policy_text: str) -> Dict:
        """
        Interpret policy document and extract key clauses.
        
        Args:
            policy_text: Full text of policy document
            
        Returns:
            Dictionary of interpreted policy terms
        """
        logger.info("Starting policy interpretation")
        
        interpretation = {
            'clauses': {},
            'terms': {},
            'exclusions': [],
            'summary': ''
        }
        
        # Rule-based extraction
        for term, keywords in self.policy_terms.items():
            clause = self._extract_clause(policy_text, keywords)
            if clause:
                interpretation['clauses'][term] = clause
        
        # LLM-based interpretation if available
        if self.bedrock_client:
            try:
                llm_interpretation = self._llm_interpret(policy_text)
                interpretation.update(llm_interpretation)
            except Exception as e:
                logger.warning(f"LLM interpretation failed: {e}")
        
        logger.info(f"Extracted {len(interpretation['clauses'])} policy clauses")
        return interpretation
    
    def _extract_clause(self, text: str, keywords: List[str]) -> Optional[str]:
        """Extract clause containing any of the keywords"""
        import re
        
        text_lower = text.lower()
        
        for keyword in keywords:
            if keyword in text_lower:
                # Find the sentence/paragraph containing the keyword
                pattern = r'[^.]*' + re.escape(keyword) + r'[^.]*\.'
                match = re.search(pattern, text_lower)
                if match:
                    return match.group(0).strip()
        
        return None
    
    def _llm_interpret(self, policy_text: str) -> Dict:
        """Use LLM for detailed policy interpretation"""
        
        prompt = f"""Analyze this insurance policy and extract key terms:

Policy text (excerpt):
{policy_text[:3000]}

Provide a structured interpretation in JSON format:
{{
    "elimination_period": "number of days/description",
    "benefit_period": "duration description",
    "benefit_percentage": "percentage of earnings",
    "own_occupation_period": "duration",
    "any_occupation_after": "when it applies",
    "pre_existing_condition_clause": "full description",
    "exclusions": ["list of exclusions"],
    "key_terms": {{"term": "definition"}},
    "summary": "brief policy summary"
}}"""

        try:
            response = self.bedrock_client.invoke_model(
                modelId='anthropic.claude-3-sonnet-20240229-v1:0',
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1500,
                    "temperature": 0.1
                })
            )
            
            result = json.loads(response['body'].read())
            content = result['content'][0]['text']
            return json.loads(content)
        
        except Exception as e:
            logger.error(f"LLM policy interpretation error: {e}")
            raise
    
    def compare_to_claim(self, policy_interpretation: Dict, claim_data: Dict) -> Dict:
        """
        Compare policy terms to claim data for adjudication support.
        
        Args:
            policy_interpretation: Interpreted policy terms
            claim_data: Extracted claim data
            
        Returns:
            Comparison results with potential issues
        """
        comparison = {
            'matches': [],
            'issues': [],
            'recommendations': []
        }
        
        # Check elimination period
        if 'elimination_period' in policy_interpretation.get('clauses', {}):
            comparison['matches'].append('Elimination period identified')
        
        # Check for exclusions that might apply
        exclusions = policy_interpretation.get('exclusions', [])
        diagnosis = claim_data.get('diagnosis', '').lower()
        
        for exclusion in exclusions:
            if any(word in diagnosis for word in exclusion.lower().split()):
                comparison['issues'].append(f"Potential exclusion applies: {exclusion}")
        
        return comparison
