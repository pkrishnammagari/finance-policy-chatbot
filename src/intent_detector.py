"""
Intent detection and policy selection for Finance House Policy Chatbot
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import json
import re
from typing import List, Dict, Any, Tuple
from collections import Counter
from langchain_ollama import ChatOllama
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from src.utils import Config, Timer, logger


class IntentAnalysis(BaseModel):
    """Output format for intent analysis"""
    primary_intent: str = Field(description="Main intent of the query")
    domain: str = Field(description="Policy domain (HR, IT, Finance, Corporate)")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Brief explanation of the classification")


class PolicySelection(BaseModel):
    """Output format for policy selection"""
    selected_policy: str = Field(description="Selected policy number (e.g., POL-HR-002)")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Why this policy was selected")
    related_policies: List[str] = Field(description="Other relevant policies", default_factory=list)


class IntentDetector:
    """
    Detects user intent and classifies queries by domain
    """
    
    def __init__(self):
        self.llm = ChatOllama(
            model=Config.LLM_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=0.2,
            format='json'  # Request JSON output
        )
        
        self.intent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at analyzing user queries about organizational policies.

POLICY DOMAINS:
1. Human Resources (HR): Remote work, leave/vacation, performance reviews, professional development, training budgets
2. Information Technology (IT): Equipment procurement, laptops, data security, technology policies
3. Finance (FIN): Travel expenses, reimbursements, budgets, financial policies
4. Corporate/Compliance (COR): Code of conduct, ethics, conflicts of interest, whistleblower protection, gifts

Your task is to analyze the user's query and return a JSON object with:
- primary_intent: what the user is trying to find out
- domain: most relevant domain (HR, IT, FIN, or COR)
- confidence: confidence score between 0.0 and 1.0
- reasoning: brief explanation

Output ONLY valid JSON, no other text."""),
            ("user", "User query: {query}")
        ])
        
    def detect_intent(self, query: str) -> Tuple[IntentAnalysis, float]:
        """
        Detect the intent of a user query
        
        Args:
            query: User query
            
        Returns:
            Tuple of (IntentAnalysis object, duration)
        """
        start_time = time.time()
        
        try:
            prompt = self.intent_prompt.format_messages(query=query)
            response = self.llm.invoke(prompt)
            
            # Extract JSON from response
            content = response.content.strip()
            
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                intent = IntentAnalysis(
                    primary_intent=data.get("primary_intent", "Unknown"),
                    domain=data.get("domain", "Unknown"),
                    confidence=float(data.get("confidence", 0.5)),
                    reasoning=data.get("reasoning", "")
                )
            else:
                # Fallback: try to parse entire content as JSON
                data = json.loads(content)
                intent = IntentAnalysis(**data)
            
            duration = time.time() - start_time
            logger.info(f"Intent detected: {intent.domain} (confidence: {intent.confidence:.2f}) in {duration:.2f}s")
            
            return intent, duration
            
        except Exception as e:
            logger.warning(f"Error detecting intent: {e}, using heuristic fallback")
            duration = time.time() - start_time
            
            # Heuristic fallback based on keywords
            query_lower = query.lower()
            
            if any(word in query_lower for word in ['remote', 'home', 'work from home', 'leave', 'vacation', 'performance', 'training', 'development']):
                domain = "HR"
            elif any(word in query_lower for word in ['laptop', 'equipment', 'computer', 'device', 'security', 'data', 'technology']):
                domain = "IT"
            elif any(word in query_lower for word in ['expense', 'travel', 'reimbursement', 'budget', 'cost', 'financial']):
                domain = "FIN"
            elif any(word in query_lower for word in ['gift', 'ethics', 'conduct', 'conflict', 'compliance', 'whistleblower']):
                domain = "COR"
            else:
                domain = "Unknown"
            
            intent = IntentAnalysis(
                primary_intent="Policy information request",
                domain=domain,
                confidence=0.7,
                reasoning=f"Classified using keyword heuristic due to parsing error"
            )
            
            return intent, duration


class PolicySelector:
    """
    Selects the most relevant policy from retrieved documents
    """
    
    def __init__(self):
        self.llm = ChatOllama(
            model=Config.LLM_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=0.2,
            format='json'
        )
        
        self.selection_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at selecting the most relevant policy document for a user's question.

You will be given:
1. The user's original query
2. The detected intent and domain
3. A list of retrieved policy documents with metadata

Your task is to select THE SINGLE MOST RELEVANT policy that directly answers the user's question.

Return a JSON object with:
- selected_policy: the policy number (e.g., "POL-HR-002")
- confidence: confidence score (0.0 to 1.0)
- reasoning: why this policy was selected
- related_policies: list of other relevant policy numbers

Output ONLY valid JSON, no other text."""),
            ("user", """Query: {query}
Detected Intent: {intent}
Domain: {domain}

Retrieved Policies:
{policy_list}

Select the most relevant policy.""")
        ])
        
    def select_policy(self, query: str, intent: IntentAnalysis, docs: List[Document]) -> Tuple[PolicySelection, float]:
        """
        Select the best policy from retrieved documents
        
        Args:
            query: User query
            intent: Detected intent
            docs: Retrieved documents
            
        Returns:
            Tuple of (PolicySelection object, duration)
        """
        start_time = time.time()
        
        try:
            # Build policy list from documents
            policy_info = {}
            for doc in docs:
                policy_num = doc.metadata.get("policy_number", "Unknown")
                if policy_num not in policy_info:
                    policy_info[policy_num] = {
                        "policy_number": policy_num,
                        "domain": doc.metadata.get("domain", "Unknown"),
                        "filename": doc.metadata.get("filename", "Unknown"),
                        "chunk_count": 0
                    }
                policy_info[policy_num]["chunk_count"] += 1
            
            # Format policy list
            policy_list = "\n".join([
                f"- {info['policy_number']} ({info['domain']}): {info['chunk_count']} relevant sections"
                for info in policy_info.values()
            ])
            
            prompt = self.selection_prompt.format_messages(
                query=query,
                intent=intent.primary_intent,
                domain=intent.domain,
                policy_list=policy_list
            )
            
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # Extract JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
            else:
                data = json.loads(content)
            
            selection = PolicySelection(
                selected_policy=data.get("selected_policy", "Unknown"),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
                related_policies=data.get("related_policies", [])
            )
            
            duration = time.time() - start_time
            logger.info(f"Selected policy: {selection.selected_policy} (confidence: {selection.confidence:.2f}) in {duration:.2f}s")
            
            return selection, duration
            
        except Exception as e:
            logger.warning(f"Error selecting policy: {e}, using frequency heuristic")
            duration = time.time() - start_time
            
            # Fallback: use most frequent policy
            policy_counts = Counter([doc.metadata.get("policy_number", "Unknown") for doc in docs])
            most_common_policy = policy_counts.most_common(1)[0][0] if policy_counts else "Unknown"
            
            return PolicySelection(
                selected_policy=most_common_policy,
                confidence=0.7,
                reasoning=f"Selected based on frequency (appears {policy_counts[most_common_policy]} times)",
                related_policies=list(policy_info.keys())
            ), duration
    
    def filter_documents_by_policy(self, docs: List[Document], policy_number: str) -> List[Document]:
        """
        Filter documents to only those from the selected policy
        
        Args:
            docs: All retrieved documents
            policy_number: Selected policy number
            
        Returns:
            Filtered list of documents
        """
        filtered = [doc for doc in docs if doc.metadata.get("policy_number") == policy_number]
        logger.info(f"Filtered to {len(filtered)} documents from {policy_number}")
        return filtered


def test_intent_detection():
    """Test intent detection and policy selection"""
    print("=" * 60)
    print("TESTING INTENT DETECTION & POLICY SELECTION")
    print("=" * 60)
    
    detector = IntentDetector()
    
    test_queries = [
        "Can I work from home full-time?",
        "What laptop budget do I have?",
        "How do I claim travel expenses?",
        "Can I accept gifts from clients?"
    ]
    
    for query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print("=" * 60)
        
        intent, duration = detector.detect_intent(query)
        
        print(f"\nIntent Analysis:")
        print(f"  Primary Intent: {intent.primary_intent}")
        print(f"  Domain: {intent.domain}")
        print(f"  Confidence: {intent.confidence:.2%}")
        print(f"  Reasoning: {intent.reasoning}")
        print(f"  Duration: {duration:.2f}s")


if __name__ == "__main__":
    test_intent_detection()