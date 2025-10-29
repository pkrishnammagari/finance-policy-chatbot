"""
Complete RAG Chain orchestration for Finance House Policy Chatbot - IMPROVED
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import json
import re
from typing import Dict, Any, List, Tuple
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from pydantic import BaseModel, Field

from src.utils import Config, Timer, TraceCollector, logger
from src.embeddings import VectorStoreManager
from src.retriever import PolicyRetriever
from src.intent_detector import IntentDetector, PolicySelector


class PolicyAnswer(BaseModel):
    """Structured answer output"""
    answer: str = Field(description="Clear, concise answer to the user's question")
    policy_number: str = Field(description="Primary policy number cited")
    policy_owner: str = Field(description="Policy owner/department")
    relevant_clause: str = Field(description="Specific clause or section reference")
    interpretation: str = Field(description="Brief interpretation of the policy")


class FinanceHousePolicyChain:
    """
    Complete RAG chain for Finance House Policy Assistant
    """
    
    def __init__(self, vector_store: Chroma = None):
        # Initialize components
        self.vector_store = vector_store or self._load_vector_store()
        self.retriever = PolicyRetriever(self.vector_store)
        self.intent_detector = IntentDetector()
        self.policy_selector = PolicySelector()
        
        # Initialize LLM for answer generation
        self.llm = ChatOllama(
            model=Config.LLM_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=0.1,  # Lower for more factual
            format='json'
        )
        
        # IMPROVED answer generation prompt
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a precise policy assistant for Finance House. Your task is to provide detailed, specific answers based on the policy document content.

CRITICAL INSTRUCTIONS:
1. Provide SPECIFIC details from the policy (numbers, criteria, requirements)
2. If there are eligibility requirements (Band levels, tenure, amounts), STATE THEM EXPLICITLY
3. Cite the ACTUAL section number (e.g., "Section 2.1", "Clause 3.2"), NOT "Chunk X"
4. Be detailed and helpful - users need actionable information
5. If the policy has conditions or exceptions, mention them

BAD ANSWER: "Yes, you can work from home if you meet requirements."
GOOD ANSWER: "Yes, employees in Band 3-5 with minimum 12 months tenure can work from home. You must have manager approval and maintain a secure home office setup per IT security requirements."

Return JSON with:
- answer: Detailed, specific answer with concrete details (3-5 sentences)
- policy_number: Policy number (e.g., "POL-HR-002")
- policy_owner: Department
- relevant_clause: Actual section/clause number from the policy
- interpretation: Brief summary

Output ONLY valid JSON."""),
            ("user", """Question: {question}

Policy Context:
{context}

Selected Policy: {selected_policy}
Domain: {domain}

Provide a detailed, specific answer with concrete details from the policy.""")
        ])
        
        # IMPROVED related questions prompt
        self.related_questions_prompt = ChatPromptTemplate.from_messages([
            ("system", """Generate 3-5 specific, practical related questions that users might ask about this policy.

Questions should:
- Be directly answerable from the policy content shown
- Be specific and actionable
- Cover different aspects (eligibility, process, exceptions, requirements)
- Sound like real employee questions

Return JSON: {"questions": ["Question 1?", "Question 2?", "Question 3?"]}

Output ONLY valid JSON with a "questions" array."""),
            ("user", """Policy: {policy_number}
Original Question: {original_question}

Policy Content:
{context}

Generate 3-5 related questions.""")
        ])
        
        # Trace collector
        self.trace_collector = TraceCollector()
        
    def _load_vector_store(self) -> Chroma:
        """Load the vector store"""
        logger.info("Loading vector store...")
        embeddings = OllamaEmbeddings(
            model=Config.EMBEDDING_MODEL,
            base_url=Config.OLLAMA_BASE_URL
        )
        
        vector_store = Chroma(
            persist_directory=Config.CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=Config.COLLECTION_NAME
        )
        return vector_store
    
    def _extract_json_from_response(self, content: str) -> Dict[str, Any]:
        """Extract JSON from LLM response"""
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                return json.loads(content)
        except Exception as e:
            logger.error(f"Failed to extract JSON: {e}")
            return {}
    
    def generate_answer(
        self, 
        question: str, 
        intent: Any, 
        selected_policy: str,
        docs: List[Document]
    ) -> Tuple[PolicyAnswer, float]:
        """Generate structured answer from documents"""
        start_time = time.time()
        
        try:
            # Prepare context - use MORE context for better answers
            context = "\n\n".join([
                f"[Section {i+1}]\n{doc.page_content}" 
                for i, doc in enumerate(docs[:8])  # Use top 8 chunks for more context
            ])
            
            # Generate answer
            prompt = self.answer_prompt.format_messages(
                question=question,
                context=context,
                selected_policy=selected_policy,
                domain=intent.domain
            )
            
            response = self.llm.invoke(prompt)
            data = self._extract_json_from_response(response.content)
            
            # Extract policy metadata
            policy_metadata = {}
            for doc in docs:
                if doc.metadata.get("policy_number") == selected_policy:
                    policy_metadata = doc.metadata
                    break
            
            answer = PolicyAnswer(
                answer=data.get("answer", "Unable to generate detailed answer. Please refer to the policy document."),
                policy_number=data.get("policy_number", selected_policy),
                policy_owner=data.get("policy_owner", policy_metadata.get("policy_owner", "Unknown")),
                relevant_clause=data.get("relevant_clause", "See policy document"),
                interpretation=data.get("interpretation", "")
            )
            
            duration = time.time() - start_time
            logger.info(f"Generated answer in {duration:.2f}s")
            
            return answer, duration
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            duration = time.time() - start_time
            
            # Fallback answer
            return PolicyAnswer(
                answer="I found relevant information but encountered an error generating a detailed answer. Please refer to the policy document.",
                policy_number=selected_policy,
                policy_owner="Unknown",
                relevant_clause="See policy document",
                interpretation=""
            ), duration
    
    def generate_related_questions(
        self,
        question: str,
        policy_number: str,
        docs: List[Document]
    ) -> Tuple[List[str], float]:
        """Generate related questions from policy content"""
        start_time = time.time()
        
        try:
            # Use first 3 chunks for context
            context = "\n\n".join([doc.page_content for doc in docs[:3]])
            
            prompt = self.related_questions_prompt.format_messages(
                policy_number=policy_number,
                original_question=question,
                context=context
            )
            
            response = self.llm.invoke(prompt)
            data = self._extract_json_from_response(response.content)
            
            # Extract questions - handle both list and dict formats
            questions = data.get("questions", [])
            if not questions:
                questions = data.get("related_questions", [])
            
            # Clean questions
            cleaned = []
            for q in questions:
                if isinstance(q, dict):
                    q_text = q.get('text', q.get('question', str(q)))
                    cleaned.append(q_text)
                elif isinstance(q, str):
                    cleaned.append(q)
            
            # Ensure 3-5 questions
            if len(cleaned) < 3:
                cleaned = [
                    f"What are the key requirements in {policy_number}?",
                    f"Who should I contact about {policy_number}?",
                    f"Are there any exceptions to {policy_number}?"
                ]
            
            cleaned = cleaned[:5]
            
            duration = time.time() - start_time
            logger.info(f"Generated {len(cleaned)} related questions in {duration:.2f}s")
            
            return cleaned, duration
            
        except Exception as e:
            logger.error(f"Error generating related questions: {e}")
            duration = time.time() - start_time
            
            return [
                f"What are the eligibility criteria in {policy_number}?",
                f"What documents do I need for {policy_number}?",
                f"How long does the {policy_number} process take?"
            ], duration
    
    def query(self, question: str, use_multi_query: bool = True) -> Dict[str, Any]:
        """Main query method - complete RAG pipeline"""
        logger.info("=" * 60)
        logger.info(f"PROCESSING QUERY: {question}")
        logger.info("=" * 60)
        
        # Start trace collection
        self.trace_collector.start()
        overall_start = time.time()
        
        try:
            # STEP 1: Intent Detection
            intent, intent_duration = self.intent_detector.detect_intent(question)
            self.trace_collector.add_step(
                "Query Understanding",
                {
                    "original_query": question,
                    "detected_intent": intent.primary_intent,
                    "domain": intent.domain,
                    "confidence": f"{intent.confidence:.0%}"
                },
                intent_duration
            )
            
            # STEP 2: Multi-Query Retrieval
            docs, retrieval_metadata = self.retriever.retrieve(
                question, 
                k=8,  # Get more chunks for better context
                use_multi_query=use_multi_query
            )
            
            self.trace_collector.add_step(
                "Multi-Query Generation & Retrieval",
                {
                    "generated_queries": retrieval_metadata.get("generated_queries", []),
                    "total_docs_retrieved": retrieval_metadata.get("total_docs_retrieved", 0),
                    "unique_docs": retrieval_metadata.get("unique_docs", 0)
                },
                retrieval_metadata.get("total_duration", 0)
            )
            
            # STEP 3: Policy Selection
            policy_selection, selection_duration = self.policy_selector.select_policy(
                question, intent, docs
            )
            
            # Get policy distribution
            policy_summary = self.retriever.get_policy_summary(docs)
            
            self.trace_collector.add_step(
                "Policy Selection & Intent Matching",
                {
                    "selected_policy": policy_selection.selected_policy,
                    "confidence": f"{policy_selection.confidence:.0%}",
                    "reasoning": policy_selection.reasoning,
                    "policy_distribution": policy_summary.get("policy_distribution", {})
                },
                selection_duration
            )
            
            # Filter documents to selected policy
            filtered_docs = self.policy_selector.filter_documents_by_policy(
                docs, policy_selection.selected_policy
            )
            
            if not filtered_docs:
                filtered_docs = docs
            
            # STEP 4: Answer Generation (with more context)
            answer, answer_duration = self.generate_answer(
                question, intent, policy_selection.selected_policy, filtered_docs
            )
            
            self.trace_collector.add_step(
                "Answer Generation",
                {
                    "policy_cited": answer.policy_number,
                    "policy_owner": answer.policy_owner,
                    "relevant_clause": answer.relevant_clause,
                    "answer_length": f"{len(answer.answer)} characters"
                },
                answer_duration
            )
            
            # STEP 5: Related Questions
            related_questions, related_duration = self.generate_related_questions(
                question, policy_selection.selected_policy, filtered_docs
            )
            
            self.trace_collector.add_step(
                "Related Questions Generation",
                {
                    "num_questions": len(related_questions),
                    "questions": related_questions
                },
                related_duration
            )
            
            # Complete trace
            trace = self.trace_collector.complete()
            total_duration = time.time() - overall_start
            
            # Build response
            response = {
                "success": True,
                "question": question,
                "answer": {
                    "text": answer.answer,
                    "policy_number": answer.policy_number,
                    "policy_owner": answer.policy_owner,
                    "relevant_clause": answer.relevant_clause,
                    "interpretation": answer.interpretation,
                    "confidence": policy_selection.confidence
                },
                "related_questions": related_questions,
                "metadata": {
                    "intent": intent.domain,
                    "total_duration": total_duration,
                    "num_policies_searched": len(policy_summary.get("policy_distribution", {})),
                    "chunks_used": len(filtered_docs)
                },
                "trace": trace
            }
            
            logger.info(f"Query completed in {total_duration:.2f}s")
            logger.info("=" * 60)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            
            return {
                "success": False,
                "question": question,
                "error": str(e),
                "answer": {
                    "text": "I encountered an error processing your question. Please try rephrasing or contact support.",
                    "policy_number": "N/A",
                    "policy_owner": "N/A",
                    "relevant_clause": "N/A",
                    "interpretation": "",
                    "confidence": 0.0
                },
                "related_questions": [],
                "metadata": {
                    "total_duration": time.time() - overall_start
                },
                "trace": {}
            }