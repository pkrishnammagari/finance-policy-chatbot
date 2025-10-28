# Quick fix for related questions formatting
"""
Complete RAG Chain orchestration for Finance House Policy Chatbot
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


class RelatedQuestions(BaseModel):
    """Related questions output"""
    questions: List[str] = Field(description="List of 3-5 related questions from the policy")


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
            temperature=0.3,
            format='json'
        )
        
        # Answer generation prompt
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledgeable policy assistant for Finance House.

Your task is to provide clear, accurate answers based ONLY on the provided policy documents.

INSTRUCTIONS:
1. Answer the user's question directly and concisely (2-3 sentences)
2. Cite the specific policy number, owner, and clause
3. Provide interpretation in business-friendly language (not technical jargon)
4. If information is unclear, say so - DO NOT make up information
5. Be helpful and professional

Return a JSON object with these fields:
- answer: Direct answer to the question (2-3 sentences)
- policy_number: The policy number (e.g., "POL-HR-002")
- policy_owner: Department that owns the policy
- relevant_clause: Specific section/clause reference
- interpretation: Business-friendly interpretation

Output ONLY valid JSON."""),
            ("user", """Question: {question}

Intent: {intent}
Domain: {domain}
Selected Policy: {selected_policy}

Relevant Policy Content:
{context}

Provide a complete answer based on the policy content above.""")
        ])
        
        # Related questions prompt
        self.related_questions_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are helping generate related questions that users might want to ask about a policy.

Based on the policy content provided, generate 3-5 related questions that:
1. Are directly answerable from THIS policy document
2. Cover different aspects of the policy
3. Are practical and commonly asked
4. Are phrased as questions a new employee might ask

Return a JSON object with:
- questions: Array of 3-5 question strings (just the text, no objects)

Example format:
{"questions": ["Question 1?", "Question 2?", "Question 3?"]}

Output ONLY valid JSON."""),
            ("user", """Policy: {policy_number}
User's Original Question: {original_question}

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
        """
        Generate structured answer from documents
        
        Args:
            question: User's question
            intent: Detected intent
            selected_policy: Selected policy number
            docs: Retrieved documents
            
        Returns:
            Tuple of (PolicyAnswer, duration)
        """
        start_time = time.time()
        
        try:
            # Prepare context from documents
            context = "\n\n".join([
                f"[Chunk {i+1}]\n{doc.page_content}" 
                for i, doc in enumerate(docs[:5])  # Use top 5 chunks
            ])
            
            # Generate answer
            prompt = self.answer_prompt.format_messages(
                question=question,
                intent=intent.primary_intent,
                domain=intent.domain,
                selected_policy=selected_policy,
                context=context
            )
            
            response = self.llm.invoke(prompt)
            data = self._extract_json_from_response(response.content)
            
            # Extract policy metadata from documents
            policy_metadata = {}
            for doc in docs:
                if doc.metadata.get("policy_number") == selected_policy:
                    policy_metadata = doc.metadata
                    break
            
            answer = PolicyAnswer(
                answer=data.get("answer", "Unable to generate answer"),
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
                answer="I found relevant information but encountered an error generating the answer. Please refer to the policy document.",
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
        """
        Generate related questions from policy content
        
        Args:
            question: Original question
            policy_number: Policy number
            docs: Retrieved documents
            
        Returns:
            Tuple of (list of questions, duration)
        """
        start_time = time.time()
        
        try:
            # Prepare context
            context = "\n\n".join([
                doc.page_content for doc in docs[:3]
            ])
            
            prompt = self.related_questions_prompt.format_messages(
                policy_number=policy_number,
                original_question=question,
                context=context
            )
            
            response = self.llm.invoke(prompt)
            data = self._extract_json_from_response(response.content)
            
            questions = data.get("questions", [])
            
            # Ensure questions are strings, not dicts
            cleaned_questions = []
            for q in questions:
                if isinstance(q, dict):
                    # Extract text from dict
                    q_text = q.get('text', str(q))
                    cleaned_questions.append(q_text)
                else:
                    cleaned_questions.append(str(q))
            
            # Ensure we have 3-5 questions
            if len(cleaned_questions) < 3:
                cleaned_questions.extend([
                    f"What are the key requirements in {policy_number}?",
                    f"Who should I contact about {policy_number}?",
                    f"When was {policy_number} last updated?"
                ])
            
            cleaned_questions = cleaned_questions[:5]  # Max 5 questions
            
            duration = time.time() - start_time
            logger.info(f"Generated {len(cleaned_questions)} related questions in {duration:.2f}s")
            
            return cleaned_questions, duration
            
        except Exception as e:
            logger.error(f"Error generating related questions: {e}")
            duration = time.time() - start_time
            
            # Fallback questions
            return [
                f"What are the requirements in {policy_number}?",
                f"Who manages {policy_number}?",
                f"How do I comply with {policy_number}?"
            ], duration
    
    def query(self, question: str, use_multi_query: bool = True) -> Dict[str, Any]:
        """
        Main query method - complete RAG pipeline
        
        Args:
            question: User's question
            use_multi_query: Whether to use multi-query retrieval
            
        Returns:
            Complete response dictionary with answer and trace
        """
        logger.info("=" * 60)
        logger.info(f"PROCESSING QUERY: {question}")
        logger.info("=" * 60)
        
        # Start trace collection
        self.trace_collector.start()
        overall_start = time.time()
        
        try:
            # STEP 1: Intent Detection
            step_start = time.time()
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
                k=5, 
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
                filtered_docs = docs  # Fallback to all docs
            
            # STEP 4: Answer Generation
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
            
            # Error response
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


def test_rag_chain():
    """Test the complete RAG chain"""
    print("=" * 80)
    print("TESTING COMPLETE RAG CHAIN")
    print("=" * 80)
    
    # Initialize chain
    chain = FinanceHousePolicyChain()
    
    # Test queries
    test_queries = [
        "Can I work from home full-time?",
        "What laptop budget do I have as a Band 3 employee?",
        "Can I accept gifts from clients?"
    ]
    
    for query in test_queries:
        print(f"\n{'=' * 80}")
        print(f"QUERY: {query}")
        print("=" * 80)
        
        response = chain.query(query)
        
        if response["success"]:
            print(f"\n📋 ANSWER:")
            print(f"{response['answer']['text']}")
            print(f"\n📖 POLICY REFERENCE:")
            print(f"  Policy: {response['answer']['policy_number']}")
            print(f"  Owner: {response['answer']['policy_owner']}")
            print(f"  Clause: {response['answer']['relevant_clause']}")
            print(f"  Confidence: {response['answer']['confidence']:.0%}")
            
            print(f"\n💡 RELATED QUESTIONS:")
            for i, q in enumerate(response['related_questions'], 1):
                print(f"  {i}. {q}")
            
            print(f"\n⏱️  PERFORMANCE:")
            print(f"  Total Time: {response['metadata']['total_duration']:.2f}s")
            print(f"  Policies Searched: {response['metadata']['num_policies_searched']}")
            print(f"  Chunks Used: {response['metadata']['chunks_used']}")
            
            print(f"\n🔍 TRACE SUMMARY:")
            trace = response['trace']
            for i, step in enumerate(trace.get('steps', []), 1):
                print(f"  Step {i}: {step['step_name']} ({step['duration']:.2f}s)")
        else:
            print(f"\n❌ ERROR: {response['error']}")
        
        print("\n" + "=" * 80)
        input("\nPress Enter to continue to next query...")


if __name__ == "__main__":
    test_rag_chain()