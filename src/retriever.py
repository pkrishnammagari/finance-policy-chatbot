"""
Multi-query retriever and advanced retrieval strategies for Finance House Policy Chatbot
"""
import time
from typing import List, Dict, Any, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from src.utils import Config, Timer, logger
from langchain_groq import ChatGroq


class MultiQueryOutput(BaseModel):
    """Output format for multi-query generation"""
    queries: List[str] = Field(description="List of alternative query formulations")
    reasoning: str = Field(description="Brief explanation of query variations")


class MultiQueryRetriever:
    """
    Generates multiple query variations and retrieves documents
    from different semantic perspectives
    """
    
    def __init__(self, vector_store: Chroma = None):
        self.vector_store = vector_store
        self.llm = ChatGroq(model_name="llama3-8b-8192", model_kwargs={"response_format": {"type": "json_object"}})
        self.parser = PydanticOutputParser(pydantic_object=MultiQueryOutput)
        
        # Multi-query generation prompt
        self.multi_query_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant helping to improve search results for a policy document database.
Your task is to generate {num_queries} alternative versions of the user's query to search from different perspectives.

CONTEXT: Finance House has policies covering:
- Human Resources (HR): Remote work, leave, performance reviews, professional development
- Information Technology (IT): Equipment procurement, data security
- Finance (FIN): Travel expenses, reimbursements
- Corporate/Compliance (COR): Code of conduct, ethics, conflicts of interest, whistleblower protection

GUIDELINES:
1. Generate queries that explore different semantic angles
2. Use domain-specific terminology (HR, IT, Finance, Compliance)
3. Include both specific and general formulations
4. Focus on policy-related keywords
5. Keep queries concise (less than 10 words each)

{format_instructions}"""),
            ("user", "Original query: {query}")
        ])
        
    def generate_multi_queries(self, query: str, num_queries: int = None) -> Tuple[List[str], str, float]:
        """
        Generate multiple query variations
        
        Args:
            query: Original user query
            num_queries: Number of variations to generate
            
        Returns:
            Tuple of (list of queries, reasoning, duration)
        """
        num_queries = num_queries or Config.MULTI_QUERY_COUNT
        
        start_time = time.time()
        
        try:
            prompt = self.multi_query_prompt.format_messages(
                query=query,
                num_queries=num_queries,
                format_instructions=self.parser.get_format_instructions()
            )
            
            response = self.llm.invoke(prompt)
            
            # Parse the response
            try:
                parsed = self.parser.parse(response.content)
                queries = parsed.queries
                reasoning = parsed.reasoning
            except Exception as parse_error:
                logger.warning(f"Failed to parse structured output, using fallback: {parse_error}")
                # Fallback: simple line-based parsing
                lines = [line.strip() for line in response.content.split('\n') if line.strip()]
                queries = [line for line in lines if not line.startswith('{') and not line.startswith('}')][:num_queries]
                reasoning = "Generated query variations"
            
            # Always include the original query
            if query not in queries:
                queries.insert(0, query)
            
            # Ensure we don't exceed the requested number
            queries = queries[:num_queries]
            
            duration = time.time() - start_time
            logger.info(f"Generated {len(queries)} query variations in {duration:.2f}s")
            
            return queries, reasoning, duration
            
        except Exception as e:
            logger.error(f"Error generating multi-queries: {e}")
            # Fallback to original query
            return [query], "Using original query due to generation error", time.time() - start_time
    
    def retrieve_with_multi_query(self, query: str, k: int = None) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Retrieve documents using multi-query strategy
        
        Args:
            query: User query
            k: Number of documents to retrieve per query
            
        Returns:
            Tuple of (unique documents, retrieval metadata)
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        k = k or Config.TOP_K_RETRIEVAL
        
        with Timer("Multi-Query Retrieval"):
            # Step 1: Generate multiple queries
            queries, reasoning, gen_duration = self.generate_multi_queries(query)
            
            # Step 2: Retrieve documents for each query
            all_docs = []
            query_results = {}
            
            retrieval_start = time.time()
            for q in queries:
                try:
                    docs = self.vector_store.similarity_search(q, k=k)
                    all_docs.extend(docs)
                    query_results[q] = len(docs)
                    logger.debug(f"Query '{q}' retrieved {len(docs)} documents")
                except Exception as e:
                    logger.error(f"Error retrieving for query '{q}': {e}")
                    query_results[q] = 0
            
            retrieval_duration = time.time() - retrieval_start
            
            # Step 3: Deduplicate documents (by content hash)
            unique_docs = []
            seen_content = set()
            
            for doc in all_docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_docs.append(doc)
            
            logger.info(f"Retrieved {len(all_docs)} total docs, {len(unique_docs)} unique")
            
            # Metadata for tracing
            metadata = {
                "original_query": query,
                "generated_queries": queries,
                "query_reasoning": reasoning,
                "num_queries": len(queries),
                "total_docs_retrieved": len(all_docs),
                "unique_docs": len(unique_docs),
                "query_results": query_results,
                "generation_duration": gen_duration,
                "retrieval_duration": retrieval_duration,
                "total_duration": gen_duration + retrieval_duration
            }
            
            return unique_docs, metadata


class PolicyRetriever:
    """
    High-level retriever with policy-specific intelligence
    """
    
    def __init__(self, vector_store: Chroma = None):
        self.vector_store = vector_store
        self.multi_query_retriever = MultiQueryRetriever(vector_store)
        
    def retrieve(self, query: str, k: int = None, use_multi_query: bool = True) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Main retrieval method
        
        Args:
            query: User query
            k: Number of documents to retrieve
            use_multi_query: Whether to use multi-query strategy
            
        Returns:
            Tuple of (documents, metadata)
        """
        if use_multi_query:
            return self.multi_query_retriever.retrieve_with_multi_query(query, k)
        else:
            # Simple single-query retrieval
            start_time = time.time()
            docs = self.vector_store.similarity_search(query, k=k or Config.TOP_K_RETRIEVAL)
            duration = time.time() - start_time
            
            metadata = {
                "original_query": query,
                "generated_queries": [query],
                "num_queries": 1,
                "unique_docs": len(docs),
                "total_duration": duration
            }
            
            return docs, metadata
    
    def get_policy_summary(self, docs: List[Document]) -> Dict[str, Any]:
        """
        Summarize which policies were retrieved
        
        Args:
            docs: List of retrieved documents
            
        Returns:
            Summary dictionary
        """
        policy_counts = {}
        domain_counts = {}
        
        for doc in docs:
            policy_num = doc.metadata.get("policy_number", "Unknown")
            domain = doc.metadata.get("domain", "Unknown")
            
            policy_counts[policy_num] = policy_counts.get(policy_num, 0) + 1
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        return {
            "total_chunks": len(docs),
            "unique_policies": len(policy_counts),
            "policy_distribution": policy_counts,
            "domain_distribution": domain_counts
        }


def test_retriever():
    """Test the retriever module"""
    from langchain_community.embeddings import HuggingFaceEmbeddings # <-- Import change
    from langchain_chroma import Chroma      # <-- Also update this import
    
    print("=" * 60)
    print("TESTING MULTI-QUERY RETRIEVER")
    print("=" * 60)
    
    # Load vector store
    vector_store = Chroma(
        persist_directory=Config.CHROMA_PERSIST_DIR,
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        collection_name=Config.COLLECTION_NAME
        )

if __name__ == "__main__":
    test_retriever()