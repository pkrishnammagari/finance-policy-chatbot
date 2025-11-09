"""
Document loading, chunking, and embedding management for Finance House Policy Chatbot
"""
import os
from pathlib import Path
from typing import List, Dict, Any
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.schema import Document
from src.utils import Config, Timer, logger, extract_policy_metadata


class PolicyDocumentLoader:
    """Loads and processes policy documents"""
    
    def __init__(self, policies_dir: Path = None):
        self.policies_dir = policies_dir or Config.POLICIES_DIR
        self.documents: List[Document] = []
        
    def load_documents(self) -> List[Document]:
        """
        Load all policy documents from the policies directory
        
        Returns:
            List of Document objects
        """
        with Timer("Document Loading"):
            policy_files = list(self.policies_dir.glob("*.txt"))
            
            if not policy_files:
                raise FileNotFoundError(f"No .txt files found in {self.policies_dir}")
            
            logger.info(f"Found {len(policy_files)} policy files")
            
            for policy_file in policy_files:
                try:
                    loader = TextLoader(str(policy_file), encoding='utf-8')
                    docs = loader.load()
                    
                    # Extract metadata from filename and content
                    filename = policy_file.name
                    content = docs[0].page_content if docs else ""
                    metadata = extract_policy_metadata(content, filename)
                    
                    # Add metadata to document
                    for doc in docs:
                        doc.metadata.update(metadata)
                        doc.metadata['source'] = str(policy_file)
                    
                    self.documents.extend(docs)
                    logger.info(f"Loaded: {filename} ({len(content)} chars)")
                    
                except Exception as e:
                    logger.error(f"Error loading {policy_file}: {e}")
                    continue
            
            logger.info(f"Total documents loaded: {len(self.documents)}")
            return self.documents
    
    def get_documents(self) -> List[Document]:
        """Get loaded documents"""
        return self.documents
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of loaded documents"""
        return {
            "total_documents": len(self.documents),
            "total_chars": sum(len(doc.page_content) for doc in self.documents),
            "policies": [doc.metadata.get("filename", "Unknown") for doc in self.documents]
        }


class PolicyChunker:
    """Chunks policy documents for embedding"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents into smaller pieces
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        with Timer("Document Chunking"):
            chunks = self.text_splitter.split_documents(documents)
            
            # Add chunk index to metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_id'] = i
                
            logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
            logger.info(f"Average chunk size: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} chars")
            
            return chunks


class VectorStoreManager:
    """Manages ChromaDB vector store for policy embeddings"""
    
    def __init__(self, persist_dir: str = None, collection_name: str = None):
        self.persist_dir = persist_dir or Config.CHROMA_PERSIST_DIR
        self.collection_name = collection_name or Config.COLLECTION_NAME
        # THIS IS THE LINE THAT FIXES THE CRASH
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 
        self.vector_store = None
        
    def create_vector_store(self, chunks: List[Document], force_recreate: bool = False) -> Chroma:
        """
        Create or load vector store from document chunks
        
        Args:
            chunks: List of document chunks
            force_recreate: If True, delete existing and create new
            
        Returns:
            Chroma vector store
        """
        # Check if vector store already exists
        persist_path = Path(self.persist_dir)
        
        if persist_path.exists() and not force_recreate:
            logger.info(f"Loading existing vector store from {self.persist_dir}")
            with Timer("Vector Store Loading"):
                self.vector_store = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name
                )
            return self.vector_store
        
        # Create new vector store
        logger.info(f"Creating new vector store with {len(chunks)} chunks")
        with Timer("Vector Store Creation"):
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_dir,
                collection_name=self.collection_name
            )
        
        logger.info(f"Vector store created and persisted to {self.persist_dir}")
        return self.vector_store
    
    def get_vector_store(self) -> Chroma:
        """Get the vector store instance"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call create_vector_store first.")
        return self.vector_store
    
    def search(self, query: str, k: int = None) -> List[Document]:
        """
        Search vector store for relevant documents
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        k = k or Config.TOP_K_RETRIEVAL
        
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        with Timer(f"Vector Search (k={k})"):
            results = self.vector_store.similarity_search(query, k=k)
        
        return results


def build_vector_database(force_recreate: bool = False) -> VectorStoreManager:
    """
    Main function to build the complete vector database
    
    Args:
        force_recreate: If True, recreate even if exists
        
    Returns:
        VectorStoreManager instance
    """
    logger.info("=" * 60)
    logger.info("BUILDING VECTOR DATABASE")
    logger.info("=" * 60)
    
    # Step 1: Load documents
    loader = PolicyDocumentLoader()
    documents = loader.load_documents()
    summary = loader.get_summary()
    logger.info(f"Loaded {summary['total_documents']} policies, {summary['total_chars']:,} total chars")
    
    # Step 2: Chunk documents
    chunker = PolicyChunker()
    chunks = chunker.chunk_documents(documents)
    
    # Step 3: Create vector store
    vector_manager = VectorStoreManager()
    vector_manager.create_vector_store(chunks, force_recreate=force_recreate)
    
    logger.info("=" * 60)
    logger.info("VECTOR DATABASE BUILD COMPLETE")
    logger.info("=" * 60)
    
    return vector_manager


if __name__ == "__main__":
    # Test the module
    vector_manager = build_vector_database(force_recreate=True)
    
    # Test search
    print("\nTesting search functionality:")
    results = vector_manager.search("remote work policy", k=3)
    print(f"Found {len(results)} results for 'remote work policy'")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc.metadata.get('policy_number', 'Unknown')} - {doc.metadata.get('filename', 'Unknown')}")
        print(f"   Preview: {doc.page_content[:150]}...")
