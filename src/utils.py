"""
Utility functions for Finance House Policy Chatbot
"""
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration management"""
    
    # Project
    PROJECT_NAME = os.getenv("PROJECT_NAME", "finance-house-policy-chatbot")
    VERSION = os.getenv("VERSION", "1.0.0")
    
    
    # ChromaDB
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "finance_house_policies")
    
    # Retrieval
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))
    MULTI_QUERY_COUNT = int(os.getenv("MULTI_QUERY_COUNT", "4"))
    
    # LLM
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
    
    # Paths
    POLICIES_DIR = Path("data/policies")
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        if os.environ.get("STREAMLIT_CLOUD", "false").lower() == "true":
            # Skip policies directory check on Streamlit Cloud deployment
            return
        if not cls.POLICIES_DIR.exists():
            raise FileNotFoundError(f"Policies directory not found: {cls.POLICIES_DIR}")

        logger.info(f"Configuration validated successfully")
        return True


class Timer:
    """Simple timer for measuring execution time"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"{self.name} started")
        return self
        
    def __exit__(self, *args):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        logger.info(f"{self.name} completed in {elapsed:.2f} seconds")
        
    def elapsed(self) -> float:
        """Get elapsed time"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


class TraceCollector:
    """Collects trace information for observability"""
    
    def __init__(self):
        self.traces: List[Dict[str, Any]] = []
        self.current_trace: Dict[str, Any] = {}
        self.start_time = None
        
    def start(self):
        """Start trace collection"""
        self.start_time = time.time()
        self.current_trace = {
            "timestamp": datetime.now().isoformat(),
            "steps": []
        }
        
    def add_step(self, step_name: str, data: Dict[str, Any], duration: float = None):
        """Add a step to current trace"""
        step = {
            "step_name": step_name,
            "data": data,
            "duration": duration
        }
        self.current_trace["steps"].append(step)
        logger.debug(f"Trace step added: {step_name}")
        
    def complete(self) -> Dict[str, Any]:
        """Complete trace and return"""
        total_duration = time.time() - self.start_time if self.start_time else 0
        self.current_trace["total_duration"] = total_duration
        self.traces.append(self.current_trace)
        
        trace = self.current_trace.copy()
        self.current_trace = {}
        return trace
        
    def get_last_trace(self) -> Dict[str, Any]:
        """Get the last completed trace"""
        return self.traces[-1] if self.traces else {}


def extract_policy_metadata(content: str, filename: str) -> Dict[str, str]:
    """
    Extract metadata from policy content
    
    Args:
        content: Policy document content
        filename: Name of the policy file
        
    Returns:
        Dictionary with policy metadata
    """
    metadata = {
        "filename": filename,
        "policy_number": "",
        "policy_owner": "",
        "effective_date": "",
        "domain": ""
    }
    
    # Extract policy number from filename (e.g., POL-HR-002)
    if filename.startswith("POL-"):
        parts = filename.split("_")[0]
        metadata["policy_number"] = parts
        
        # Extract domain from policy number
        if "-HR-" in parts:
            metadata["domain"] = "Human Resources"
        elif "-IT-" in parts:
            metadata["domain"] = "Information Technology"
        elif "-FIN-" in parts:
            metadata["domain"] = "Finance"
        elif "-COR-" in parts:
            metadata["domain"] = "Corporate/Compliance"
    
    # Try to extract from content
    lines = content.split('\n')[:20]  # Check first 20 lines
    
    for line in lines:
        line_lower = line.lower()
        
        if "policy number" in line_lower and not metadata["policy_number"]:
            # Try to extract policy number
            if "POL-" in line:
                start = line.index("POL-")
                metadata["policy_number"] = line[start:start+11].strip()
                
        if "policy owner" in line_lower and not metadata["policy_owner"]:
            # Extract policy owner
            parts = line.split(":")
            if len(parts) > 1:
                metadata["policy_owner"] = parts[1].strip()
                
        if "effective date" in line_lower and not metadata["effective_date"]:
            # Extract effective date
            parts = line.split(":")
            if len(parts) > 1:
                metadata["effective_date"] = parts[1].strip()
    
    logger.debug(f"Extracted metadata: {metadata}")
    return metadata


def format_trace_for_display(trace: Dict[str, Any]) -> str:
    """
    Format trace data for display in UI
    
    Args:
        trace: Trace dictionary
        
    Returns:
        Formatted string for display
    """
    if not trace or "steps" not in trace:
        return "No trace data available"
    
    output = []
    output.append("=" * 60)
    output.append("AI REASONING PROCESS")
    output.append("=" * 60)
    
    for i, step in enumerate(trace.get("steps", []), 1):
        step_name = step.get("step_name", "Unknown Step")
        duration = step.get("duration", 0)
        data = step.get("data", {})
        
        output.append(f"\nSTEP {i}: {step_name} ({duration:.2f}s)")
        output.append("-" * 60)
        
        for key, value in data.items():
            if isinstance(value, list):
                output.append(f"{key}:")
                for item in value:
                    output.append(f"  • {item}")
            else:
                output.append(f"{key}: {value}")
    
    total = trace.get("total_duration", 0)
    output.append(f"\n{'=' * 60}")
    output.append(f"TOTAL PROCESSING TIME: {total:.2f} seconds")
    output.append("=" * 60)
    
    return "\n".join(output)


# Initialize configuration on import
try:
    Config.validate()
except Exception as e:
    logger.error(f"Configuration validation failed: {e}")
    raise
