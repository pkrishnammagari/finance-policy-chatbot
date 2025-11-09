"""
Streamlit Cloud Optimized Entry Point for Finance House Policy Chatbot
"""
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run your UI app
from ui.app import main

if __name__ == "__main__":
    main()
