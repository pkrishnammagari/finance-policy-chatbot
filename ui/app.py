"""
Finance House Policy Assistant - Streamlit Cloud Optimized UI
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import os

# Import factory chain functions and environment check
from src.chain import create_inference_chain, create_full_chain, is_streamlit_cloud

st.set_page_config(
    page_title="Finance House Policy Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #1e1e1e !important;
    }
    .stTextInput label, .stButton button {
        color: #1e1e1e !important;
    }
    .related-question {
        background-color: #f0f2f6;
        border-left: 3px solid #4CAF50;
        padding: 10px;
        margin: 5px 0;
        cursor: pointer;
        border-radius: 5px;
        transition: all 0.3s;
    }
    .related-question:hover {
        background-color: #e1e4e8;
        border-left-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_rag_chain():
    if is_streamlit_cloud():
        return create_inference_chain()
    else:
        return create_full_chain()

if 'rag_chain' not in st.session_state:
    with st.spinner("🔄 Initializing Policy Assistant..."):
        st.session_state.rag_chain = load_rag_chain()
        
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'query_count' not in st.session_state:
    st.session_state.query_count = 0

if 'selected_question' not in st.session_state:
    st.session_state.selected_question = None

# Sidebar
with st.sidebar:
    st.markdown("## 📚 About")
    
    # Show deployment mode
    if is_streamlit_cloud():
        st.info("🌐 Running on Streamlit Cloud (Inference Mode)")
    else:
        st.success("💻 Running Locally (Full Mode)")
    
    st.markdown("""
    **Finance House Policy Assistant** uses advanced AI to help you understand and navigate organizational policies.
    
    ### Features:
    - 🔍 Multi-Query RAG
    - 🎯 Intent Detection  
    - 📋 10 Policy Documents
    - 🔄 Real-time Processing
    
    ### How to Use:
    1. Type your policy question below
    2. Get instant, accurate answers
    3. Review policy references
    4. Check AI reasoning process
    """)
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.query_count = 0
        st.session_state.selected_question = None
        st.rerun()

# Main content
st.markdown("""
<h1 style='text-align: center; color: #1e1e1e;'>🏦 Finance House Policy Assistant</h1>
<p style='text-align: center; color: #1e1e1e; font-size: 1.2em;'>Your intelligent guide to organizational policies and procedures</p>
""", unsafe_allow_html=True)

st.markdown("---")

# Query input
query = st.text_input(
    "💬 Ask a policy question:",
    placeholder="e.g., Can I work from home? What laptop can I get?",
    key="query_input",
    value=st.session_state.get('selected_question', '')
)

# Handle query submission
if query:
    st.session_state.query_count += 1
    
    with st.spinner("🤔 Processing your question..."):
        try:
            # Query the RAG chain
            response = st.session_state.rag_chain.query(query)
            
            if response.get('success'):
                # Store in chat history
                st.session_state.chat_history.append({
                    'query': query,
                    'response': response
                })
                
                # Clear selected question
                st.session_state.selected_question = None
                
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            st.info("Please ensure Ollama is running locally or configured correctly.")

# Display chat history
if st.session_state.chat_history:
    for idx, chat in enumerate(reversed(st.session_state.chat_history), 1):
        with st.container():
            # Question
            st.markdown(f"### 🙋 Question {len(st.session_state.chat_history) - idx + 1}")
            st.markdown(f"**{chat['query']}**")
            
            # Answer
            response = chat['response']
            answer_data = response.get('answer', {})
            
            st.markdown("### 💡 Answer")
            st.markdown(f"**{answer_data.get('text', 'No answer available')}**")
            
            # Metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**📋 Policy:** {answer_data.get('policy_number', 'N/A')}")
            with col2:
                st.markdown(f"**🏢 Owner:** {answer_data.get('policy_owner', 'N/A')}")
            with col3:
                confidence = answer_data.get('confidence', 0)
                st.markdown(f"**📊 Confidence:** {confidence:.0%}")
            
            # Relevant clause
            if answer_data.get('relevant_clause'):
                st.markdown(f"**📌 Relevant Clause:** {answer_data['relevant_clause']}")
            
            # Related questions
            related_questions = response.get('related_questions', [])
            if related_questions:
                st.markdown("### 🔗 Related Questions")
                for rq in related_questions:
                    if st.button(f"➤ {rq}", key=f"rq_{idx}_{rq[:30]}", use_container_width=True):
                        st.session_state.selected_question = rq
                        st.rerun()
            
            # Expander for trace/reasoning
            with st.expander("🔍 View AI Reasoning Process"):
                trace = response.get('trace', {})
                if trace.get('steps'):
                    for step_idx, step in enumerate(trace['steps'], 1):
                        st.markdown(f"**Step {step_idx}: {step.get('step_name', 'Unknown')}** ({step.get('duration', 0):.2f}s)")
                        step_data = step.get('data', {})
                        for key, value in step_data.items():
                            if isinstance(value, list):
                                st.markdown(f"- **{key}:**")
                                for item in value[:3]:  # Show first 3 items
                                    st.markdown(f"  - {item}")
                            else:
                                st.markdown(f"- **{key}:** {value}")
                        st.markdown("---")
                    
                    total_time = trace.get('total_duration', 0)
                    st.markdown(f"**⏱️ Total Processing Time:** {total_time:.2f}s")
                else:
                    st.info("No trace data available")
            
            st.markdown("---")

# Example questions
if not st.session_state.chat_history:
    st.markdown("### 💭 Try These Example Questions:")
    
    example_questions = [
        "Can I work from home?",
        "What are the laptop procurement requirements?",
        "How many vacation days do I get?",
        "What is the travel expense reimbursement policy?",
        "How do I report a code of conduct violation?"
    ]
    
    cols = st.columns(2)
    for idx, eq in enumerate(example_questions):
        with cols[idx % 2]:
            if st.button(eq, key=f"example_{idx}", use_container_width=True):
                st.session_state.selected_question = eq
                st.rerun()

def main():
    """Main entry point"""
    pass

if __name__ == "__main__":
    main()
