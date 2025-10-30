"""
Finance House Policy Assistant - WITH CLICKABLE RELATED QUESTIONS
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from src.chain import FinanceHousePolicyChain

st.set_page_config(
    page_title="Finance House Policy Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS - ALL TEXT DARK + RELATED QUESTIONS STYLING
st.markdown("""
<style>
    /* GLOBAL FORCE: WHITE BACKGROUND, DARK TEXT */
    .stApp, .main, .block-container, body, html {
        background-color: #ffffff !important;
        color: #1f2937 !important;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
        padding: 2rem 1rem;
        border-right: 1px solid #e0e0e0;
    }
    
    section[data-testid="stSidebar"] > div {
        overflow-y: hidden !important;
    }
    
    section[data-testid="stSidebar"] *, 
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] h5,
    section[data-testid="stSidebar"] h6 {
        color: #1f2937 !important;
    }
    
    /* FORCE ALL STREAMLIT TEXT ELEMENTS TO DARK */
    .stMarkdown, 
    .stMarkdown *,
    .stText,
    .stText *,
    .stCaption,
    .stCaption *,
    div[data-testid="stMarkdownContainer"],
    div[data-testid="stMarkdownContainer"] *,
    div[data-testid="stText"],
    div[data-testid="stText"] * {
        color: #1f2937 !important;
    }
    
    /* SPINNER TEXT */
    .stSpinner,
    .stSpinner > div,
    .stSpinner > div > div,
    .stSpinner *,
    div[data-testid="stStatusWidget"],
    div[data-testid="stStatusWidget"] *,
    div[data-testid="stStatusWidget"] div,
    div[data-testid="stStatusWidget"] p {
        color: #1f2937 !important;
    }
    
    /* EXPANDER */
    div[data-testid="stExpander"],
    div[data-testid="stExpander"] *,
    div[data-testid="stExpander"] p,
    div[data-testid="stExpander"] div,
    div[data-testid="stExpander"] span {
        color: #1f2937 !important;
    }
    
    div[data-testid="stExpander"] {
        border: 2px solid #3b82f6 !important;
        border-radius: 8px !important;
        background-color: #ffffff !important;
        margin: 1rem 0 !important;
    }
    
    div[data-testid="stExpander"] summary {
        background-color: #10b981 !important;
        color: white !important;
        padding: 0.75rem 1rem !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }
    
    /* FORCE EXPANDER HEADER TEXT TO WHITE */
    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary * {
        color: white !important;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white !important;
        margin: 0;
        font-size: 2rem;
        font-weight: 600;
    }
    
    .main-header p {
        color: #e0f2fe !important;
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }
    
    /* Question */
    .user-question {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin: 1.5rem 0;
        border-radius: 8px;
    }
    
    .user-question .label {
        color: #1e40af !important;
        font-weight: 600;
        font-size: 0.875rem;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    .user-question .content {
        color: #1e40af !important;
        font-size: 1.1rem;
    }
    
    /* Answer */
    .assistant-answer {
        background-color: white;
        border: 1px solid #e5e7eb;
        border-left: 4px solid #10b981;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .assistant-answer .label {
        color: #059669 !important;
        font-weight: 600;
        font-size: 0.875rem;
        text-transform: uppercase;
        margin-bottom: 1rem;
        display: block;
    }
    
    .assistant-answer .content {
        color: #1f2937 !important;
        font-size: 1.05rem;
        line-height: 1.7;
    }
    
    /* Policy reference */
    .policy-reference {
        background-color: #fef3c7;
        border: 1px solid #fde047;
        border-left: 4px solid #eab308;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 6px;
    }
    
    .policy-reference .title {
        color: #713f12 !important;
        font-weight: 600;
        font-size: 0.875rem;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
        display: block;
    }
    
    .policy-reference .item {
        color: #854d0e !important;
        margin: 0.4rem 0;
        font-size: 0.95rem;
    }
    
    .policy-reference .item strong {
        color: #713f12 !important;
    }
    
    .confidence-high { color: #059669 !important; font-weight: 600; }
    .confidence-medium { color: #d97706 !important; font-weight: 600; }
    
    /* Metrics */
    .metrics {
        color: #1f2937 !important;
        font-size: 0.95rem;
        font-weight: 500;
        margin: 1rem 0;
        padding: 0.75rem;
        background-color: #f9fafb;
        border-radius: 6px;
        border: 1px solid #e5e7eb;
    }
    
    /* RELATED QUESTIONS SECTION */
    .related-questions {
        background-color: #f0f9ff;
        border: 1px solid #bae6fd;
        border-left: 4px solid #0ea5e9;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 8px;
    }
    
    .related-questions .title {
        color: #0c4a6e !important;
        font-weight: 600;
        font-size: 0.875rem;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
        display: block;
    }
    
    /* Text input */
    .stTextInput input {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #d1d5db !important;
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
        font-size: 1rem !important;
    }
    
    .stTextInput input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    .stTextInput input::placeholder {
        color: #9ca3af !important;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #1e40af !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        cursor: pointer !important;
    }
    
    .stButton button:hover {
        background-color: #1e3a8a !important;
    }
    
    /* FORCE ALL BUTTON TEXT TO WHITE (override global dark text) */
    .stButton button,
    .stButton button *,
    .stButton button span,
    .stButton button p,
    .stButton button div {
        color: white !important;
    }
    
    /* Trace */
    .trace-step {
        background-color: #ffffff;
        border-left: 3px solid #3b82f6;
        padding: 0.75rem;
        margin: 0.75rem 0;
        border-radius: 4px;
        border: 1px solid #e5e7eb;
    }
    
    .trace-step-title {
        color: #1f2937 !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }
    
    .trace-step-detail,
    .trace-step-detail * {
        color: #1f2937 !important;
        font-size: 0.875rem;
        margin: 0.25rem 0;
        font-family: 'Courier New', monospace;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_chain' not in st.session_state:
    with st.spinner("🔄 Initializing Policy Assistant..."):
        st.session_state.rag_chain = FinanceHousePolicyChain()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'query_count' not in st.session_state:
    st.session_state.query_count = 0

if 'selected_question' not in st.session_state:
    st.session_state.selected_question = None

# Sidebar
with st.sidebar:
    st.markdown("## 📚 About")
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
<div class="main-header">
    <h1>🏦 Finance House Policy Assistant</h1>
    <p>Your intelligent guide to organizational policies and procedures</p>
</div>
""", unsafe_allow_html=True)

# Display chat history
for entry in st.session_state.chat_history:
    st.markdown(f"""
    <div class="user-question">
        <span class="label">Your Question</span>
        <div class="content">{entry['question']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    response = entry['response']
    answer = response['answer']
    
    st.markdown(f"""
    <div class="assistant-answer">
        <span class="label">Policy Assistant Answer</span>
        <div class="content">{answer['text']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    conf_class = "confidence-high" if answer.get('confidence', 0) >= 0.8 else "confidence-medium"
    st.markdown(f"""
    <div class="policy-reference">
        <span class="title">📋 POLICY REFERENCE</span>
        <div class="item"><strong>Policy:</strong> {answer.get('policy_number', 'N/A')}</div>
        <div class="item"><strong>Owner:</strong> {answer.get('policy_owner', 'N/A')}</div>
        <div class="item"><strong>Section:</strong> {answer.get('relevant_clause', 'N/A')}</div>
        <div class="item"><strong>Confidence:</strong> <span class="{conf_class}">{answer.get('confidence', 0):.0%}</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    metadata = response.get('metadata', {})
    st.markdown(f"""
    <div class="metrics">
        ⏱️ {metadata.get('total_duration', 0):.1f}s | 
        📊 Policies Searched: {metadata.get('num_policies_searched', 0)} | 
        📄 Sections Analyzed: {metadata.get('chunks_used', 0)}
    </div>
    """, unsafe_allow_html=True)
    
    # AI REASONING TRACE
    with st.expander("🔍 View AI Reasoning Process", expanded=False):
        trace = response.get('trace', {})
        if trace and 'steps' in trace:
            trace_html = ""
            for i, step in enumerate(trace['steps'], 1):
                step_name = step.get('step_name', 'Unknown')
                duration = step.get('duration', 0)
                data = step.get('data', {})
                
                trace_html += f'<div class="trace-step">'
                trace_html += f'<div class="trace-step-title">Step {i}: {step_name} ({duration:.2f}s)</div>'
                
                for key, value in data.items():
                    if isinstance(value, list) and len(value) > 0:
                        trace_html += f'<div class="trace-step-detail"><strong>{key}:</strong></div>'
                        for item in value[:3]:
                            trace_html += f'<div class="trace-step-detail">  • {item}</div>'
                    elif isinstance(value, dict):
                        trace_html += f'<div class="trace-step-detail"><strong>{key}:</strong> {str(value)[:100]}...</div>'
                    else:
                        trace_html += f'<div class="trace-step-detail"><strong>{key}:</strong> {value}</div>'
                
                trace_html += '</div>'
            
            st.markdown(trace_html, unsafe_allow_html=True)
        else:
            st.info("No trace data available for this query.")

    
    # RELATED QUESTIONS - CLICKABLE BUTTONS
    related_questions = response.get('related_questions', [])
    if related_questions:
        st.markdown("""
        <div class="related-questions">
            <span class="title">💡 Related Questions</span>
        </div>
        """, unsafe_allow_html=True)
        
        cols = st.columns(1)
        for i, question in enumerate(related_questions):
            if st.button(f"❓ {question}", key=f"rq_{entry['question'][:20]}_{i}", use_container_width=True):
                st.session_state.selected_question = question
                st.session_state.query_count += 1
                st.rerun()
    
# Input section
st.markdown("---")
st.markdown('<h3 style="color: #1f2937 !important; font-weight: 600;">💬 Ask a Question</h3>', unsafe_allow_html=True)

col1, col2 = st.columns([5, 1])

with col1:
    user_question = st.text_input(
        "Type your policy question here...",
        value="",
        key=f"user_input_{st.session_state.query_count}",
        placeholder="e.g., Can I work from home?",
        label_visibility="collapsed"
    )

with col2:
    ask_button = st.button("Send", type="primary", use_container_width=True)

# Auto-process if related question was clicked
if st.session_state.selected_question and not ask_button:
    user_question = st.session_state.selected_question
    st.session_state.selected_question = None  # Clear it now
    ask_button = True  # Simulate button click

# Process the question (either manual or auto-triggered)
if ask_button and user_question:
    with st.spinner("🔄 Processing your question..."):
        response = st.session_state.rag_chain.query(user_question)
        
        if response.get('success'):
            st.session_state.chat_history.append({
                'question': user_question,
                'response': response
            })
            st.session_state.query_count += 1
            st.rerun()
        else:
            st.error(f"❌ Error: {response.get('error', 'Unknown error occurred')}")
