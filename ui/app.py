"""
Finance House Policy Assistant - Clean Professional UI (Sidebar Fixed)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import time
from typing import Dict, Any
from src.chain import FinanceHousePolicyChain

# Page config
st.set_page_config(
    page_title="Finance House Policy Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean Professional CSS - Sidebar Enhanced
st.markdown("""
<style>
    /* Force clean white background */
    .stApp {
        background-color: #FFFFFF;
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }
    
    /* Professional header */
    .main-header {
        background: linear-gradient(135deg, #1A3A5C 0%, #2C5282 100%);
        padding: 2.5rem 2rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .main-header h1 {
        color: #FFFFFF;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: #E8F0F8;
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
        font-weight: 400;
    }
    
    /* User message */
    .user-msg {
        background-color: #F8FAFC;
        border-left: 3px solid #1A3A5C;
        padding: 1.2rem 1.5rem;
        margin: 1.5rem 0;
        border-radius: 4px;
    }
    
    .user-msg .label {
        color: #1A3A5C;
        font-weight: 600;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    .user-msg .content {
        color: #2C3E50;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    
    /* Assistant message */
    .bot-msg {
        background-color: #FFFFFF;
        border: 1px solid #E0E7EF;
        border-left: 3px solid #10B981;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .bot-msg .label {
        color: #10B981;
        font-weight: 600;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 1rem;
        display: block;
    }
    
    .bot-msg .content {
        color: #2C3E50;
        font-size: 1.05rem;
        line-height: 1.7;
    }
    
    /* Policy reference box */
    .policy-box {
        background-color: #FFFBF0;
        border: 1px solid #F0E5CC;
        border-left: 3px solid #C9A961;
        padding: 1.2rem 1.5rem;
        margin: 1.5rem 0;
        border-radius: 4px;
    }
    
    .policy-box .title {
        color: #8B7355;
        font-weight: 600;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 1rem;
        display: block;
    }
    
    .policy-box .item {
        color: #5A4A3A;
        font-size: 0.95rem;
        line-height: 1.8;
        margin: 0.3rem 0;
    }
    
    .policy-box .item strong {
        color: #3A2A1A;
        font-weight: 600;
    }
    
    .confidence-high {
        color: #059669;
        font-weight: 600;
    }
    
    .confidence-med {
        color: #D97706;
        font-weight: 600;
    }
    
    /* Related questions box */
    .related-box {
        background-color: #F0F9FF;
        border: 1px solid #DBEAFE;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    .related-box .title {
        color: #1E40AF;
        font-weight: 600;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.8rem;
        display: block;
    }
    
    .related-box .question {
        color: #3B5998;
        font-size: 0.95rem;
        line-height: 1.6;
        margin: 0.5rem 0;
    }
    
    /* Metrics */
    .metrics {
        color: #64748B;
        font-size: 0.85rem;
        margin: 1rem 0;
        padding: 0.5rem 0;
        border-top: 1px solid #F1F5F9;
    }
    
    /* SIDEBAR STYLING - ENHANCED */
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
        padding-top: 1rem !important;
    }
    
    /* Sidebar headings - BIGGER and BOLDER */
    section[data-testid="stSidebar"] h3 {
        color: #1A3A5C !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
        margin: 1.5rem 0 1rem 0 !important;
        padding-bottom: 0.6rem !important;
        border-bottom: 3px solid #1A3A5C !important;
        letter-spacing: -0.3px !important;
    }
    
    /* First heading (About) - extra spacing at top */
    section[data-testid="stSidebar"] h3:first-of-type {
        margin-top: 0 !important;
    }
    
    /* Sidebar info box - ENHANCED with darker border */
    section[data-testid="stSidebar"] .stAlert {
        background-color: #FFFFFF !important;
        border: 2px solid #2C5282 !important;
        border-radius: 8px !important;
        padding: 1.2rem !important;
        margin-bottom: 1.5rem !important;
        box-shadow: 0 1px 4px rgba(26, 58, 92, 0.1) !important;
    }
    
    section[data-testid="stSidebar"] .stAlert p {
        color: #2C3E50 !important;
        font-size: 0.9rem !important;
        line-height: 1.7 !important;
        margin: 0.4rem 0 !important;
    }
    
    section[data-testid="stSidebar"] .stAlert strong {
        color: #1A3A5C !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar sample question buttons - DARKER BOXES */
    section[data-testid="stSidebar"] .stButton button {
        background-color: #FFFFFF !important;
        color: #2C3E50 !important;
        border: 2px solid #94A3B8 !important;
        border-radius: 6px !important;
        padding: 0.7rem 1rem !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        transition: all 0.2s !important;
        text-align: left !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    }
    
    section[data-testid="stSidebar"] .stButton button:hover {
        background-color: #E8F2FC !important;
        border-color: #1A3A5C !important;
        color: #1A3A5C !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 6px rgba(26, 58, 92, 0.15) !important;
    }
    
    /* Input section heading */
    h3 {
        color: #1A3A5C !important;
        font-weight: 600 !important;
        margin-top: 2rem !important;
    }
    
    /* Input area */
    .stTextInput input {
        border-radius: 4px !important;
        border: 1px solid #CBD5E1 !important;
        padding: 0.75rem 1rem !important;
        font-size: 1rem !important;
        color: #2C3E50 !important;
    }
    
    .stTextInput input:focus {
        border-color: #1A3A5C !important;
        box-shadow: 0 0 0 2px rgba(26, 58, 92, 0.1) !important;
    }
    
    /* Send button */
    .stButton button[kind="primary"] {
        background-color: #1A3A5C !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.2s !important;
    }
    
    .stButton button[kind="primary"]:hover {
        background-color: #2C5282 !important;
        box-shadow: 0 2px 8px rgba(26, 58, 92, 0.2) !important;
    }
    
    /* Success message */
    .stSuccess {
        background-color: #D1FAE5 !important;
        color: #065F46 !important;
        font-weight: 600 !important;
        border: 1px solid #10B981 !important;
        padding: 0.75rem 1rem !important;
        border-radius: 4px !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Clean scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #F1F5F9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #CBD5E1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94A3B8;
    }
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


# Header
st.markdown("""
<div class="main-header">
    <h1>🏦 Finance House Policy Assistant</h1>
    <p>Your intelligent guide to organizational policies and procedures</p>
</div>
""", unsafe_allow_html=True)


# Sidebar - About at the very top
with st.sidebar:
    st.markdown("### 📚 About")
    st.info("""
    **Finance House Policy Assistant** uses advanced AI to provide accurate, policy-based answers.
    
    **Features:**
    
    • Multi-Query RAG  
    • Intent Detection  
    • 10 Policy Documents
    """)
    
    st.markdown("### 💡 Sample Questions")
    
    sample_qs = [
        "Can I work from home?",
        "What laptop budget do I have?",
        "How do I claim travel expenses?",
        "Can I accept gifts from clients?",
        "How much vacation do I have?"
    ]
    
    for q in sample_qs:
        if st.button(q, key=f"btn_{q}", use_container_width=True):
            st.session_state.pending_query = q
            st.rerun()
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", st.session_state.query_count)
    with col2:
        st.metric("Policies", "10")
    
    st.markdown("")
    if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
        st.session_state.chat_history = []
        st.session_state.query_count = 0
        st.rerun()


# Display chat history
for msg in st.session_state.chat_history:
    if msg['type'] == 'user':
        st.markdown(f"""
        <div class="user-msg">
            <span class="label">Your Question</span>
            <div class="content">{msg['content']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    elif msg['type'] == 'assistant':
        resp = msg['response']
        ans = resp['answer']
        
        # Answer
        st.markdown(f"""
        <div class="bot-msg">
            <span class="label">Policy Assistant Answer</span>
            <div class="content">{ans['text']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Policy Reference
        conf_class = "confidence-high" if ans['confidence'] >= 0.8 else "confidence-med"
        st.markdown(f"""
        <div class="policy-box">
            <span class="title">📋 Policy Reference</span>
            <div class="item"><strong>Policy:</strong> {ans['policy_number']}</div>
            <div class="item"><strong>Owner:</strong> {ans['policy_owner']}</div>
            <div class="item"><strong>Section:</strong> {ans['relevant_clause']}</div>
            <div class="item"><strong>Confidence:</strong> <span class="{conf_class}">{ans['confidence']:.0%}</span></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Related Questions
        if resp.get('related_questions'):
            questions_html = "\n".join([
                f'<div class="question">{i}. {q}</div>' 
                for i, q in enumerate(resp['related_questions'], 1)
            ])
            st.markdown(f"""
            <div class="related-box">
                <span class="title">💡 Related Questions</span>
                {questions_html}
            </div>
            """, unsafe_allow_html=True)
        
        # Metrics
        meta = resp['metadata']
        st.markdown(f"""
        <div class="metrics">
            ⏱️ Response Time: {meta.get('total_duration', 0):.1f}s  |  
            📊 Policies Searched: {meta.get('num_policies_searched', 0)}  |  
            📄 Sections Analyzed: {meta.get('chunks_used', 0)}
        </div>
        """, unsafe_allow_html=True)


# Handle pending query from sidebar
if 'pending_query' in st.session_state and st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None
    
    # Add to history
    st.session_state.chat_history.append({
        'type': 'user',
        'content': query
    })
    
    # Process
    with st.spinner("Processing your question..."):
        response = st.session_state.rag_chain.query(query)
    
    if response['success']:
        st.session_state.chat_history.append({
            'type': 'assistant',
            'response': response
        })
        st.session_state.query_count += 1
    else:
        st.error(f"Error: {response.get('error', 'Unknown error')}")
    
    st.rerun()


# INPUT AT BOTTOM
st.markdown("---")
st.markdown("### 💬 Ask a Question")

col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_input(
        "Type your question here...",
        placeholder="e.g., What are the requirements for remote work?",
        label_visibility="collapsed",
        key="user_input_box"
    )

with col2:
    send_clicked = st.button("Send", use_container_width=True, type="primary")

if send_clicked and user_input:
    # Add to history
    st.session_state.chat_history.append({
        'type': 'user',
        'content': user_input
    })
    
    # Process with progress
    with st.spinner("Processing your question..."):
        progress = st.progress(0)
        status = st.empty()
        
        status.info("🔍 Understanding your question...")
        progress.progress(20)
        time.sleep(0.2)
        
        status.info("🔄 Generating search strategies...")
        progress.progress(40)
        time.sleep(0.2)
        
        status.info("📚 Searching policy documents...")
        progress.progress(60)
        
        response = st.session_state.rag_chain.query(user_input)
        
        status.info("🎯 Selecting most relevant policy...")
        progress.progress(80)
        time.sleep(0.2)
        
        status.info("✍️ Generating detailed answer...")
        progress.progress(100)
        time.sleep(0.2)
        
        status.empty()
        progress.empty()
    
    if response['success']:
        st.session_state.chat_history.append({
            'type': 'assistant',
            'response': response
        })
        st.session_state.query_count += 1
    else:
        st.error(f"❌ Error: {response.get('error', 'Unknown error')}")
    
    st.rerun()