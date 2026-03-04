# streamlit_app.py

import streamlit as st
import os
import sys
agent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'agent')) 
if agent_dir not in sys.path: 
    sys.path.insert(0, agent_dir)
from agent_implementation import SensorDataAgent, CSV_PATH, DATA_PATH
import time

def normalize_user_prompt(prompt: str) -> str:
    """
    Light normalization to help the agent + tools.
    This does NOT execute logic, only rewrites intent.
    """
    p = prompt.lower().strip()

    # ---- Feature analysis shortcuts
    if any(k in p for k in ["top features", "feature importance", "important features"]):
        return "show top features"

    # ---- Dataset inspection shortcuts
    if any(k in p for k in ["dataset", "structure", "available sensors", "info"]):
        return "show dataset info"

    # ---- Plot intent normalization
    if p.startswith("plot ") or p.startswith("show "):
        # strip leading verbs, tools don't need them
        p = p.replace("plot ", "").replace("show ", "")

    return p


st.set_page_config(
    page_title="Sensor Data AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🤖 Sensor Data Analysis Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Statistical AI Agent for OK/KO Dataset Analysis</div>', unsafe_allow_html=True)

# Initialize agent (cached)
@st.cache_resource
def load_agent():
    """Load and cache the AI agent."""
    with st.spinner("🔄 Loading AI model... This may take up to a minute on first run."):
        try:
            agent = SensorDataAgent(
                df_path=str(CSV_PATH),
                path_to_data=str(DATA_PATH),
            )
            return agent, None
        except Exception as e:
            return None, str(e)

agent, error = load_agent()

if error:
    st.error(f"❌ Failed to load agent: {error}")
    st.stop()

# -----------------------------
# Session state initialization
# -----------------------------
if "plots" not in st.session_state:
    st.session_state["plots"] = []

if "last_query" not in st.session_state:
    st.session_state["last_query"] = None

# -------------------------------------------------
# Sidebar: Dataset overview + quick actions
# -------------------------------------------------
with st.sidebar:
    st.markdown("### 📊 Dataset Overview")

    try:
        summary = agent.get_dataset_summary()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Samples", summary["total_samples"])
        with col2:
            st.metric("Features", len(summary["columns"]))

        col3, col4 = st.columns(2)
        with col3:
            st.metric("OK Samples", summary["ok_samples"], delta_color="off")
        with col4:
            st.metric("KO Samples", summary["ko_samples"], delta_color="off")

    except Exception as e:
        st.warning(f"Could not load dataset summary: {e}")

    st.markdown("---")

    # -----------------------------
    # Quick Actions (tool-aligned)
    # -----------------------------
    st.markdown("### 🚀 Quick Actions")

    if st.button("📋 Dataset Info", use_container_width=True):
        st.session_state.pending_query = "show dataset info"
        st.rerun()

    if st.button("⭐ Top Features", use_container_width=True):
        st.session_state.pending_query = "show top features"
        st.rerun()

    if st.button("📈 ACC – OK – Time", use_container_width=True):
        st.session_state.pending_query = "acc OK time"
        st.rerun()

    if st.button("📊 MIC – KO – Frequency", use_container_width=True):
        st.session_state.pending_query = "mic KO frequency"
        st.rerun()

    st.markdown("---")

    # -----------------------------
    # Example Queries (executable)
    # -----------------------------
    st.markdown("### 💡 Example Queries")

    examples = [
        "acc OK time",
        "mic KO frequency",
        "gyro KO time vel-fissa",
        "mag OK frequency",
        "show top features",
        "show dataset info",
    ]

    for example in examples:
        if st.button(f"📌 {example}", key=f"ex_{example}", use_container_width=True):
            st.session_state.pending_query = example
            st.rerun()

    st.markdown("---")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# Main chat interface
st.markdown("### 💬 Chat with Your Data")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------------------------------
# Render persisted plots (CRITICAL POSITION)
# -------------------------------------------------
if st.session_state.plots:
    st.markdown("### 📈 Generated Plots")
    for fig in st.session_state.plots:
        st.pyplot(fig, clear_figure=False)

# -----------------------------
# Plot persistence (CRITICAL)
# -----------------------------
if "plots" not in st.session_state:
    st.session_state.plots = []

if "last_query" not in st.session_state:
    st.session_state.last_query = None

# Handle pending query from sidebar
if st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None

    # 🔥 Clear old plots if this is a new query
    if st.session_state.last_query != query:
        st.session_state.plots = []
        st.session_state.last_query = query

    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            start_time = time.time()
            response = agent.run(query)
            elapsed = time.time() - start_time
            
            st.markdown(response)
            st.caption(f"⏱️ Response time: {elapsed:.2f}s")
    
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# Chat input
if prompt := st.chat_input("Ask about your sensor data... (e.g., 'plot acc OK time')"):

    normalized = normalize_user_prompt(prompt)

    if st.session_state.last_query != normalized:
        st.session_state.plots = []
        st.session_state.last_query = normalized

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyzing your request..."):
            start_time = time.time()
            normalized = normalize_user_prompt(prompt)
            response = agent.run(normalized)

            elapsed = time.time() - start_time
            
            st.markdown(response)
            st.caption(f"⏱️ Response time: {elapsed:.2f}s")
    
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})

# Help section at the bottom
with st.expander("ℹ️ How to Use This Agent"):
    st.markdown("""
    ### Plotting Commands
    Use this format:

    `<sensor|sensor_type> <OK/KO> <time|frequency> [condition] [rpm|stwin]`

    **Examples**
    - `acc OK time`
    - `mic KO frequency`
    - `gyro KO time vel-fissa`
    - `iis3dwb_acc OK time no-load-cycles STWIN_00012`
    
    ### Feature Analysis
    Simply ask:
    - `"Show me the top features"`
    - `"What features discriminate OK from KO?"`
    - `"Display feature importance"`
    
    ### Dataset Information
    Ask questions like:
    - `"What sensors are available?"`
    - `"Show dataset structure"`
    - `"How many OK and KO samples?"`
    """)

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666; font-size: 0.9rem;">'
    '🎓 System and Device Programming - Project Q3 | Powered by qwen2.5:7b-instruct'
    '</div>',
    unsafe_allow_html=True
)