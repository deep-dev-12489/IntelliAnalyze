import streamlit as st
import pandas as pd
import os
import base64
from agents.orchestrator import create_orchestrator
from utils.rag_handler import update_index, get_retriever
from tools.pandas_tools import get_df_info
from dotenv import load_dotenv

load_dotenv()

# ── 1. Page Configuration & Theme ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Analyst Dashboard",
    page_icon="🤖",
    layout="wide",
)

# Dark Theme Aesthetics (Custom CSS)
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stChatMessage {
        background-color: #1e2130 !important;
        border-radius: 10px;
    }
    .stSidebar {
        background-color: #161b22;
    }
    </styles>
    """, unsafe_allow_html=True)

# ── 2. Session State Initialization ───────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "df" not in st.session_state:
    st.session_state.df = None

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = create_orchestrator()

# ── 3. Sidebar (File Uploads) ─────────────────────────────────────────────────
with st.sidebar:
    st.title("🗂 Data Center")
    st.markdown("---")
    
    # 3.1 Structured Data (Pandas)
    st.subheader("📊 Structured Data")
    uploaded_data = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    
    if uploaded_data:
        if uploaded_data.name.endswith('.csv'):
            st.session_state.df = pd.read_csv(uploaded_data)
        else:
            st.session_state.df = pd.read_excel(uploaded_data)
        st.success(f"Loaded: {uploaded_data.name}")
        with st.expander("Show Schema"):
            st.code(get_df_info(st.session_state.df))

    st.markdown("---")
    
    # 3.2 Unstructured Data (RAG)
    st.subheader("📄 Knowledge Base")
    uploaded_pdf = st.file_uploader("Upload PDF Documents", type=["pdf"])
    
    if uploaded_pdf:
        # Save temp file for processing
        temp_path = os.path.join("knowledge_base", uploaded_pdf.name)
        if not os.path.exists("knowledge_base"):
            os.makedirs("knowledge_base")
            
        with open(temp_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())
            
        with st.spinner("Indexing document..."):
            if update_index(temp_path):
                st.success(f"Indexed: {uploaded_pdf.name}")
            else:
                st.error("Failed to index.")

    st.markdown("---")
    
    # 3.3 Model Configuration
    st.subheader("⚙️ Model Settings")
    selected_model = st.selectbox(
        "Choose Engine",
        ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest"],
        index=0,
        help="Switch models if you hit rate limits."
    )

# ── 4. Chat Interface ─────────────────────────────────────────────────────────
st.title("🤖 AI Data Analyst Agent")
st.caption("Ask questions about your datasets or document knowledge base.")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "plot" in message and message["plot"]:
            st.image(base64.b64decode(message["plot"]), caption="Generated Plot")
        if "table" in message and message["table"]:
            st.dataframe(message["table"])

# Handle User Input
if prompt := st.chat_input("Ex: What was the total revenue in Q3?"):
    # 1. Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Run Agent Logic
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Prepare state
            metadata = get_df_info(st.session_state.df) if st.session_state.df is not None else "No data loaded."
            
            initial_state = {
                "user_query": prompt,
                "pandas_metadata": metadata,
                "rag_context": "",
                "generated_code": "",
                "pandas_result": "",
                "error_log": "",
                "retry_count": 0,
                "final_response": "",
                "selected_route": "",
                "df_context": st.session_state.df,
                "model_name": selected_model
            }
            
            # Execute Graph
            try:
                final_state = st.session_state.orchestrator.invoke(initial_state)
                
                # --- SHOW AGENT PROCESS ---
                with st.expander("🛠️ Agent Process Log", expanded=False):
                    st.write(f"**Selected Route:** `{final_state.get('selected_route', 'Unknown')}`")
                    if final_state.get('generated_code'):
                        st.write("**Generated Code:**")
                        st.code(final_state['generated_code'], language='python')
                    if final_state.get('rag_context'):
                        st.write("**Retrieved Context:**")
                        st.info(final_state['rag_context'][:500] + "...") # Snippet
                # ---------------------------

                # 3. Handle Dual-Display Logic
                response_content = final_state['final_response']
                st.markdown(response_content)
                
                # Extract Results
                plot_b64 = None
                result_df = None
                pandas_out = final_state.get('pandas_result')
                
                if isinstance(pandas_out, dict):
                    plot_b64 = pandas_out.get('plot')
                    result_df = pandas_out.get('table')
                
                # Display Plot
                if plot_b64:
                    st.image(base64.b64decode(plot_b64), caption="Generated Visualization")
                
                # Display Table
                if result_df is not None:
                    st.dataframe(result_df)
                
                # Store in history
                message_entry = {
                    "role": "assistant",
                    "content": response_content,
                    "plot": plot_b64,
                    "table": result_df
                }
                st.session_state.messages.append(message_entry)

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    st.warning("⚠️ **API Rate Limit Exceeded.** The Free Tier limit has been reached. Please wait 60 seconds or try switching to a different model in the sidebar.")
                else:
                    st.error(f"❌ **An unexpected error occurred:** {error_msg}")

# ── 5. Footer/Quick Info ──────────────────────────────────────────────────────
if st.session_state.df is not None:
    with st.expander("View Raw Data"):
        st.dataframe(st.session_state.df.head(10))
