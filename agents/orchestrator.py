import os
import time
from typing import TypedDict, Literal, Optional, Union, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# Real Tool Imports
from tools.pandas_tools import execute_pandas_code
from utils.rag_handler import get_retriever, update_index

load_dotenv()

# ── 1. Define State Management ────────────────────────────────────────────────
class State(TypedDict):
    user_query: str
    pandas_metadata: str      # Schema only
    rag_context: str
    generated_code: str
    pandas_result: Union[str, Dict[str, Any]]
    error_log: str
    retry_count: int
    final_response: str
    selected_route: str       # Internal tracking for routing
    df_context: Any           # The actual DataFrame object (not serialized)
    model_name: str           # User-selected model

# ── 2. Initialize Model ───────────────────────────────────────────────────────
def get_model(model_name="gemini-2.5-flash"):
    """Initializes the model with robust retry logic for 429 errors."""
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.1,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        max_retries=3  # Automatically handles transient "Resource Exhausted" errors
    )

# ── 3. Graph Nodes ────────────────────────────────────────────────────────────

def router_node(state: State):
    """Analyzes the query and decides the next step."""
    print("--- NODE: ROUTER ---")
    llm = get_model(state.get('model_name', 'gemini-2.5-flash'))
    
    prompt = f"""
    You are a routing assistant for an AI Data Analyst.
    User Query: {state['user_query']}
    
    Determine if this query requires:
    1. 'pandas': For numerical calculations, trends, or structured data analysis.
    2. 'rag': For retrieving context from documents, policies, or text notes.
    3. 'hybrid': If both are needed (e.g., comparing data trends against a policy).
    
    Respond with ONLY one word: 'pandas', 'rag', or 'hybrid'.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content
    if isinstance(content, list):
        content = content[0] if content else ""
    
    route = str(content).strip().lower()
    
    # Validation
    if route not in ['pandas', 'rag', 'hybrid']:
        route = 'pandas' # Default fallback
        
    return {"selected_route": route}

def pandas_coder_node(state: State):
    """Generates Python/Pandas code."""
    print("--- NODE: PANDAS CODER ---")
    llm = get_model(state.get('model_name', 'gemini-2.5-flash'))
    
    error_context = f"\nPrevious Error: {state['error_log']}" if state['error_log'] else ""
    
    prompt = f"""
    You are a Python Data Analyst. Write code to answer the user query using the provided metadata.
    
    User Query: {state['user_query']}
    Data Schema: {state['pandas_metadata']}
    {error_context}
    
    Instructions:
    - Use the variable 'df' (it is already loaded).
    - Use 'print()' to show numerical results.
    - If a plot is needed, create it but DO NOT use plt.show().
    - If the result is a table or multiple rows, assign it to a variable named 'result_df'.
    - Provide ONLY the Python code without markdown blocks or explanations.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content
    if isinstance(content, list):
        content = content[0] if content else ""
        
    code = str(content).replace("```python", "").replace("```", "").strip()
    
    return {"generated_code": code}

def pandas_executor_node(state: State):
    """Handles execution and self-healing logic."""
    print("--- NODE: PANDAS EXECUTOR ---")
    
    df = state.get("df_context")
    if df is None:
        return {"error_log": "No dataset loaded.", "pandas_result": "Please upload a CSV/Excel file first."}
        
    result = execute_pandas_code(df, state['generated_code'])
    
    if result.get('error'):
        return {
            "error_log": result['error'],
            "retry_count": state['retry_count'] + 1,
            "pandas_result": result['error']
        }
    
    return {
        "pandas_result": result, # Return the full dict (result + plot + table)
        "error_log": "",
        "retry_count": state['retry_count']
    }

def rag_retriever_node(state: State):
    """Retrieves document context from the real FAISS vector store."""
    print("--- NODE: RAG RETRIEVER ---")
    
    retriever = get_retriever()
    if not retriever:
        return {"rag_context": "No documents found in knowledge base."}
        
    docs = retriever.invoke(state['user_query'])
    context = "\n".join([doc.page_content for doc in docs])
    
    return {"rag_context": context}

def summarizer_node(state: State):
    """Synthesizes the final answer."""
    print("--- NODE: SUMMARIZER ---")
    llm = get_model(state.get('model_name', 'gemini-2.5-flash'))
    
    prompt = f"""
    You are an AI Data Analyst giving a final report.
    User Query: {state['user_query']}
    Pandas Result: {state['pandas_result']}
    RAG Context: {state['rag_context']}
    
    Synthesize a clear, multi-line, professional response. 
    1. If both findings are present, integrate them logically.
    2. If a plot was generated (noted in Pandas Result), mention it briefly.
    3. If a table was generated (noted in Pandas Result), refer to it as "the data below".
    4. Ensure the output is clean and easy to read.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content
    if isinstance(content, list):
        content = "\n".join([str(c) for c in content])
        
    return {"final_response": str(content)}

# ── 4. Define Graph Logic (The Flow) ──────────────────────────────────────────

def create_orchestrator():
    workflow = StateGraph(State)
    
    # Add Nodes
    workflow.add_node("router", router_node)
    workflow.add_node("pandas_coder", pandas_coder_node)
    workflow.add_node("pandas_executor", pandas_executor_node)
    workflow.add_node("rag_retriever", rag_retriever_node)
    workflow.add_node("summarizer", summarizer_node)
    
    # Define Edges
    workflow.set_entry_point("router")
    
    # Conditional edge from router
    def router_decision(state: State):
        if state['selected_route'] in ['pandas', 'hybrid']:
            return "pandas_coder"
        return "rag_retriever"
    
    workflow.add_conditional_edges("router", router_decision)
    
    # Linear edge from coder to executor
    workflow.add_edge("pandas_coder", "pandas_executor")
    
    # Conditional edge from executor for self-healing and routing
    def executor_decision(state: State):
        if state['error_log'] and state['retry_count'] < 3:
            return "pandas_coder"
        if state['selected_route'] == 'hybrid':
            return "rag_retriever"
        return "summarizer"
    
    workflow.add_conditional_edges("pandas_executor", executor_decision)
    
    # Linear edge from rag to summarizer
    workflow.add_edge("rag_retriever", "summarizer")
    
    workflow.add_edge("summarizer", END)
    
    return workflow.compile()
