import os
import pandas as pd
from agents.orchestrator import create_orchestrator
from utils.rag_handler import ingest_knowledge_base, get_retriever
from tools.pandas_tools import get_df_info
import asyncio
import time

async def run_test():
    print("--- 1. Preparing Test Environment ---")
    
    # Create test data
    test_csv = 'test_sales_data.csv'
    df = pd.DataFrame({
        'Product': ['Apple', 'Banana', 'Cherry'],
        'Revenue': [100, 200, 300]
    })
    df.to_csv(test_csv, index=False)
    
    # Create test RAG directory and file
    test_rag_dir = 'knowledge_base'
    if not os.path.exists(test_rag_dir):
        os.makedirs(test_rag_dir)
        
    # Initialize Orchestrator
    orchestrator = create_orchestrator()
    
    # --- 2. Test Cases ---
    test_cases = [
        "What was the total revenue?", # Pandas
    ]
    
    for i, query in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: '{query}' ---")
        
        # Prepare initial state
        metadata = get_df_info(df)
        initial_state = {
            "user_query": query,
            "pandas_metadata": metadata,
            "rag_context": "",
            "generated_code": "",
            "pandas_result": "",
            "error_log": "",
            "retry_count": 0,
            "final_response": "",
            "selected_route": "",
            "df_context": df,
            "model_name": "gemini-flash-latest"
        }
        
        # LangGraph invocation
        final_state = orchestrator.invoke(initial_state)
        
        # Display Final Message
        print(f"ROUTE: {final_state.get('selected_route')}")
        print(f"CODE: {final_state.get('generated_code')}")
        print(f"RESPONSE:\n{final_state.get('final_response')}")

    # Cleanup
    if os.path.exists(test_csv):
        os.remove(test_csv)
    print("\n--- 3. Verification Sample Complete ---")

if __name__ == "__main__":
    asyncio.run(run_test())
