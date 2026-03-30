import pandas as pd
from agents.orchestrator import create_orchestrator
from utils.data_handler import get_df_info
import os
from dotenv import load_dotenv

load_dotenv()

# Setup dummy data
df = pd.DataFrame({
    'Product': ['Apple', 'Banana', 'Cherry'],
    'Revenue': [100, 200, 300]
})

def test_pandas_flow():
    print("\n--- Testing Pandas Flow ---")
    orchestrator = create_orchestrator()
    
    initial_state = {
        "user_query": "What is the total revenue and show me the data table.",
        "pandas_metadata": get_df_info(df),
        "rag_context": "",
        "generated_code": "",
        "pandas_result": "",
        "error_log": "",
        "retry_count": 0,
        "final_response": "",
        "selected_route": "",
        "df_context": df
    }
    
    final_state = orchestrator.invoke(initial_state)
    
    import json
    def serialize_df(obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        return str(obj)

    results = {
        "selected_route": final_state['selected_route'],
        "generated_code": final_state['generated_code'],
        "final_response": final_state['final_response'],
        "pandas_result": {k: serialize_df(v) for k, v in final_state['pandas_result'].items()} if isinstance(final_state['pandas_result'], dict) else final_state['pandas_result']
    }
    
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to test_results.json")

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in .env")
    else:
        test_pandas_flow()
