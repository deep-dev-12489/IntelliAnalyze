import traceback
import pandas as pd
from langchain_core.messages import HumanMessage
from agents.orchestrator import create_orchestrator, call_model

# Prepare dummy state
df = pd.DataFrame({'test': [1, 2, 3]})
state = {
    "messages": [HumanMessage(content="Hello")],
    "df": df,
    "retriever": None
}

print("Invoking call_model node manual check...")
try:
    result = call_model(state)
    print("SUCCESS")
    print(result)
except Exception as e:
    print(f"FAILURE: {e}")
    traceback.print_exc()
