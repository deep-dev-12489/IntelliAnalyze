import os
from dotenv import load_dotenv
import traceback

# 1. Force load from root
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
print(f"API KEY Found: {'Yes (ends with ' + api_key[-4:] + ')' if api_key else 'No'}")

from agents.orchestrator import create_orchestrator
from langchain_core.messages import HumanMessage
import pandas as pd

async def run_test():
    try:
        df = pd.DataFrame({'a': [1]})
        graph = create_orchestrator()
        state = {"messages": [HumanMessage(content="hi")], "df": df, "retriever": None}
        print("Invoking graph...")
        res = graph.invoke(state)
        print("Success!")
        print(res['messages'][-1].content)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_test())
