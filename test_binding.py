from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
import os
from dotenv import load_dotenv

load_dotenv()

@tool
def analyze_data(code: str):
    """Analyze data."""
    pass

@tool
def query_knowledge_base(query: str):
    """Query knowledge base."""
    pass

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

print("Attempting to bind tools...")
try:
    llm_with_tools = llm.bind_tools([analyze_data, query_knowledge_base])
    print("SUCCESS")
except Exception as e:
    print(f"FAILURE: {e}")
    import traceback
    traceback.print_exc()
