from langchain_core.tools import tool
from typing import List, Dict, Any

@tool
def query_knowledge_base(query: str):
    """
    Search the knowledge base for contextual information from PDF reports,
    meeting notes, and text documents.
    Use this for qualitative data, background context, and non-numerical insights.
    """
    # Note: 'retriever' must be provided in the tool's execution context
    pass
