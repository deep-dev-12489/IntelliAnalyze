import pandas as pd
import sys
import io
import contextlib
import matplotlib.pyplot as plt
from typing import Dict, Any, Union
from utils.data_handler import plot_to_base64, get_df_info

def get_data_metadata(df: pd.DataFrame) -> str:
    """
    Returns the dataset schema (column names, types, and null counts).
    This is the only information passed to the LLM context about the data.
    """
    return get_df_info(df)

def execute_pandas_code(df: pd.DataFrame, code: str) -> Dict[str, Any]:
    """
    Executes pandas code locally on the provided DataFrame.
    Returns a dictionary containing:
    - 'result': The string output of the code (printed or returned).
    - 'plot': Base64 string of the plot if one was generated.
    - 'error': Error message if execution failed.
    """
    output = io.StringIO()
    result_data = {"result": "", "plot": None, "table": None, "error": None}
    
    # Define the local environment for code execution
    local_vars = {"df": df, "pd": pd, "plt": plt}
    
    try:
        with contextlib.redirect_stdout(output):
            # Use local_vars for both globals and locals to ensure capture
            exec(code, local_vars)
        
        result_data["result"] = output.getvalue()
        
        # DEBUG: Print keys in local_vars
        # print(f"DEBUG: local_vars keys: {list(local_vars.keys())}", file=sys.stderr)
        
        # Capture result_df if it exists and is a DataFrame
        if "result_df" in local_vars:
            # print(f"DEBUG: result_df type: {type(local_vars['result_df'])}", file=sys.stderr)
            if isinstance(local_vars["result_df"], pd.DataFrame):
                result_data["table"] = local_vars["result_df"]
        
        # Check if a plot was generated in the current figure
        if plt.get_fignums():
            fig = plt.gcf()
            result_data["plot"] = plot_to_base64(fig)
            plt.close('all') # Clear all figures after conversion
            
    except Exception as e:
        result_data["error"] = str(e)
        
    return result_data

from langchain_core.tools import tool
from typing import Dict, Any, Union

@tool
def analyze_data(code: str):
    """
    Execute Python/Pandas code on the loaded dataset 'df'.
    Use this for calculations, trends, aggregations, and generating charts.
    The code should assume 'df' is the main DataFrame.
    Returns: {'result': str, 'plot': str (base64 if any), 'error': str}
    """
    # Note: 'df' must be provided in the tool's execution context
    pass
