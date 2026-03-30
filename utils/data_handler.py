import pandas as pd
import io
import base64
import matplotlib.pyplot as plt

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV or Excel file into a Pandas DataFrame.
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

def plot_to_base64(fig) -> str:
    """
    Converts a Matplotlib figure to a base64 encoded string.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)  # Close the figure to free up memory
    return img_str

def get_df_info(df: pd.DataFrame) -> str:
    """
    Returns a string representation of df.info() for the LLM context.
    """
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()
