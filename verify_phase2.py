import pandas as pd
import matplotlib.pyplot as plt
from utils.data_handler import load_data, plot_to_base64, get_df_info
from tools.pandas_tools import execute_pandas_code, get_data_metadata

# 1. Create a sample CSV for testing
df_test = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']
})
test_csv = 'test_data.csv'
df_test.to_csv(test_csv, index=False)

print("Created test_data.csv")

# 2. Test loading and metadata
df = load_data(test_csv)
print("\n--- Metadata ---")
print(get_data_metadata(df))

# 3. Test executing pandas code (analysis)
analysis_code = """
print(df.describe())
"""
print("\n--- Running Analysis ---")
result = execute_pandas_code(df, analysis_code)
print(result['result'])

# 4. Test plot generation and Base64 conversion
plot_code = """
import matplotlib.pyplot as plt
df.plot(kind='bar', x='Name', y='Age')
plt.title('Sample Plot')
"""
print("\n--- Generating Plot ---")
result_plot = execute_pandas_code(df, plot_code)
if result_plot['plot']:
    print(f"Plot converted to Base64 (length: {len(result_plot['plot'])} characters)")
else:
    print("No plot generated")

if result_plot['error']:
    print(f"Error: {result_plot['error']}")

# Cleanup
import os
if os.path.exists(test_csv):
    os.remove(test_csv)
print("\nCleaned up test files.")
