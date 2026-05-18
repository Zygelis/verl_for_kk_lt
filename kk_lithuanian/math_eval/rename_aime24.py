import pandas as pd

# Define file paths
parquet_file = "kk_lithuanian/math_eval/math_datasets/aime2024.parquet"
jsonl_file = "kk_lithuanian/math_eval/math_datasets/aime2024.jsonl"

# Read the Parquet file into a DataFrame
df = pd.read_parquet(parquet_file)

df['solution'] = df['solution'].str.extract(r'\\boxed{([^}]+)}')

df['solution'] = pd.to_numeric(df['solution'], errors='coerce').astype('Int64')

# Dictionary mapping old field names to new field names
# Syntax: {'old_name': 'new_name'}
rename_mapping = {
    'solution': 'answer'
}

# Rename the columns
df = df.rename(columns=rename_mapping)

desired_columns_order = ['problem', 'answer', 'id']  # Specify the desired order of columns
df = df[desired_columns_order]

# Export to JSON Lines format
df.to_json(jsonl_file, orient='records', lines=True)
