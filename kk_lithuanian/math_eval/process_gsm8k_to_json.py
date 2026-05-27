import pandas as pd
import re

INPUT_FILE = "kk_lithuanian/math_eval/math_datasets/gsm8k/socratic/test-00000-of-00001.parquet"
OUTPUT_FILE = "kk_lithuanian/math_eval/math_datasets/gsm8k/gsm8k_socratic.jsonl"
N = 100

# 1. Load the parquet file
df = pd.read_parquet(INPUT_FILE).head(N)  # Load only the first N rows for testing

# 2. Rename 'question' to 'problem'
df = df.rename(columns={'question': 'problem'})

# 3. Define a function to split the answer field
def process_answer(row):
    full_answer = str(row['answer'])
    
    # Split by the specific delimiter \n####
    if '####' in full_answer:
        parts = full_answer.split('\n####')
        text_reasoning = parts[0].strip()
        # Extract the numeric part (handling potential whitespace)
        numeric_answer = parts[1].strip()
        
        # Convert to numeric (float or int) if possible
        try:
            numeric_answer = pd.to_numeric(numeric_answer)
        except ValueError:
            pass 
            
        return text_reasoning, numeric_answer
    return full_answer, None

# 4. Apply the splitting logic
df[['text_reasoning', 'answer']] = df.apply(
    lambda x: pd.Series(process_answer(x)), axis=1
)

# 5. Keep only the requested fields
final_df = df[['problem', 'text_reasoning', 'answer']]

# 6. Export to JSONL (one JSON object per line)
final_df.to_json(OUTPUT_FILE, orient='records', lines=True)

print(f"Conversion complete: {OUTPUT_FILE}")