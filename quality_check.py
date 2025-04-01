import openai
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Set your OpenAI API key
openai.api_key = "your-api-key-here"

def plot_distributions(original_df, synthetic_df, columns):
    for col in columns:
        plt.figure(figsize=(6, 4))
        sns.kdeplot(original_df[col].dropna(), label="Original", fill=True)
        sns.kdeplot(synthetic_df[col].dropna(), label="Synthetic", fill=True)
        plt.title(f"Distribution Comparison for '{col}'")
        plt.legend()
        plt.tight_layout()
        plt.show()

def evaluate_column_with_llm(col_name, original_sample, synthetic_sample, model="gpt-4"):
    prompt = f"""
You are a data validation expert.

Compare the real and synthetic samples for the column '{col_name}'.

Original values:
{original_sample}

Synthetic values:
{synthetic_sample}

Please:
- Assess distribution similarity
- Identify potential anomalies or issues
- Score quality from 1 (poor) to 10 (excellent)
- Suggest improvements if needed

Respond in JSON:
{{"column": "{col_name}", "assessment": "...", "score": X, "suggestions": "..."}}
"""

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You evaluate synthetic data quality for each column."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=600
    )

    return response['choices'][0]['message']['content']

def evaluate_synthetic_data_full(original_df, synthetic_df, columns=None):
    if columns is None:
        columns = list(set(original_df.columns).intersection(set(synthetic_df.columns)))

    results = []

    print(f"\n📊 Plotting column distributions...")
    plot_distributions(original_df, synthetic_df, columns)

    print(f"\n🤖 Evaluating synthetic data column by column with LLM...\n")
    for col in columns:
        orig_sample = original_df[col].dropna().sample(min(10, len(original_df))).tolist()
        synth_sample = synthetic_df[col].dropna().sample(min(10, len(synthetic_df))).tolist()

        llm_output = evaluate_column_with_llm(col, orig_sample, synth_sample)
        
        try:
            result = json.loads(llm_output)
            results.append(result)
            print(f"✅ Column '{col}': Score {result['score']}")
        except json.JSONDecodeError:
            print(f"⚠️ Failed to parse JSON for column '{col}'. Raw response:\n{llm_output}")
    
    return results

# === Run Example ===
original_df = pd.read_csv("original_data.csv")
synthetic_df = pd.read_csv("synthetic_data.csv")

columns_to_evaluate = ["age", "income", "gender"]
evaluation_results = evaluate_synthetic_data_full(original_df, synthetic_df, columns_to_evaluate)

print("\n📝 Final Evaluation Report:")
for r in evaluation_results:
    print(json.dumps(r, indent=2))
