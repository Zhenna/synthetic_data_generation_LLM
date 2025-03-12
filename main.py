# %%
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# %%
# Load an open-source LLM from Hugging Face
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# "mistralai/Mistral-7B-Instruct"  # Alternative: "facebook/opt-1.3b", "gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

# %%
# Function to generate synthetic data using the model
def generate_synthetic_data(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    output_tokens = model.generate(**inputs, max_length=max_length, do_sample=True, temperature=0.7, top_p=0.9)
    return tokenizer.decode(output_tokens[0], skip_special_tokens=True)

# %%
# Load the CSV file
input_file = "AIL-5-minute-data copy.csv"
# "AIL-5-minute-data.csv"
# "friends/friends_table.csv"
# "input_data.csv"  # Change to your input file
df = pd.read_csv(input_file)

# %%
print(df)

# %%
# for col in df.columns:
#     print(col)

'''
StartTime
EndTime
Actual Internal Load
'''
# %%
# Generate synthetic rows
synthetic_rows = []
for _, row in df.iterrows():
    prompt = f"First, analyze the dataset to understand its schema, constraints, and patterns."
    # f"Generate a synthetic customer transaction record similar to:\nCustomer ID: {row['customer_id']}, Name: {row['name']}, Age: {row['age']}, Email: {row['email']}, Transaction: ${row['transaction_amount']} on {row['transaction_date']}."
    synthetic_text = generate_synthetic_data(prompt)
    synthetic_rows.append(synthetic_text)

# %%
# Convert synthetic data into DataFrame
df_synthetic = pd.DataFrame({"synthetic_record": synthetic_rows})

print(df_synthetic)
# # Save the synthetic dataset
# output_file = "synthetic_output.csv"
# df_synthetic.to_csv(output_file, index=False)

# print(f"✅ Synthetic data saved to {output_file}")
