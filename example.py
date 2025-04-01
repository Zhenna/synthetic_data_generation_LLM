import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# Choose an open-source model
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# "mistralai/Mistral-7B-Instruct"  # Alternatives: "facebook/opt-1.3b", "gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

# Structured prompt for tabular data generation
prompt = """Generate 5 synthetic Chinese names in JSON format."""

# """Generate 5 synthetic customer transaction records in JSON format with the following fields:
# - customer_id: Unique integer starting from 2001
# - name: Random full name
# - age: Integer between 18 and 65
# - email: A valid email format
# - transaction_amount: Decimal between 10.00 and 500.00
# - transaction_date: In YYYY-MM-DD format

# Return the output strictly in JSON array format.

# Example Output:
# [
#   {"customer_id": 2001, "name": "Emily Johnson", "age": 29, "email": "emilyj@example.com", "transaction_amount": 78.50, "transaction_date": "2024-03-05"},
#   {"customer_id": 2002, "name": "Michael Smith", "age": 42, "email": "michaelsmith@example.com", "transaction_amount": 150.00, "transaction_date": "2024-03-08"}
# ]
# """

# Generate synthetic data
inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
output_tokens = model.generate(**inputs, max_length=500, do_sample=True, temperature=0.7, top_p=0.9)
generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

# print(generated_text)

# synthetic_data = json.loads(generated_text)
# print(synthetic_data)

# Parse JSON response
try:
    synthetic_data = json.loads(generated_text)
except json.JSONDecodeError:
    print("❌ Error parsing JSON output. Please refine the prompt.")
    synthetic_data = []

# # Save synthetic data to CSV
# import pandas as pd
# df_synthetic = pd.DataFrame(synthetic_data)
# # df_synthetic.to_csv("synthetic_output.csv", index=False)
# print(df_synthetic)

# print("✅ Synthetic data saved to synthetic_output.csv")
