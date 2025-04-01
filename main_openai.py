
import json
import os
from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# print(f"OpenAI API Key: {openai.api_key}")

def generate_synthetic_data(prompt, model="gpt-4", temperature=0.7, max_tokens=500):
    response = client.chat.completions.create(model=model,
    messages=[
        {"role": "system", "content": "You are a data generator that outputs strictly JSON format."},
        {"role": "user", "content": prompt}
    ],
    temperature=temperature,
    max_tokens=max_tokens)
    message = response.choices[0].message.content
    try:
        # Validate that the output is valid JSON
        json_data = json.loads(message)
        return json_data
    except json.JSONDecodeError:
        print("Response was not valid JSON:")
        print(message)
        return None

# Example prompt to generate synthetic user profiles
prompt = """
Generate 5 synthetic user profiles with the following fields:
- id (integer)
- name (string)
- email (string)
- age (integer)
- signup_date (ISO format YYYY-MM-DD)

Output in JSON format as a list of user objects.
"""

# age (integer between 18 and 65)

# Generate the data
synthetic_data = generate_synthetic_data(prompt)

# Print the output
print(json.dumps(synthetic_data, indent=2))

# convert to CSV
import pandas as pd
df_synthetic = pd.DataFrame(synthetic_data)
# df_synthetic.to_csv("synthetic_output.csv", index=False)
# print("✅ Synthetic data saved to synthetic_output.csv")


# print every column to list
for col in df_synthetic.columns:
    print(col)
    print(df_synthetic[col].to_list())