import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Set your API key
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def generate_values_from_profile(profile_text, num_values=5):
    prompt = f"""
        Based on the following data profile report, generate {num_values} synthetic values 
        that match the same value type, distribution, pattern, and statistical properties.

        Data Profile:
        {profile_text}

        Only return the generated values as a JSON array.
        """

    response = client.chat.completions.create(model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a data generation assistant."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.4,
    max_tokens=200)

    return response.choices[0].message.content

if __name__ == "__main__":

    # Example profile of an 'age' field
    example_profile = """
    Summary Statistics:
    - Count: 1000
    - Min: 18
    - Max: 65
    - Mean: 37.5
    - Median: 36
    - Standard Deviation: 10.2
    - Most common values: 25, 30, 35, 40

    Pattern:
    - Normally distributed with a peak around ages 30–40
    """

    # Generate values
    synthetic_values = generate_values_from_profile(example_profile)
    # Print result
    print("Synthetic Values:", synthetic_values)
    # Convert to JSON
    
    try:
        synthetic_data = json.loads(synthetic_values)
        print("Type:", type(synthetic_data))
    except json.JSONDecodeError:
        print("Response was not valid JSON:")
        print(synthetic_values)
        synthetic_data = None

    
    
