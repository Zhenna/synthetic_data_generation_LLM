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

        Only return the generated values in JSON format.
        """

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",  # "gpt-4" does not support json output
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a data generation assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=1000,
    )

    return response.choices[0].message.content


if __name__ == "__main__":

    example_profile = """
    Value Type: 
    The given list of values appears to be a list of full names. 

    Summary Statistics: 
    - Count: The list contains 100 names. 
    - Unique: There are 97 unique names in the list, suggesting that there are 3 duplicates. 

    Patterns or Distribution Notes: 
    - The names do not follow a specific pattern or distribution as they are categorical and non-numeric. 
    - The names appear to be from various cultural backgrounds, suggesting a diverse dataset. 

    Data Quality Issues or Anomalies: 
    - There are 3 duplicate names in the list which may be an issue if each entry is supposed to represent a unique individual. 
    - The list contains full names which might pose a privacy concern if not handled properly.
    - There are no apparent anomalies such as null values, incorrect data type, or inconsistent formatting in the list. However, without additional context, it's hard to definitively identify all potential issues. 

    """

    try:
        synthetic_values = generate_values_from_profile(example_profile)
        print("Synthetic Values:", synthetic_values)
    except Exception as e:
        print("Error generating values:", e)

    try:
        synthetic_data = json.loads(synthetic_values)
        print("Parsing JSON response...")
    except json.JSONDecodeError:
        print("Response was not valid JSON:")
        synthetic_data = None

    # print("Type:", type(synthetic_values))
    # print(synthetic_values)
