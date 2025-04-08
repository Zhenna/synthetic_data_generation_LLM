import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def profile_values(values, description=""):
    prompt = f"""
        You are a data analyst. Profile the following list of values.
        {description if description else ""}

        List of values:
        {values}

        Provide:
        - Value type (e.g., name, email, date in a specific datetime format, age, etc)
        - Summary statistics (e.g., count, unique, min, max, avg if numeric)
        - Patterns or distribution notes (e.g., normal, skewed, whether sequential if numeric)
        - Data quality issues or anomalies

        Output in plain text.
        """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant specialized in data analysis.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=500,
    )
    return response.choices[0].message.content


if __name__ == "__main__":

    # Example list: can be numeric, categorical, dates, etc.
    sample_values = [
        "johndoe@gmail.com",
        "janesmith@gmail.com",
        "bob.johnson@gmail.com",
        "alice.williams@gmail.com",
        "charlie.brown@gmail.com",
    ]

    # ['2020-01-01', '2021-02-15', '2019-03-20', '2020-11-05', '2021-06-30']
    # ['johndoe@gmail.com', 'janesmith@gmail.com', 'bob.johnson@gmail.com', 'alice.williams@gmail.com', 'charlie.brown@gmail.com']
    # ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Williams', 'Charlie Brown']
    # [22, 25, 25, 30, 30, 30, 1000, 35, 40, None, 45]

    # Call the function
    profile_report = profile_values(
        sample_values, description=""
    )  # This is a list of ages collected from a survey.")

    # Print result
    print(profile_report)
