# %%
import pandas as pd
import random

from data_profiler import profile_values
from data_generator import generate_values_from_profile

# %%
# 1. analyze the csv file column by column, to understand its column type, range constraints, and patterns

# %%
# Load the CSV file
input_file = "synthetic_output.csv"
# "AIL-5-minute-data copy.csv"
# "AIL-5-minute-data.csv"
# "friends/friends_table.csv"
# "input_data.csv"  # Change to your input file
df = pd.read_csv(input_file)
print(df)


# %%

# create a dataframe with the same columns as the original dataframe
synthetic_values = pd.DataFrame(columns=df.columns)
# Loop through each column in the DataFrame
# and generate synthetic values based on the profile
# of the original values

for col in df.columns:

    values = df[col].tolist()

    # if length of values is larger than 1000, randomly sample 1000 values
    if len(values) > 1000:
        values = random.sample(values, 1000)
    # show warning if length of value is less than 5
    elif len(values) < 5:
        print(f"Warning: {col} has less than 5 values. Sample size is too small.")


    # Generate 5 synthetic values for each column
    synthetic_values[col] = generate_values_from_profile(profile_values(df[col].tolist()), num_values=5)

    # Call the function
    profile_report = profile_values(values, description="") #This is a list of ages collected from a survey.")

    # Print result
    print(profile_report)

    # Generate values
    synthetic_values = generate_values_from_profile(profile_report)

    # Print result
    print("Synthetic Values:", synthetic_values)
    print(type(synthetic_values))