import os
import argparse
import pandas as pd
import random
import json

from column_profiler import profile_values
from data_generator import generate_values_from_profile


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate synthetic data from a CSV file using LLM."
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        required=True,
        help="Directory containing CSV files",
    )
    parser.add_argument(
        "-o",
        "--save_output",
        type=bool,
        required=False,
        help="If True, the output will be saved to the default directory output/ with the original table name. If none is provided, the output will be not be saved.",
    )
    parser.add_argument(
        "-n",
        "--num_rows",
        type=int,
        required=False,
        default=5,
        help="Number of rows to generate. Please test with a small row number like 5 first.",
    )
    parser.add_argument(
        "-s",
        "--sample_size",
        type=int,
        required=False,
        default=10,
        help="Sample size for each column. Please keep this number small.",
    )
    parser.add_argument(
        "-c",
        "--columns2drop",
        type=str,
        required=False,
        nargs="+",
        help="Exclude columns to save money.",
    )

    args = parser.parse_args()

    # loop through csv file in the selected directory
    for filename in os.listdir(args.directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(args.directory, filename)
            print(f"Processing file: {file_path}")

            # Read the CSV file of real data
            df = pd.read_csv(file_path)
            # create a dataframe with the same columns as the original dataframe
            synthetic_data = pd.DataFrame(columns=df.columns)

            # Loop through each column in the DataFrame
            for col in df.columns:
                if col not in args.columns2drop:

                    values = df[col].tolist()
                    # if length of values is large, randomly sample fewer values
                    if len(values) > args.sample_size:
                        values = random.sample(values, args.sample_size)
                    # show warning if length of value is less than 5
                    elif len(values) < 5:
                        print(
                            f"Warning: {col} has less than 5 values. Sample size is too small."
                        )

                    try:
                        profile_report = profile_values(values, description="")
                        # print(profile_report)
                    except Exception as e:
                        print(e)
                        print(f"Error profiling column {col}.")
                        profile_report = ""

                    try:
                        # Generate values
                        synthetic_values = generate_values_from_profile(
                            profile_report, num_values=args.num_rows
                        )
                        # print("Synthetic Values:", synthetic_values)
                    except Exception as e:
                        print("Error generating values:", e)
                        synthetic_values = None

                    try:
                        # convert Json
                        synthetic_values = json.loads(synthetic_values)
                        # append to dataframe
                        print(f"Saving Column {col} to synthetic data...")
                        synthetic_data[col] = pd.DataFrame(synthetic_values)
                    except json.JSONDecodeError:
                        print("Response was not valid JSON:")
                        synthetic_data[col] = pd.NaT

                else:
                    print(f"Skipping column: {col}.")
                    synthetic_data[col] = pd.NaT

            # print the synthetic data
            print(synthetic_data)

            # # save the synthetic data to a csv file
            if args.save_output:
                os.makedirs("output", exist_ok=True)

                print(
                    f"Saving synthetic data to output/synthetic_output_LLM_{filename}..."
                )
                # save the synthetic data to a csv file
                synthetic_data.to_csv(
                    os.path.join("output", f"synthetic_output_LLM_{filename}"),
                    index=False,
                )
