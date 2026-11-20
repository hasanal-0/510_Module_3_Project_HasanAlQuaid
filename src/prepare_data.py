"""
AIPI 510: Module 3 Project: Data/ML Engineering
- src/prepare_data.py
Hasan Al-Quaid
=================================================
Downloads the Diamonds dataset from Hugging Face, cleans it, and uploads the
cleaned file to Google Cloud Storage:
  1. Checks the raw data for missing values, bad sizes, and duplicates
  2. Removes the extra quote marks from the column names ('x' -> x)
  3. Removes the b'...' wrapping from the text columns (b'Ideal' -> Ideal)
  4. Removes diamonds with a length, width, or depth of 0 mm (impossible values)
  5. Removes exact duplicate rows
  6. Flags outliers with the IQR rule (reported, not removed)
  7. Removes diamonds with a size over 15 mm (impossible values found in the outlier report)
  8. Checks the clean data again, then saves and uploads it

Output:
    data/raw/diamonds.csv               (raw data that was downloaded)
    data/processed/diamonds_clean.csv   (cleaded data)
    gs://<bucket>/data/diamonds_clean.csv
"""
import argparse
import os
import pandas as pd
import yaml
from google.cloud import storage

# global values
TEXT_COLS = ["cut", "color", "clarity"]
SIZE_COLS = ["x", "y", "z"]
MAX_SIZE_MM = 15
OUTLIER_COLS = ["price", "carat", "depth", "table", "x", "y", "z"]


def download_data(source_url, raw_path):
    """
    Download the raw data and save an untouched copy

    Parameters:
        source_url (str): Link to the raw CSV on Hugging Face
        raw_path (str): Where to save the untouched copy

    Returns:
        pd.DataFrame: The raw data
    """
    print("-------------------------- Start: Download Raw Data ----------------------------")
    df = pd.read_csv(source_url)
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    df.to_csv(raw_path, index=False)
    print(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
    print(f"Saved untouched copy to: {raw_path}")
    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))
    print("-------------------------- End: Download Raw Data------------------------------\n")

    return df


def check_data(df, title):
    """
    Check the data for missing values, 0 mm sizes, and duplicate rows

    Parameters:
        df (pd.DataFrame): The diamonds data
        title (str): Label printed in the header, e.g. "Raw data checks"

    Returns:
        bool: True if all checks passed, else False
    """
    print(f"-------------------------- Start: {title} --------------------------")
    size_cols = [c for c in df.columns if c.strip("'") in SIZE_COLS]

    missing = df.isnull().sum().sum()
    zero_size = (df[size_cols] == 0).any(axis=1).sum()
    duplicates = df.duplicated().sum()

    if missing == 0 and zero_size == 0 and duplicates == 0:
        all_passed = True
    else:
        all_passed = False

    print(f"Rows: {len(df)}  |  Columns: {list(df.columns)}")
    print(f"Missing values:         {missing}")
    print(f"Rows with a 0 mm size:  {zero_size}")
    print(f"Duplicate rows:         {duplicates}")
    print(f"Price range:            ${df['price'].min():,.0f} to ${df['price'].max():,.0f}")
    print(f"All checks passed: {all_passed}")
    print(f"-------------------------- End: {title} ----------------------------\n")

    return all_passed


def clean_column_names(df):
    """
    Remove the extra quote marks from the column names

    Parameters:
        df (pd.DataFrame): The diamonds data

    Returns:
        pd.DataFrame: The data with clean column names
    """
    print("-------------------------- Start: Fixing column names ------------------------")
    # Looking at the raw file, the size columns are named 'x', 'y', 'z' (with quote marks),
    # which makes them awkward to use, so the quote marks are removed
    print(f"Before: {list(df.columns)}")
    df = df.copy()
    df.columns = [c.strip().strip("'") for c in df.columns]
    print(f"After:  {list(df.columns)}")
    print("-------------------------- End: Fixing column names --------------------------\n")

    return df


def clean_text_values(df, allowed_values):
    """
    Remove the b'...' wrapping from the cut, color, and clarity values

    Parameters:
        df (pd.DataFrame): The diamonds data
        allowed_values (dict): Valid values for each text column, from config.yaml

    Returns:
        pd.DataFrame: The data with clean text values
    """
    print("-------------------------- Start: Text values -------------------------")
    # Looking at the raw file, the text values are stored like b'Ideal' instead of Ideal,
    # which looks like leftover formatting from how the file was created
    print(f"Before: {sorted(df['cut'].unique())}")
    print(f"Before: {sorted(df['color'].unique())}")
    print(f"Before: {sorted(df['clarity'].unique())}")

    for col in TEXT_COLS:
        df[col] = df[col].astype(str).str.replace(r"^b'(.*)'$", r"\1", regex=True)
    print(f"\nAfter:  {sorted(df['cut'].unique())}")
    print(f"After:  {sorted(df['color'].unique())}")
    print(f"After:  {sorted(df['clarity'].unique())}")

    # verify every value is one of the allowed values in config.yaml, should all be True
    print("\nAll values valid (should all be True):")
    for col in TEXT_COLS:
        is_valid = df[col].isin(allowed_values[col]).all()
        print(f"  {col}: {is_valid}")
    print("-------------------------- End: Text values ---------------------------\n")

    return df


def remove_zero_size_rows(df):
    """
    Remove diamonds with a length, width, or depth of 0 mm

    Parameters:
        df (pd.DataFrame): The diamonds data

    Returns:
        pd.DataFrame: The data without the 0 mm rows
    """
    print("-------------------------- Start: Remove 0 mm sizes --------------------------")
    # A diamond can't have a size of 0 mm, so these are probably recording errors.
    # It is a very small part of the data, so I decided to drop them.
    zero_size = (df[SIZE_COLS] == 0).any(axis=1)
    rows_num_before = len(df)
    print(f"Rows with a 0 mm size: {zero_size.sum()}\n")
    print(df.loc[zero_size, ["carat", "price", "x", "y", "z"]].to_string())

    print("\ndf.shape:", df.shape)
    df = df[~zero_size]
    
    print(f"df.shape after (expect {zero_size.sum()} fewer rows):", df.shape)
    print(f"{rows_num_before} - {zero_size.sum()} = {len(df)}: {rows_num_before - zero_size.sum() == len(df)}")
    print("-------------------------- End: Remove 0 mm sizes ----------------------------\n")
    return df


def remove_duplicate_rows(df):
    """
    Remove exact duplicate rows

    Parameters:
        df (pd.DataFrame): The diamonds data

    Returns:
        pd.DataFrame: The data without duplicates
    """
    print("-------------------------- Start: Duplicates --------------------------")
    # Duplicates could end up in both the training and test sets, which would make
    # the test results look better than they really are, so they are removed
    duplicates = df.duplicated()
    rows_num_before = len(df)
    print(f"Duplicate rows: {duplicates.sum()}")
    print("df.shape:", df.shape)
    df = df[~duplicates].reset_index(drop=True)
    print(f"df.shape after (expect {duplicates.sum()} fewer rows):", df.shape)
    print(f"{rows_num_before} - {duplicates.sum()} = {len(df)}: {rows_num_before - duplicates.sum() == len(df)}")
    print("-------------------------- End: Duplicates ----------------------------\n")

    return df


def report_outliers(df):
    """
    Flag outliers in each numeric column with the IQR rule and print a summary.
    Outliers are reported, not removed. Most of them are large, expensive diamonds,
    since prices are skewed (most diamonds are cheap, a few are very expensive).
    They are real diamonds, not errors, and removing them would stop the model
    from learning to predict prices for expensive stones.

    Parameters:
        df (pd.DataFrame): The diamonds data

    Returns:
        None
    """
    print("-------------------------- Start: Outliers (IQR rule) ------------------")
    print("Normal range for each column (Q1 - 1.5 x IQR to Q3 + 1.5 x IQR):")
    for col in OUTLIER_COLS:
        q1, q3, iqr, lower, upper = iqr_range(df[col])
        n_outliers = flag_outliers(df[col]).sum()
        print(f"\n{col}:")
        print(f"  Q1:           {q1:.2f}")
        print(f"  Q3:           {q3:.2f}")
        print(f"  IQR:          {iqr:.2f}")
        print(f"  Normal range: {lower:.2f} to {upper:.2f}")
        print(f"  Outliers:     {n_outliers} of {len(df)} rows (kept in the data, not removed)")
    print("-------------------------- End: Outliers -------------------------------\n")

def iqr_range(values):
    """
    Calculate the normal range for one column using the IQR rule,
    with k being 1.5 as its conventional (Tukey's rule).

    Parameters:
        values (pd.Series): The values of one column

    Returns:
        tuple: (q1, q3, iqr, lower, upper)
    """
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return q1, q3, iqr, lower, upper


def flag_outliers(values):
    """
    Flag outliers in one column using the IQR rule

    Parameters:
        values (pd.Series): The values of one column

    Returns:
        pd.Series: True where a value is outside the normal range
    """
    q1, q3, iqr, lower, upper = iqr_range(values)
    too_low = values < lower
    too_high = values > upper

    return too_low | too_high

 
def remove_impossible_sizes(df):
    """
    Remove diamonds with a length, width, or depth over 15 mm
 
    Parameters:
        df (pd.DataFrame): The diamonds data
 
    Returns:
        pd.DataFrame: The data without the impossible sizes
    """
    print("-------------------------- Start: Remove impossible sizes --------------------------")
    # A few values are impossible, the largest diamond in the data (5.01 carats) is only about 10.7 mm, so any size over
    # 15 mm can't be real. For example, a 0.51 carat diamond listed as 31.8 mm has its other
    # sides at about 5.1 mm, so 31.8 is most likely a typo. These are recording errors, so I decided to drop them.
    too_big = (df[SIZE_COLS] > MAX_SIZE_MM).any(axis=1)
    rows_before = len(df)
    print(f"Rows with a size over {MAX_SIZE_MM} mm: {too_big.sum()}\n")
    print(df.loc[too_big, ["carat", "price", "x", "y", "z"]].sort_index().to_string())
 
    print("\ndf.shape:", df.shape)
    df = df[~too_big]
 
    print(f"df.shape after (expect {too_big.sum()} fewer rows):", df.shape)
    print(f"{rows_before} - {too_big.sum()} = {len(df)}: {rows_before - too_big.sum() == len(df)}")
    print("-------------------------- End: Remove impossible sizes ----------------------------\n")
    return df

def save_clean_data(df, processed_path):
    """
    Save the cleaned data locally

    Parameters:
        df (pd.DataFrame): The cleaned diamonds data
        processed_path (str): Where to save the cleaned CSV

    Returns:
        None
    """
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"Saved {processed_path}  ({df.shape[0]} rows, {df.shape[1]} columns)\n")


def upload_to_gcs(local_path, project, bucket, gcs_path):
    """
    Upload file to Google Cloud Storage

    Parameters:
        local_path (str): File to upload
        project (str): Google Cloud project ID
        bucket (str): GCS bucket name
        gcs_path (str): Destination path inside the bucket

    Returns:
        None
    """
    print("-------------------------- Start: Upload to GCS -----------------------")
    blob = storage.Client(project=project).bucket(bucket).blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to gs://{bucket}/{gcs_path}")
    print(f"\nPublic URL: {blob.public_url}")


    # verify the file is now in the bucket, should be True
    print(f"File exists in bucket (should be True): {blob.exists()}")
    print("-------------------------- End: Upload to GCS -------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="Download, clean, and upload the diamonds data.")
    parser.add_argument("--config", default="config.yaml", help="Path to the config file")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    ds, st = cfg["dataset"], cfg["storage"]

    # download the raw data and keep an untouched copy
    df = download_data(ds["source_url"], ds["raw_path"])

    # do an initial data check for missing values, bad sizes, and duplicates
    check_data(df, "Initial raw data checks")

    # fix the formatting problems found in the raw file
    df = clean_column_names(df)
    df = clean_text_values(df, ds["features"]["ordinal"])

    # remove the bad rows
    df = remove_zero_size_rows(df)
    df = remove_duplicate_rows(df)

    # flag outliers, these are reported but kept in the data
    report_outliers(df)

    # the outlier functions shows a few impossible sizes (over 15 mm), so I will remove those
    df = remove_impossible_sizes(df)

    # check again, everything should pass now
    check_data(df, "Clean data checks")

    # save the clean data and 
    save_clean_data(df, ds["processed_path"])

    # upload it to the cloud
    if st["use_gcs"]:
        upload_to_gcs(ds["processed_path"], st["project"], st["bucket"], st["data_path"])


if __name__ == "__main__":
    main()
