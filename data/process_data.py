import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads and merges messages and categories from the specified CSV files into a DataFrame
    :param messages_filepath: Path of the CSV file containing messages
    :param categories_filepath: Path of the CSV file containing categories
    :return: DataFrame containing messages and categories
    """
    # Load messages and categories datasets into DataFrames
    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)

    # Merge messages and categories into one DataFrame. Merge on the id column, present on both datasets.
    # Do an inner-join, so the result does not include uncategorized messages or categorizations of missing messages
    df_merged = df_messages.merge(df_categories, how="inner", on="id")

    # Make categories DataFrame made up of all the individual categories, which are originally in a single column
    categories = df_merged["categories"].str.split(";", expand=True)

    # Use the first row to extract the name of all categories.
    # Note that in the categories DataFrame, values are in the format: categoryname-n, with n a number
    category_column_names = categories.iloc[0, :].apply(lambda e: e[:-2])

    # Add a column per category to the categories DataFrame
    categories.columns = category_column_names

    # The actual value of the category is at the last character and is an integer
    for column in categories:
        categories[column] = categories[column].str[-1:]
        categories[column] = categories[column].astype(np.int16)

    # Now we can drop the original categories column from the DataFrame
    df_merged.drop("categories", axis=1, inplace=True)

    # Concatenate the original dataframe with the new `categories` dataframe
    df_result = pd.concat([df_merged, categories], axis=1)

    return df_result


def clean_data(df):
    """
    Cleans the DataFrame of categorized messages. That is removes missing data, useless columns and records
    :param df: DataFrame of categorized messages to be cleaned
    :return: Clean DataFrame of categorized messages
    """
    # Drop messages with no classification
    df_clean = df.dropna(how="all", subset=df.columns[4:])

    # Drop categories that have a single value
    single_valued_cols = [column for column, num_vals in df_clean.iloc[:, 4:].nunique().iteritems() if num_vals <= 1]
    df_clean.drop(labels=single_valued_cols, axis=1, inplace=True)

    # Drop duplicated messages
    print(f"Dropping a total of: {df.duplicated()} messages")
    df_clean.drop_duplicates(inplace=True)

    return df_clean


def save_data(df, database_filename):
    """
    Stores the DataFrame of messages in a database at the path specified
    :param df: DataFrame of categorized messages to be stored in a database
    :param database_filename: Path to the database file
    """
    db_engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql("Message", db_engine, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(f"Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}")
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)
        
        print(f"Saving data...\n    DATABASE: {database_filepath}")
        save_data(df, database_filepath)
        
        print("Cleaned data saved to database!")
    
    else:
        print(f"Please provide the filepaths of the messages and categories "
              f"datasets as the first and second argument respectively, as "
              f"well as the filepath of the database to save the cleaned data ",
              f"to as the third argument. \n\nExample: python process_data.py "
              f"disaster_messages.csv disaster_categories.csv "
              f"DisasterResponse.db")


if __name__ == "__main__":
    main()
