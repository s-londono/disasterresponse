import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

# load messages dataset
messages = pd.read_csv("tmp_experiments/data/messages.csv")
messages.head()

# Load categories dataset
categories = pd.read_csv("tmp_experiments/data/categories.csv")
categories.head()

# Merge datasets.
# We merge on the id column, present on both datasets. We do an inner-join, so that neither uncategorized messages
# nor categorizations of missing messages are included in the result
df = messages.merge(categories, how="inner", on="id")

# Note that the resulting DataFrame has more records than the merged datasets. As we did a left-join,
# this means that some messages are associated to more than one categories record
print(f"Shapes: {messages.shape}, {categories.shape}, {df.shape}")

df.head()

# Create a Series with the total number of category elements that result from splitting each categories record
category_counts = categories["categories"].str.split(";").str.len()

# The fact that the min. and max. number of categories in all records are equal,
# suggests that all categories are present in each record
print(f"Max. number of categories: {category_counts.max()}. Min. number of categories: {category_counts.min()}")

# Create a dataframe of the 36 individual category columns
categories = df["categories"].str.split(";", expand=True)
categories.head()

# Use the first row to extract the name of each category.
# one way is to apply a lambda function that takes everything
# up to the second to last character of each string with slicing
row = categories.iloc[0, :]
category_colnames = row.apply(lambda e: e[:-2])
print(category_colnames[0:10])

# Rename the columns of `categories`
categories.columns = category_colnames
categories.head()

for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].str[-1:]

    # convert column from string to numeric
    categories[column] = categories[column].astype(np.int16)
categories.head()

# Drop the original categories column from `df`
df.drop("categories", axis=1, inplace=True)
df.head()

# concatenate the original dataframe with the new `categories` dataframe
df = pd.concat([df, categories], axis=1)
df.head()

# check number of duplicates
all_duplicates = df.duplicated().sum()

original_duplicates = df["original"].nunique()

message_duplicates = df["message"].nunique()

both_duplicates = df[["message", "original"]].duplicated().sum()

print(f"Shape: {df.shape}")
print(f"All columns are duplicates: {all_duplicates}")
print(f"Original column duplicates: {original_duplicates}")
print(f"Message column duplicates:  {message_duplicates}")
print(f"Message and Original column duplicates: {both_duplicates}")

# drop messages with no categorization
df.dropna(how="all", subset=df.columns[4:], inplace=True)

# drop duplicates
df.drop_duplicates(inplace=True)

# check number of duplicates
print(f"Shape: {df.shape}")
print(f"Duplicates now: {df.duplicated().sum()}")

# Load the clean dataset into an SQLite DB
engine = create_engine('sqlite:///data/DisasterResponse.db')
df.to_sql('Message', engine, index=False, if_exists='replace')

# Use the template file attached in the Resources folder to write a script that runs the steps above to create a
# database based on new datasets specified by the user. Alternatively, you can complete etl_pipeline.py in the
# classroom on the Project Workspace IDE coming later
