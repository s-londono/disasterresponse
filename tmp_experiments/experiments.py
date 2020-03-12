import re
import pandas as pd
import nltk
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics.classification import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
import pickle

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')

# Regex to find URLs
url_regex = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"

# Lemmatizer & StopWords
lemmatzr = WordNetLemmatizer()
eng_stopwords = stopwords.words("english")

# load data from database
engine = create_engine("sqlite:///data/DisasterResponse.db")
df = pd.read_sql_table("Message", engine)
df.head()

# Note that for all target variables, false (0.0) observations are heavily under-represented
for column in df.columns[4:]:
    print(f"\nColumn: {column}")
    print(df[column].value_counts())

# Count messages per category
df_total_msgs_x_category = df.iloc[:, 4:].sum()

category_names = [str(ix) for ix in df_total_msgs_x_category.index]

msg_lengths = df["message"].apply(len)

df_msg_genre = df.loc[:, ["message", "genre"]]
df_msg_genre["msg_length"] = df_msg_genre["message"].apply(len)

group_lens = df_msg_genre[["msg_length", "genre"]].groupby(by="genre").groups

for genre in df["genre"].unique():
    print(f"{genre} - {df[df['genre'] == genre]['message'].apply(len)}")
