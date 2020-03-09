import os
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
from time import time

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')

if not os.getcwd().endswith("tmp_experiments"):
    os.chdir("./tmp_experiments")

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

X = df["message"].values
Y = df.iloc[:, 4:].values


def tokenize(text):
    """Tokenization function to process text data"""
    # Replace all URLs with a placeholder
    text_no_urls = re.sub(url_regex, "urlplaceholder", text)

    # Clean and normalize
    text_norm = re.sub(r"[^0-9a-zA-Z]", " ", text_no_urls.lower())

    # Tokenize and remove heading and trailing spaces
    tokens = [t.strip() for t in word_tokenize(text_norm)]

    # Lemmatize tokens and remove stopwords
    return [lemmatzr.lemmatize(t) for t in tokens if t not in eng_stopwords]


# 1. RANDOM FOREST

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# We have a multiple classifications per input text, that is, we have a two-dimensional target variable.
# Use the MultiOutputClassifier to train a different Classifier per target variable
clsfr = Pipeline([
    ("vctzr", CountVectorizer(analyzer="word")),
    ("tfidf", TfidfTransformer()),
    ("clsfr", MultiOutputClassifier(RandomForestClassifier()))
])

parameters = {
    "vctzr__tokenizer": [tokenize, None],
    "tfidf__smooth_idf": [True],
    "tfidf__use_idf": [True],
    # "tfidf__sublinear_tf": [True, False],
    "clsfr__estimator__criterion": ["gini", "entropy"],
    "clsfr__estimator__max_depth": [30, 50],
    # "clsfr__estimator__min_samples_split": [2, 4, 5],
    "clsfr__estimator__n_estimators": [75, 100],
    # "clsfr__estimator__warm_start": [False, True]
}

print(f"\nTunning parameters using GridSearch with Cross-Validation...")

cv = GridSearchCV(clsfr, param_grid=parameters, cv=5, n_jobs=-1)

cv.fit(X_train, Y_train)

print(f"- BEST PARAMETERS: \n{cv.best_params_}\n")

Y_pred = cv.predict(X_test)

# Compute f-score, recall and precision of each class
for j, column in enumerate(df.columns[4:]):
    cur_class_report = classification_report(Y_test[:, j], Y_pred[:, j], output_dict=False)
    print(f"Classification report for {column}: \n\n{cur_class_report}\n")

# Manually compute and display accuracies of each class. Note that accuracies are pretty good
accuracies = []

for j, column in enumerate(df.columns[4:]):
    right_preds = (Y_test[:, j] == Y_pred[:, j]).sum()
    total_preds = Y_test.shape[0]
    accuracies.append((column, (right_preds / total_preds)))

for column_name, accuracy in accuracies:
    print(f"- {column_name}. Accuracy: {accuracy}")

# Pickle the model. For maximum compatibility, open file in mode write-binary
with open(f"multioutput-randomforest-gridsearchcv-{round(time())}.pickle", mode="wb") as pickle_file:
    pickle.dump(clsfr, pickle_file)

# 1st run. Best Params:
# - BEST PARAMETERS:
# {'clsfr__estimator__criterion': 'gini', 'clsfr__estimator__n_estimators': 20,
# 'tfidf__smooth_idf': False, 'tfidf__use_idf': False}

# 2nd run. Best Params:
# - BEST PARAMETERS:
# {'clsfr__estimator__criterion': 'entropy', 'clsfr__estimator__max_depth': 20,
# 'clsfr__estimator__n_estimators': 50, 'tfidf__smooth_idf': False, 'tfidf__use_idf': True}

# 3rd run. Best Params:
# - BEST PARAMETERS:
# {'clsfr__estimator__criterion': 'entropy', 'clsfr__estimator__max_depth': 20,
# 'clsfr__estimator__n_estimators': 50, 'tfidf__smooth_idf': False, 'tfidf__use_idf': True,
# 'vctzr__tokenizer': <function tokenize at 0x7f1a5a60c8c0>}

# 4th run. Best Params:
# - BEST PARAMETERS:
# {'clsfr__estimator__criterion': 'entropy', 'clsfr__estimator__max_depth': 20, 'clsfr__estimator__n_estimators': 50,
# 'tfidf__smooth_idf': False, 'tfidf__use_idf': True, 'vctzr__tokenizer': None}

# 5th run. Best Params:
# {'clsfr__estimator__criterion': 'gini', 'clsfr__estimator__max_depth': 30, 'clsfr__estimator__n_estimators': 75,
# 'tfidf__smooth_idf': True, 'tfidf__use_idf': True, 'vctzr__tokenizer': <function tokenize at 0x7fe83a40d8c0>}

# 6th run. Best Params:
# {'clsfr__estimator__criterion': 'entropy', 'clsfr__estimator__max_depth': 50, 'clsfr__estimator__n_estimators': 100,
# 'tfidf__smooth_idf': True, 'tfidf__use_idf': True, 'vctzr__tokenizer': <function tokenize at 0x7f8cbb3808c0>}

print(tokenize)
