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


def compute_accuracies(df, Y_test, Y_pred):
    # Manually compute and display accuracies of each class
    accuracies = []

    for j, column in enumerate(df.columns[4:]):
        right_preds = (Y_test[:, j] == Y_pred[:, j]).sum()
        total_preds = Y_test.shape[0]
        accuracies.append((column, (right_preds / total_preds)))

    for column_name, accuracy in accuracies:
        print(f"- {column_name}. Accuracy: {accuracy}")


def compute_fscores(df, Y_test, Y_pred):
    # Compute f-score, recall and precision of each class
    for j, column in enumerate(df.columns[4:]):
        cur_class_report = classification_report(Y_test[:, j], Y_pred[:, j])
        print(f"Classification report for {column}: \n\n{cur_class_report}")


def fit_simple_pipeline():
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    # We have a multiple classifications per input text, that is, we have a two-dimensional target variable.
    # Use the MultiOutputClassifier to train a different Classifier per target variable
    pipeline = Pipeline([
        ('vectr', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clsfr', MultiOutputClassifier(RandomForestClassifier(n_estimators=5)))
    ])

    pipeline.fit(X_train, Y_train)

    Y_simple_pred = pipeline.predict(X_test)


def tune_pipeline(X_train, X_test, Y_train, Y_test, pipeline, parameters):
    tunned_pipeline = GridSearchCV(pipeline, param_grid=parameters, cv=5)
    tunned_pipeline.fit(X_train, Y_train)

    return tunned_pipeline


# 1. RANDOM FOREST

def tune_random_forest_pipeline():
    # We have a multiple classifications per input text, that is, we have a two-dimensional target variable.
    # Use the MultiOutputClassifier to train a different Classifier per target variable
    pipeline = Pipeline([
        ('vectr', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clsfr', MultiOutputClassifier(RandomForestClassifier(n_estimators=5)))
    ])

    # Print parameters
    # print(pipeline.get_params())

    parameters = {
        "tfidf__smooth_idf": [True, False],
        "tfidf__use_idf": [True, False],
        # "tfidf__sublinear_tf": [True, False],
        "clsfr__estimator__criterion": ["gini", "entropy"],
        # "clsfr__estimator__max_depth": [None, 10, 20, 100],
        # "clsfr__estimator__min_samples_split": [2, 4, 5],
        "clsfr__estimator__n_estimators": [10, 20],
        # "clsfr__estimator__warm_start": [False, True]
    }

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    tunned_random_forest = tune_pipeline(X_train, X_test, Y_train, Y_test, pipeline, parameters)

    # score = tunned_random_forest.score(X_test, Y_test)
    # print(f"Mean score Random Forest: {score}")

    Y_pred = tunned_random_forest.predict(X_test)

    compute_fscores(df, Y_test, Y_pred)


tune_random_forest_pipeline()
