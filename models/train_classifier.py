import re
import sys
import pickle
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Regex to match URLs
url_regex = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"

# Lemmatizer based on the WordNet corpus
lemmatzr = WordNetLemmatizer()

stopwords = stopwords.words("english")


def load_data(database_filepath):
    """
    Loads categorized messages from an Sqlite database file, specifically, from the Message table
    :database_filepath: Path of the Sqlite database to load from
    :return: Triple with the following components:
             X: Numpy array, messages loaded from the Message table
             Y:
    """
    engine = create_engine(f"sqlite:///{database_filepath}")
    df_messages = pd.read_sql_table("Message", engine)

    X = df_messages["message"].values
    Y = df_messages.iloc[:, 4:].values
    categories = [str(category_name) for category_name in df_messages.iloc[0, 4:].index]

    return X, Y, categories


def tokenize(text):
    """
    Cleans, normalizes and converts the text into an array of lemmatized tokens
    :text: Piece to be tokenized
    :return: List of tokens corresponding to the lemmatized words of the text
    """
    # Replace URLs with a fixed placeholder
    clean_text = re.sub(url_regex, "urlplaceholder", text)

    # Make the text all lowercase and clean it of any remaining special characters
    clean_text = re.sub(r"[^0-9a-zA-Z]", " ", clean_text.lower())

    # Convert the text into a list of tokens, each corresponding to a word and remove heading and trailing spaces
    word_tokens = [word.strip() for word in word_tokenize(clean_text)]

    return [lemmatzr.lemmatize(word) for word in word_tokens if word not in stopwords]


def build_model():
    """
    Builds a pipelined, supervised classification model. The resulting model receives a set of categorized text pieces
    and their corresponding categories as training input. Afterwards, the trained model predicts the category of a
    new piece of text. Note that the model automatically tokenizes and vectorizes the input text.
    The parameters of the model were tunned using GridSearchCV
    :return: Text classification model to predict the category of raw pieces of text
    """
    # Classifier to predict categories based on the vectorized text. The default kernel function of the SVC is RBF
    svc = SVC(class_weight="balanced", gamma="scale")

    # We have a multiple classifications per input text, that is, we have a two-dimensional target variable.
    # Use the MultiOutputClassifier to train a different Classifier per target variable
    clsfr = Pipeline([
        ("vctzr", CountVectorizer(tokenizer=tokenize)),
        ("tfidf", TfidfTransformer(use_idf=True, smooth_idf=True)),
        ("clsfr", MultiOutputClassifier(svc))
    ])

    return clsfr


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Computes metrics such as precision, recall, accuracy and f-score on the specified model and test dataset
    :param model: Model to be evaluated
    :param X_test: Test dataset. Independent variables
    :param Y_test: Test dataset. Dependent variables
    :param category_names: Array containing the names of the dependent variables
    :return: Nothing just prints out the results
    """
    # Predict categories on the test dataset
    Y_pred = model.predict(X_test)

    print("MODEL RESULTS")
    print("-----------------------------------------------------------------------------------------------------------")

    # Compute f-score, recall and precision of each class
    for j, category in enumerate(category_names):
        cur_category_report = classification_report(Y_test[:, j], Y_pred[:, j], output_dict=False)
        print(f"- {category}: \n\n{cur_category_report}\n")

    print("-----------------------------------------------------------------------------------------------------------")

    # Compute mean accuracy over all categories
    print(f"\nMEAN ACCURACY: {model.score(X_test, Y_test)}\n")


def save_model(model, model_filepath):
    """
    Stores the trained model in a pickle file, at the specified location
    :param model: Model to be saved
    :param model_filepath: Path to write the pickle file to
    :return: Nothing
    """
    with open(model_filepath, mode="wb") as pickle_file:
        pickle.dump(model, pickle_file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print(f"Loading data...\n    DATABASE: {database_filepath}")
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print("Building model...")
        model = build_model()
        
        print("Training model...")
        model.fit(X_train, Y_train)
        
        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print(f"Saving model...\n    MODEL: {model_filepath}")
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(f"Please provide the filepath of the disaster messages database "
              f"as the first argument and the filepath of the pickle file to "
              f"save the model to as the second argument. \n\nExample: python "
              f"train_classifier.py ../data/DisasterResponse.db classifier.pkl")


if __name__ == "__main__":
    main()
